"""
Run Evaluation — Script CLI pour lancer l'évaluation complète AgriConnect.
==========================================================================

Usage :
    # Évaluation complète (tous les agents, toutes les catégories)
    python scripts/run_evaluation.py

    # Seulement un agent
    python scripts/run_evaluation.py --agent sentinelle

    # Seulement les tests de sécurité
    python scripts/run_evaluation.py --category security

    # Uploader les datasets vers LangSmith (sans exécuter)
    python scripts/run_evaluation.py --upload-only

    # Mode verbeux
    python scripts/run_evaluation.py -v

Pré-requis :
    - .env configuré avec LANGCHAIN_TRACING_V2=true et LANGCHAIN_API_KEY
    - Agents fonctionnels (Groq API key valide)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Setup path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.langsmith.datasets import ALL_DATASETS, get_dataset_stats, get_flat_examples
from tests.langsmith.evaluators import (
    compute_aggregate_score,
    run_all_evaluators,
)

logger = logging.getLogger("run_evaluation")


# ======================================================================
# UPLOAD DATASETS VERS LANGSMITH
# ======================================================================

def upload_datasets_to_langsmith():
    """Pousse tous les datasets vers LangSmith."""
    from agriconnect.core.tracing import get_ls_client, get_or_create_dataset

    client = get_ls_client()
    if client is None:
        logger.error("❌ LangSmith non configuré. Vérifiez votre .env")
        return False

    for dataset_name, examples in ALL_DATASETS.items():
        ds_full_name = f"agriconnect-{dataset_name}"
        ds = get_or_create_dataset(
            ds_full_name,
            description=f"AgriConnect eval: {dataset_name} ({len(examples)} examples)",
        )
        if ds is None:
            logger.error("❌ Impossible de créer le dataset: %s", ds_full_name)
            continue

        uploaded = 0
        for ex in examples:
            try:
                client.create_example(
                    dataset_id=ds.id,
                    inputs=ex["inputs"],
                    outputs=ex.get("expected", {}),
                )
                uploaded += 1
            except Exception as e:
                logger.debug("Skip example: %s", e)

        logger.info("✅ %s: %d/%d examples uploadés", ds_full_name, uploaded, len(examples))

    return True


# ======================================================================
# ÉVALUATION COMPLÈTE
# ======================================================================

def run_evaluation(
    category_filter: Optional[str] = None,
    agent_filter: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Exécute l'évaluation complète.

    Returns:
        Dict avec les résultats agrégés par catégorie.
    """
    from agriconnect.graphs.message_flow import MessageResponseFlow

    logger.info("🚀 Initialisation de l'orchestrateur...")
    orchestrator = MessageResponseFlow()
    logger.info("✅ Orchestrateur prêt")

    # Filtrer les exemples
    all_examples = get_flat_examples()
    if category_filter:
        all_examples = [ex for ex in all_examples if ex["category"] == category_filter]
    if agent_filter:
        all_examples = [ex for ex in all_examples if ex["category"] == agent_filter]

    if not all_examples:
        logger.error("❌ Aucun exemple trouvé pour le filtre donné")
        return {}

    logger.info("📋 %d exemples à évaluer", len(all_examples))

    results_by_category: Dict[str, List[Dict]] = {}
    total_pass = 0
    total_fail = 0
    total_errors = 0

    # Délai entre tests pour éviter les rate-limits Groq (429)
    INTER_TEST_DELAY = 10  # secondes
    MAX_RETRIES = 3        # tentatives par test
    RATE_LIMIT_WAIT = 30   # attente après 429

    for i, ex in enumerate(all_examples, 1):
        category = ex["category"]
        inputs = ex["inputs"]
        expected = ex["expected"]

        # Construire le state
        query_key = "user_query" if category in ("sentinelle", "formation", "market", "marketplace") else "requete_utilisateur"
        query = inputs.get(query_key, inputs.get("requete_utilisateur", ""))
        if not query:
            continue

        # Skip cas spéciaux
        if expected.get("status") == "ERROR":
            continue

        state = {
            "requete_utilisateur": query,
            "zone_id": inputs.get("zone_id", inputs.get("location_profile", {}).get("village", "Bobo")),
            "crop": inputs.get("crop", inputs.get("learner_profile", {}).get("culture_actuelle", "Maïs")),
            "user_level": inputs.get("user_level", inputs.get("learner_profile", {}).get("niveau", "debutant")),
            "user_phone": inputs.get("user_phone", ""),
        }

        logger.info("[%d/%d] %s — '%s...'", i, len(all_examples), category, query[:50])

        # Retry loop pour gérer les rate-limits Groq (429)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start = time.perf_counter()
                result = orchestrator.run(state)
                duration = time.perf_counter() - start

                evaluations = run_all_evaluators(result, expected, inputs, duration)
                aggregate = compute_aggregate_score(evaluations)

                entry = {
                    "query": query[:80],
                    "grade": aggregate["grade"],
                    "score": aggregate["aggregate_score"],
                    "duration": round(duration, 1),
                    "details": {k: v["score"] for k, v in aggregate["details"].items()},
                    "response_preview": (result.get("final_response", "") or "")[:200],
                    "execution_path": result.get("execution_path", []),
                }

                results_by_category.setdefault(category, []).append(entry)

                icon = "✅" if entry["score"] >= 0.5 else "❌"
                logger.info(
                    "  %s [%s] score=%.2f ⏱️%.1fs",
                    icon, entry["grade"], entry["score"], duration,
                )

                if verbose and entry["score"] < 0.7:
                    weak = {k: v for k, v in entry["details"].items() if v < 0.7}
                    if weak:
                        logger.info("    ⚠️  Faiblesses: %s", weak)

                if entry["score"] >= 0.5:
                    total_pass += 1
                else:
                    total_fail += 1

                # Pause entre tests pour respecter les rate-limits Groq
                if i < len(all_examples):
                    time.sleep(INTER_TEST_DELAY)

                break  # succès, sortir de la boucle retry

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate" in err_str or "too many" in err_str

                if is_rate_limit and attempt < MAX_RETRIES:
                    wait = RATE_LIMIT_WAIT * attempt
                    logger.warning(
                        "  ⏳ Rate-limit (429) — tentative %d/%d, attente %ds...",
                        attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue  # retry

                logger.error("  💥 Erreur: %s", e)
                results_by_category.setdefault(category, []).append({
                    "query": query[:80],
                    "grade": "F",
                    "score": 0.0,
                    "error": str(e),
                })
                total_errors += 1
                total_fail += 1
                # Pause plus longue après erreur
                if i < len(all_examples):
                    time.sleep(RATE_LIMIT_WAIT)
                break  # échec définitif

    return {
        "results_by_category": results_by_category,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "total_errors": total_errors,
        "timestamp": datetime.now().isoformat(),
    }


# ======================================================================
# RAPPORT
# ======================================================================

def generate_report(evaluation: Dict[str, Any]) -> str:
    """Génère un rapport textuel à partir des résultats."""
    results = evaluation.get("results_by_category", {})
    total_pass = evaluation.get("total_pass", 0)
    total_fail = evaluation.get("total_fail", 0)
    total = total_pass + total_fail

    lines = [
        "=" * 72,
        "  🌾 RAPPORT D'ÉVALUATION AGRICONNECT — LangSmith",
        f"  📅 {evaluation.get('timestamp', 'N/A')}",
        "=" * 72,
        "",
        f"  Total: {total} tests",
        f"  ✅ Pass (≥0.5): {total_pass}",
        f"  ❌ Fail (<0.5): {total_fail}",
        f"  💥 Erreurs: {evaluation.get('total_errors', 0)}",
        f"  📊 Taux de réussite: {total_pass / max(1, total) * 100:.1f}%",
        "",
    ]

    # Par catégorie
    for category, entries in sorted(results.items()):
        avg = sum(e["score"] for e in entries) / len(entries) if entries else 0
        cat_icon = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.5 else "❌"
        lines.append(f"\n{cat_icon} {category.upper()} — {len(entries)} tests, moyenne: {avg:.2f}")
        lines.append("-" * 60)

        for e in entries:
            icon = "✅" if e["score"] >= 0.5 else "❌"
            lines.append(
                f"  {icon} [{e['grade']}] {e['query'][:55]:<55} "
                f"{e['score']:.2f} "
                f"{'⏱️' + str(e.get('duration', '?')) + 's' if 'duration' in e else ''}"
            )

            if e.get("error"):
                lines.append(f"      💥 {e['error'][:70]}")
            elif e["score"] < 0.7 and "details" in e:
                weak = {k: v for k, v in e["details"].items() if v < 0.7}
                if weak:
                    lines.append(f"      ⚠️ {weak}")

    # Scores moyens par critère
    all_scores: Dict[str, List[float]] = {}
    for entries in results.values():
        for e in entries:
            for k, v in e.get("details", {}).items():
                all_scores.setdefault(k, []).append(v)

    lines.append("\n" + "=" * 72)
    lines.append("📊 SCORES MOYENS PAR CRITÈRE D'ÉVALUATION:")
    lines.append("-" * 40)
    for criterion, scores in sorted(all_scores.items(), key=lambda x: sum(x[1]) / len(x[1])):
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        icon = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.5 else "❌"
        lines.append(f"  {icon} {criterion:<20} {bar} {avg:.2f}")

    # Recommandations
    lines.append("\n" + "=" * 72)
    lines.append("💡 RECOMMANDATIONS:")
    lines.append("-" * 40)

    for criterion, scores in all_scores.items():
        avg = sum(scores) / len(scores)
        if avg < 0.5:
            lines.append(f"  🔴 {criterion}: Score critique ({avg:.2f}) — action urgente requise")
        elif avg < 0.7:
            lines.append(f"  🟡 {criterion}: Score insuffisant ({avg:.2f}) — amélioration nécessaire")

    # Faiblesses par agent
    for category, entries in results.items():
        failures = [e for e in entries if e["score"] < 0.5]
        if len(failures) > len(entries) * 0.3:
            lines.append(f"  🔴 Agent '{category}': {len(failures)}/{len(entries)} échecs — revoir cet agent")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🌾 AgriConnect — Évaluation complète des agents avec LangSmith",
    )
    parser.add_argument(
        "--agent",
        choices=["sentinelle", "formation", "market", "marketplace"],
        help="Évaluer un seul agent",
    )
    parser.add_argument(
        "--category",
        choices=list(ALL_DATASETS.keys()),
        help="Évaluer une seule catégorie de tests",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Uploader les datasets vers LangSmith sans exécuter",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mode verbeux",
    )
    parser.add_argument(
        "-o", "--output",
        default="evaluation_report.txt",
        help="Chemin du fichier de rapport (défaut: evaluation_report.txt)",
    )
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Stats
    stats = get_dataset_stats()
    logger.info("📋 Datasets chargés: %s", stats)

    # Upload only
    if args.upload_only:
        logger.info("📤 Upload des datasets vers LangSmith...")
        success = upload_datasets_to_langsmith()
        sys.exit(0 if success else 1)

    # Évaluation
    logger.info("🚀 Lancement de l'évaluation...")
    evaluation = run_evaluation(
        category_filter=args.category,
        agent_filter=args.agent,
        verbose=args.verbose,
    )

    # Rapport
    report = generate_report(evaluation)
    print(report)

    # Sauvegarder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(project_root, args.output)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("📝 Rapport sauvegardé: %s", report_path)

    # Sauvegarder aussi le JSON brut
    json_path = report_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False, default=str)
    logger.info("📝 JSON brut sauvegardé: %s", json_path)

    # Code de sortie basé sur le taux de réussite
    total = evaluation.get("total_pass", 0) + evaluation.get("total_fail", 0)
    success_rate = evaluation.get("total_pass", 0) / max(1, total)
    sys.exit(0 if success_rate >= 0.5 else 1)


if __name__ == "__main__":
    main()
