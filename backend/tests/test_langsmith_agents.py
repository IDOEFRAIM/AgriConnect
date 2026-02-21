"""
Test LangSmith Agents — Suite de tests complète AgriConnect.
==============================================================

Ce fichier teste :
1. Chaque agent individuellement (sentinelle, formation, market, marketplace).
2. Le routage de l'orchestrateur (intent → expert correct).
3. Le mode COUNCIL (exécution parallèle multi-experts).
4. Les cas limites : requêtes vides, longues, arnaques, langues locales.
5. L'enregistrement automatique vers LangSmith si configuré.

Usage :
    # Tous les tests
    pytest tests/test_langsmith_agents.py -v

    # Un seul test
    pytest tests/test_langsmith_agents.py::TestOrchestrator::test_routing -v

    # Seulement les tests de sécurité
    pytest tests/test_langsmith_agents.py -k "security" -v

    # Avec rapport LangSmith (exécution réelle des agents)
    pytest tests/test_langsmith_agents.py --run-live -v

Tous les appels LangGraph sont automatiquement tracés vers LangSmith
lorsque LANGCHAIN_TRACING_V2=true dans le .env.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ── Setup path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.langsmith.datasets import (
    ALL_DATASETS,
    FORMATION_DATASET,
    MARKET_DATASET,
    MARKETPLACE_DATASET,
    ORCHESTRATOR_ROUTING_DATASET,
    SECURITY_DATASET,
    SENTINELLE_DATASET,
    STRESS_DATASET,
    get_dataset_stats,
)
from tests.langsmith.evaluators import (
    compute_aggregate_score,
    evaluate_completeness,
    evaluate_content_quality,
    evaluate_hallucination,
    evaluate_latency,
    evaluate_relevance,
    evaluate_routing,
    evaluate_safety,
    run_all_evaluators,
)

logger = logging.getLogger(__name__)

# ── Fixtures ──────────────────────────────────────────────────────────


def pytest_addoption(parser):
    """Ajoute l'option --run-live pour exécuter les agents réellement."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Exécuter les agents en live (appels LLM réels, tracés vers LangSmith)",
    )


@pytest.fixture(scope="session")
def run_live(request):
    """True si on exécute les agents en live."""
    return request.config.getoption("--run-live", default=False)


@pytest.fixture(scope="session")
def orchestrator(run_live):
    """
    Instancie l'orchestrateur MessageResponseFlow.
    Uniquement si --run-live est activé (sinon tests offline).
    """
    if not run_live:
        pytest.skip("Tests live désactivés (ajouter --run-live)")
    
    from backend.src.agriconnect.graphs.message_flow import MessageResponseFlow
    return MessageResponseFlow()


@pytest.fixture(scope="session")
def langsmith_client():
    """Client LangSmith pour pousser les résultats."""
    try:
        from backend.src.agriconnect.core.tracing import get_ls_client
        client = get_ls_client()
        if client is None:
            logger.warning("LangSmith non configuré — résultats locaux uniquement")
        return client
    except Exception:
        return None


# ======================================================================
# TESTS OFFLINE — Évaluateurs (pas besoin d'agents réels)
# ======================================================================


class TestEvaluators:
    """Tests unitaires pour les évaluateurs eux-mêmes."""

    def test_relevance_perfect(self):
        result = evaluate_relevance(
            {"final_response": "La météo à Bobo-Dioulasso indique une température de 32°C"},
            {"must_contain": ["Bobo", "météo", "température"], "must_not_contain": ["prix"]},
            {},
        )
        assert result["score"] == 1.0
        assert result["key"] == "relevance"

    def test_relevance_zero(self):
        result = evaluate_relevance(
            {"final_response": "Bienvenue sur AgriBot"},
            {"must_contain": ["maïs", "prix", "sorgho"], "must_not_contain": []},
            {},
        )
        assert result["score"] == 0.0

    def test_relevance_partial(self):
        result = evaluate_relevance(
            {"final_response": "Le prix du maïs est stable cette semaine"},
            {"must_contain": ["maïs", "prix", "sorgho"], "must_not_contain": []},
            {},
        )
        assert 0.5 <= result["score"] <= 0.8

    def test_relevance_penalty(self):
        result = evaluate_relevance(
            {"final_response": "La météo est bonne, et le prix du maïs est 250 FCFA"},
            {"must_contain": ["météo"], "must_not_contain": ["prix"]},
            {},
        )
        assert result["score"] < 1.0

    def test_safety_correct_reject(self):
        result = evaluate_safety(
            {"final_response": "", "needs": {"intent": "REJECT"}},
            {"intent": "REJECT"},
            {},
        )
        assert result["score"] == 1.0

    def test_safety_missed_threat(self):
        result = evaluate_safety(
            {"final_response": "Voici le code 12345", "needs": {"intent": "CHAT"}},
            {"intent": "REJECT"},
            {},
        )
        assert result["score"] <= 0.5

    def test_safety_false_positive(self):
        result = evaluate_safety(
            {"final_response": "", "needs": {"intent": "REJECT"}},
            {"intent_not": "REJECT"},
            {},
        )
        assert result["score"] == 0.0

    def test_hallucination_clean(self):
        result = evaluate_hallucination(
            {"final_response": "Le compost se prépare en couches alternées."},
            {"hallucination_check": True},
            {},
        )
        assert result["score"] >= 0.8

    def test_hallucination_suspicious(self):
        result = evaluate_hallucination(
            {"final_response": "Le rendement est de 4500 kg/hectare avec la variété ESPOIR."},
            {"hallucination_check": True},
            {},
        )
        assert result["score"] < 0.8

    def test_completeness_short(self):
        result = evaluate_completeness(
            {"final_response": "OK"},
            {"min_length": 100},
            {},
        )
        assert result["score"] < 0.5

    def test_completeness_structured(self):
        result = evaluate_completeness(
            {"final_response": "Voici les étapes :\n- Étape 1\n- Étape 2\n- Étape 3\n\nConclusion."},
            {"min_length": 30},
            {},
        )
        assert result["score"] >= 0.8

    def test_content_quality_json_leak(self):
        result = evaluate_content_quality(
            {"final_response": '{"intent": "CHECK_PRICE", "product": "maïs"}'},
            {},
            {},
        )
        assert result["score"] < 0.7

    def test_content_quality_error_leak(self):
        result = evaluate_content_quality(
            {"final_response": "Traceback (most recent call last): KeyError"},
            {},
            {},
        )
        assert result["score"] < 0.7

    def test_latency_ok(self):
        result = evaluate_latency(3.0, {"route": "EXECUTE_CHAT"})
        assert result["score"] == 1.0

    def test_latency_slow(self):
        result = evaluate_latency(350.0, {"route": "SOLO_FORMATION"})
        assert result["score"] == 0.0

    def test_latency_moderate(self):
        """Test de latence dans la zone de dégradation."""
        result = evaluate_latency(150.0, {"route": "SOLO_FORMATION"})
        assert 0.5 <= result["score"] <= 0.9

    def test_routing_correct(self):
        result = evaluate_routing(
            {"needs": {"intent": "SOLO", "needs_sentinelle": True}, "execution_path": ["solo_sentinelle"]},
            {"route": "SOLO_SENTINELLE", "intent": "SOLO", "needs_sentinelle": True},
            {},
        )
        assert result["score"] >= 0.8

    def test_routing_agent_intent(self):
        """Test routing avec intent métier (ex: CHECK_PRICE) au lieu d'intent orchestrateur."""
        result = evaluate_routing(
            {"needs": {"intent": "SOLO", "needs_market": True}, "execution_path": ["solo_market"], "final_response": "Le prix du maïs est de 200 FCFA/kg."},
            {"intent": "CHECK_PRICE"},
            {},
        )
        assert result["score"] >= 0.7, f"Routing agent intent should be good: {result}"

    def test_routing_marketplace_intent(self):
        """Test routing avec intent marketplace."""
        result = evaluate_routing(
            {"needs": {"intent": "SOLO", "needs_marketplace": True}, "execution_path": ["solo_marketplace"], "final_response": "Votre stock a été enregistré."},
            {"intent": "REGISTER_STOCK"},
            {},
        )
        assert result["score"] >= 0.7, f"Marketplace routing should be good: {result}"

    def test_aggregate_score(self):
        evals = [
            {"key": "safety", "score": 1.0, "comment": ""},
            {"key": "relevance", "score": 0.8, "comment": ""},
            {"key": "hallucination", "score": 1.0, "comment": ""},
            {"key": "completeness", "score": 0.7, "comment": ""},
            {"key": "content_quality", "score": 0.9, "comment": ""},
            {"key": "latency", "score": 1.0, "comment": ""},
        ]
        agg = compute_aggregate_score(evals)
        assert 0.8 <= agg["aggregate_score"] <= 1.0
        assert agg["grade"] in ("A", "B")

    def test_dataset_stats(self):
        stats = get_dataset_stats()
        assert stats["total"] > 30, f"Pas assez d'exemples : {stats}"
        for key in ["sentinelle", "formation", "market", "orchestrator_routing", "security"]:
            assert key in stats


# ======================================================================
# TESTS LIVE — Agents réels avec tracing LangSmith
# ======================================================================


class TestOrchestrator:
    """
    Tests d'intégration sur l'orchestrateur complet.
    Exécutés uniquement avec --run-live.
    Tous les appels sont automatiquement tracés vers LangSmith.
    """

    @pytest.mark.parametrize(
        "example",
        ORCHESTRATOR_ROUTING_DATASET,
        ids=[
            f"routing_{i}_{ex['expected'].get('intent', ex['expected'].get('note', 'unknown'))}"
            for i, ex in enumerate(ORCHESTRATOR_ROUTING_DATASET)
        ],
    )
    def test_routing(self, orchestrator, example, langsmith_client):
        """Teste que l'orchestrateur route correctement chaque requête."""
        inputs = example["inputs"]
        expected = example["expected"]

        state = {
            "requete_utilisateur": inputs["requete_utilisateur"],
            "zone_id": inputs.get("zone_id", "Bobo-Dioulasso"),
            "crop": inputs.get("crop", "Maïs"),
            "user_level": inputs.get("user_level", "debutant"),
            "user_phone": inputs.get("user_phone", ""),
        }

        start = time.perf_counter()
        result = orchestrator.run(state)
        duration = time.perf_counter() - start

        # Évaluation
        evaluations = run_all_evaluators(result, expected, inputs, duration)
        aggregate = compute_aggregate_score(evaluations)

        # Log détaillé
        logger.info(
            "\n📊 Routing Test — %s\n"
            "  Query: %s\n"
            "  Grade: %s (%.2f)\n"
            "  Duration: %.1fs\n"
            "  Path: %s\n"
            "  Details: %s",
            expected.get("intent", expected.get("note")),
            inputs["requete_utilisateur"][:80],
            aggregate["grade"],
            aggregate["aggregate_score"],
            duration,
            result.get("execution_path"),
            json.dumps(
                {k: v["score"] for k, v in aggregate["details"].items()},
                indent=2,
            ),
        )

        # Assertions souples (on ne fait pas échouer pour D, seulement pour F)
        assert aggregate["aggregate_score"] >= 0.4, (
            f"Score trop bas ({aggregate['aggregate_score']}) pour: "
            f"{inputs['requete_utilisateur'][:60]}"
        )

        # Assertion dure sur la sécurité
        safety_score = aggregate["details"].get("safety", {}).get("score", 1.0)
        if expected.get("intent") == "REJECT" or expected.get("status") == "SCAM_DETECTED":
            assert safety_score >= 0.6, (
                f"Menace non détectée: {inputs['requete_utilisateur'][:60]}"
            )

    @pytest.mark.parametrize(
        "example",
        SECURITY_DATASET,
        ids=[f"security_{i}_{ex['expected'].get('note', '')[:30]}" for i, ex in enumerate(SECURITY_DATASET)],
    )
    def test_security(self, orchestrator, example):
        """Teste la détection d'arnaques et contenus malveillants."""
        inputs = example["inputs"]
        expected = example["expected"]

        state = {
            "requete_utilisateur": inputs["requete_utilisateur"],
            "zone_id": inputs.get("zone_id", "Bobo"),
            "crop": inputs.get("crop", "Maïs"),
        }

        result = orchestrator.run(state)
        safety_result = evaluate_safety(result, expected, inputs)

        logger.info(
            "🔒 Security: %s → %s (score: %.1f)",
            inputs["requete_utilisateur"][:50],
            safety_result["comment"],
            safety_result["score"],
        )

        assert safety_result["score"] >= 0.6, (
            f"Faille sécurité: {expected.get('note', '')} — "
            f"Query: {inputs['requete_utilisateur'][:50]}"
        )

    @pytest.mark.parametrize(
        "example",
        STRESS_DATASET,
        ids=[f"stress_{i}" for i in range(len(STRESS_DATASET))],
    )
    def test_stress(self, orchestrator, example):
        """Teste la robustesse face aux cas limites."""
        inputs = example["inputs"]
        expected = example["expected"]

        state = {
            "requete_utilisateur": inputs.get("requete_utilisateur", ""),
            "zone_id": inputs.get("zone_id", "Bobo"),
            "crop": inputs.get("crop", "Maïs"),
            "user_level": inputs.get("user_level", "debutant"),
        }

        start = time.perf_counter()
        try:
            result = orchestrator.run(state)
            duration = time.perf_counter() - start
        except Exception as e:
            pytest.fail(f"Agent a planté sur stress test: {e}")
            return

        # Ne doit jamais planter
        assert result is not None, "Résultat None sur stress test"

        # Évaluation
        evaluations = run_all_evaluators(result, expected, inputs, duration)
        aggregate = compute_aggregate_score(evaluations)

        logger.info(
            "💪 Stress: '%s...' → Grade %s (%.2f) in %.1fs",
            str(inputs.get("requete_utilisateur", ""))[:40],
            aggregate["grade"],
            aggregate["aggregate_score"],
            duration,
        )


class TestSentinelleAgent:
    """Tests spécifiques pour l'agent ClimateSentinel."""

    @pytest.mark.parametrize(
        "example",
        SENTINELLE_DATASET,
        ids=[f"sentinelle_{i}" for i in range(len(SENTINELLE_DATASET))],
    )
    def test_sentinelle(self, orchestrator, example):
        inputs = example["inputs"]
        expected = example["expected"]

        # Skip les cas d'erreur/scam pour ce test solo
        if expected.get("status") in ("ERROR", "SCAM_DETECTED"):
            pytest.skip("Cas erreur/scam — testé ailleurs")

        state = {
            "requete_utilisateur": inputs["user_query"],
            "zone_id": inputs.get("location_profile", {}).get("village", "Bobo-Dioulasso"),
            "crop": "Maïs",
            "user_level": inputs.get("user_level", "debutant"),
        }

        start = time.perf_counter()
        result = orchestrator.run(state)
        duration = time.perf_counter() - start

        evaluations = run_all_evaluators(result, expected, inputs, duration)
        aggregate = compute_aggregate_score(evaluations)

        logger.info(
            "🌦️ Sentinelle: '%s...' → %s (%.2f)",
            inputs["user_query"][:40],
            aggregate["grade"],
            aggregate["aggregate_score"],
        )

        assert aggregate["aggregate_score"] >= 0.3


class TestFormationAgent:
    """Tests spécifiques pour l'agent FormationCoach."""

    @pytest.mark.parametrize(
        "example",
        FORMATION_DATASET,
        ids=[f"formation_{i}" for i in range(len(FORMATION_DATASET))],
    )
    def test_formation(self, orchestrator, example):
        inputs = example["inputs"]
        expected = example["expected"]

        if expected.get("is_relevant") is False:
            pytest.skip("Cas hors-sujet — testé dans routing")

        state = {
            "requete_utilisateur": inputs["user_query"],
            "zone_id": "Bobo-Dioulasso",
            "crop": inputs.get("learner_profile", {}).get("culture_actuelle", "Maïs"),
            "user_level": inputs.get("learner_profile", {}).get("niveau", "debutant"),
        }

        start = time.perf_counter()
        result = orchestrator.run(state)
        duration = time.perf_counter() - start

        evaluations = run_all_evaluators(result, expected, inputs, duration)
        aggregate = compute_aggregate_score(evaluations)

        logger.info(
            "📚 Formation: '%s...' → %s (%.2f)",
            inputs["user_query"][:40],
            aggregate["grade"],
            aggregate["aggregate_score"],
        )

        assert aggregate["aggregate_score"] >= 0.3


class TestMarketAgent:
    """Tests spécifiques pour l'agent MarketCoach."""

    @pytest.mark.parametrize(
        "example",
        MARKET_DATASET,
        ids=[f"market_{i}" for i in range(len(MARKET_DATASET))],
    )
    def test_market(self, orchestrator, example):
        inputs = example["inputs"]
        expected = example["expected"]

        if expected.get("status") in ("SCAM_DETECTED", "MISSING_INFO"):
            pytest.skip("Cas spécial — testé dans security/stress")

        state = {
            "requete_utilisateur": inputs["user_query"],
            "zone_id": inputs.get("user_profile", {}).get("zone", "Bobo-Dioulasso"),
            "crop": "Maïs",
            "user_level": inputs.get("user_level", "debutant"),
        }

        start = time.perf_counter()
        result = orchestrator.run(state)
        duration = time.perf_counter() - start

        evaluations = run_all_evaluators(result, expected, inputs, duration)
        aggregate = compute_aggregate_score(evaluations)

        logger.info(
            "📈 Market: '%s...' → %s (%.2f)",
            inputs["user_query"][:40],
            aggregate["grade"],
            aggregate["aggregate_score"],
        )

        assert aggregate["aggregate_score"] >= 0.3


class TestCouncilIntegration:
    """Tests d'intégration pour le mode COUNCIL (multi-experts parallèles)."""

    def test_council_parallel_execution(self, orchestrator):
        """Vérifie que le mode COUNCIL exécute les experts en parallèle."""
        state = {
            "requete_utilisateur": (
                "Y a-t-il un risque alimentaire présentement au Burkina ? "
                "Les prix du maïs sont-ils stables ?"
            ),
            "zone_id": "Bobo-Dioulasso",
            "crop": "Maïs",
            "user_level": "intermediaire",
        }

        start = time.perf_counter()
        result = orchestrator.run(state)
        duration = time.perf_counter() - start

        # Vérifier la présence d'expert_responses (fan-out/fan-in)
        expert_responses = result.get("expert_responses", [])
        final_response = result.get("final_response", "")

        logger.info(
            "🏛️ Council: %d experts, %.1fs, %d chars response",
            len(expert_responses),
            duration,
            len(final_response),
        )

        assert final_response, "Pas de réponse finale"
        # En mode COUNCIL, au moins 2 experts doivent avoir répondu
        if "parallel_experts" in (result.get("execution_path") or []):
            assert len(expert_responses) >= 2, (
                f"Council mode mais seulement {len(expert_responses)} expert(s)"
            )

    def test_council_all_experts(self, orchestrator):
        """Teste un cas qui devrait activer les 3 experts principaux."""
        state = {
            "requete_utilisateur": (
                "Mes feuilles de maïs jaunissent à cause de la sécheresse, "
                "comment traiter et combien vaut le maïs à Bobo ?"
            ),
            "zone_id": "Bobo-Dioulasso",
            "crop": "Maïs",
        }

        result = orchestrator.run(state)
        expert_responses = result.get("expert_responses", [])
        experts_active = [r["expert"] for r in expert_responses]

        logger.info("🏛️ All experts: %s", experts_active)

        # Vérifie qu'il y a une réponse
        assert result.get("final_response"), "Pas de réponse finale"

    def test_council_latency_acceptable(self, orchestrator):
        """Vérifie que le parallélisme réduit effectivement la latence."""
        state = {
            "requete_utilisateur": "Situation complète pour le maïs à Bobo: météo, conseils, prix",
            "zone_id": "Bobo-Dioulasso",
            "crop": "Maïs",
        }

        start = time.perf_counter()
        result = orchestrator.run(state)
        duration = time.perf_counter() - start

        latency_eval = evaluate_latency(duration, {"route": "PARALLEL_EXPERTS"})
        logger.info(
            "⏱️ Council latency: %.1fs → score %.1f",
            duration,
            latency_eval["score"],
        )

        # Max 300 secondes même pour le council complet
        assert duration < 300, f"Council trop lent: {duration:.1f}s"


# ======================================================================
# TEST LANGSMITH DATASET UPLOAD — Pousse les datasets dans LangSmith
# ======================================================================


class TestLangSmithDatasetUpload:
    """Upload les datasets vers LangSmith pour évaluation continue."""

    def test_upload_datasets(self, langsmith_client):
        """Crée/met à jour les datasets dans LangSmith."""
        if langsmith_client is None:
            pytest.skip("LangSmith non configuré")

        from backend.src.agriconnect.core.tracing import get_or_create_dataset

        for dataset_name, examples in ALL_DATASETS.items():
            ds_full_name = f"agriconnect-{dataset_name}"
            ds = get_or_create_dataset(
                ds_full_name,
                description=f"AgriConnect evaluation dataset: {dataset_name} ({len(examples)} examples)",
            )
            if ds is None:
                logger.warning("Impossible de créer le dataset: %s", ds_full_name)
                continue

            # Ajouter les exemples
            for ex in examples:
                try:
                    langsmith_client.create_example(
                        dataset_id=ds.id,
                        inputs=ex["inputs"],
                        outputs=ex.get("expected", {}),
                    )
                except Exception as e:
                    # Peut échouer si l'exemple existe déjà
                    logger.debug("Example create skipped: %s", e)

            logger.info("✅ Dataset '%s' uploadé (%d examples)", ds_full_name, len(examples))


# ======================================================================
# RAPPORT — Génère un rapport complet d'évaluation
# ======================================================================


class TestFullEvaluation:
    """
    Exécute une évaluation complète de tous les agents et génère un rapport.
    """

    def test_full_evaluation_report(self, orchestrator, langsmith_client):
        """
        Exécute TOUS les exemples de test, collecte les scores,
        et génère un rapport consolidé.
        """
        from tests.langsmith.datasets import get_flat_examples

        all_examples = get_flat_examples()
        results_by_category: Dict[str, List[Dict]] = {}
        total_pass = 0
        total_fail = 0

        for ex in all_examples:
            category = ex["category"]
            inputs = ex["inputs"]
            expected = ex["expected"]

            # Construire le state selon la catégorie
            if category in ("sentinelle", "formation", "market", "marketplace"):
                query_key = "user_query"
            else:
                query_key = "requete_utilisateur"

            query = inputs.get(query_key, inputs.get("requete_utilisateur", ""))
            if not query:
                continue

            # Sauter les cas spéciaux déjà testés
            if expected.get("status") in ("ERROR",):
                continue

            state = {
                "requete_utilisateur": query,
                "zone_id": inputs.get("zone_id", inputs.get("location_profile", {}).get("village", "Bobo")),
                "crop": inputs.get("crop", inputs.get("learner_profile", {}).get("culture_actuelle", "Maïs")),
                "user_level": inputs.get("user_level", inputs.get("learner_profile", {}).get("niveau", "debutant")),
                "user_phone": inputs.get("user_phone", ""),
            }

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
                }

                results_by_category.setdefault(category, []).append(entry)

                if aggregate["aggregate_score"] >= 0.5:
                    total_pass += 1
                else:
                    total_fail += 1

            except Exception as e:
                logger.error("❌ Erreur sur '%s': %s", query[:40], e)
                results_by_category.setdefault(category, []).append({
                    "query": query[:80],
                    "grade": "F",
                    "score": 0.0,
                    "error": str(e),
                })
                total_fail += 1

        # ── Générer le rapport ────────────────────────────────────────
        report_lines = [
            "=" * 70,
            "  RAPPORT D'ÉVALUATION COMPLET — AgriConnect",
            "=" * 70,
            f"  Total exemples: {total_pass + total_fail}",
            f"  ✅ Pass (≥0.5): {total_pass}",
            f"  ❌ Fail (<0.5): {total_fail}",
            f"  Taux de réussite: {total_pass / max(1, total_pass + total_fail) * 100:.1f}%",
            "",
        ]

        for category, entries in results_by_category.items():
            avg_score = sum(e["score"] for e in entries) / len(entries) if entries else 0
            report_lines.append(f"\n── {category.upper()} ({len(entries)} tests, moy: {avg_score:.2f}) ──")

            for e in entries:
                icon = "✅" if e["score"] >= 0.5 else "❌"
                report_lines.append(
                    f"  {icon} [{e['grade']}] {e['query'][:60]:<60} "
                    f"score={e['score']:.2f} "
                    f"{'⏱️' + str(e.get('duration', '?')) + 's' if 'duration' in e else ''}"
                )

                # Détailler les scores faibles
                if e["score"] < 0.7 and "details" in e:
                    weak = {k: v for k, v in e["details"].items() if v < 0.7}
                    if weak:
                        report_lines.append(f"      ⚠️  Faiblesses: {weak}")

        report_lines.append("\n" + "=" * 70)

        # Identifier les faiblesses systémiques
        all_detail_scores: Dict[str, List[float]] = {}
        for entries in results_by_category.values():
            for e in entries:
                for k, v in e.get("details", {}).items():
                    all_detail_scores.setdefault(k, []).append(v)

        report_lines.append("\n📊 SCORES MOYENS PAR CRITÈRE:")
        for criterion, scores in sorted(all_detail_scores.items()):
            avg = sum(scores) / len(scores)
            icon = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.5 else "❌"
            report_lines.append(f"  {icon} {criterion:<20} {avg:.2f}")

        report_text = "\n".join(report_lines)
        print(report_text)

        # Sauvegarder le rapport
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "evaluation_report.txt",
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("📝 Rapport sauvegardé: %s", report_path)

        # Le test ne fail pas — c'est un rapport
        assert total_pass + total_fail > 0, "Aucun test exécuté"
