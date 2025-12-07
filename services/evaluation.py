import logging
import datetime
import json
import random
import hashlib
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
# Import UTC for DeprecationWarning fix
from datetime import UTC 
from rag.re_ranker import Reranker
from rag.vector_store import VectorStoreHandler
from rag.retriever import AgentRetriever
from rag.metrics import RAGMetrics
from services.utils.embedding import EmbeddingService
from services.utils.cache import StorageManager
# ==============================================================================
# MOCK DEPENDENCIES for a self-contained, runnable file
# In a real project, these would be imported from your 'services' and 'rag' directories.
# ==============================================================================

# Mock config
class MockConfig:
    EMBEDDING_MODEL = "mock-multilingual-L12"
config = MockConfig()


# RETRIEVAL EVALUATOR CLASS (User's Code with minor cleanup)
# ==============================================================================

logger = logging.getLogger("rag.evaluator")
logger.setLevel(logging.INFO)


class RetrievalEvaluator:
    """
    Classe dédiée à l'évaluation de la pertinence du système RAG.
    - Lit des scénarios JSON
    - Interroge le retriever (simulé ici par MockAgentRetriever)
    - Calcule plusieurs métriques et sauvegarde un rapport JSON
    """

    def __init__(
        self,
        test_file_path: str = "test_scenarios.json",
        results_path: str = "evaluation_results.json",
        retriever: Optional[AgentRetriever] = None,
    ):
        self.test_file_path = Path(test_file_path)
        self.results_path = Path(results_path)

        # Initialize the retriever with the mock dependencies
        self.retriever = retriever or AgentRetriever(
            store=VectorStoreHandler,
            embedder=EmbeddingService,
            reranker=Reranker,
            storage=StorageManager
        )
        logger.info(f"🔄 Évaluateur prêt. Scénarios: {self.test_file_path}")

    def load_test_suite(self) -> List[Dict[str, Any]]:
        """Charge la suite de tests à partir du fichier JSON; crée un exemple si absent."""
        if not self.test_file_path.exists():
            logger.warning(f"Fichier de scénarios non trouvé: {self.test_file_path}. Création d'un fichier d'exemple.")
            
            # --- Default Scenario Creation ---
            default_scenarios = [
                {
                    "id": "C_001",
                    "role": "CROP",
                    "query": "Impact de la sécheresse sur les cultures de maïs et recommandations de semis.",
                    "filters": {"location": "Ouest"},
                    "expected_keywords": ["maïs", "sécheresse", "semis", "rendement"],
                    "expected_entities": ["maïs", "sécheresse"]
                },
                {
                    "id": "M_002",
                    "role": "METEO",
                    "query": "Quel est le niveau d'alerte des barrages dans la région Sud-Ouest ?",
                    "filters": {"location": "Sud-Ouest"},
                    "expected_keywords": ["barrages", "niveau", "alerte", "inondation"],
                    "expected_entities": ["barrage", "alerte"]
                },
                {
                    "id": "L_003",
                    "role": "LOGISTICS", # Renamed from MARKET for better semantic fit
                    "query": "Les ponts et routes principales sont-ils accessibles après les intempéries ?",
                    "filters": None,
                    "expected_keywords": ["routes", "ponts", "impraticable", "transport"],
                    "expected_entities": ["pont", "routes"]
                },
                # Added 2 more for better statistical averaging
                {
                    "id": "S_004",
                    "role": "SOIL",
                    "query": "Quelles sont les meilleures pratiques de drainage pour nos terres après les fortes pluies ?",
                    "filters": None,
                    "expected_keywords": ["drainage", "sols", "compaction", "infiltration"],
                    "expected_entities": ["drainage", "sols"]
                },
                {
                    "id": "H_005",
                    "role": "HEALTH",
                    "query": "Y a-t-il un risque d'épidémie de choléra dans les zones sinistrées ?",
                    "filters": None,
                    "expected_keywords": ["choléra", "risques", "hygiène", "épidémie"],
                    "expected_entities": ["choléra", "épidémie"]
                }
            ]
            # --- End Default Scenario Creation ---

            self.test_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.test_file_path, 'w', encoding='utf-8') as f:
                json.dump(default_scenarios, f, indent=4, ensure_ascii=False)
            logger.info("✅ Fichier d'exemple créé. Éditez-le pour vos tests réels.")
            return default_scenarios

        try:
            with open(self.test_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.error("Format JSON invalide : la racine doit être une liste de scénarios.")
                    return []
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Erreur JSON: {e}")
            return []

    def run_evaluation(self, top_k: int = 3, save_results: bool = True) -> Dict[str, Any]:
        """Exécute les tests et retourne un rapport structuré."""
        test_suite = self.load_test_suite()
        if not test_suite:
            logger.error("Aucun scénario de test valide. Annulation.")
            return {}

        logger.info(f"🚀 DÉMARRAGE DE L'ÉVALUATION ({len(test_suite)} scénarios)")

        results: List[Dict[str, Any]] = []
        # Accumulators for aggregate metrics
        accum = {
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "rouge_l": 0.0,
            "bertscore": 0.0,
            "entity_recall": 0.0,
            "perfect_hits": 0
        }

        for idx, case in enumerate(test_suite, start=1):
            scenario_id = case.get("id", f"case_{idx}")
            role = case.get("role", "Coordinateur")
            query = case.get("query", "")
            filters = case.get("filters")
            expected_keywords = case.get("expected_keywords", [])
            expected_entities = case.get("expected_entities", expected_keywords) # Fallback to keywords

            logger.info(f"[{idx}/{len(test_suite)}] Scénario {scenario_id} — role={role} | Query: {query[:50]}...")

            start = datetime.datetime.now(UTC)
            context = "" # Initialize context outside try
            try:
                # Retrieve the context using the mock retriever
                context = self.retriever.retrieve_for_agent(
                    query,
                    agent_role=role,
                    filters=filters,
                    limit=top_k, 
                )
            except Exception as e:
                logger.error(f"Erreur retrieval pour {scenario_id}: {e}")
                # context remains empty string

            elapsed = (datetime.datetime.now(UTC) - start).total_seconds()

            if not context or not context.strip():
                logger.warning(f"Contexte vide pour {scenario_id}.")
                # Record the failure
                result = {
                    "id": scenario_id,
                    "role": role,
                    "query": query,
                    "success": False,
                    "error": "empty_context",
                    "elapsed_s": elapsed,
                    "expected_keywords": expected_keywords,
                }
                results.append(result)
                continue

            # Compute metrics
            reference_text = " ".join(expected_keywords)
            # RAGMetrics calls (using mock static methods)
            recall = RAGMetrics.calculate_keyword_recall(expected_keywords, context)
            f1 = RAGMetrics.calculate_f1(reference_text, context)
            jaccard = RAGMetrics.calculate_jaccard(reference_text, context)
            rouge_l = RAGMetrics.calculate_rouge_l(reference_text, context)
            bscore = RAGMetrics.bertscore_placeholder(reference_text, context)
            ent_recall = RAGMetrics.calculate_entity_recall(expected_entities, context)
            perfect = RAGMetrics.is_perfect_hit(recall)

            # Accumulate scores
            accum["recall"] += recall
            accum["f1"] += f1
            accum["jaccard"] += jaccard
            accum["rouge_l"] += rouge_l
            accum["bertscore"] += bscore
            accum["entity_recall"] += ent_recall
            if perfect:
                accum["perfect_hits"] += 1

            result = {
                "id": scenario_id,
                "role": role,
                "query": query,
                "success": True,
                "elapsed_s": elapsed,
                "metrics": {
                    "keyword_recall": recall,
                    "f1": f1,
                    "jaccard": jaccard,
                    "rouge_l": rouge_l,
                    "bertscore": bscore,
                    "entity_recall": ent_recall,
                    "perfect_hit": perfect
                },
                "expected_keywords": expected_keywords,
                "expected_entities": expected_entities,
                "context_excerpt": context[:1000].strip() + "..." # keep report compact
            }
            results.append(result)

            # Log summary per case
            status = "✅" if perfect else ("⚠️" if recall > 0.5 else "❌")
            logger.info(f" {status} K-Recall={recall:.2f} F1={f1:.2f} ROUGE-L={rouge_l:.2f} E-Recall={ent_recall:.2f} (t={elapsed:.2f}s)")

        # Aggregate
        n = len(test_suite)
        summary = {
            "n_scenarios": n,
            "avg_keyword_recall": accum["recall"] / n if n else 0.0,
            "avg_f1": accum["f1"] / n if n else 0.0,
            "avg_jaccard": accum["jaccard"] / n if n else 0.0,
            "avg_rouge_l": accum["rouge_l"] / n if n else 0.0,
            "avg_bertscore": accum["bertscore"] / n if n else 0.0,
            "avg_entity_recall": accum["entity_recall"] / n if n else 0.0,
            "perfect_hit_rate": (accum["perfect_hits"] / n) if n else 0.0,
            "generated_at": datetime.datetime.now(UTC).isoformat()
        }

        report = {
            "summary": summary,
            "results": results
        }

        if save_results:
            try:
                self.results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.results_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Rapport sauvegardé : {self.results_path}")
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder le rapport: {e}")

        # Print compact summary to stdout
        logger.info("\n=== FINAL EVALUATION SUMMARY ===")
        logger.info(f"Total Scénarios: {summary['n_scenarios']}")
        logger.info(f"Perfect Hit Rate: {summary['perfect_hit_rate']*100:.1f}%")
        logger.info(f"Avg Keyword Recall: {summary['avg_keyword_recall']*100:.1f}%")
        logger.info(f"Avg F1 Score: {summary['avg_f1']:.3f}")
        logger.info(f"Avg ROUGE-L Score: {summary['avg_rouge_l']:.3f}")
        logger.info("==================================\n")

        return report


def main():
    # Setup logging to stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
    
    # Run the evaluation
    # Note: 'evaluation_set.json' will be created if it doesn't exist.
    evaluator = RetrievalEvaluator(test_file_path="evaluation_set.json", results_path="evaluation_results.json")
    report = evaluator.run_evaluation(top_k=3, save_results=True)
    
    # Optionally print the JSON summary
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    # Add a handler to stdout for the mock logging inside the class
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)

    main()