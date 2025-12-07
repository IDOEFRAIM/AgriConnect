import logging
import sys
import os
import shutil
from typing import List, Dict

# Configuration des chemins pour importer les modules voisins
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from retriever import AgentRetriever
    from services.utils.indexer import UniversalIndexer
    from metrics import RAGMetrics
    from vector_store import VectorStoreHandler
except ImportError:
    from .retriever import AgentRetriever
    from .indexer import UniversalIndexer
    from .metrics import RAGMetrics
    from .vector_store import VectorStoreHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("rag.evaluator")

class RetrievalEvaluator:
    """
    Classe dédiée à l'évaluation de la pertinence du système RAG.
    """

    def __init__(self):
        self.indexer = UniversalIndexer()
        # Le retriever sera initialisé plus tard pour garantir qu'il charge les données fraîches
        self.retriever = None 

    def clear_knowledge_base(self):
        """
        Nettoie la base vectorielle pour garantir un test propre.
        Supprime les fichiers d'index locaux.
        """
        logger.info("🧹 Nettoyage de la base de connaissances...")
        try:
            # On recrée un store juste pour supprimer les fichiers
            store = VectorStoreHandler()
            if os.path.exists(store.index_file):
                os.remove(store.index_file)
            if os.path.exists(store.meta_file):
                os.remove(store.meta_file)
            logger.info("✅ Base vide.")
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage : {e}")

    def setup_ground_truth(self):
        """
        Injecte les données de test.
        """
        self.clear_knowledge_base()
        
        logger.info("🛠️ Injection des données de Vérité Terrain...")
        
        # On ré-instancie l'indexer pour qu'il crée un nouvel index propre
        self.indexer = UniversalIndexer()

        # Donnée 1 : Inondation (Pour Hydrologue/Logisticien)
        flood_data = [{
            "properties": {
                "name": "Pont de la Léraba",
                "severity_level": "Rouge",
                "description": "Crue exceptionnelle. Le pont est submergé et impraticable. Cote d'alerte dépassée de 50cm."
            }
        }]
        self.indexer.index_meteo_data(flood_data, "INONDATIONS", "test_source_flood")

        # Donnée 2 : Agriculture (Pour Agronome)
        agri_doc = {
            "title": "Alerte Chenilles",
            "period": "Novembre 2025",
            "download_url": "http://test.com/doc.pdf",
            "text_content": "Une invasion de chenilles légionnaires menace les cultures de maïs dans l'Ouest. Il est urgent de traiter les parcelles."
        }
        self.indexer.index_document(agri_doc)
        
        logger.info("✅ Données injectées et sauvegardées sur disque.")

    def run_evaluation(self, test_suite: List[Dict]):
        """Exécute les tests."""
        
        # CRITIQUE : On initialise le Retriever ICI, après l'injection des données.
        # Cela force le chargement du fichier d'index qui vient d'être créé.
        logger.info("🔄 Chargement du Retriever avec les nouvelles données...")
        self.retriever = AgentRetriever()
        
        logger.info(f"\n🚀 DÉMARRAGE DE L'ÉVALUATION ({len(test_suite)} scénarios)")
        
        total_recall = 0.0
        total_f1 = 0.0
        total_jaccard = 0.0
        perfect_hits = 0
        total = len(test_suite)
        
        for i, case in enumerate(test_suite):
            query = case["query"]
            role = case["role"]
            expected_keywords = case["expected"]
            
            print(f"\n[{i+1}/{total}] Test Agent {role.upper()} : '{query}'")
            
            # 1. Interrogation
            response_context = self.retriever.retrieve_for_agent(query, agent_role=role, top_k=1)
            
            # 2. Métriques
            reference_text = " ".join(expected_keywords)
            recall = RAGMetrics.calculate_keyword_recall(expected_keywords, response_context)
            f1_score = RAGMetrics.calculate_f1(reference_text, response_context)
            jaccard = RAGMetrics.calculate_jaccard(reference_text, response_context)
            
            is_hit = RAGMetrics.is_perfect_hit(recall)
            
            # Stats
            total_recall += recall
            total_f1 += f1_score
            total_jaccard += jaccard
            if is_hit: perfect_hits += 1
            
            # Affichage
            status = "✅" if is_hit else "⚠️" if recall > 0 else "❌"
            print(f"   {status} Rappel : {recall*100:.0f}% | F1 : {f1_score:.2f} | Jaccard : {jaccard:.2f}")
            
            if not is_hit:
                missing = [kw for kw in expected_keywords if kw.lower() not in response_context.lower()]
                print(f"   -> Manquant : {missing}")
                # print(f"   -> Contexte (Debug) : {response_context[:100]}...")

        # Rapport
        print("\n" + "="*40)
        print(f"📊 RAPPORT FINAL")
        print(f"   - Hit Rate (Perfect) : {(perfect_hits/total)*100:.1f}%")
        print(f"   - Précision Moyenne  : {(total_recall/total)*100:.1f}%")
        print(f"   - Score F1 Moyen     : {(total_f1/total):.3f}")
        print("="*40)

if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    evaluator.setup_ground_truth()
    
    scenarios = [
        {
            "role": "Logisticien",
            "query": "Est-ce qu'on peut passer ?", 
            "expected": ["pont", "submergé", "impraticable"] 
        },
        {
            "role": "Agronome",
            "query": "Y a-t-il des menaces sur les cultures ?",
            "expected": ["chenilles", "maïs"]
        },
        {
            "role": "Hydrologue",
            "query": "Quel est le niveau de l'eau ?",
            "expected": ["crue", "cote d'alerte"]
        }
    ]
    
    evaluator.run_evaluation(scenarios)