import sys
import os
import shutil
import logging
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.components.vector_store import VectorStoreHandler
from rag.components.retriever import AgentRetriever
from rag.components.re_ranker import Reranker
from services.utils.ingestor import DataIngestor
import services.utils.embedding as embedding_module

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestRAG")

from services.utils.cache import StorageManager

TEST_DATA_DIR = "data/test_vector_store"
TEST_DB_PATH = "data/test_cache.db"

def setup_test_environment():
    """Clean up previous test data"""
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR)
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def mock_meteo_data() -> List[Dict[str, Any]]:
    """Create sample meteo data mimicking Highcharts format found in scraping"""
    return [
        {
            "timestamp": "2025-01-10",
            "metadata": {
                "source_type": "METEO_VECTOR",
                "city": "Dédougou",
                "raw_data": [
                    {
                        "series": [
                            {
                                "data": [
                                    {"x": 0, "y": 18.5}, # Jan
                                    {"x": 4, "y": 35.2}, # Mai (Chaud)
                                    {"x": 7, "y": 28.0}, # Aout (Pluie/Frais)
                                ]
                            }
                        ]
                    }
                ]
            }
        }
    ]

def test_pipeline():
    print("🚀 DÉMARRAGE DU TEST RAG DE PRODUCTION\n")
    setup_test_environment()

    # --- 1. CONFIGURATION ---
    # Override paths for testing to avoid polluting real DB
    store_handler = VectorStoreHandler(
        index_path=f"{TEST_DATA_DIR}/test.index",
        metadata_path=f"{TEST_DATA_DIR}/metadata.json"
    )
    
    # Initialize Ingestor (ETL)
    ingestor = DataIngestor()
    # Inject our test store handler
    ingestor.store_handler = store_handler 
    
    # --- 2. INGESTION (ETL) ---
    print("📥 ÉTAPE 1: Ingestion & Textualisation...")
    raw_data = mock_meteo_data()
    
    # This triggers: MeteoProcessor -> Textualization -> TextChunker -> Embedding -> FAISS
    ingestor.ingest_data_from_orchestrator(raw_data)
    
    # Verify Ingestion
    print(f"\n✅ Documents indexés: {store_handler.index.ntotal}")
    
    # Inspecter le contenu généré (Preuve de textualisation)
    # On regarde le premier doc
    first_id = list(store_handler.metadata.keys())[0]
    content = store_handler.metadata[first_id].get("text_content", "")
    print(f"📄 Contenu textualisé (Extrait): '{content[:100]}...'")
    
    if "Dédougou" not in content and "températures" not in content:
        print("❌ ECHEC: La textualisation n'a pas fonctionné comme prévu.")
        return

    # --- 3. RETRIEVAL (Interrogation) ---
    print("\n🔍 ÉTAPE 2: Recherche (Retrieval + Reranking)...")
    
    # Setup Retriever Components
    # Note: We reuse the embedding service from ingestor for consistency
    embedder = ingestor.embedder 
    reranker = Reranker() # Will load Cross-Encoder
    storage_manager = StorageManager(db_path=TEST_DB_PATH)
    
    retriever = AgentRetriever(
        store=store_handler,
        embedder=embedder,
        reranker=reranker,
        storage=storage_manager
    )
    
    # Test Queries (Test with redundancy to verify cache)
    queries = [
        ("Quelle est la situation thermique à Dédougou ?", "METEO"),
        ("Quelle est la situation thermique à Dédougou ?", "METEO"), # Should trigger cache
        ("Risque de canicule en mai ?", "METEO")
    ]
    
    for query, role in queries:
        print(f"\n❓ Question: '{query}' (Rôle: {role})")
        results = retriever.retrieve_for_agent(query, role, zone_id="Dédougou")
        
        for i, res in enumerate(results):
            print(f"   {i+1}. [Score: {res['score']:.4f}] {res['content'][:80]}...")
            
    print("\n🎉 TEST RAG TERMINÉ AVEC SUCCÈS")

if __name__ == "__main__":
    test_pipeline()
