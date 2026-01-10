import sys
import os
import shutil
import logging
import time
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.components.vector_store import VectorStoreHandler
from rag.components.retriever import AgentRetriever
from rag.components.re_ranker import Reranker
from services.utils.ingestor import DataIngestor
from services.utils.cache import StorageManager

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s') # Simplified format for benchmark
logger = logging.getLogger("Benchmark")

TEST_DATA_DIR = "data/benchmark_vector_store"
TEST_DB_PATH = "data/benchmark_cache.db"

def setup_benchmark_environment():
    """Clean up previous test data"""
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR)
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except:
            pass

def mock_comprehensive_data() -> List[Dict[str, Any]]:
    """Create diverse sample data covering all domains"""
    return [
        # --- METEO ---
        {
            "timestamp": "2026-01-10",
            "metadata": {
                "source": "Climatologie de Dédougou",
                "source_type": "METEO_VECTOR",
                "city": "Dédougou",
                "raw_data": [
                    {"series": [{"data": [{"x": 0, "y": 18.5}, {"x": 4, "y": 42.0}, {"x": 7, "y": 28.0}]}]}
                ]
            }
        },
        {
            "timestamp": "2026-05-15",
            "title": "Alerte Inondation Ouagadougou",
            "content": "Alerte Rouge : Le barrage n°3 menace de céder. Les quartiers de Tanghin et Kilwin sont à risque d'inondation majeure. Évacuation conseillée.",
            "metadata": {"source": "Alerte Civile", "source_type": "METEO_ALERT", "city": "Ouagadougou"}
        },
        # --- MARKET ---
        {
            "timestamp": "2026-01-09",
            "title": "Rapport Hebdomadaire SIM - Bobo",
            "content": "À Bobo-Dioulasso, le prix du maïs blanc est en hausse à 22 000 FCFA le sac de 100kg. Le sorgho rouge reste stable à 19 000 FCFA. Grande disponibilité de l'igname.",
            "metadata": {"source": "SIM Bobo", "source_type": "MARKET_REPORT", "city": "Bobo-Dioulasso"}
        },
        {
            "timestamp": "2026-01-09",
            "title": "Prix Céréales Fada N'Gourma",
            "content": "Pénurie de mil à Fada. Les prix flambent à 28 000 FCFA. Les commerçants attendent les stocks du Bénin.",
            "metadata": {"source": "SIM Fada", "source_type": "MARKET_REPORT", "city": "Fada N'Gourma"}
        },
        # --- CROP ---
        {
            "timestamp": "2025-11-20",
            "title": "Guide Technique: Culture du Maïs",
            "content": "Le semis du maïs doit se faire après une pluie utile de 20mm. Ecartement recommandé : 80cm x 40cm. Variété recommandée pour la zone soudano-sahélienne : Barka.",
            "metadata": {"source": "ITRA Maïs", "source_type": "AGRI_REPORT", "crop": "Maïs"}
        },
        {
            "timestamp": "2025-10-10",
            "title": "Gestion du Coton Bt",
            "content": "Pour le Coton, le traitement contre les jassides doit être préventif. Réduisez l'épandage en cas de vent fort.",
            "metadata": {"source": "Sofitex Guide", "source_type": "AGRI_REPORT", "crop": "Coton"}
        },
        # --- SOIL ---
        {
            "timestamp": "2025-06-01",
            "title": "Amender les sols argileux",
            "content": "Les sols argileux de la Vallée du Sourou nécessitent un apport en matière organique pour améliorer le drainage. Évitez le labour profond quand le sol est trop humide.",
            "metadata": {"source": "BUNASOLS", "source_type": "AGRI_REPORT", "topic": "Soil"}
        },
        # --- HEALTH ---
        {
            "timestamp": "2026-01-05",
            "title": "Alerte Chenille Légionnaire",
            "content": "Foyers de chenilles légionnaires signalés dans la Boucle du Mouhoun. Traitement urgent avec Emamectine Benzoate recommandé le soir.",
            "metadata": {"source": "DGPV Alerte", "source_type": "AGRI_REPORT", "topic": "Pest"}
        },
        # --- SUBSIDY ---
        {
            "timestamp": "2026-01-02",
            "title": "Subvention Engrais 2026",
            "content": "Le Ministère de l'Agriculture lance l'opération 'Engrais pour Tous'. Subvention de 50% sur le NPK et l'Urée. Inscription auprès des CRA avant le 15 mars.",
            "metadata": {"source": "Communique Ministère", "source_type": "OFFICIAL_NOTICE", "topic": "Subsidy"}
        },
        {
            "timestamp": "2026-01-03",
            "title": "Aide Équipement",
            "content": "Subvention pour l'achat de charrues et tracteurs. Financement à hauteur de 30% pour les coopératives enregistrées.",
            "metadata": {"source": "Journal Officiel", "source_type": "OFFICIAL_NOTICE", "topic": "Subsidy"}
        }
    ]

QUESTIONS_BENCHMARK = [
    # -- METEO --
    {"q": "Quelle est la température maximale prévue à Dédougou ?", "role": "METEO", "zone": "Dédougou"},
    {"q": "Y a-t-il des risques d'inondation à Ouagadougou ?", "role": "METEO", "zone": "Ouagadougou"},
    {"q": "Est-il prévu qu'il fasse chaud en mai à Dédougou ?", "role": "METEO", "zone": "Dédougou"},
    {"q": "Quelle est la pluviométrie attendue en Août ?", "role": "METEO", "zone": "Dédougou"}, # Infer from visual/textualizer
    
    # -- MARKET --
    {"q": "Quel est le prix actuel du maïs à Bobo ?", "role": "MARKET", "zone": "Bobo-Dioulasso"},
    {"q": "Y a-t-il une pénurie de céréales à Fada ?", "role": "MARKET", "zone": "Fada N'Gourma"},
    {"q": "Combien coûte le sorgho rouge ?", "role": "MARKET", "zone": "Bobo-Dioulasso"},
    {"q": "Où trouver de l'igname en abondance ?", "role": "MARKET", "zone": "Bobo-Dioulasso"},
    {"q": "Pourquoi le prix du mil augmente à Fada ?", "role": "MARKET", "zone": "Fada N'Gourma"},

    # -- CROP (Agronomie) --
    {"q": "Quelle distance respecter pour le semis de maïs ?", "role": "CROP", "zone": "General"},
    {"q": "Quelle variété de maïs planter en zone sahélienne ?", "role": "CROP", "zone": "General"},
    {"q": "Comment traiter les jassides sur le coton ?", "role": "CROP", "zone": "General"},
    {"q": "Quand faut-il semer le maïs ?", "role": "CROP", "zone": "General"},

    # -- SOIL --
    {"q": "Comment améliorer un sol argileux ?", "role": "SOIL", "zone": "Vallée du Sourou"},
    {"q": "Peut-on labourer un sol humide ?", "role": "SOIL", "zone": "General"},

    # -- HEALTH (Phyto) --
    {"q": "Quel produit utiliser contre la chenille légionnaire ?", "role": "HEALTH", "zone": "Boucle du Mouhoun"},
    {"q": "Quels ravageurs sont signalés dans la Boucle du Mouhoun ?", "role": "HEALTH", "zone": "Boucle du Mouhoun"},

    # -- SUBSIDY --
    {"q": "Comment obtenir de l'engrais subventionné ?", "role": "SUBSIDY", "zone": "General"},
    {"q": "Quelle est la date limite pour la subvention engrais ?", "role": "SUBSIDY", "zone": "General"},
    {"q": "Y a-t-il des aides pour acheter un tracteur ?", "role": "SUBSIDY", "zone": "General"}
]

def run_benchmark():
    print("📊 DÉMARRAGE DU BENCHMARK RETRIEVER (20 QUESTIONS)\n")
    setup_benchmark_environment()

    # 1. SETUP & INGESTION
    store_handler = VectorStoreHandler(
        index_path=f"{TEST_DATA_DIR}/bench.index",
        metadata_path=f"{TEST_DATA_DIR}/metadata.json"
    )
    ingestor = DataIngestor()
    ingestor.store_handler = store_handler # Inject store
    
    print("📥 Ingestion des données de test...")
    mock_data = mock_comprehensive_data()
    ingestor.ingest_data_from_orchestrator(mock_data)
    print(f"✅ {store_handler.index.ntotal} documents vectorisés.\n")

    # 2. INITIALIZE RETRIEVER
    storage_manager = StorageManager(db_path=TEST_DB_PATH)
    retriever = AgentRetriever(
        store=store_handler,
        embedder=ingestor.embedder,
        reranker=Reranker(),
        storage=storage_manager
    )

    # 3. RUN QUESTIONS
    total_score = 0
    start_time = time.time()

    # Prepare Report
    report_path = "benchmark_results.md"
    md_lines = [
        "# 📊 Rapport de Benchmark RAG",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| ID | Rôle | Question | Source Trouvée | Score |",
        "|----|------|----------|----------------|-------|"
    ]

    print(f"{'ID':<3} | {'ROLE':<8} | {'TOPIC DU DOCUMENT TROUVÉ (Top 1)':<50} | {'SCORE':<8}")
    print("-" * 80)

    for i, item in enumerate(QUESTIONS_BENCHMARK):
        q_text = item["q"]
        role = item["role"]
        zone = item["zone"]
        
        # Call Retriever
        results = retriever.retrieve_for_agent(q_text, role, zone_id=zone, limit=1)
        
        # Display Result
        top_doc = results[0]["content"] if results else "AUCUN RÉSULTAT"
        title = results[0].get("source") or "Inconnu"
        score = results[0]["score"] if results else 0.0
        
        # Terminal Display
        display_doc = (str(title) + ": " + str(top_doc))[:47] + "..."
        print(f"{i+1:<3} | {role:<8} | {display_doc:<50} | {score:.4f}")
        
        # Report Line
        doc_preview = (str(title) + ": " + str(top_doc)).replace("\n", " ")[:100].replace("|", "-")
        md_lines.append(f"| {i+1} | {role} | {q_text} | {doc_preview} | {score:.4f} |")

    end_time = time.time()
    duration = end_time - start_time
    
    md_lines.append("")
    md_lines.append(f"**Temps total:** {duration:.2f}s")
    md_lines.append(f"**Moyenne:** {duration/len(QUESTIONS_BENCHMARK):.2f}s/req")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("-" * 80)
    print(f"\n⏱️ Benchmark terminé en {duration:.2f}s")
    print(f"📄 Rapport détaillé sauvegardé dans : {os.path.abspath(report_path)}")

if __name__ == "__main__":
    run_benchmark()
