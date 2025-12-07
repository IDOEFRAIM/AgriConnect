import sys
import os
import json

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.utils.indexer import UniversalIndexer
from retriever import AgentRetriever

def run_full_test():
    print("🚀 DÉMARRAGE DU TEST INTÉGRAL DU SYSTÈME RAG\n")

    # --- 1. PHASE D'INGESTION (Simulation Scraping) ---
    print("📥 PHASE 1 : INGESTION DE DONNÉES SIMULÉES")
    indexer = UniversalIndexer()

    # Donnée 1 : Une alerte inondation (Intéressant pour Hydrologue et Coordinateur)
    mock_flood_data = [{
        "properties": {
            "name": "Ouagadougou",
            "severity_level": "Orange",
            "description": "Niveau du barrage n°3 critique. Débordements possibles dans les quartiers bas."
        }
    }]
    indexer.index_meteo_data(mock_flood_data, "INONDATIONS", "http://fanfar.test")

    # Donnée 2 : Un bulletin agricole (Intéressant pour Agronome)
    mock_agri_doc = {
        "title": "Bulletin Décadaire N°25",
        "period": "Août 2025",
        "download_url": "http://meteo.bf/doc.pdf",
        "text_content": "Les sols sont saturés en eau. Risque élevé de pourrissement des racines du maïs et du sorgho. Il est conseillé de drainer les champs rapidement. Les routes vers le sud sont praticables."
    }
    indexer.index_document(mock_agri_doc)

    # Donnée 3 : Info logistique (Intéressant pour Logisticien)
    mock_road_data = [{
        "properties": {
            "name": "Route Nationale 1",
            "severity_level": "Rouge",
            "description": "Pont submergé à 15km de la capitale. Accès impossible pour les camions."
        }
    }]
    indexer.index_meteo_data(mock_road_data, "INFRASTRUCTURE", "http://fanfar.test")

    print("\n✅ Données indexées avec succès.\n")

    # --- 2. PHASE DE RÉCUPÉRATION (Retrieval) ---
    print("🔍 PHASE 2 : INTERROGATION PAR LES AGENTS")
    retriever = AgentRetriever()
    question = "Quelle est la situation actuelle ?"

    # Test AGENT AGRONOME
    print("\n--- 🧑‍🌾 Agent AGRONOME ---")
    context_agro = retriever.retrieve_for_agent(question, agent_role="Agronome")
    print(context_agro)
    # On s'attend à voir le bulletin sur le maïs et le drainage en premier

    # Test AGENT LOGISTICIEN
    print("\n--- 🚚 Agent LOGISTICIEN ---")
    context_logi = retriever.retrieve_for_agent(question, agent_role="Logisticien")
    print(context_logi)
    # On s'attend à voir l'info sur le pont submergé et la route RN1 en premier

    # Test AGENT HYDROLOGUE
    print("\n--- 💧 Agent HYDROLOGUE ---")
    context_hydro = retriever.retrieve_for_agent(question, agent_role="Hydrologue")
    print(context_hydro)
    # On s'attend à voir l'info sur le barrage en premier

if __name__ == "__main__":
    run_full_test()