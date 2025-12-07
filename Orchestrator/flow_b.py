from typing import Literal
from typing import Dict, Literal, Any
from langgraph.graph import StateGraph, END, START
from datetime import date

# Importation des modules locaux (assumant que vous les avez créés)
from state import GlobalAgriState, Severity, Alert
from agents import MeteoAgent, SoilAgent, HealthAgent, MarketAgent, CultureAgent, AgriAgentsExecutor
from services.utils import WeatherScraperService, SoilDataService, SymptomDataService, MarketDataService

def query_classifier_router(state: GlobalAgriState) -> Literal["meteo_check", "health_check", "soil_check", "market_check", "synthesis"]:
    """
    Classificateur qui route la requête de l'utilisateur vers l'agent pertinent.
    Ceci simule un petit modèle de langage (LLM) ou une logique de mots-clés.
    """
    query = state["requete_utilisateur"].lower()

    # Règle 1 : Urgences Météo/Climat
    if any(keyword in query for keyword in ["chaud", "froid", "pluie", "canicule", "vent", "température"]):
        return "meteo_check"

    # Règle 2 : Problèmes Biologiques/Maladies
    if any(keyword in query for keyword in ["feuilles", "jaunes", "taches", "maladie", "insecte", "ravageur", "trous"]):
        return "health_check"

    # Règle 3 : Sol/Eau/Nutriments
    if any(keyword in query for keyword in ["ph", "eau", "arroser", "sec", "irrigation", "engrais", "sol"]):
        return "soil_check"
    
    # Règle 4 : Économie/Administration
    if any(keyword in query for keyword in ["prix", "vendre", "subvention", "aide", "crédit", "marché"]):
        return "market_check"

    # Règle 5 : Question Générale / Non classifiée
    # Si non spécifique, on lance la routine Météo pour établir le contexte environnemental de base.
    return "meteo_check"



# --- CONSTRUCTION DU GRAPHE DU FLUX 2 (RÉACTIF) ---

# Dans flux_b.py

# --- Nouvelle Fonction pour l'initialisation du flux ---
def init_flow_node(state: GlobalAgriState) -> Dict:
    """Nœud initial, ne fait qu'ajouter le chemin et passe l'état."""
    return {"execution_path": ["init_flow"]}

def build_flow2_graph(meteo_agent, soil_agent, health_agent, market_agent, culture_agent):
    workflow = StateGraph(GlobalAgriState)
    executors = AgriAgentsExecutor(meteo_agent, soil_agent, health_agent, market_agent, culture_agent)

    # 1. Ajout des nœuds
    # AJOUT du Nœud initial
    workflow.add_node("init_flow", init_flow_node) 
    
    # SUPPRESSION du nœud "router" (il devient une transition)
    # workflow.add_node("router", query_classifier_router) # <-- Ceci est supprimé
    
    workflow.add_node("meteo_agent", executors.meteo_node_exec)
    workflow.add_node("health_agent", executors.health_node_exec)
    workflow.add_node("soil_agent", executors.soil_node_exec)
    workflow.add_node("market_agent", executors.market_node_exec)
    workflow.add_node("synthesis", executors.synthesis_node)

    # 2. Point d'entrée : On passe à l'initialisation
    workflow.add_edge(START, "init_flow")

    # 3. Transitions Conditionnelles du Router : C'EST LA CLÉ
    # On passe de "init_flow" à l'agent ciblé en utilisant le routeur
    workflow.add_conditional_edges(
        "init_flow", # Le noeud sortant
        query_classifier_router, # La fonction de routage qui retourne la chaîne de destination
        {
            "meteo_check": "meteo_agent",
            "health_check": "health_agent",
            "soil_check": "soil_agent",
            "market_check": "market_agent",
            "synthesis": "synthesis" 
        }
    )

    # 4. Sortie des Agents : L'Agent ciblé va directement à la synthèse
    workflow.add_edge("meteo_agent", "synthesis")
    workflow.add_edge("health_agent", "synthesis")
    workflow.add_edge("soil_agent", "synthesis")
    workflow.add_edge("market_agent", "synthesis")

    # 5. Fin
    workflow.add_edge("synthesis", END)

    return workflow.compile()
# --- 3. EXÉCUTION DU FLUX B (Exemple) ---

if __name__ == "__main__":
    # 1. Instanciation des services
    meteo_service = WeatherScraperService()
    soil_service = SoilDataService()
    health_service = SymptomDataService()
    market_service = MarketDataService()

    # 2. Instanciation des agents (tous nécessaires pour l'AgriAgentsExecutor)
    meteo_agent_real = MeteoAgent(meteo_service)
    soil_agent_real = SoilAgent(soil_service)
    health_agent_real = HealthAgent(health_service)
    market_agent_real = MarketAgent(market_service)
    culture_agent_real = CultureAgent() # Non utilisé par ce flux, mais inclus pour l'Executor

    # 3. Construction du graphe (Flux B)
    app_flow2 = build_flow2_graph(meteo_agent_real, soil_agent_real, health_agent_real, market_agent_real, culture_agent_real)

    # Cas A : Requête Santé (Doit aller à health_agent) tout ceci est envoye par l api
    initial_state_health = {
        "zone_id": "Berrechid",
        "requete_utilisateur": "J'ai vu des petites taches jaunes sur les feuilles de maïs, que dois-je faire ?",
        "global_alerts": [],
        "execution_path": [],
        "health_raw_data": {"infestation_rate_pct": 0.15} # Simuler une détection CRITICAL,obtenu grace a resnet par exemple:scan d une feuille malade
    }
    
    print("\n--- 🚀 Démarrage du Flux B : Réponse à l'Utilisateur (Cas Santé) ---")
    result_flow2_health = app_flow2.invoke(initial_state_health)
    
    print(f"\nChemin exécuté : {result_flow2_health['execution_path']}")
    print(f"Agent ciblé : {result_flow2_health['execution_path'][1]}")
    print(f"Conseil : {result_flow2_health['final_report']['advice']}")
    
    print("-" * 30)

    # Cas B : Requête Marché (Doit aller à market_agent)
    initial_state_market = {
        "zone_id": "Berrechid",
        "requete_utilisateur": "Quel est le prix actuel du mil et y a-t-il une subvention pour l'engrais ?",
        "global_alerts": [],
        "execution_path": [],
    }
    
    print("\n--- 🚀 Démarrage du Flux B : Réponse à l'Utilisateur (Cas Marché) ---")
    result_flow2_market = app_flow2.invoke(initial_state_market)
    
    print(f"\nChemin exécuté : {result_flow2_market['execution_path']}")
    print(f"Agent ciblé : {result_flow2_market['execution_path'][1]}")
    print(f"Conseil : {result_flow2_market['final_report']['advice']}")