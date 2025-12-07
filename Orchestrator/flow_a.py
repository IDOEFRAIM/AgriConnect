from typing import Dict, Literal, Any
from langgraph.graph import StateGraph, END, START
from datetime import date

# Importation des modules locaux
from state import GlobalAgriState, Severity, Alert
from agents import MeteoAgent, SoilAgent, HealthAgent, MarketAgent, CultureAgent
from services.utils import WeatherScraperService, SoilDataService, SymptomDataService, MarketDataService

# --- EXÉCUTEURS DES AGENTS (NODES) ---

class AgriAgentsExecutor:
    """Encapsule les Agents pour l'exécution dans le graphe."""

    def __init__(self, meteo: MeteoAgent, soil: SoilAgent, health: HealthAgent, market: MarketAgent, culture: CultureAgent):
        self.meteo = meteo
        self.soil = soil
        self.health = health
        self.market = market
        self.culture = culture
    
    def meteo_node_exec(self, state: GlobalAgriState) -> Dict:
        print("--- [Agent Météo] Exécution... ---")
        update = self.meteo.fetch_data_and_calculate_indicators(state)
        update["execution_path"] = ["meteo_agent"]
        return update

    def health_node_exec(self, state: GlobalAgriState) -> Dict:
        print("--- [Agent Santé] Exécution... ---")
        update = self.health.preliminary_diagnosis(state)
        update["execution_path"] = ["health_agent"]
        return update

    def soil_node_exec(self, state: GlobalAgriState) -> Dict:
        print("--- [Agent Sol] Exécution... ---")
        update = self.soil.analyze_soil_status(state)
        update["execution_path"] = ["soil_agent"]
        return update

    def culture_agent_exec(self, state: GlobalAgriState) -> Dict:
        print("--- [Agent Culture] Exécution... ---")
        update = self.culture.advise_on_planning(state)
        update["execution_path"] = ["culture_agent"]
        return update

    def market_node_exec(self, state: GlobalAgriState) -> Dict:
        print("--- [Agent Marché] Exécution... ---")
        update = self.market.analyze_and_advise(state)
        update["execution_path"] = ["market_agent"]
        return update

    def synthesis_node(self, state: GlobalAgriState) -> Dict:
        """Nœud final d'agrégation des résultats."""
        print("--- [Synthèse] Génération du rapport... ---")
        alerts = state["global_alerts"]
        
        critical_alerts = [a for a in alerts if a["severity"] == Severity.CRITICAL]
        high_alerts = [a for a in alerts if a["severity"] == Severity.HIGH]

        if critical_alerts:
            status = "CRITICAL"
            advice = "INTERVENTION URGENTE REQUISE. Concentrez-vous sur le traitement immédiat."
        elif high_alerts:
            status = "HIGH_ALERT"
            advice = f"Attention: {high_alerts[0]['source']} a détecté un risque. Suivez les instructions spécifiques."
        else:
            status = "NORMAL"
            advice = "Situation sous contrôle. Suivez les conseils d'irrigation, de planification et de marché."

        report = {
            "status": status,
            "advice": advice,
            "summary_alerts": [f"[{a['source']} | {a['severity'].name}] {a['message']}" for a in alerts],
            "path_taken": state["execution_path"],
            "planning_advice": state.get("culture_data", {}).get("planning_advice", "N/A"),
            "market_info": state.get("market_data")
        }
        
        return {"final_report": report}

# --- SUPERVISEUR INTELLIGENT (ROUTEUR DU FLUX A) ---

def supervisor_router_flow1(state: GlobalAgriState) -> Literal["health_check", "soil_check", "culture_check", "market_check", "emergency_stop"]:
    """
    Routeur du Flux 1 : Séquence Météo -> Santé -> Sol -> Culture -> Marché.
    Court-circuite la routine si une alerte CRITICAL est détectée.
    """
    alerts = state.get("global_alerts", [])
    path = state.get("execution_path", [])
    last_step = path[-1] if path else "start"

    # RÈGLE D'URGENCE: Si CRITICAL, on arrête tout et on va à la synthèse
    if any(a["severity"] == Severity.CRITICAL for a in alerts):
        return "emergency_stop"

    # RÈGLE SÉQUENTIELLE
    if last_step == "meteo_agent":
        return "health_check"
    if last_step == "health_agent":
        return "soil_check"
    if last_step == "soil_agent":
        return "culture_check"
    if last_step == "culture_agent":
        return "market_check"

    return "emergency_stop" 

# --- CONSTRUCTION DU GRAPHE DU FLUX 1 ---

def build_flow1_graph(meteo_agent, soil_agent, health_agent, market_agent, culture_agent):
    workflow = StateGraph(GlobalAgriState)
    executors = AgriAgentsExecutor(meteo_agent, soil_agent, health_agent, market_agent, culture_agent)

    workflow.add_node("meteo_agent", executors.meteo_node_exec)
    workflow.add_node("health_agent", executors.health_node_exec)
    workflow.add_node("soil_agent", executors.soil_node_exec)
    workflow.add_node("culture_agent", executors.culture_agent_exec)
    workflow.add_node("market_agent", executors.market_node_exec)
    workflow.add_node("synthesis", executors.synthesis_node)

    workflow.add_edge(START, "meteo_agent")

    # Météo -> Suite
    workflow.add_conditional_edges(
        "meteo_agent", supervisor_router_flow1, 
        { "health_check": "health_agent", "emergency_stop": "synthesis" }
    )

    # Santé -> Suite
    workflow.add_conditional_edges(
        "health_agent", supervisor_router_flow1, 
        { "soil_check": "soil_agent", "emergency_stop": "synthesis" }
    )

    # Sol -> Suite
    workflow.add_conditional_edges(
        "soil_agent", supervisor_router_flow1, 
        { "culture_check": "culture_agent", "emergency_stop": "synthesis" }
    )

    # Culture -> Suite
    workflow.add_conditional_edges(
        "culture_agent", supervisor_router_flow1, 
        { "market_check": "market_agent", "emergency_stop": "synthesis" }
    )
    
    # Fin de chaîne
    workflow.add_edge("market_agent", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()

# --- EXÉCUTION DU FLUX 1 (Routine Complète) ---

if __name__ == "__main__":
    # 1. Instanciation des services
    meteo_service = WeatherScraperService()
    soil_service = SoilDataService()
    health_service = SymptomDataService()
    market_service = MarketDataService()

    # 2. Instanciation des agents
    meteo_agent_real = MeteoAgent(meteo_service)
    soil_agent_real = SoilAgent(soil_service)
    health_agent_real = HealthAgent(health_service)
    market_agent_real = MarketAgent(market_service)
    culture_agent_real = CultureAgent()

    # 3. Construction du graphe (Flux 1)
    app_flow1 = build_flow1_graph(meteo_agent_real, soil_agent_real, health_agent_real, market_agent_real, culture_agent_real)

    # Cas : Routine Complète (Aucune alerte critique simulée)
    initial_state_routine = {
        "zone_id": "Berrechid",
        "requete_utilisateur": "Génération du rapport hebdomadaire.",
        "global_alerts": [],
        "execution_path": [],
        # L'Agent Santé est configuré pour ne pas détecter d'urgence CRITICAL ici
        "health_raw_data": {"infestation_rate_pct": 0.05} 
    }

    print("--- 🚀 Démarrage du Flux A : Conseil Hebdomadaire (Routine complète) ---")
    result_flow1 = app_flow1.invoke(initial_state_routine)
    
    print("\n--- RÉSULTAT DU FLUX A ---")
    print(f"Chemin exécuté : {result_flow1['execution_path']}")
    print("Synthèse des alertes :")
    for alert in result_flow1['global_alerts']:
        print(f"  - [{alert['source']} | {alert['severity'].name}] {alert['message']}")

    print("\nRAPPORT FINAL :")
    print(f"Statut : {result_flow1['final_report']['status']}")
    print(f"Conseil : {result_flow1['final_report']['advice']}")