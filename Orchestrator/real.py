import operator
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Union,Optional
from enum import Enum
from langgraph.graph import StateGraph, END, START
from datetime import date

# ----------------------------------------------------------------------
# SIMULATION DES CLASSES DES AGENTS PRÉCÉDENTS (pour l'exécution)
# Dans un environnement réel, ces classes seraient importées.

class WeatherScraperService:
    def get_forecast(self, zone_id: str) -> Dict[str, Any]:
        return {"temp_max_c": 38.0, "pluie_mm_prevu": 0, "humidite_relative_pct": 40}
    def get_official_alerts(self, zone_id: str) -> List[str]: return []

class SoilDataService:
    def normalize_input_data(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        return {"texture": "limoneux", "ph": 7.0, "moisture_pct": 15.0, "root_depth_cm": 30}

class SymptomDataService:
    def process_raw_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        return {"symptoms_list": [], "infestation_rate_pct": 0.15, "main_pest_identified": "chenille_legionnaire"}

class MarketDataService:
    def get_price(self, product_name: str) -> Optional[float]: return 250.0

# Logique de calcul simple pour simuler la sortie de l'Agent Météo
def calculate_agri_values(T_max, T_min, hum, vent, pluie_mm_prevu, config) -> Dict[str, float | str]:
    # Logique simplifiée pour retourner des indicateurs
    return {"T_moyenne_C": (T_max + T_min) / 2, "GDD_jour": (T_max + T_min) / 2 - 10, "ET0_mm_jour": 7.2}

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Alert(TypedDict):
    source: str
    message: str
    severity: Severity

class GlobalAgriState(TypedDict):
    # --- Context ---
    zone_id: str
    requete_utilisateur: str
    
    # --- Data Lake (Données collectées) ---
    meteo_data: Dict[str, Any]
    soil_data: Dict[str, Any]
    health_data: Dict[str, Any]
    market_data: Dict[str, Any]
    
    # --- Memory & Outputs ---
    # operator.add permet d'ajouter à la liste existante au lieu de la remplacer
    global_alerts: Annotated[List[Alert], operator.add]
    
    # Trace de décision (pour debugger le chemin pris par le graphe)
    execution_path: Annotated[List[str], operator.add]
    
    # Rapport final
    final_report: Dict[str, Any]

# Simplification des Agents pour l'intégration au Supervisor
class MeteoAgent:
    def __init__(self, scraper_service): self.scraper = scraper_service
    def fetch_data_and_calculate_indicators(self, state: GlobalAgriState) -> Dict:
        raw_data = self.scraper.get_forecast(state["zone_id"])
        indicators = calculate_agri_values(
            raw_data["temp_max_c"], 
            20, # T_min simulée
            raw_data["humidite_relative_pct"], 
            10, # Vent simulé
            raw_data["pluie_mm_prevu"], # Utilisation de la clé correcte
            {"t_base": 10}
        )
        alerts = [{"source": "METEO", "message": "Canicule détectée > 35°C", "severity": Severity.HIGH}] if raw_data["temp_max_c"] > 35 else []
        return {"meteo_data": {"raw": raw_data, "indicators": indicators}, "global_alerts": alerts}
    
    
class SoilAgent:
    def __init__(self, data_service): self.service = data_service
    def analyze_soil_status(self, state: GlobalAgriState) -> Dict:
        # Utilise l'ET0 du MeteoAgent
        et0 = state.get("meteo_data", {}).get("indicators", {}).get("ET0_mm_jour", 0)
        clean_data = self.service.normalize_input_data({})
        deficit = et0 * 1.2 * 30 / 100 # Simplifié
        
        alerts = [{"source": "SOL", "message": f"Stress hydrique critique. Déficit: {deficit:.1f}mm", "severity": Severity.MEDIUM}] if deficit > 5 else []
        return {"soil_data": {"deficit": deficit, "moisture": 15}, "global_alerts": alerts}

class HealthAgent:
    def __init__(self, data_service): self.service = data_service
    def preliminary_diagnosis(self, state: GlobalAgriState) -> Dict:
        clean_data = self.service.process_raw_input(state.get("health_raw_data", {}))
        
        alerts = []
        if clean_data["infestation_rate_pct"] > 0.10:
             alerts.append({"source": "SANTE", "message": "Attaque Chenille Légionnaire (15% infest.)", "severity": Severity.CRITICAL})
        
        return {"health_data": clean_data, "global_alerts": alerts}

class MarketAgent:
    def __init__(self, data_service): self.service = data_service
    def analyze_and_advise(self, state: GlobalAgriState) -> Dict:
        price = self.service.get_price("mil")
        return {"market_data": {"prix_mil": price, "subvention": True}}
# ----------------------------------------------------------------------

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Alert(TypedDict):
    source: str
    message: str
    severity: Severity

# État Global (comme défini par l'utilisateur)
class GlobalAgriState(TypedDict):
    zone_id: str
    requete_utilisateur: str
    meteo_data: Dict[str, Any]
    soil_data: Dict[str, Any]
    health_data: Dict[str, Any]
    market_data: Dict[str, Any]
    global_alerts: Annotated[List[Alert], operator.add]
    execution_path: Annotated[List[str], operator.add]
    final_report: Dict[str, Any]
    health_raw_data: Optional[Dict[str, Any]] # Ajouté pour les inputs
    

class AgriAgentsExecutor:
    """Nœuds du Superviseur appelant les Agents Spécialisés."""

    def __init__(self, meteo_agent: MeteoAgent, soil_agent: SoilAgent, health_agent: HealthAgent, market_agent: MarketAgent):
        self.meteo = meteo_agent
        self.soil = soil_agent
        self.health = health_agent
        self.market = market_agent
        
    def meteo_node_exec(self, state: GlobalAgriState) -> Dict:
        print("--- [Agent Météo] Exécution... ---")
        # Le MeteoAgent retourne un Dict qui est fusionné dans l'état global
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

    def market_node_exec(self, state: GlobalAgriState) -> Dict:
        print("--- [Agent Marché] Exécution... ---")
        update = self.market.analyze_and_advise(state)
        update["execution_path"] = ["market_agent"]
        return update

    def synthesis_node(self, state: GlobalAgriState) -> Dict:
        """Agrégation finale et génération du conseil (Identique à votre logique)."""
        print("--- [Synthèse] Génération du rapport... ---")
        alerts = state["global_alerts"]
        critical_alerts = [a for a in alerts if a["severity"] == Severity.CRITICAL]
        
        if critical_alerts:
            advice = "INTERVENTION URGENTE REQUISE. Traitement phytosanitaire immédiat ou gestion de crise climatique."
            status = "CRITICAL"
        elif any(a["severity"] == Severity.HIGH for a in alerts):
            advice = "Suivre l'alerte Canicule/Déficit et irriguer prioritairement."
            status = "HIGH_ALERT"
        else:
            advice = "Suivre le plan d'irrigation, de fertilisation et de rotation standard."
            status = "NORMAL"

        report = {
            "status": status,
            "advice": advice,
            "summary_alerts": [f"[{a['source']}] {a['message']}" for a in alerts],
            "path_taken": state["execution_path"]
        }
        
        return {"final_report": report}

# --- 2. LE SUPERVISEUR INTELLIGENT (ROUTEUR) ---

def supervisor_router(state: GlobalAgriState) -> Literal["health_check", "soil_check", "market_check", "emergency_stop"]:
    """
    Le Cerveau : Analyse l'état actuel pour décider de la prochaine étape.
    """
    alerts = state.get("global_alerts", [])
    path = state.get("execution_path", [])
    last_step = path[-1] if path else "start"

    severities = [a["severity"] for a in alerts]

    # --- RÈGLE 1 : Urgence Sanitaire/Destructive ---
    if Severity.CRITICAL in severities:
        # Si une urgence menace la récolte (Santé), on va directement à la synthèse pour conseiller
        return "emergency_stop"

    # --- RÈGLE 2 : Flux Standard et Interdépendances ---
    if last_step == "meteo_agent":
        # Après la météo (température, humidité), on vérifie la Santé (risque fongique)
        return "health_check"
    
    if last_step == "health_agent":
        # Si la santé est vérifiée (pas de CRITICAL), on vérifie le sol (irrigation)
        return "soil_check"
    
    if last_step == "soil_agent":
        # Si l'état physique (Météo/Sol) est établi, on passe à l'économie
        return "market_check"

    # Fallback par sécurité
    return "emergency_stop" 

# --- 3. CONSTRUCTION DU GRAPHE FINAL ---

def build_advanced_agri_graph(meteo_agent, soil_agent, health_agent, market_agent):
    workflow = StateGraph(GlobalAgriState)
    
    # Exécuteurs de l'agent
    executors = AgriAgentsExecutor(meteo_agent, soil_agent, health_agent, market_agent)

    # Ajout des nœuds
    workflow.add_node("meteo_agent", executors.meteo_node_exec)
    workflow.add_node("health_agent", executors.health_node_exec)
    workflow.add_node("soil_agent", executors.soil_node_exec)
    workflow.add_node("market_agent", executors.market_node_exec)
    workflow.add_node("synthesis", executors.synthesis_node)

    # Définition du flux
    workflow.add_edge(START, "meteo_agent")

    # Transitions Météo
    workflow.add_conditional_edges(
        "meteo_agent",
        supervisor_router,
        { "health_check": "health_agent", "emergency_stop": "synthesis" }
    )

    # Transitions Santé
    workflow.add_conditional_edges(
        "health_agent",
        supervisor_router,
        { "soil_check": "soil_agent", "emergency_stop": "synthesis" }
    )

    # Transitions Sol
    workflow.add_conditional_edges(
        "soil_agent",
        supervisor_router,
        { "market_check": "market_agent", "emergency_stop": "synthesis" }
    )

    # Fin de chaîne
    workflow.add_edge("market_agent", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()

# --- 4. EXÉCUTION AVEC LES VRAIS AGENTS SIMULÉS ---

if __name__ == "__main__":
    # 1. Instanciation des services
    meteo_service = WeatherScraperService()
    soil_service = SoilDataService()
    health_service = SymptomDataService()
    market_service = MarketDataService()

    # 2. Instanciation des agents (avec injection de dépendance)
    meteo_agent_real = MeteoAgent(meteo_service)
    soil_agent_real = SoilAgent(soil_service)
    health_agent_real = HealthAgent(health_service)
    market_agent_real = MarketAgent(market_service)

    # 3. Construction du graphe avec les agents réels
    app = build_advanced_agri_graph(meteo_agent_real, soil_agent_real, health_agent_real, market_agent_real)

    # État initial (Input utilisateur)
    initial_state = {
        "zone_id": "Berrechid",
        "requete_utilisateur": "J'ai l'impression que la canicule stresse mes cultures. Dois-je m'inquiéter?",
        "global_alerts": [],
        "execution_path": [],
        "health_raw_data": {"observation_text": "Trous dans les feuilles"}
    }

    print("🚀 Démarrage du Superviseur Agricole (Exécution de la séquence complète + Alerte Critique)...")
    result = app.invoke(initial_state)
    
    print("\n--- RÉSULTAT FINAL ---")
    print(f"Chemin exécuté : {result['execution_path']}")
    print("Synthèse des alertes :")
    for alert in result['global_alerts']:
        print(f"  - [{alert['source']} | {alert['severity'].name}] {alert['message']}")

    print("\nRAPPORT FINAL :")
    print(f"Statut : {result['final_report']['status']}")
    print(f"Conseil : {result['final_report']['advice']}")