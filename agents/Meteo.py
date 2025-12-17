import logging
from datetime import datetime
from typing import TypedDict, Dict, Any, Optional

# --- Importations LangChain & LangGraph ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from Tools.meteo.basis_tools import SahelAgriAdvisor,SoilType

# Configuration du Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SahelAgent.Robust")

# ==============================================================================
# 1. ÉTAT DE L'AGENT AVEC GESTION D'ERREUR
# ==============================================================================
class AgentState(TypedDict):
    user_query: str
    weather_data: Dict[str, Any]
    culture_info: Dict[str, Any]
    raw_diagnosis: Optional[Dict[str, Any]]
    final_response: str
    error_log: list[str]

# ==============================================================================
# 2. L'AGENT ROBUSTE
# ==============================================================================
class MeteoAgent:
    def __init__(self, OLLAMA_MODEL="llama3:8b", ollama_host="http://localhost:11434",llm_client=None):
        self.advisor = SahelAgriAdvisor()
        self.llm_name = OLLAMA_MODEL
        
        # Initialisation prudente du LLM
        try:
            self.llm_client = llm_client if llm_client else ChatOllama(model=self.llm_name, base_url=ollama_host, temperature=0.1) # adapte la temperature
        except Exception as e:
            logger.error(f"Échec initialisation Ollama: {e}")
            self.llm = None

    # --- ÉTAPE 1: VALIDATION ET CALCUL ---
    def validate_and_calculate(self, state: AgentState):
        """Vérifie la présence des données et effectue les calculs techniques."""
        logger.info("Validation des données d'entrée...")
        errors = []
        
        # Check données météo
        required_weather = ["t_min", "t_max", "rh", "precip"]
        w = state.get("weather_data", {})
        for param in required_weather:
            if param not in w or w[param] is None:
                errors.append(f"Paramètre météo manquant: {param}")

        # Check données culture
        c = state.get("culture_info", {})
        if "crop_name" not in c:
            errors.append("Nom de la culture manquant dans culture_info")

        if errors:
            return {"error_log": errors, "raw_diagnosis": None}

        try:
            # Appel sécurisé de ton Tool
            diagnosis = self.advisor.get_daily_diagnosis(
                crop_key=c["crop_name"],
                soil=c.get("soil_type", SoilType.STANDARD),
                t_min=float(w["t_min"]),
                t_max=float(w["t_max"]),
                rh=float(w["rh"]),
                precip=float(w["precip"]),
                doy=datetime.now().timetuple().tm_yday,
                lat=float(c.get("lat", 14.0))
            )
            return {"raw_diagnosis": diagnosis, "error_log": []}
        except Exception as e:
            logger.error(f"Erreur lors du calcul technique: {e}")
            return {"error_log": [f"Erreur technique: {str(e)}"], "raw_diagnosis": None}

    # --- ÉTAPE 2: GÉNÉRATION DE RÉPONSE (AVEC FALLBACK) ---
    def generate_expert_response(self, state: AgentState):
        """Génère la réponse finale via LLM ou via Template si erreur."""
        
        # CAS D'ERREUR PRÉALABLE
        if state["error_log"]:
            error_msg = " | ".join(state["error_log"])
            return {"final_response": f"⚠️ Désolé, je ne peux pas calculer de conseil précis : {error_msg}. Veuillez vérifier vos capteurs."}

        diag = state["raw_diagnosis"]
        
        # CAS OÙ LE LLM EST INDISPONIBLE
        if not self.llm:
            logger.warning("Mode Fallback : Ollama indisponible.")
            return {"final_response": self._fallback_template(diag)}

        # CAS NORMAL : LLM EXPERT
        system_prompt = "Tu es un agronome expert. Transforme les données techniques en conseils pratiques."
        human_content = f"Données: {diag}. Question: {state['user_query']}"

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_content)
            ])
            return {"final_response": response.content}
        except Exception as e:
            logger.error(f"Erreur LLM: {e}")
            return {"final_response": self._fallback_template(diag)}

    def _fallback_template(self, diag: Dict) -> str:
        """Réponse de secours structurée si le LLM crash."""
        return (
            f"📢 [CONSEIL TECHNIQUE AUTOMATIQUE]\n"
            f"- Culture : {diag['culture']}\n"
            f"- Besoin Eau : {diag['besoin_eau_etc_mm']}mm\n"
            f"- Bilan : {diag['bilan_hydrique_mm']}mm ({diag['conseil_irrigation']})\n"
            f"- Traitement : {diag['pulverisation']} (Delta T: {diag['delta_t']})"
        )

    # --- WORKFLOW ---
    def build(self):
        graph = StateGraph(AgentState)
        graph.add_node("logic", self.validate_and_calculate)
        graph.add_node("expert", self.generate_expert_response)
        
        graph.set_entry_point("logic")
        graph.add_edge("logic", "expert")
        graph.add_edge("expert", END)
        
        return graph.compile()