import logging
from datetime import datetime
from typing import TypedDict, Dict, Any, Optional

# --- Importations LangChain & LangGraph ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from tools.meteo.basis_tools import SahelAgriAdvisor,SoilType
from tools.meteo.flood_risk import FloodRiskTool

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
    flood_risk: Optional[Dict[str, Any]]
    final_response: str
    error_log: list[str]

# ==============================================================================
# 2. L'AGENT VIGILANCE CLIMATIQUE
# ==============================================================================
class ClimateVigilance:
    def __init__(self, OLLAMA_MODEL="llama3:8b", ollama_host="http://localhost:11434",llm_client=None):
        self.advisor = SahelAgriAdvisor()
        self.flood_tool = FloodRiskTool()
        self.llm_name = OLLAMA_MODEL
        
        # Initialisation prudente du LLM
        try:
            self.llm = llm_client if llm_client else ChatOllama(model=self.llm_name, base_url=ollama_host, temperature=0.1) # adapte la temperature
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
            return {"error_log": errors, "raw_diagnosis": None, "flood_risk": None}

        try:
            # 1. Calcul Agronomique (Tool existant)
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
            
            # 2. Vérification des Risques d'Inondation (Nouveau Tool)
            # On utilise la localisation fournie ou une valeur par défaut
            location = c.get("location", "Zone Agricole")
            lat = float(c.get("lat", 12.37))
            lon = float(c.get("lon", -1.52))
            
            flood_risk = self.flood_tool.check_flood_risk(location, lat, lon)
            
            return {"raw_diagnosis": diagnosis, "flood_risk": flood_risk, "error_log": []}
        except Exception as e:
            logger.error(f"Erreur lors du calcul technique: {e}")
            return {"error_log": [f"Erreur technique: {str(e)}"], "raw_diagnosis": None, "flood_risk": None}

    # --- ÉTAPE 2: GÉNÉRATION DE RÉPONSE (AVEC FALLBACK) ---
    def generate_expert_response(self, state: AgentState):
        """Génère la réponse finale via LLM ou via Template si erreur."""
        
        # CAS D'ERREUR PRÉALABLE
        if state["error_log"]:
            error_msg = " | ".join(state["error_log"])
            return {"final_response": f"⚠️ Désolé, je ne peux pas calculer de conseil précis : {error_msg}. Veuillez vérifier vos capteurs."}

        diag = state["raw_diagnosis"]
        flood = state.get("flood_risk", {})
        
        # CAS OÙ LE LLM EST INDISPONIBLE
        if not self.llm:
            logger.warning("Mode Fallback : Ollama indisponible.")
            return {"final_response": self._fallback_template(diag, flood)}

        # CAS NORMAL : LLM EXPERT
        system_prompt = (
            "Tu es un assistant agricole intelligent pour le Sahel (Burkina Faso), expert en météo et agronomie. "
            "Ton objectif est de répondre DANS LE CONTEXTE de la question de l'utilisateur.\n\n"
            "RÈGLES D'OR POUR LA GÉNÉRATION DE RÉPONSE :\n"
            "1. 🎯 RÉPONSE DIRECTE : Si l'utilisateur pose une question précise (ex: 'Va-t-il pleuvoir ?'), réponds par OUI ou NON + les prévisions exactes. Ne tourne pas autour du pot.\n"
            "2. 💡 DÉCISION DRIVEN : Ne donne pas juste la météo. Donne l'IMPACT sur la culture. (ex: 'Pas de pluie = Stress Hydrique -> Irriguer ce soir').\n"
            "3. 🚫 PAS DE TEMPLATE RIGIDE : Adapte la structure à la question. Pas besoin de sections 'Plan d'action' si la question est simple.\n"
            "4. 📉 PREUVE : Cite les chiffres clés (mm, °C) pour justifier ton conseil.\n\n"
            "Ton : Expert, Directif, Bienveillant. Tu parles à un agriculteur, sois clair et concis."
        )
        human_content = f"CONTEXTE ACTUEL:\n- Données Agronomiques: {diag}\n- Risque Inondation: {flood}\n- QUESTION UTILISATEUR: '{state['user_query']}'\n\nInstructions: Réponds à la question de l'utilisateur en utilisant les données ci-dessus. Sois bref et précis."

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_content)
            ])
            return {"final_response": response.content}
        except Exception as e:
            logger.error(f"Erreur LLM: {e}")
            return {"final_response": self._fallback_template(diag, flood)}

    def _fallback_template(self, diag: Optional[Dict], flood: Optional[Dict] = None) -> str:
        """Réponse de secours structurée si le LLM crash."""
        flood_msg = ""
        if flood and flood.get("risk_level") in ["Élevé", "Critique"]:
            flood_msg = f"\n⚠️ ALERTE INONDATION ({flood['risk_level']}) : {flood['alert_message']}\n"
        elif flood:
            flood_msg = f"\nℹ️ Info Inondation : {flood['alert_message']}\n"

        if not diag or "error" in diag:
            return (
                f"📢 [CONSEIL TECHNIQUE AUTOMATIQUE]\n"
                f"{flood_msg}"
                f"⚠️ Impossible de calculer le diagnostic agronomique précis.\n"
                f"Raison : {diag.get('error', 'Données manquantes') if diag else 'Données manquantes'}"
            )

        return (
            f"📢 [CONSEIL TECHNIQUE AUTOMATIQUE]\n"
            f"{flood_msg}"
            f"- Culture : {diag.get('culture', 'N/A')}\n"
            f"- Besoin Eau : {diag.get('besoin_eau_etc_mm', 'N/A')}mm\n"
            f"- Bilan : {diag.get('bilan_hydrique_mm', 'N/A')}mm ({diag.get('conseil_irrigation', 'N/A')})\n"
            f"- Traitement : {diag.get('pulverisation', 'N/A')} (Delta T: {diag.get('delta_t', 'N/A')})"
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