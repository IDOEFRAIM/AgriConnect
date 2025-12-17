import logging
import math
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# --- Importations LangChain & LangGraph ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

# ==============================================================================
# 1. TON TOOL (SahelAgriAdvisor & Math)
# ==============================================================================
class SoilType(Enum):
    SABLEUX = "sableux"
    ARGILEUX = "argileux"
    LIMONNEUX = "limonneux"
    FERRUGINEUX = "ferrugineux"
    STANDARD = "standard"

@dataclass(frozen=True)
class CropProfile:
    name: str
    t_base: float
    t_max_optimal: float
    kc: Dict[str, float]
    cycle_days: int
    drought_sensitive: bool

class SahelAgroMath:
    GSC = 0.0820
    @staticmethod
    def calculate_hargreaves_et0(t_min: float, t_max: float, lat: float, doy: int) -> float:
        phi = math.radians(lat)
        dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365.0)
        delta = 0.409 * math.sin(2 * math.pi * doy / 365.0 - 1.39)
        x = -math.tan(phi) * math.tan(delta)
        omega_s = math.acos(max(-1.0, min(1.0, x)))
        ra = (24 * 60 / math.pi) * 0.0820 * dr * (
            omega_s * math.sin(phi) * math.sin(delta) +
            math.cos(phi) * math.cos(delta) * math.sin(omega_s)
        )
        t_mean = (t_max + t_min) / 2
        et0 = 0.0023 * 0.408 * ra * (t_mean + 17.8) * math.sqrt(max(0, t_max - t_min))
        return round(et0, 2)

    @staticmethod
    def calculate_delta_t(temp: float, rh: float) -> Tuple[float, str]:
        tw = (temp * math.atan(0.151977 * math.sqrt(rh + 8.313659)) + 
              math.atan(temp + rh) - math.atan(rh - 1.676331) + 
              0.00391838 * (rh**1.5) * math.atan(0.023101 * rh) - 4.686035)
        delta_t = round(temp - tw, 1)
        if 2 <= delta_t <= 8: advice = "OPTIMAL"
        elif delta_t > 10: advice = "DANGER_EVAPORATION"
        else: advice = "RISQUE_LESSIVAGE"
        return delta_t, advice

class SahelAgriAdvisor:
    def __init__(self):
        self.math = SahelAgroMath()
        self.crops = {
            "maïs": CropProfile("Maïs", 10, 35, {'ini': 0.3, 'mid': 1.2, 'end': 0.6}, 90, True),
            "mil": CropProfile("Mil", 12, 42, {'ini': 0.3, 'mid': 1.0, 'end': 0.5}, 100, False),
            "niébé": CropProfile("Niébé", 12, 36, {'ini': 0.4, 'mid': 1.0, 'end': 0.35}, 75, False)
        }

    def get_daily_diagnosis(self, crop_key: str, soil: SoilType, 
                            t_min: float, t_max: float, rh: float, 
                            precip: float, doy: int, lat: float) -> dict:
        crop = self.crops.get(crop_key.lower())
        if not crop: return {"error": "Culture non reconnue"}
        et0 = self.math.calculate_hargreaves_et0(t_min, t_max, lat, doy)
        etc = et0 * crop.kc['mid']
        pe = self._calculate_pe(precip, soil)
        water_balance = round(pe - etc, 2)
        heat_alert = t_max > crop.t_max_optimal
        delta_t, spray_status = self.math.calculate_delta_t(t_max, rh)

        return {
            "culture": crop.name,
            "besoin_eau_etc_mm": etc,
            "bilan_hydrique_mm": water_balance,
            "conseil_irrigation": "NECESSAIRE" if water_balance < -3 else "SURVEILLANCE",
            "alerte_chaleur": heat_alert,
            "pulverisation": spray_status,
            "delta_t": delta_t
        }

    def _calculate_pe(self, rain: float, soil: SoilType) -> float:
        coeffs = {SoilType.SABLEUX: 0.5, SoilType.ARGILEUX: 0.7, SoilType.STANDARD: 0.6}
        return round(rain * coeffs.get(soil, 0.6), 2) if rain > 5 else 0.0

# ==============================================================================
# 2. CONFIGURATION DE L'AGENT LANGGRAPH
# ==============================================================================

class AgentState(TypedDict):
    user_query: str
    weather_data: Dict[str, Any]
    culture_info: Dict[str, Any]
    raw_diagnosis: Dict[str, Any]
    final_response: str

class SahelAgent:
    def __init__(self, ollama_model="llama3:8b"):
        self.advisor = SahelAgriAdvisor()
        self.llm = ChatOllama(model=ollama_model, temperature=0.2)
        logger = logging.getLogger("agent")

    def call_agri_tool_node(self, state: AgentState):
        """Utilise le moteur mathématique pour obtenir des données brutes."""
        w = state["weather_data"]
        c = state["culture_info"]
        
        # On exécute le diagnostic technique
        diagnosis = self.advisor.get_daily_diagnosis(
            crop_key=c["crop_name"],
            soil=c["soil_type"],
            t_min=w["t_min"],
            t_max=w["t_max"],
            rh=w["rh"],
            precip=w["precip"],
            doy=datetime.now().timetuple().tm_yday,
            lat=c["lat"]
        )
        return {"raw_diagnosis": diagnosis}

    def ollama_expert_node(self, state: AgentState):
        """Ollama traduit les chiffres en conseils humains et bienveillants."""
        diag = state["raw_diagnosis"]
        
        system_prompt = (
            "Tu es un expert agronome sahélien. Ton but est d'expliquer les résultats "
            "scientifiques (ETc, Bilan hydrique, Delta T) de manière simple à un paysan. "
            "Sois fraternel, utilise des emojis et donne des conseils pratiques basés sur les chiffres."
        )
        
        human_content = f"""
        Voici le diagnostic technique :
        - Culture : {diag['culture']}
        - Besoin en eau (ETc) : {diag['besoin_eau_etc_mm']} mm
        - Bilan hydrique : {diag['bilan_hydrique_mm']} mm
        - Conseil Irrigation : {diag['conseil_irrigation']}
        - Alerte Chaleur : {'OUI' if diag['alerte_chaleur'] else 'NON'}
        - Condition Pulvérisation (Delta T) : {diag['pulverisation']} (Valeur: {diag['delta_t']})

        Réponds à la question de l'utilisateur : "{state['user_query']}"
        """
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ])
        return {"final_response": response.content}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("calculate", self.call_agri_tool_node)
        workflow.add_node("explain", self.ollama_expert_node)
        
        workflow.set_entry_point("calculate")
        workflow.add_edge("calculate", "explain")
        workflow.add_edge("explain", END)
        
        return workflow.compile()

# ==============================================================================
# 3. TEST DE L'AGENT
# ==============================================================================
if __name__ == "__main__":
    agent_instance = SahelAgent()
    app = agent_instance.build_graph()

    # Simulation de données reçues (ex: via une API météo ou un capteur)
    inputs = {
        "user_query": "Est-ce que c'est le bon moment pour traiter mon champ de maïs et comment va l'irrigation ?",
        "weather_data": {
            "t_min": 22.0, "t_max": 38.0, "rh": 40.0, "precip": 0.0
        },
        "culture_info": {
            "crop_name": "maïs",
            "soil_type": SoilType.SABLEUX,
            "lat": 14.5  # Latitude typique (ex: Sénégal/Mali)
        }
    }

    result = app.invoke(inputs)
    print("\n=== RÉPONSE DE L'AGENT EXPERT ===\n")
    print(result["final_response"])