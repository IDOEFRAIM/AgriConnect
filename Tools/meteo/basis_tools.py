import logging
import math
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

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

    def _compute_confiance(self, distance_km: float, inc_rain: float) -> tuple:
        if distance_km < 10 and inc_rain < 0.2:
            return ("🟢", "Confiance élevée")
        elif distance_km < 20:
            return ("🟠", "Confiance moyenne")
        else:
            return ("🔴", "Confiance faible (Station éloignée)")

    def get_daily_diagnosis(self, crop_key: str, soil: SoilType, 
                            t_min: float, t_max: float, rh: float, 
                            precip: float, doy: int, lat: float,
                            distance_km: float = 0.0) -> dict:
        crop = self.crops.get(crop_key.lower())
        if not crop: return {"error": "Culture non reconnue"}
        
        et0 = self.math.calculate_hargreaves_et0(t_min, t_max, lat, doy)
        etc = et0 * crop.kc['mid']
        pe = self._calculate_pe(precip, soil)
        water_balance = round(pe - etc, 2)
        heat_alert = t_max > crop.t_max_optimal
        delta_t, spray_status = self.math.calculate_delta_t(t_max, rh)

        inc_rain = 0.0
        needs_ground_check = False
        recos = []

        if distance_km >= 20:
            inc_rain = 0.5
            needs_ground_check = True
            recos.append("Distance importante : Appliquer la procédure de vérification manuelle de l'humidité.")
        elif distance_km >= 10:
            inc_rain = 0.3

        emoji, conf_txt = self._compute_confiance(distance_km, inc_rain)
        
        return {
            "culture": crop.name,
            "etc": etc,
            "bilan": water_balance,
            "conseil_irrigation": "NECESSAIRE" if water_balance < -3 else "SURVEILLANCE",
            "alerte_chaleur": heat_alert,
            "pulverisation": spray_status,
            "delta_t": delta_t,
            "distance": distance_km,
            "recommandations": recos,
            "emoji": emoji,
            "confiance": conf_txt,
            "check_terrain": needs_ground_check,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

    def _calculate_pe(self, rain: float, soil: SoilType) -> float:
        coeffs = {SoilType.SABLEUX: 0.5, SoilType.ARGILEUX: 0.7, SoilType.STANDARD: 0.6}
        return round(rain * coeffs.get(soil, 0.6), 2) if rain > 5 else 0.0

class AgentState(TypedDict):
    user_query: str
    weather_data: Dict[str, Any]
    culture_info: Dict[str, Any]
    raw_diagnosis: Dict[str, Any]
    final_response: str

class SahelAgent:
    def __init__(self, ollama_model="llama3:8b"):
        self.advisor = SahelAgriAdvisor()
        self.llm = ChatOllama(model=ollama_model, temperature=0.1)

    def call_agri_tool_node(self, state: AgentState):
        w = state["weather_data"]
        c = state["culture_info"]
        diag = self.advisor.get_daily_diagnosis(
            crop_key=c["crop_name"],
            soil=c["soil_type"],
            t_min=w["t_min"],
            t_max=w["t_max"],
            rh=w["rh"],
            precip=w["precip"],
            doy=datetime.now().timetuple().tm_yday,
            lat=c["lat"],
            distance_km=c.get("distance_km", 0.0)
        )
        return {"raw_diagnosis": diag}

    def ollama_expert_node(self, state: AgentState):
        diag = state["raw_diagnosis"]
        
        system_prompt = (
            "Tu es un expert agronome sahélien leader. Ton ton est fraternel mais ferme. "
            "Tu transformes les incertitudes de distance en rigueur de terrain. "
            "Si check_terrain est VRAI, tu imposes la vérification manuelle comme une étape obligatoire. "
            "Structure : 1. État global (emoji), 2. Action immédiate, 3. Sécurité terrain si besoin."
        )
        
        human_content = f"""
        Diagnostic {diag['emoji']} ({diag['confiance']}) :
        - Culture : {diag['culture']}
        - Besoin Eau : {diag['etc']} mm | Bilan : {diag['bilan']} mm
        - Irrigation : {diag['conseil_irrigation']}
        - Pulvérisation : {diag['pulverisation']} (Delta T: {diag['delta_t']})
        - Alerte Chaleur : {'OUI' if diag['alerte_chaleur'] else 'NON'}
        - Vérification terrain requise : {diag['check_terrain']}
        - Distance station : {diag['distance']} km

        Question : {state['user_query']}
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

if __name__ == "__main__":
    agent = SahelAgent()
    app = agent.build_graph()

    inputs = {
        "user_query": "La station est loin, est-ce que je peux faire confiance au conseil d'irrigation ?",
        "weather_data": {"t_min": 23.0, "t_max": 39.5, "rh": 30.0, "precip": 0.0},
        "culture_info": {
            "crop_name": "maïs",
            "soil_type": SoilType.SABLEUX,
            "lat": 14.0,
            "distance_km": 28.0
        }
    }

    result = app.invoke(inputs)
    print(result["final_response"])