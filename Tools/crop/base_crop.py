import json
import os
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

@dataclass
class SahelianCropProfile:
    name: str
    varieties: Dict[str, List[str]]
    cycle_days: int
    seeding_density: str
    depth_cm: int
    organic_matter_min_tha: float
    mineral_fertilizer: Dict[str, str]
    water_strategy: str

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

class FloodRiskTool:
    def __init__(self):
        self.risk_levels = {1: "Faible", 2: "Modéré", 3: "Élevé", 4: "Critique"}

    def check_flood_risk(self, location: str, lat: float, lon: float) -> Dict[str, Any]:
        current_month = datetime.now().month
        is_rainy_season = 6 <= current_month <= 9
        base_risk = "Faible"
        advice = "Aucun risque majeur d'inondation détecté."
        if is_rainy_season:
            base_risk = "Modéré"
            advice = "Saison des pluies : surveillez les bas-fonds."
        return {
            "location": location,
            "risk_level": base_risk,
            "is_rainy_season": is_rainy_season,
            "alert_message": advice
        }

    def get_prevention_advice(self, risk_level: str) -> List[str]:
        advice_map = {
            "Faible": ["Curage des caniveaux.", "Surveillance normale."],
            "Modéré": ["Sécuriser les stocks.", "Créer des saignées d'évacuation."],
            "Élevé": ["ÉVACUATION des zones basses.", "Sacs de sable requis."],
            "Critique": ["DANGER DE MORT.", "Rejoindre les points de rassemblement."]
        }
        return advice_map.get(risk_level, ["Restez vigilant."])

class BurkinaCropTool:
    def __init__(self):
        self._CROPS = {
            "maïs": SahelianCropProfile(
                "Maïs", {"Nord": ["Komsaya"], "Centre": ["Barka", "Bondofa"], "Sud": ["Espoir"]}, 
                90, "80cm x 40cm", 5, 5.0, {"JAS15": "NPK", "JAS30": "Urée"}, "Zaï et Cordons pierreux"
            ),
            "niébé": SahelianCropProfile(
                "Niébé", {"Nord": ["KVX"], "Centre": ["Komcallé"], "Sud": ["Nafi"]}, 
                70, "50cm x 20cm", 3, 2.5, {"JAS0": "Foscapel"}, "Bandes enherbées"
            )
        }

    def get_technical_sheet(self, crop: str, zone: str) -> str:
        p = self._CROPS.get(crop.lower())
        if not p: return f"Culture '{crop}' non répertoriée."
        vars_zone = p.varieties.get(zone.capitalize(), p.varieties.get("Centre", []))
        inera_seed = f"INERA-{crop[:3].upper()}-Hybrid"
        return (
            f"📍 **FICHE TECHNIQUE : {p.name.upper()} ({zone.upper()})**\n"
            f"--- \n"
            f"🧬 **Semence INERA :** {inera_seed}\n"
            f"🌾 **Variétés locales :** {', '.join(vars_zone)}\n"
            f"⏱️ **Cycle :** {p.cycle_days} jours\n"
            f"📏 **Semis :** {p.seeding_density} (Prof: {p.depth_cm}cm)\n"
            f"💩 **Fumure :** {p.organic_matter_min_tha} t/ha ({int(p.organic_matter_min_tha * 5)} charrettes)\n"
            f"💧 **Eau :** {p.water_strategy}"
        )

    def calculate_inputs(self, crop: str, surface_ha: float) -> Dict[str, Any]:
        p = self._CROPS.get(crop.lower())
        if not p: return {}
        npk = 100 * surface_ha if "maïs" in crop.lower() else 50 * surface_ha
        urea = 50 * surface_ha if "maïs" in crop.lower() else 0
        return {
            "NPK_sacs": round(npk / 50, 1),
            "Uree_sacs": round(urea / 50, 1),
            "Charrettes": int((p.organic_matter_min_tha * surface_ha) / 0.2)
        }

class SahelAgriAdvisor:
    def __init__(self):
        self.math = SahelAgroMath()
        self.flood_tool = FloodRiskTool()
        self.burkina_tool = BurkinaCropTool()
        self.crops_math = {
            "maïs": CropProfile("Maïs", 10, 35, {'ini': 0.3, 'mid': 1.2, 'end': 0.6}, 90, True),
            "niébé": CropProfile("Niébé", 12, 36, {'ini': 0.4, 'mid': 1.0, 'end': 0.35}, 70, False)
        }

    def get_full_diagnosis(self, crop_key: str, soil: SoilType, 
                           t_min: float, t_max: float, rh: float, 
                           precip: float, doy: int, lat: float, lon: float,
                           zone: str, surface: float, location_name: str, 
                           distance_km: float = 0.0) -> dict:
        crop = self.crops_math.get(crop_key.lower())
        if not crop: return {"error": "Culture non reconnue"}
        
        et0 = self.math.calculate_hargreaves_et0(t_min, t_max, lat, doy)
        etc = et0 * crop.kc['mid']
        delta_t, spray_status = self.math.calculate_delta_t(t_max, rh)
        flood_data = self.flood_tool.check_flood_risk(location_name, lat, lon)
        
        tech_sheet = self.burkina_tool.get_technical_sheet(crop_key, zone)
        inputs = self.burkina_tool.calculate_inputs(crop_key, surface)

        return {
            "culture": crop.name,
            "etc": etc,
            "bilan": round((precip * 0.6) - etc, 2),
            "irrigation": "NECESSAIRE" if ((precip * 0.6) - etc) < -3 else "SURVEILLANCE",
            "pulverisation": spray_status,
            "flood_risk": flood_data["risk_level"],
            "tech_sheet": tech_sheet,
            "inputs": inputs,
            "distance": distance_km,
            "check_terrain": distance_km >= 20,
            "date": datetime.now().strftime("%Y-%m-%d")
        }

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

    def calculate_node(self, state: AgentState):
        w, c = state["weather_data"], state["culture_info"]
        diag = self.advisor.get_full_diagnosis(
            crop_key=c["crop_name"], soil=c["soil_type"],
            t_min=w["t_min"], t_max=w["t_max"], rh=w["rh"], precip=w["precip"],
            doy=datetime.now().timetuple().tm_yday, lat=c["lat"], lon=c["lon"],
            zone=c["zone"], surface=c["surface"], location_name=c["location"],
            distance_km=c.get("distance_km", 0.0)
        )
        return {"raw_diagnosis": diag}

    def explain_node(self, state: AgentState):
        diag = state["raw_diagnosis"]
        system_prompt = (
            "Tu es un expert agronome burkinabè. Ton ton est fraternel, direct et expert. "
            "Priorise la sécurité (inondation/chaleur) puis les conseils techniques (NPK/Urée)."
        )
        human_content = f"""
        Diagnostic {diag['date']} :
        - Culture : {diag['culture']} | Risque Inondation : {diag['flood_risk']}
        - Conseil Eau : {diag['irrigation']} | Pulvérisation : {diag['pulverisation']}
        - Besoins : {diag['inputs']['NPK_sacs']} sacs NPK, {diag['inputs']['Uree_sacs']} Urée, {diag['inputs']['Charrettes']} charrettes de fumure.
        - Fiche Technique : {diag['tech_sheet']}
        - Vérification terrain : {'OBLIGATOIRE' if diag['check_terrain'] else 'NON'} (Distance: {diag['distance']}km)
        
        Question : {state['user_query']}
        """
        response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_content)])
        return {"final_response": response.content}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("calculate", self.calculate_node)
        workflow.add_node("explain", self.explain_node)
        workflow.set_entry_point("calculate")
        workflow.add_edge("calculate", "explain")
        workflow.add_edge("explain", END)
        return workflow.compile()

if __name__ == "__main__":
    agent = SahelAgent()
    app = agent.build_graph()
    inputs = {
        "user_query": "Comment préparer mon champ de 2 hectares de maïs à Ouaga ?",
        "weather_data": {"t_min": 24, "t_max": 38, "rh": 45, "precip": 0},
        "culture_info": {
            "crop_name": "maïs", "soil_type": SoilType.FERRUGINEUX, 
            "lat": 12.3, "lon": -1.5, "zone": "Centre", "surface": 2.0, 
            "location": "Ouagadougou", "distance_km": 15
        }
    }
    result = app.invoke(inputs)
    print(result["final_response"])