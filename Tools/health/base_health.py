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

# ==============================================================================
# 1. STRUCTURES DE DONNÉES & PROFILS (SAHEL / BURKINA)
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

@dataclass
class DiseaseProfile:
    name: str
    local_names: List[str]
    symptoms_keywords: List[str]
    risk_level: str
    threshold_pct: int
    bio_recipe: str
    chemical_ref: str
    prevention: str

# ==============================================================================
# 2. OUTILS TECHNIQUES (MATH, FLOOD, HEALTH, BURKINA)
# ==============================================================================

class SahelAgroMath:
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
        et0 = 0.0023 * 0.408 * ra * (t_mean + 17.8) * math.sqrt(max(0.001, t_max - t_min))
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

class HealthDoctorTool:
    def __init__(self):
        # Simulation d'une base de données locale
        self._DATA = {
            "maïs": [
                DiseaseProfile("Chenille Légionnaire d'Automne", ["Spodoptera", "Chenille"], ["feuilles trouées", "sciure", "cœur mangé"], 
                               "CRITIQUE", 15, "Solution au Neem ou Piment/Ail", "Emamectine benzoate", "Semis précoces"),
                DiseaseProfile("Striga (Wongo)", ["Wongo", "Striga"], ["fleurs violettes", "jaunissement", "croissance arrêtée"], 
                               "CRITIQUE", 1, "Arrachage manuel avant floraison", "N/A", "Fumure organique massive")
            ]
        }

    def diagnose(self, crop: str, observations: str, rate: float = 0.0) -> Dict[str, Any]:
        candidates = self._DATA.get(crop.lower(), [])
        best_match = None
        score = 0
        obs = observations.lower()
        
        for d in candidates:
            match_score = sum(1 for k in d.symptoms_keywords if k in obs)
            if match_score > score:
                score = match_score
                best_match = d
        
        if not best_match: return {"status": "Inconnu"}
        
        return {
            "nom": best_match.name,
            "alerte": best_match.risk_level,
            "bio": best_match.bio_recipe,
            "chimique": best_match.chemical_ref if rate >= best_match.threshold_pct else "Non requis",
            "prevention": best_match.prevention,
            "diagramme": "Cycle du Striga" if "Striga" in best_match.name else "Cycle de la Chenille Légionnaire"
        }

# ==============================================================================
# 3. AGENT SAHELIEN INTEGRÉ (LANGGRAPH)
# ==============================================================================

class SahelAgriAdvisor:
    def __init__(self):
        self.math = SahelAgroMath()
        self.health = HealthDoctorTool()
        self.crops_math = {
            "maïs": CropProfile("Maïs", 10, 35, {'ini': 0.3, 'mid': 1.2, 'end': 0.6}, 90, True),
            "niébé": CropProfile("Niébé", 12, 36, {'ini': 0.4, 'mid': 1.0, 'end': 0.35}, 70, False)
        }

    def get_full_diagnosis(self, state: Dict) -> dict:
        w, c = state["weather_data"], state["culture_info"]
        crop = self.crops_math.get(c["crop_name"].lower())
        
        # 1. Calculs Hydriques
        et0 = self.math.calculate_hargreaves_et0(w["t_min"], w["t_max"], c["lat"], datetime.now().timetuple().tm_yday)
        etc = et0 * crop.kc['mid']
        delta_t, spray_status = self.math.calculate_delta_t(w["t_max"], w["rh"])
        
        # 2. Diagnostic Santé
        health_diag = self.health.diagnose(c["crop_name"], state.get("user_obs", ""), state.get("infestation_rate", 0.0))
        
        return {
            "culture": crop.name,
            "etc": etc,
            "pulverisation": spray_status,
            "sante_plante": health_diag,
            "distance": c.get("distance_km", 0.0),
            "check_terrain": c.get("distance_km", 0.0) >= 20
        }

class SahelAgent:
    def __init__(self):
        self.advisor = SahelAgriAdvisor()
        self.llm = ChatOllama(model="llama3:8b", temperature=0.1)

    def process(self, state: Dict):
        diag = self.advisor.get_full_diagnosis(state)
        
        prompt = (
            f"Tu es l'Expert AgConnect. Réponds avec fraternité et précision.\n"
            f"Culture: {diag['culture']} | Pulvérisation: {diag['pulverisation']}\n"
            f"Santé détectée: {diag['sante_plante'].get('nom', 'RAS')}\n"
            f"Action Bio: {diag['sante_plante'].get('bio', 'N/A')}\n"
            f"Note Distance: {diag['distance']} km (Terrain requis: {diag['check_terrain']})\n"
            f"Question utilisateur: {state['user_query']}"
        )
        
        response = self.llm.invoke([SystemMessage(content="Expert Agronome Sahélien"), HumanMessage(content=prompt)])
        return response.content

# ==============================================================================
# 4. EXÉCUTION & CONSEILS EXPERTS
# ==============================================================================

if __name__ == "__main__":
    agent = SahelAgent()
    inputs = {
        "user_query": "Il y a des fleurs violettes partout dans mon maïs et les feuilles jaunissent. Que faire ?",
        "user_obs": "fleurs violettes, jaunissement, maïs petit",
        "infestation_rate": 20,
        "weather_data": {"t_min": 24, "t_max": 37, "rh": 40, "precip": 0},
        "culture_info": {"crop_name": "maïs", "lat": 12.3, "lon": -1.5, "distance_km": 12}
    }
    
    print("=== RÉPONSE DE L'AGENT DOCTEUR ===\n")
    print(agent.process(inputs))