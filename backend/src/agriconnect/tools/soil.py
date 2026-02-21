import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Any
from .shared_math import SoilType

logger = logging.getLogger("SoilDoctorTool")

@dataclass(frozen=True)
class SahelSoilProfile:
    name: str
    local_term: str
    characteristics: str
    pwp_fc: tuple # Point de flétrissement / Capacité au champ
    crops: List[str]  # Cultures adaptées
    ces_technique: str
    ces_description: str

class SoilDoctorTool:
    """
    Outil d'expertise pédologique (Sols) adapté au Sahel.
    Interprète les données SoilGrids ou les observations manuelles.
    """
    def __init__(self):
        self._SOILS = {
            SoilType.SABLEUX: SahelSoilProfile(
                "Sableux (Arenosols)", "Sénon", 
                "Filtrant, risque de lixiviation élevé, pauvre en MO.", 
                (5, 12), ["Mil", "Niébé", "Arachide"], 
                "Zaï & Compost", "Concentration de la fertilité dans des poquets."
            ),
            SoilType.FERRUGINEUX: SahelSoilProfile(
                "Ferrugineux (Lixisols)", "Kolloko", 
                "Sols tropicaux classiques, sensibles à l'érosion et à la battance.", 
                (10, 22), ["Sorgho", "Maïs", "Coton"], 
                "Cordons Pierreux", "Freine le ruissellement et favorise l'infiltration."
            ),
            SoilType.ARGILEUX: SahelSoilProfile(
                "Argileux (Vertisols)", "Bolo", 
                "Riche mais difficile (fentes de retrait), excellente rétention.", 
                (15, 30), ["Riz", "Maraîchage", "Banane"], 
                "Demi-lunes & Drainage", "Gestion de l'excès d'eau en hivernage."
            ),
            # Fallback
            SoilType.LIMONNEUX: SahelSoilProfile(
                "Limoneux", "Baongo",
                "Sol de bas-fond riche mais risque d'engorgement.",
                (12, 25), ["Riz", "Maraîchage"],
                "Drainage", "Evacuation des excès d'eau."
            ),
             SoilType.STANDARD: SahelSoilProfile(
                "Standard (Moyen)", "Tinkunté",
                "Sol équilibré.",
                (10, 20), ["Maïs", "Sorgho"],
                "Bandes enherbées", "Protection contre l'érosion."
            )
        }

    def _get_fertility_score(self, soc: float, nitro: float, cec: float) -> Dict[str, str]:
        """Analyse la richesse réelle du sol (Indice de nutrition)."""
        # SOC (Carbon) en g/kg : < 6 = Critique, > 15 = Bon au Sahel
        # Nitrogen en cg/kg : < 50 = Carence
        # CEC en mmol(c)/kg : < 50 = Sol 'passoire', > 150 = Sol 'réservoir'
        
        score = 0
        if soc > 100: score += 1  # SOC > 10g/kg (SoilGrids unit conversion often needed, assuming dg/kg or similar)
        if nitro > 60: score += 1 
        if cec > 100: score += 1 
        
        status = ["Faible", "Moyen", "Bon", "Excellent"][score]
        advice = "Besoin urgent de fumure organique." if score < 2 else "Maintenir le taux de carbone."
        return {"niveau": status, "conseil": advice}

    def get_diagnosis_from_soilgrids(self, sg_json: Dict[str, Any], observation: str = "sec") -> Dict[str, Any]:
        layers = sg_json.get("layers", {})
        
        # Extraction sécurisée 
        get_val = lambda key: layers.get(key, {}).get("0-5cm", {}).get("mean", 0)
        
        sand = get_val("sand")
        clay = get_val("clay")
        ph = get_val("phh2o") / 10.0 # SoilGrids often pH*10
        soc = get_val("soc")
        nitro = get_val("nitrogen")
        cec = get_val("cec")

        # 1. Classification précise (Heuristique simplifiée du triangle des textures)
        # SoilGrids units: g/kg (0-1000)
        if sand > 650: 
            soil_enum = SoilType.SABLEUX
        elif clay > 350: 
            soil_enum = SoilType.ARGILEUX
        elif sand < 400 and clay < 400:
             soil_enum = SoilType.LIMONNEUX
        else: 
            soil_enum = SoilType.FERRUGINEUX
        
        # Fallback if specific type not in dict (though we covered most)
        profile = self._SOILS.get(soil_enum, self._SOILS[SoilType.FERRUGINEUX])
        
        fertility = self._get_fertility_score(soc, nitro, cec)

        # 2. Calcul du déficit hydrique
        moisture = 5.0 if "sec" in observation.lower() else 15.0
        # Estimation eau requise pour atteindre la capacité au champ (profil pwp_fc)
        water_needed = max(0, (profile.pwp_fc[1] - moisture) / 100) * 300 #mm pour 30cm de sol

        return {
            "identite_pedologique": {
                "type_code": soil_enum.value,
                "nom": profile.name,
                "nom_local": profile.local_term,
                "atouts": profile.characteristics,
                "cultures_recommandees": profile.crops
            },
            "bilan_sante": {
                "fertilite": fertility["niveau"],
                "stockage_nutriments_cec": "Élevé (Réservoir)" if cec > 150 else "Faible (Passoire)",
                "action_organique": fertility["conseil"],
                "alerte_ph": self._analyze_ph(ph)
            },
            "gestion_eau": {
                "besoin_irrigation_estime": f"{round(water_needed, 1)} mm",
                "strategie": profile.ces_technique,
                "details_ces": profile.ces_description
            }
        }

    def _analyze_ph(self, ph: float) -> str:
        if ph < 5.5: return f"Acide ({ph}). Risque de toxicité aluminique. Amender avec de la chaux ou des cendres."
        if ph > 7.8: return f"Alcalin ({ph}). Blocage du Fer et du Zinc."
        return f"Équilibré ({ph})."

# --- TEST AVEC LES DONNÉES DE LOUMBILA ---
if __name__ == "__main__":
    loumbila_raw = {
        "layers": {
            "sand": {"0-5cm": {"mean": 593}}, "clay": {"0-5cm": {"mean": 180}},
            "phh2o": {"0-5cm": {"mean": 6.6}}, "soc": {"0-5cm": {"mean": 82}},
            "nitrogen": {"0-5cm": {"mean": 74}}, "cec": {"0-5cm": {"mean": 92}}
        }
    }
    print(json.dumps(SoilDoctorTool().get_diagnosis_from_soilgrids(loumbila_raw), indent=4, ensure_ascii=False))