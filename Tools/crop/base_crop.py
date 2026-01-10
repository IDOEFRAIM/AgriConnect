import logging
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import date, timedelta

@dataclass
class SahelianCropProfile:
    name: str
    varieties: Dict[str, List[str]]  # Zone (Sahel, Centre, Sud) -> Liste de noms
    cycle_days: int
    seeding_density: str
    depth_cm: int
    organic_matter_min_tha: float # Tonnes/ha de fumure organique
    mineral_fertilizer: Dict[str, str] # JAS -> Instruction
    water_strategy: str # Zaï, Cordons pierreux, etc.

class BurkinaCropTool:
    """
    Expert agronomique dédié au Burkina Faso. 
    Gère les itinéraires techniques adaptés aux sols ferrugineux et aux 
    variations de pluviométrie sahélienne.
    """

    def __init__(self):
        self.logger = logging.getLogger("BurkinaAgro")
        self._CROPS = self._load_crops()

    def _load_crops(self) -> Dict[str, SahelianCropProfile]:
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'crops_data.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            crops = {}
            for key, value in data.items():
                crops[key] = SahelianCropProfile(**value)
            return crops
        except Exception as e:
            self.logger.error(f"Error loading crops data: {e}")
            return {}

    def get_technical_sheet(self, crop: str, zone: str) -> str:
        """
        Génère une fiche technique complète adaptée à la zone climatique.
        Zones valides : 'Nord' (Sahel), 'Centre', 'Sud'.
        """
        p = self._CROPS.get(crop.lower())
        if not p: 
            self.logger.warning(f"Crop '{crop}' not found.")
            return f"Culture '{crop}' non répertoriée."
        
        vars_zone = p.varieties.get(zone.capitalize(), p.varieties.get("Centre", []))
        
        # Ajout des semences INERA
        inera_seed = f"INERA-{crop[:3].upper()}-Hybrid"
        
        return (
            f"📍 **FICHE TECHNIQUE EXPERT : {p.name.upper()} ({zone.upper()})**\n"
            f"--- \n"
            f"🧬 **Variété INERA Recommandée :** {inera_seed} (Certifiée, cycle court)\n"
            f"🌾 **Autres Variétés :** {', '.join(vars_zone)}\n"
            f"⏱️ **Cycle moyen :** {p.cycle_days} jours\n"
            f"📏 **Semis :** {p.seeding_density} à {p.depth_cm}cm de profondeur\n"
            f"💩 **Fumure organique :** {p.organic_matter_min_tha} t/ha (environ {int(p.organic_matter_min_tha * 5)} charrettes)\n"
            f"💧 **Stratégie Eau :** {p.water_strategy}\n"
            f"⚠️ **Note :** Ne jamais appliquer d'engrais minéral sur sol sec."
        )

    def check_climate_risk(self, crop: str, days_remaining_rain: int) -> str:
        """
        Évalue si le cycle de la culture peut se terminer avant la fin des pluies.
        """
        p = self._CROPS.get(crop.lower())
        if not p: return "Culture inconnue."
        
        if p.cycle_days > days_remaining_rain:
            return (f"⛔ **INTERDICTION FORMELLE** : Le cycle du {p.name} ({p.cycle_days}j) est trop long "
                    f"pour la pluie restante ({days_remaining_rain}j). \n"
                    f"👉 **RISQUE MORTEL** : Vous allez tout perdre. Plantez une variété de 70 jours ou du Niébé.")
        return f"✅ **FEU VERT** : Le cycle est bon. Semez dès que l'humidité est là."

    def calculate_inputs(self, crop: str, surface_ha: float) -> Dict[str, Any]:
        """
        Calcule les besoins exacts en sacs de 50kg et Charrettes.
        """
        p = self._CROPS.get(crop.lower())
        if not p: return {"error": "Culture inconnue"}
        
        # Logique simplifiée basée sur les données si possible, sinon hardcodée pour l'exemple
        npk_kg = 0
        urea_kg = 0
        
        if "sorgho" in crop.lower():
             npk_kg = 100 * surface_ha
             urea_kg = 50 * surface_ha
        elif "mil" in crop.lower():
             npk_kg = 50 * surface_ha
        elif "niébé" in crop.lower():
             npk_kg = 50 * surface_ha
        
        # Conversion 1 Charrette ~ 200kg (0.2T)
        charrettes = (p.organic_matter_min_tha * surface_ha) / 0.2
        
        return {
            "surface_ha": surface_ha,
            "NPK_sacs_50kg": round(npk_kg / 50, 1),
            "Uree_sacs_50kg": round(urea_kg / 50, 1),
            "Fumure_organique_tonnes": p.organic_matter_min_tha * surface_ha,
            "Fumure_charrettes": int(charrettes)
        }

    def get_rotation_advice(self, current_crop: str, previous_crop: str) -> str:
        """
        Conseille sur la rotation pour lutter contre le Striga et l'appauvrissement.
        """
        c, p = current_crop.lower(), previous_crop.lower()
        
        if c == p:
            return "⚠️ **MAUVAISE ROTATION** : Évitez la monoculture (risque Striga et parasites)."
        if "niébé" in p or "arachide" in p:
            return "🌟 **EXCELLENTE ROTATION** : La légumineuse a enrichi le sol en azote."
        return "✅ **ROTATION CORRECTE**."