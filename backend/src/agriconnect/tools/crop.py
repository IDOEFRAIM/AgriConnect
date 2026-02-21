import logging
from typing import Dict, List, Any, Optional
from .shared_math import SahelianCropProfile, CropProfile

logger = logging.getLogger("BurkinaCropTool")

class BurkinaCropTool:
    """
    Outil spécialisé contenant la base de connaissance technique
    des cultures au Burkina Faso (INERA).
    """
    def __init__(self):
        # Base de connaissance INERA simplifiée
        self._SAHELIAN_CROPS = {
            "maïs": SahelianCropProfile(
                "Maïs", {"Nord": ["Komsaya"], "Centre": ["Barka", "Bondofa"], "Sud": ["Espoir"]}, 
                90, "80cm x 40cm", 5, 5.0, {"JAS15": "NPK", "JAS30": "Urée"}, "Zaï et Cordons pierreux"
            ),
            "niébé": SahelianCropProfile(
                "Niébé", {"Nord": ["KVX"], "Centre": ["Komcallé"], "Sud": ["Nafi"]}, 
                70, "50cm x 20cm", 3, 2.5, {"JAS0": "Foscapel"}, "Bandes enherbées"
            ),
            "sorgho": SahelianCropProfile(
                "Sorgho", {"Nord": ["Sariaso 14"], "Centre": ["Kapelga"], "Sud": ["Pisnou"]}, 
                110, "80cm x 40cm", 4, 3.0, {"JAS20": "NPK"}, "Demi-lunes"
            )
        }
        
        # Profils agronomiques pour calculs
        self.MATH_PROFILES = {
            "maïs": CropProfile("Maïs", 10, 35, {'ini': 0.3, 'mid': 1.2, 'end': 0.6}, 90, True),
            "niébé": CropProfile("Niébé", 12, 36, {'ini': 0.4, 'mid': 1.0, 'end': 0.35}, 70, False),
            "sorgho": CropProfile("Sorgho", 10, 40, {'ini': 0.3, 'mid': 1.1, 'end': 0.55}, 110, False)
        }

    def get_technical_sheet(self, crop: str, zone: str) -> str:
        """Récupère la fiche technique pour une culture donnée."""
        p = self._SAHELIAN_CROPS.get(crop.lower())
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
        """Calcule les intrants nécessaires."""
        p = self._SAHELIAN_CROPS.get(crop.lower())
        if not p: return {}
        
        is_mais = "maïs" in crop.lower()
        npk = 100 * surface_ha if is_mais else 50 * surface_ha
        urea = 50 * surface_ha if is_mais else 0
        
        return {
            "NPK_sacs": round(npk / 50, 1),
            "Uree_sacs": round(urea / 50, 1),
            "Charrettes": int((p.organic_matter_min_tha * surface_ha) / 0.2)
        }
    
    def get_math_profile(self, crop: str) -> Optional[CropProfile]:
        return self.MATH_PROFILES.get(crop.lower())
