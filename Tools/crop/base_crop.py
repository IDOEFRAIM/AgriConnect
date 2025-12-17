import logging
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

    _CROPS = {
        "sorgho": SahelianCropProfile(
            name="Sorgho (Blanc/Rouge)",
            varieties={
                "Nord": ["Sariaso 14", "Soubatimi"],
                "Centre": ["Kapèlga", "Sariaso 11"],
                "Sud": ["Sariaso 18", "Framida"]
            },
            cycle_days=110,
            seeding_density="80cm x 40cm (31 250 poquets/ha)",
            depth_cm=3,
            organic_matter_min_tha=5.0,
            mineral_fertilizer={
                "15": "NPK (15-15-15) : 100 kg/ha après démariage.",
                "45": "Urée (46%) : 50 kg/ha si l'humidité du sol est suffisante."
            },
            water_strategy="Cordons pierreux ou Zaï sur sols dégradés (Zipellé)."
        ),
        "mil": SahelianCropProfile(
            name="Petit Mil",
            varieties={
                "Nord": ["IKP", "HKP"],
                "Centre": ["MISARI"],
                "Sud": ["SOSAT C88"]
            },
            cycle_days=85,
            seeding_density="1m x 1m (10 000 poquets/ha)",
            depth_cm=3,
            organic_matter_min_tha=3.0,
            mineral_fertilizer={"20": "NPK : 50 kg/ha (optionnel si fumure organique forte)."},
            water_strategy="Demi-lunes pour la récupération des eaux de ruissellement."
        ),
        "niébé": SahelianCropProfile(
            name="Niébé (Haricot)",
            varieties={
                "Nord": ["Kom-callé", "KVX 61-1"],
                "Centre": ["Tiligré", "KVX 396-4-5-2D"],
                "Sud": ["KVX 442-3-25"]
            },
            cycle_days=70,
            seeding_density="50cm x 20cm (Sableux) ou 60cm x 30cm",
            depth_cm=4,
            organic_matter_min_tha=2.5,
            mineral_fertilizer={"0": "NPK : 50 kg/ha au semis. Pas d'Urée (fixe l'azote)."},
            water_strategy="Culture en pur ou associé (Mil/Niébé) pour couvrir le sol."
        )
    }

    def __init__(self):
        self.logger = logging.getLogger("BurkinaAgro")

    def get_technical_sheet(self, crop: str, zone: str) -> str:
        """
        Génère une fiche technique complète adaptée à la zone climatique.
        Zones valides : 'Nord' (Sahel), 'Centre', 'Sud'.
        """
        p = self._CROPS.get(crop.lower())
        if not p: return f"Culture '{crop}' non répertoriée."
        
        vars_zone = p.varieties.get(zone.capitalize(), p.varieties["Centre"])
        
        return (
            f"📍 **FICHE TECHNIQUE : {p.name.upper()} ({zone.upper()})**\n"
            f"--- \n"
            f"🌾 **Variétés conseillées :** {', '.join(vars_zone)}\n"
            f"⏱️ **Cycle moyen :** {p.cycle_days} jours\n"
            f"📏 **Semis :** {p.seeding_density} à {p.depth_cm}cm de profondeur\n"
            f"💩 **Fumure organique :** {p.organic_matter_min_tha} t/ha avant labour\n"
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
            return (f"🚨 **RISQUE ÉLEVÉ** : Le cycle ({p.cycle_days}j) dépasse la fin "
                    f"estimée des pluies ({days_remaining_rain}j). Risque de flétrissement au stade grain.")
        return f"✅ **SÉCURITÉ** : Le cycle de la culture est compatible avec la saison restante."

    def calculate_inputs(self, crop: str, surface_ha: float) -> Dict[str, Any]:
        """
        Calcule les besoins exacts en sacs de 50kg pour une surface donnée.
        """
        p = self._CROPS.get(crop.lower())
        if not p: return {"error": "Culture inconnue"}
        
        # Exemple simplifié pour le Sorgho (100kg NPK, 50kg Urée)
        npk_kg = 100 * surface_ha if "sorgho" in crop.lower() else 50 * surface_ha
        urea_kg = 50 * surface_ha if "sorgho" in crop.lower() else 0
        
        return {
            "surface_ha": surface_ha,
            "NPK_sacs_50kg": round(npk_kg / 50, 1),
            "Uree_sacs_50kg": round(urea_kg / 50, 1),
            "Fumure_organique_tonnes": p.organic_matter_min_tha * surface_ha
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