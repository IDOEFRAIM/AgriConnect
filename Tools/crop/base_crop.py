from dataclasses import dataclass
from typing import Dict, Optional

# --- ITINÉRAIRES TECHNIQUES SAHEL (Burkina Faso, zone soudano-sahélienne) ---

@dataclass
class CropTechProfile:
    name: str
    cycle_days: int
    seeding_density: str     # Espacement adapté aux sols tropicaux
    seeds_per_hole: str      # Nombre de graines par poquet
    sowing_depth_cm: int
    fertilizer_schedule: Dict[str, str] # Jour après semis -> Action

class SahelAgronomyDB:
    """Base de données des itinéraires techniques adaptés au Burkina Faso (Sahel tropical)."""
    
    DATA = {
        "maïs": CropTechProfile(
            name="Maïs (Variétés composites locales)",
            cycle_days=95,
            seeding_density="80cm x 40cm (sols ferrugineux tropicaux)",
            seeds_per_hole="2 graines par poquet",
            sowing_depth_cm=5,
            fertilizer_schedule={
                "15": "NPK (14-23-14 ou 15-15-15) : 200 kg/ha. Appliquer après un bon orage (>20 mm).",
                "45": "Urée 46% : 100 kg/ha au début de la montaison. Enfouir et refermer pour éviter volatilisation."
            }
        ),
        "sorgho": CropTechProfile(
            name="Sorgho (Blanc/Rouge, variétés locales)",
            cycle_days=110,
            seeding_density="80cm x 40cm (cordons pierreux conseillés)",
            seeds_per_hole="Démarier à 3 plants maximum",
            sowing_depth_cm=3,
            fertilizer_schedule={
                "21": "NPK 15-15-15 : 100 kg/ha au premier sarclage (sol humide).",
                "45": "Urée 46% : 50 kg/ha si sol pauvre ou après pluie efficace."
            }
        ),
        "coton": CropTechProfile(
            name="Coton (zones SOFITEX)",
            cycle_days=150,
            seeding_density="80cm x 30cm en ligne continue ou 40cm en poquets",
            seeds_per_hole="Démarier à 2 plants vigoureux",
            sowing_depth_cm=3,
            fertilizer_schedule={
                "15": "NPK-SB (Spécial Coton) : 150 kg/ha après démariage.",
                "40": "Urée 46% : 50 kg/ha au début de la floraison.",
                "60": "Urée 46% : 50 kg/ha en pleine floraison."
            }
        ),
        "niébé": CropTechProfile(
            name="Niébé (variétés locales résistantes)",
            cycle_days=70,
            seeding_density="60cm x 30cm (sols sableux ou ferrugineux)",
            seeds_per_hole="2 graines par poquet",
            sowing_depth_cm=3,
            fertilizer_schedule={
                "0": "NPK 15-15-15 : 100 kg/ha au semis (fond).",
                "30": "Pas d'Urée (le niébé fixe l'azote naturellement)."
            }
        )
    }

class CropManagerTool:
    """
    Outil utilisé par l'Agent CROP pour gérer l'itinéraire technique sahélien.
    """
    
    def get_seeding_advice(self, crop_name: str) -> str:
        """Instructions de semis adaptées au Sahel burkinabè."""
        profile = SahelAgronomyDB.DATA.get(crop_name.lower())
        if not profile:
            return f"Pas de données techniques pour '{crop_name}'."
            
        return (f"Guide Semis {profile.name} :\n"
                f"- Espacement : {profile.seeding_density}\n"
                f"- Densité : {profile.seeds_per_hole}\n"
                f"- Profondeur : {profile.sowing_depth_cm} cm (ne pas dépasser, sols tropicaux sensibles à la sécheresse)\n"
                f"- Conseil : Semer après une pluie utile (>20 mm) pour assurer la levée.")

    def check_fertilizer_status(self, crop_name: str, days_after_sowing: int) -> str:
        """Vérifie si un apport d'engrais est nécessaire aujourd'hui."""
        profile = SahelAgronomyDB.DATA.get(crop_name.lower())
        if not profile:
            return "Culture inconnue."

        for day_str, action in profile.fertilizer_schedule.items():
            target_day = int(day_str)
            if target_day - 3 <= days_after_sowing <= target_day + 5:
                return (f"ACTION REQUISE (Jour {days_after_sowing}) :\n"
                        f"{action}\n"
                        f"Note : Appliquer uniquement sur sol humide pour éviter pertes par volatilisation ou ruissellement.")
        
        next_steps = [int(d) for d in profile.fertilizer_schedule.keys() if int(d) > days_after_sowing]
        if next_steps:
            return f"Aucune action aujourd'hui. Prochain apport prévu dans {min(next_steps) - days_after_sowing} jours."
        
        return "Fertilisation terminée pour cette saison."

    def estimate_harvest(self, crop_name: str, sowing_date_str: str) -> str:
        """Estime la date de récolte (simplifiée)."""
        profile = SahelAgronomyDB.DATA.get(crop_name.lower())
        if profile:
            return f"Pour le {profile.name}, la récolte est prévue environ {profile.cycle_days} jours après le semis (selon pluies et sol)."
        return "Durée de cycle inconnue."
