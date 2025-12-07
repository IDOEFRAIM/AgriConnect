from datetime import date, timedelta
from typing import List, Dict, Optional

class CropScienceLib:
    """
    Logique pure de phytotechnie : phénologie, rotation et calcul de dates.
    """

    # Règles de rotation simplifiées (Précédent -> Suivants conseillés)
    ROTATION_RULES = {
        "coton": ["mais", "sorgho", "mil"],  # Coton épuise, on met une céréale
        "legumineuse": ["mais", "sorgho", "mil", "coton"], # Fixe l'azote, bon pour tout
        "cereale": ["legumineuse", "arachide", "niebe"], # On alterne avec une légumineuse
        "jachere": ["coton", "mais", "sorgho", "mil", "arachide"] # Terre reposée
    }

    @staticmethod
    def calculate_remaining_growing_season(current_date: date, end_of_season_date: date) -> int:
        """Calcule le nombre de jours de pluie restants."""
        delta = end_of_season_date - current_date
        return max(0, delta.days)

    @staticmethod
    def filter_varieties_by_cycle(varieties: List[Dict], days_available: int) -> List[Dict]:
        """
        Ne garde que les variétés qui ont le temps d'arriver à maturité.
        Si on sème tard, on doit choisir des cycles courts.
        """
        suitable = []
        for var in varieties:
            # Marge de sécurité de 10 jours
            if var["cycle_days"] <= (days_available - 10):
                suitable.append(var)
        return suitable

    @staticmethod
    def check_rotation_compatibility(previous_crop: str, target_crop_category: str) -> str:
        """Vérifie si la rotation est bonne pour le sol."""
        prev = previous_crop.lower()
        target = target_crop_category.lower()
        
        # Si on ne connait pas le précédent, on suppose que c'est OK mais avec risque
        if prev not in CropScienceLib.ROTATION_RULES:
            return "UNKNOWN_RISK"
            
        allowed = CropScienceLib.ROTATION_RULES[prev]
        # Vérification souple
        if any(t in target for t in allowed):
            return "EXCELLENT" # Rotation bénéfique
        elif prev in target: 
            return "BAD_MONOCULTURE" # Coton sur Coton = Mauvais
        else:
            return "NEUTRAL"