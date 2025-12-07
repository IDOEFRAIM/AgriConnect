import logging
from typing import List, Dict

logger = logging.getLogger("services.crops")

class CropKnowledgeServices:
    """
    Base de connaissances hybride : Scientifique + Savoir Local.
    """
    
    # Simulation d'une DB de semences adaptées au Sahel
    SEEDS_DB = [
        {
            "name": "Sorgho Sariaso 14",
            "category": "cereale",
            "cycle_days": 110,
            "yield_potential": "Haut",
            "resistance": "Moyenne",
            "local_tags": ["sorgho blanc", "bon pour tô", "cycle long"]
        },
        {
            "name": "Mil HKP (Hâtif)",
            "category": "cereale",
            "cycle_days": 85,
            "yield_potential": "Moyen",
            "resistance": "Haute sécheresse",
            "local_tags": ["mil hâtif", "soudure", "résistant"]
        },
        {
            "name": "Niébé Kom-Calle",
            "category": "legumineuse",
            "cycle_days": 70,
            "yield_potential": "Moyen",
            "resistance": "Haute",
            "local_tags": ["haricot", "rapide", "nourriture bétail"]
        },
        {
            "name": "Coton FK 37",
            "category": "coton",
            "cycle_days": 120,
            "yield_potential": "Très Haut",
            "resistance": "Faible (Demande intrants)",
            "local_tags": ["or blanc", "argent"]
        }
    ]

    def get_varieties(self, crop_type: str = None) -> List[Dict]:
        """Récupère les variétés, filtre optionnel."""
        if not crop_type:
            return self.SEEDS_DB
        return [v for v in self.SEEDS_DB if crop_type.lower() in v["category"] or crop_type.lower() in v["name"].lower()]

    def interpret_local_intent(self, user_text: str) -> Dict:
        """
        Traduit une demande paysanne en critères techniques.
        Ex: "Je veux manger vite" -> Cycle court (période de soudure).
        """
        intent = {"priority": "yield"} # Par défaut
        
        txt = user_text.lower()
        if "faim" in txt or "vite" in txt or "soudure" in txt:
            intent["priority"] = "speed" # Cycle court prioritaire
            intent["max_cycle"] = 90
        elif "argent" in txt or "vendre" in txt:
            intent["priority"] = "cash_crop"
        elif "sec" in txt or "peu de pluie" in txt:
            intent["priority"] = "resistance"
            
        return intent