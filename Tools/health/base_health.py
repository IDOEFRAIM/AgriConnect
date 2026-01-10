import logging
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class DiseaseProfile:
    """Profil expert d'une pathologie ou d'un ravageur sahélien."""
    name: str
    local_names: List[str]
    symptoms_keywords: List[str]
    risk_level: str  # CRITIQUE, ÉLEVÉ, MOYEN
    threshold_pct: int  # Seuil d'intervention économique (%)
    bio_recipe: str     # Recette détaillée (Neem, Piment, etc.)
    chemical_ref: str   # Molécule de référence (dernier recours)
    prevention: str

class SahelPathologyDB:
    """Base de connaissances spécialisée : Burkina Faso & Sahel."""
    
    def __init__(self):
        self.logger = logging.getLogger("SahelPathologyDB")
        self._DATA = self._load_diseases()

    def _load_diseases(self) -> Dict[str, List[DiseaseProfile]]:
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'diseases_data.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            diseases = {}
            for crop, profiles in data.items():
                diseases[crop] = [DiseaseProfile(**p) for p in profiles]
            return diseases
        except Exception as e:
            self.logger.error(f"Error loading diseases data: {e}")
            return {}

class HealthDoctorTool:
    """
    Outil de diagnostic et de prescription phytosanitaire.
    Optimisé pour le conseil agricole de premier niveau au Burkina Faso.
    """

    def __init__(self):
        self.db = SahelPathologyDB()
        self.logger = logging.getLogger("HealthDoctor")

    def diagnose_and_prescribe(self, crop: str, user_obs: str, infestation_rate: Optional[float] = 0.0) -> Dict[str, Any]:
        """
        Analyse les observations, identifie la menace et propose des solutions graduées.
        """
        crop_key = crop.lower().strip()
        observations = user_obs.lower()
        candidates = self.db._DATA.get(crop_key, [])
        
        best_match = None
        highest_score = 0

        # Keyword matching algorithm
        for disease in candidates:
            score = sum(1 for symp in disease.symptoms_keywords if symp in observations)
            if score > highest_score:
                highest_score = score
                best_match = disease

        # Specific logic for WONGO (Striga)
        if "wongo" in observations or "striga" in observations or "fleurs violettes" in observations:
             # Force Wongo detection even if other keywords are weak
             # Could search for Wongo object in DB, here we simulate it if not found
             if not best_match or "Striga" not in best_match.name:
                 return {
                    "diagnostique": "LE WONGO (Striga Hermonthica)",
                    "niveau_alerte": "CRITIQUE",
                    "diagramme_aide": "Cycle du Striga",
                    "prescription_bio": "Arrachage manuel AVANT la floraison. Rotation avec Coton ou Arachide.",
                    "conseil_chimique": "Aucun herbicide n'est aussi efficace que l'arrachage précoce.",
                    "prevention": "Fumure organique riche (Le Wongo aime les sols pauvres)."
                 }

        if not best_match:
            return {
                "status": "Inconnu",
                "message": "Symptômes non identifiés. Inspectez l'envers des feuilles et les racines.",
                "action": "Consultez l'agent de vulgarisation le plus proche."
            }

        # Détermination de l'urgence
        needs_chemical = infestation_rate >= best_match.threshold_pct
        
        # Sélection du diagramme contextuel
        diagram = "Diagramme générique"
        if "Chenille" in best_match.name:
            diagram = "Cycle de la Chenille Légionnaire"
        elif "Striga" in best_match.name:
            diagram = "Cycle du Striga"

        return {
            "diagnostique": best_match.name,
            "confiance": "Haute" if highest_score >= 2 else "Moyenne",
            "niveau_alerte": best_match.risk_level,
            "diagramme_aide": diagram,
            "prescription_bio": best_match.bio_recipe,
            "seuil_alerte": f"{best_match.threshold_pct}%",
            "conseil_chimique": best_match.chemical_ref if needs_chemical else "Non nécessaire à ce stade.",
            "prevention": best_match.prevention
        }

    def get_biopesticide_tutorial(self, recipe_type: str) -> str:
        """Fournit les étapes de préparation des solutions locales (Neem, Cendre, Piment)."""
        header = "🧪 **RECETTE APPROUVÉE PAR LE DOCTEUR DES PLANTES**"
        recipes = {
            "neem": f"{header}\n1. Piler 1kg de graines ou 5kg de feuilles de Neem.\n2. Mélanger dans 10L d'eau.\n3. Laisser reposer 12h (une nuit).\n4. Filtrer avec un pagne et ajouter une cuillère de savon liquide (pour coller).",
            "cendre": f"{header}\n1. Tamiser de la cendre de bois froide.\n2. Saupoudrer tôt le matin sur les feuilles humides de rosée.\n3. Répéter après chaque pluie.",
            "piment": f"{header}\n1. Piler 100g de piment mûr avec 5 gousses d'ail.\n2. Mélanger dans 10L d'eau savonneuse.\n3. ATTENTION : Porter un masque/foulard lors de la pulvérisation !"
        }
        return recipes.get(recipe_type.lower(), "Recette non trouvée.")