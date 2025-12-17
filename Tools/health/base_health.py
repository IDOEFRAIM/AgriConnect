import logging
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
    
    _DATA = {
        "maïs": [
            DiseaseProfile(
                name="Chenille Légionnaire d'Automne (CLA)",
                local_names=["Spodoptera", "Chenille du cornet"],
                symptoms_keywords=["y inversé", "trous ronds", "crottes", "cornet mangé", "larve verte"],
                risk_level="CRITIQUE",
                threshold_pct=20,
                bio_recipe="Mélanger 50g de piment pilé + 50g d'ail + 1 cuillère de savon dans 10L d'eau. Filtrer et pulvériser le soir.",
                chemical_ref="Émamectine benzoate ou Chlorantraniliprole.",
                prevention="Semis précoces et destruction des résidus de récolte."
            ),
            DiseaseProfile(
                name="Striga",
                local_names=["Herbe sorcière", "Wéogo"],
                symptoms_keywords=["fleurs violettes", "jaunissement", "rabougrissement", "parasite"],
                risk_level="CRITIQUE",
                threshold_pct=1,
                bio_recipe="Application de fumure organique massive ou culture de niébé 'faux-hôte' (variété Tiligré).",
                chemical_ref="Peu efficace. Préférer 2,4-D en post-levée dirigée.",
                prevention="Rotation stricte avec le coton et arrachage avant floraison."
            )
        ],
        "niébé": [
            DiseaseProfile(
                name="Thrips des fleurs",
                local_names=["Pucerons noirs", "Coulure des fleurs"],
                symptoms_keywords=["fleurs tombent", "boutons noirs", "insectes minuscules", "coulure"],
                risk_level="ÉLEVÉ",
                threshold_pct=10,
                bio_recipe="Extrait aqueux de feuilles de Neem (5kg feuilles pilées dans 10L d'eau pendant 12h).",
                chemical_ref="Deltaméthrine ou Lambda-cyhalothrine.",
                prevention="Éviter les semis trop denses."
            )
        ]
    }

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

        # Algorithme de matching par mots-clés
        for disease in candidates:
            score = sum(1 for symp in disease.symptoms_keywords if symp in observations)
            if score > highest_score:
                highest_score = score
                best_match = disease

        if not best_match:
            return {
                "status": "Inconnu",
                "message": "Symptômes non identifiés. Inspectez l'envers des feuilles et les racines.",
                "action": "Consultez l'agent de vulgarisation le plus proche."
            }

        # Détermination de l'urgence
        needs_chemical = infestation_rate >= best_match.threshold_pct
        
        # Sélection du diagramme contextuel
        diagram = ""
        if "Chenille" in best_match.name:
            diagram = ""
        elif "Striga" in best_match.name:
            diagram = ""

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
        recipes = {
            "neem": "1. Piler 1kg de graines ou 5kg de feuilles. 2. Mélanger dans 10L d'eau. 3. Laisser reposer 12h. 4. Filtrer et ajouter un peu de savon liquide.",
            "cendre": "Saupoudrer la cendre de bois tamisée tôt le matin sur les feuilles humides pour lutter contre les pucerons.",
            "savon": "Diluer 50g de savon de Marseille/Noir dans 5L d'eau contre les acariens."
        }
        return recipes.get(recipe_type.lower(), "Recette non trouvée.")