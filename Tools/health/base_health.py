from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DiseaseProfile:
    name: str
    symptoms: List[str]      # Mots clés pour le matching
    bio_control: str         # Solution accessible (Neem, Cendre, Piment)
    chemical_control: str    # Solution conventionnelle (Dernier recours)
    risk_level: str          # CRITIQUE, ÉLEVÉ, MOYEN

class SahelPathologyDB:
    """Base de données des ennemis des cultures au Sahel (Burkina Faso)."""
    
    DB = {
        "maïs": [
            DiseaseProfile(
                name="Chenille Légionnaire d'Automne (CLA)",
                symptoms=["larve verte", "tête noire", "y inversé", "feuilles déchiquetées", "crottes", "cornet détruit"],
                bio_control="Pulvériser un mélange Piment + Ail + Savon, ou déposer de la poudre de graines de Neem dans le cornet.",
                chemical_control="Emamectine Benzoate (si infestation > 20%).",
                risk_level="CRITIQUE"
            ),
            DiseaseProfile(
                name="Striga (Herbe Sorcière)",
                symptoms=["fleurs violettes", "plante parasite", "maïs jauni", "rabougri", "racines"],
                bio_control="Arrachage manuel AVANT floraison. Rotation avec coton ou arachide.",
                chemical_control="Herbicide spécifique en pré-levée (efficacité limitée).",
                risk_level="CRITIQUE"
            )
        ],
        "niébé": [
            DiseaseProfile(
                name="Thrips",
                symptoms=["fleurs tombent", "boutons noirs", "coulure", "petits insectes"],
                bio_control="Pulvériser un extrait aqueux de feuilles de Neem en fin de journée.",
                chemical_control="Deltaméthrine (Decis) au début de la floraison.",
                risk_level="ÉLEVÉ"
            ),
            DiseaseProfile(
                name="Pucerons (Aphis)",
                symptoms=["feuilles collantes", "miellat", "fourmis", "feuilles enroulées", "noir"],
                bio_control="Pulvériser une solution de savon noir ou saupoudrer de la cendre de bois.",
                chemical_control="Acétamipride.",
                risk_level="MOYEN"
            )
        ],
        "coton": [
            DiseaseProfile(
                name="Jassides",
                symptoms=["feuilles rouges", "bord jaune", "recroquevillé", "rougeur"],
                bio_control="Peu de solutions bio efficaces en curatif. Surveiller précocement.",
                chemical_control="Traitement systémique (Flonicamide).",
                risk_level="MOYEN"
            )
        ]
    }

class HealthDoctorTool:
    """
    Outil utilisé par l'Agent HEALTH pour diagnostiquer et prescrire des conseils adaptés au contexte sahélien.
    """
    
    def diagnose(self, crop_name: str, observation: str) -> Dict:
        """
        Analyse une observation textuelle et identifie la maladie probable.
        """
        candidates = SahelPathologyDB.DB.get(crop_name.lower(), [])
        observation = observation.lower()
        
        best_match = None
        max_score = 0
        
        for disease in candidates:
            score = 0
            for symptom in disease.symptoms:
                if symptom in observation:
                    score += 1
            if score > 0 and score > max_score:
                max_score = score
                best_match = disease
        
        if best_match:
            return {
                "found": True,
                "disease": best_match.name,
                "confidence": "Élevée" if max_score >= 2 else "Faible",
                "severity": best_match.risk_level,
                "advice": {
                    "bio": best_match.bio_control,
                    "chimique": best_match.chemical_control
                }
            }
        else:
            return {
                "found": False,
                "message": "Symptômes non reconnus dans la base Sahel. Consultez un agent technique agricole local."
            }

    def get_prevention_plan(self, crop_name: str) -> str:
        """Conseils préventifs généraux adaptés au Burkina Faso."""
        crop_name = crop_name.lower()
        if "maïs" in crop_name:
            return "Prévention Maïs : Surveillez les cornets chaque semaine. Évitez de semer après la mi-juillet pour réduire les risques de chenilles."
        if "niébé" in crop_name:
            return "Prévention Niébé : Utilisez des variétés résistantes au Striga. Surveillez attentivement l'apparition des fleurs, période critique pour les Thrips."
        if "coton" in crop_name:
            return "Prévention Coton : Inspectez régulièrement les feuilles pour détecter les jassides. Maintenez un champ propre pour limiter les hôtes."
        return "Conseil général : Gardez le champ propre par sarclage régulier afin de réduire les parasites et adventices."
