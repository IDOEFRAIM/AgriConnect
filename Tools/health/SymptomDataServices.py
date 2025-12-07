import logging
from typing import Dict, Any, List

logger = logging.getLogger("services.health")

class SymptomDataService:
    """
    Responsable de l'interprétation des données de l'agriculteur (texte, photos).
    """
    
    # Simulation d'une base de données de symptômes basiques
    SYMPTOMS_MAPPING = {
        "feuille jaune": "jaunissement foliaire",
        "tache marron": "nécrose foliaire",
        "insecte mange": "dommage physique par morsure",
        "poudre blanche": "symptôme fongique (oïdium)"
    }

    def process_raw_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise et enrichit les données de l'utilisateur.
        """
        
        # 1. Traitement des images/données ML (si disponibles)
        # Ici, on simule la sortie d'un modèle d'IA
        ml_diagnosis = raw_input.get("ml_model_output", {})
        
        # 2. Interprétation du texte libre (Symptômes visibles)
        symptoms = []
        text_report = raw_input.get("observation_text", "").lower()
        
        for key, mapped_symptom in self.SYMPTOMS_MAPPING.items():
            if key in text_report:
                symptoms.append(mapped_symptom)
        
        # 3. Fusion des sources
        if ml_diagnosis.get("disease"):
            symptoms.append(f"ML:{ml_diagnosis['disease']}")
        
        # Utilisation de la détection de l'infestation (vital pour la chenille légionnaire)
        infestation_level = ml_diagnosis.get("infestation_rate_pct", 0) # Taux par défaut 
        
        return {
            "symptoms_list": list(set(symptoms)),
            "infestation_rate_pct": infestation_level,
            "main_pest_identified": ml_diagnosis.get("pest", None) # Ex: 'chenille_legionnaire'
        }