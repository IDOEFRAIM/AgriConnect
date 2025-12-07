from typing import Dict, List, Optional

class HealthScienceLib:
    """
    Règles agronomiques pour le diagnostic des maladies et des stress.
    """
    
    # Règle de base pour la chenille légionnaire (Spodoptera frugiperda) sur le maïs
    PEST_THRESHOLDS = {
        "chenille_legionnaire": {
            "seuil_intervention_pct": 20, # 20% des plants infestés nécessitent une action
            "symptomes_cles": ["trous ronds", "défoliation", "excréments"]
        },
        "rouille": {
            "seuil_intervention_pct": 5, # Seuils plus bas pour les maladies fongiques
            "symptomes_cles": ["pustules orangées", "face inférieure feuille"]
        }
    }

    @staticmethod
    def diagnose_disease(symptoms: List[str], culture_type: str) -> List[str]:
        """
        Diagnostique les problèmes potentiels basés sur les symptômes rapportés.
        """
        diagnoses = []
        symptoms_str = " ".join(symptoms).lower()
        
        # Exemple de règles de diagnostic
        if "taches jaunes" in symptoms_str and "nervures" in symptoms_str:
            diagnoses.append("Carence en Azote (Stress Nutritionnel)")
        
        if "flétrissement" in symptoms_str or "feuilles sèches" in symptoms_str:
            diagnoses.append("Stress Hydrique Sévère")
            
        if "trous" in symptoms_str or "morsures" in symptoms_str:
            diagnoses.append("Attaque de Ravageurs (Insectes)")
            
        # Vérification spécifique Chenille Légionnaire (ex: sur maïs/sorgho)
        if culture_type in ["mais", "sorgho"] and any(k in symptoms_str for k in HealthScienceLib.PEST_THRESHOLDS["chenille_legionnaire"]["symptomes_cles"]):
            diagnoses.append("Chenille Légionnaire (Spodoptera)")

        return list(set(diagnoses))

    @staticmethod
    def check_intervention_needed(pest_name: str, infestation_pct: float) -> Optional[str]:
        """Vérifie si le seuil d'intervention économique est dépassé."""
        if pest_name in HealthScienceLib.PEST_THRESHOLDS:
            seuil = HealthScienceLib.PEST_THRESHOLDS[pest_name]["seuil_intervention_pct"]
            if infestation_pct >= seuil:
                return f"ACTION_URGENTE: Infestation de {pest_name} ({infestation_pct}%) dépasse le seuil de {seuil}%."
        return None