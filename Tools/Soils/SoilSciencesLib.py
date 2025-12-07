from typing import Dict, Tuple

class SoilScienceLib:
    """
    Logique agronomique pure pour le sol.
    Gère les propriétés physiques (rétention d'eau) et chimiques (pH).
    """

    # Propriétés hydriques approximatives par type de sol (% volumique)
    # FC: Capacité au Champ (le sol est plein mais ne ruisselle pas)
    # PWP: Point de Flétrissement (la plante ne peut plus boire)
    SOIL_PROPERTIES = {
        "sableux": {"FC": 10, "PWP": 5, "infiltr_rate": "rapide"},
        "limoneux": {"FC": 25, "PWP": 12, "infiltr_rate": "moyen"},
        "argileux": {"FC": 40, "PWP": 25, "infiltr_rate": "lent"},
        "franco-argileux": {"FC": 35, "PWP": 20, "infiltr_rate": "moyen-lent"}
    }

    @staticmethod
    def get_soil_properties(texture: str) -> Dict:
        """Récupère les propriétés ou des valeurs par défaut."""
        # Normalisation basique
        key = texture.lower()
        for k in SoilScienceLib.SOIL_PROPERTIES:
            if k in key:
                return SoilScienceLib.SOIL_PROPERTIES[k]
        return SoilScienceLib.SOIL_PROPERTIES["limoneux"] # Défaut moyen

    @staticmethod
    def calculate_water_needs(current_moisture_pct: float, texture: str, root_depth_cm: int = 30) -> Dict[str, float]:
        """
        Calcule la quantité d'eau nécessaire pour revenir à la Capacité au Champ (FC).
        Retourne : mm d'eau à apporter.
        """
        props = SoilScienceLib.get_soil_properties(texture)
        target_fc = props["FC"]
        
        # Si l'humidité actuelle est supérieure à la capacité, pas d'eau
        if current_moisture_pct >= target_fc:
            return {"mm_to_add": 0.0, "status": "SATURATED"}
        
        # Déficit en %
        deficit_pct = target_fc - current_moisture_pct
        
        # Formule : (Déficit / 100) * Profondeur_racine (mm)
        # Note: root_depth est en cm, on convertit en mm (*10)
        mm_to_add = (deficit_pct / 100) * (root_depth_cm * 10)
        
        status = "NORMAL"
        if current_moisture_pct <= props["PWP"] * 1.2: # Près du point de flétrissement
            status = "CRITICAL_DRY"
            
        return {
            "mm_to_add": round(mm_to_add, 1),
            "status": status,
            "soil_capacity_ref": target_fc
        }

    @staticmethod
    def analyze_ph(ph_value: float) -> str:
        """Analyse basique du pH."""
        if ph_value < 5.5:
            return "ACIDE_CORRECTION_REQUISE"
        elif ph_value > 7.5:
            return "ALCALIN_ATTENTION_BLOCAGE"
        else:
            return "OPTIMAL"