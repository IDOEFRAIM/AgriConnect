import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configuration du logging
logger = logging.getLogger("FloodRiskTool")

class FloodRiskTool:
    """
    Outil pour évaluer les risques d'inondation basés sur les données FANFAR
    et les prévisions météorologiques locales.
    """
    
    def __init__(self, data_path: str = "data/floods/fanfar_latest.json"):
        self.data_path = data_path
        self.risk_levels = {
            1: "Faible",
            2: "Modéré",
            3: "Élevé",
            4: "Critique"
        }

    def _load_latest_data(self) -> List[Dict[str, Any]]:
        """Charge les dernières données d'inondation disponibles."""
        if not os.path.exists(self.data_path):
            logger.warning(f"Aucune donnée d'inondation trouvée à {self.data_path}")
            return []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("features", []) # Structure GeoJSON typique
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données d'inondation: {e}")
            return []

    def check_flood_risk(self, location: str, lat: float, lon: float) -> Dict[str, Any]:
        """
        Vérifie le risque d'inondation pour une localisation donnée.
        
        Args:
            location (str): Nom de la localité (ex: "Ouagadougou")
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            Dict: Analyse du risque
        """
        # Simulation de données si le fichier n'existe pas (pour la démo/fallback)
        # Dans un système réel, on ferait une requête spatiale ou on lancerait le scraper
        
        # Logique simplifiée : Si on est en saison des pluies (Juin-Septembre)
        current_month = datetime.now().month
        is_rainy_season = 6 <= current_month <= 9
        
        base_risk = "Faible"
        advice = "Aucun risque majeur d'inondation détecté pour le moment."
        
        if is_rainy_season:
            base_risk = "Modéré"
            advice = "Saison des pluies : surveillez les niveaux d'eau dans les bas-fonds."

        # Ici, on pourrait intégrer la logique réelle de lecture du fichier JSON FANFAR
        # Pour l'instant, on retourne une structure standardisée
        
        return {
            "location": location,
            "risk_level": base_risk,
            "is_rainy_season": is_rainy_season,
            "alert_message": advice,
            "timestamp": datetime.now().isoformat(),
            "source": "FANFAR / AgConnect System"
        }

    def get_prevention_advice(self, risk_level: str) -> List[str]:
        """Retourne des conseils préventifs basés sur le niveau de risque."""
        advice_map = {
            "Faible": [
                "Curage des caniveaux autour des champs.",
                "Surveillance normale."
            ],
            "Modéré": [
                "Sécuriser les stocks de semences en hauteur.",
                "Créer des saignées pour évacuer l'excès d'eau.",
                "Éviter les semis dans les zones inondables."
            ],
            "Élevé": [
                "ÉVACUATION IMMÉDIATE des zones basses.",
                "Protéger les habitations avec des sacs de sable.",
                "Récolter prématurément si possible pour sauver la production."
            ],
            "Critique": [
                "DANGER DE MORT : Rejoindre les points de rassemblement.",
                "Ne pas tenter de traverser les eaux de crue."
            ]
        }
        return advice_map.get(risk_level, ["Restez vigilant."])

if __name__ == "__main__":
    tool = FloodRiskTool()
    print(tool.check_flood_risk("Ouagadougou", 12.37, -1.52))
