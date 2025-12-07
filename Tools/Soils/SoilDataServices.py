import logging
from typing import Dict, Any

logger = logging.getLogger("services.soil")

class SoilDataService:
    """
    Interface pour les données du sol.
    Capable de gérer des capteurs IoT simulés OU des observations manuelles.
    """
    
    def interpret_farmer_observation(self, observation_text: str) -> float:
        """
        Convertit une observation en texte ("terre sèche qui s'effrite") 
        en estimation d'humidité (%).
        C'est une heuristic simple qui remplace un capteur coûteux.
        """
        obs = observation_text.lower()
        
        # Mapping heuristique
        if "boue" in obs or "flaque" in obs or "colle" in obs:
            return 35.0 # Très humide
        elif "humide" in obs or "boulette" in obs: # Test du boudin réussi
            return 20.0 # Correct
        elif "sec" in obs or "poussière" in obs or "dur" in obs or "craquelé" in obs:
            return 8.0 # Très sec
        elif "friable" in obs:
            return 15.0
            
        return 18.0 # Valeur par défaut conservatrice

    def normalize_input_data(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nettoie et complète les données entrantes.
        """
        # 1. Gestion de l'humidité : Capteur prioritaire, sinon observation
        moisture = raw_input.get("sensor_moisture_pct")
        
        if moisture is None and raw_input.get("farmer_observation"):
            moisture = self.interpret_farmer_observation(raw_input["farmer_observation"])
            logger.info(f"Estimation humidité basée sur observation: {moisture}%")
        
        if moisture is None:
            moisture = 15.0 # Fallback dangereux, idéalement on demande à l'user
            
        return {
            "texture": raw_input.get("texture", "limoneux"),
            "ph": raw_input.get("ph", 7.0),
            "moisture_pct": moisture,
            "root_depth_cm": raw_input.get("root_depth_cm", 30) # Dépend de la culture normalement
        }