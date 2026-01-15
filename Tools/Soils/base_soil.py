import logging
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# --- CONFIGURATION EXPERTE DES SOLS DU BURKINA ---

@dataclass(frozen=True)
class SahelSoilProfile:
    name: str
    local_term: str
    description: str
    water_retention: str
    pwp_fc: tuple  # (Point de Flétrissement, Capacité au Champ) en % vol
    ces_technique: str
    ces_description: str

class SoilDoctorTool:
    """
    Expert Pédologique pour la zone sahélienne.
    Gère la santé du sol, la rétention d'eau et les techniques de conservation (CES).
    """

    def __init__(self, data_path: str = "data/soils_data.json"):
        self.logger = logging.getLogger("SoilDoctorTool")
        self.data_path = data_path
        self._SOILS = self._load_soils()

    def _load_soils(self) -> Dict[str, SahelSoilProfile]:
        # Fallback de secours si le fichier JSON est absent
        default_data = {
            "sableux": SahelSoilProfile("Sableux", "Sénon", "Sol filtrant, pauvre en nutriments", "Faible", (5, 12), "Zaï", "Creusez des poquets de 20cm de profondeur."),
            "ferrugineux": SahelSoilProfile("Ferrugineux", "Kolloko", "Sols rouges, tendance à la battance", "Moyenne", (10, 22), "Cordons Pierreux", "Lignes de pierres pour ralentir l'érosion."),
            "argileux": SahelSoilProfile("Argileux", "Bolo", "Sol lourd, fertile mais difficile à travailler", "Haute", (15, 30), "Demi-lunes", "Aménagements en arc de cercle pour capter les eaux.")
        }

        if not os.path.exists(self.data_path):
            self.logger.warning("Fichier de données sols non trouvé. Utilisation des données par défaut.")
            return default_data

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k: SahelSoilProfile(**{**v, "pwp_fc": tuple(v["pwp_fc"])}) for k, v in data.items()}
        except Exception as e:
            self.logger.error(f"Erreur de chargement des sols : {e}")
            return default_data

    def get_full_diagnosis(self, texture: str, obs_text: str, ph: float = 6.5) -> Dict[str, Any]:
        """Effectue un diagnostic complet : Texture, Eau et Chimie."""
        profile = self._SOILS.get(texture.lower(), self._SOILS.get("sableux"))
        
        # 1. Analyse de l'humidité par observation (Heuristique sahélienne)
        moisture_est = self._estimate_moisture(obs_text)
        
        # 2. Calcul du besoin en eau (en mm)
        # Formule : (Capacité au champ - Humidité actuelle) * Profondeur racine (ex: 300mm)
        # On exprime l'humidité en fraction décimale pour le calcul
        deficit = max(0, (profile.pwp_fc[1] - moisture_est) / 100)
        water_needed_mm = deficit * 300 

        # 3. Diagnostic pH et choix visuel
        ph_diag = self._analyze_ph_local(ph)
        diagram = "Diagramme Zaï" if "Zaï" in profile.ces_technique else "Diagramme Cordons Pierreux"

        return {
            "soil_type": f"{profile.name} ({profile.local_term})",
            "moisture_status": f"{moisture_est}%",
            "water_to_add": f"{round(water_needed_mm, 1)} mm",
            "ph_analysis": ph_diag,
            "ces_recommendation": {
                "technique": profile.ces_technique,
                "details": profile.ces_description,
                "visual": diagram
            }
        }

    def _estimate_moisture(self, text: str) -> float:
        """Traduit les termes paysans en données numériques d'humidité volumique."""
        t = text.lower()
        if any(w in t for w in ["poussière", "sec", "craquelé"]): return 5.0
        if any(w in t for w in ["frais", "boulette", "humide"]): return 15.0
        if any(w in t for w in ["boue", "colle", "trempé"]): return 30.0
        return 12.0

    def _analyze_ph_local(self, ph: float) -> str:
        if ph < 5.5:
            return f"🔴 **ACIDE ($pH$ {ph})** : Bloque le Phosphore. Ajoutez de la **cendre de bois**."
        if ph > 7.5:
            return f"🟠 **ALCALIN ($pH$ {ph})** : Risque de carence en Fer. Favorisez l'apport de matière organique."
        return f"🟢 **OPTIMAL ($pH$ {ph})** : Sol équilibré."

    def recommend_p_source(self, budget: str = "bas") -> str:
        if budget == "bas":
            return "Utilisez le **Burkina Phosphate (BP)** : 200kg/ha pour une fertilité durable."
        return "Utilisez le **NPK 15-15-15** : 150kg/ha pour un effet immédiat sur la culture."

    def calculate_compost_maturity(self, start_date_str: str) -> Dict[str, Any]:
        """Calcule si le compost est prêt sur une base de 90 jours minimum au Sahel."""
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            maturity_date = start_date + timedelta(days=90)
            days_remaining = (maturity_date - datetime.now()).days
            
            if days_remaining <= 0:
                return {"status": "PRÊT", "message": "Le compost est mûr. Appliquez avant le labour."}
            return {"status": "EN COURS", "days_remaining": days_remaining}
        except:
            return {"status": "ERREUR", "message": "Date invalide."}

    def check_soil_fatigue(self, crop_history: List[str]) -> str:
        if not crop_history: return "Historique vide."
        if len(crop_history) >= 2 and crop_history[-1].lower() == crop_history[-2].lower():
            return f"⛔ **ALERTE FATIGUE** : Trop de {crop_history[-1]} successifs. Risque de Striga élevé."
        return "✅ **ROTATION CORRECTE**."