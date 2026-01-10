import logging
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# --- CONFIGURATION EXPERTE DES SOLS DU BURKINA ---

@dataclass(frozen=True)
class SahelSoilProfile:
    name: str
    local_term: str
    description: str
    water_retention: str
    pwp_fc: tuple  # (Point de Flétrissement, Capacité au Champ)
    ces_technique: str
    ces_description: str

class SoilDoctorTool:
    """
    Expert Pédologique pour la zone sahélienne.
    Gère la santé du sol, la rétention d'eau et les techniques de conservation (CES).
    """

    def __init__(self):
        self.logger = logging.getLogger("SoilDoctorTool")
        self._SOILS = self._load_soils()

    def _load_soils(self) -> Dict[str, SahelSoilProfile]:
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'soils_data.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            soils = {}
            for key, value in data.items():
                # Convert list to tuple for pwp_fc
                value['pwp_fc'] = tuple(value['pwp_fc'])
                soils[key] = SahelSoilProfile(**value)
            return soils
        except Exception as e:
            self.logger.error(f"Error loading soils data: {e}")
            return {}

    def get_full_diagnosis(self, texture: str, obs_text: str, ph: float = 6.5) -> Dict[str, Any]:
        """Effectue un diagnostic complet : Texture, Eau et Chimie."""
        profile = self._SOILS.get(texture.lower(), self._SOILS.get("sableux"))
        if not profile:
             # Fallback if even default is missing (unlikely)
             return {"error": "Données de sol non disponibles."}
        
        # 1. Analyse de l'humidité par observation (Heuristique sahélienne)
        moisture_est = self._estimate_moisture(obs_text)
        
        # 2. Calcul du besoin en eau (en mm)
        # Formule : (Capacité au champ - Humidité actuelle) * Profondeur racine (300mm par défaut)
        deficit = max(0, profile.pwp_fc[1] - moisture_est)
        water_needed_mm = (deficit / 100) * 300 

        # 3. Diagnostic pH
        ph_diag = self._analyze_ph_local(ph)

        # 4. Choix du diagramme CES
        diagram = "Diagramme Zaï" if "Zaï" in profile.ces_technique else "Diagramme Cordons Pierreux"

        return {
            "soil_type": f"{profile.name} ({profile.local_term})",
            "moisture_status": f"{moisture_est}% (Estimé)",
            "water_to_add": f"{round(water_needed_mm, 1)} mm",
            "ph_analysis": ph_diag,
            "ces_recommendation": {
                "technique": profile.ces_technique,
                "details": profile.ces_description,
                "visual": diagram
            }
        }

    def _estimate_moisture(self, text: str) -> float:
        """Traduit les termes paysans en données numériques d'humidité."""
        t = text.lower()
        if any(w in t for w in ["poussière", "sec", "craquelé"]): return 5.0
        if any(w in t for w in ["frais", "boulette", "humide"]): return 15.0
        if any(w in t for w in ["boue", "colle", "trempé"]): return 30.0
        return 12.0

    def _analyze_ph_local(self, ph: float) -> str:
        """Conseils pH adaptés aux ressources du Burkina."""
        if ph < 5.5:
            return f"🔴 **ACIDE ($pH$ {ph})** : Bloque le Phosphore. Ajoutez de la **cendre de bois** ou de la chaux."
        if ph > 7.5:
            return f"🟠 **ALCALIN ($pH$ {ph})** : Risque de carence en Fer. Utilisez du sulfate d'ammoniaque."
        return f"🟢 **OPTIMAL ($pH$ {ph})** : Sol équilibré."

    def recommend_p_source(self, budget: str = "bas") -> str:
        """Priorise le Burkina Phosphate (BP) pour l'autonomie."""
        if budget == "bas":
            return "Utilisez le **Burkina Phosphate (BP)** : 200kg/ha. Moins cher et durable (effet sur 3 ans)."
        return "Utilisez le **NPK 15-15-15** : 150kg/ha pour un effet immédiat."

    def calculate_compost_maturity(self, start_date: str) -> Dict[str, Any]:
        """
        Estime la date de maturité du compost.
        start_date format: 'YYYY-MM-DD'
        """
        # (Dans un vrai système, on utiliserait datetime, ici on simule)
        return {
            "status": "EN COURS",
            "message": "Le compost doit chauffer pendant 3 semaines (phase thermophile) puis mûrir 3 mois.",
            "test_maturite": "Prenez une poignée : si ça sent la terre de forêt et que c'est noir, c'est bon.",
            "danger": "Si ça chauffe encore ou sent l'ammoniac, c'est TOXIQUE pour les racines."
        }

    def check_soil_fatigue(self, crop_history: List[str]) -> str:
        """Détecte si le sol est épuisé par la monoculture."""
        if not crop_history: return "Pas d'historique."
        last_crop = crop_history[-1].lower()
        if crop_history.count(last_crop) >= 2:
            return f"⛔ **SOL FATIGUÉ** : Trop de {last_crop} successifs. Le sol a faim. Il faut tourner avec de l'Arachide ou du Niébé."
        return "✅ **SOL SAIN** : La rotation semble respectée."