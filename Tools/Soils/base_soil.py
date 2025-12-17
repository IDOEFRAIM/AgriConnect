import logging
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

    _SOILS = {
        "sableux": SahelSoilProfile(
            name="Sol Sableux",
            local_term="Dior / Seno",
            description="Sol filtrant, pauvre en humus.",
            water_retention="Faible",
            pwp_fc=(5, 12),
            ces_technique="Zaï et Poquets fertilisés",
            ces_description="Creuser des trous de 20-30cm, y mettre du fumier pour concentrer l'humidité."
        ),
        "gravillonnaire": SahelSoilProfile(
            name="Sol Ferrugineux",
            local_term="Zipellé (si dénudé)",
            description="Sol dur avec gravillons rouges (latérite).",
            water_retention="Moyenne-Basse",
            pwp_fc=(8, 18),
            ces_technique="Cordons pierreux / Demi-lunes",
            ces_description="Barrières de pierres suivant les courbes de niveau pour freiner l'eau."
        ),
        "argileux": SahelSoilProfile(
            name="Sol Argileux",
            local_term="Bas-fond / Baogo",
            description="Sol lourd, riche mais risque d'asphyxie.",
            water_retention="Très Forte",
            pwp_fc=(20, 35),
            ces_technique="Billonnage cloisonné",
            ces_description="Créer des diguettes pour retenir ou évacuer l'eau selon la pluie."
        )
    }

    def get_full_diagnosis(self, texture: str, obs_text: str, ph: float = 6.5) -> Dict[str, Any]:
        """Effectue un diagnostic complet : Texture, Eau et Chimie."""
        profile = self._SOILS.get(texture.lower(), self._SOILS["sableux"])
        
        # 1. Analyse de l'humidité par observation (Heuristique sahélienne)
        moisture_est = self._estimate_moisture(obs_text)
        
        # 2. Calcul du besoin en eau (en mm)
        # Formule : (Capacité au champ - Humidité actuelle) * Profondeur racine (300mm par défaut)
        deficit = max(0, profile.pwp_fc[1] - moisture_est)
        water_needed_mm = (deficit / 100) * 300 

        # 3. Diagnostic pH
        ph_diag = self._analyze_ph_local(ph)

        # 4. Choix du diagramme CES
        diagram = "" if "Zaï" in profile.ces_technique else ""

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