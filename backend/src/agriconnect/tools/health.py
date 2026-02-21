import logging
from dataclasses import dataclass
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

@dataclass
class DiseaseProfile:
    name: str
    local_names: List[str]
    symptoms_keywords: List[str]
    risk_level: str
    threshold_pct: int
    bio_recipe: str
    chemical_ref: str
    prevention: str

class HealthDoctorTool:
    """
    Outil de diagnostic phytosanitaire (Maladies & Ravageurs).
    Base de connaissance : INERA / Protection des Végétaux.
    """
    def __init__(self):
        # Base de connaissance simplifiée pour démo
        # TODO: Connecter à une vraie BD ou VectorStore pour RAG spécialisé
        self._DATA = {
            "maïs": [
                DiseaseProfile("Chenille Légionnaire d'Automne", ["Spodoptera", "Chenille"], ["feuilles trouées", "sciure", "cœur mangé"], 
                               "CRITIQUE", 15, "Solution au Neem ou Piment/Ail", "Emamectine benzoate", "Semis précoces"),
                DiseaseProfile("Striga (Wongo)", ["Wongo", "Striga"], ["fleurs violettes", "jaunissement", "croissance arrêtée"], 
                               "CRITIQUE", 1, "Arrachage manuel avant floraison", "N/A", "Fumure organique massive"),
                DiseaseProfile("Charbon", ["Charbon"], ["excroissances noires", "poudre noire"], 
                               "MOYEN", 5, "Enlever et brûler les parties malades", "Thirame (traitement semence)", "Rotation des cultures")
            ],
            "niébé": [
                 DiseaseProfile("Thrips", ["Thrips", "Poux des fleurs"], ["chute des fleurs", "fleurs noires"], 
                               "ELEVÉ", 10, "Extrait de Neem", "Acetamipride", "Semis décalé"),
                 DiseaseProfile("Maruca", ["Foreuse de gousses"], ["trous dans gousses", "déjections"], 
                               "CRITIQUE", 5, "Neem + Savon", "Indoxacarbe", "Variétés résistantes")
            ]
        }

    def diagnose(self, crop: str, observations: str, rate: float = 0.0) -> Dict[str, Any]:
        """Tente d'identifier la maladie basée sur les observations."""
        candidates = self._DATA.get(crop.lower(), [])
        if not candidates:
             return {"status": "Inconnu", "message": f"Pas de données santé pour {crop}"}

        best_match = None
        score = 0
        obs = observations.lower()
        
        for d in candidates:
             # Score basique : nombre de mots-clés trouvés
            match_score = sum(1 for k in d.symptoms_keywords if k in obs)
            # Bonus si le nom local est cité
            match_score += sum(2 for name in d.local_names if name.lower() in obs)
            
            if match_score > score:
                score = match_score
                best_match = d
        
        if not best_match or score == 0: 
            return {
                "status": "Incertain", 
                "message": "Symptômes non reconnus. Consultez un technicien ou envoyez une photo."
            }
        
        return {
            "status": "Identifié",
            "nom": best_match.name,
            "alerte": best_match.risk_level,
            "traitement_bio": best_match.bio_recipe,
            "traitement_chimique": best_match.chemical_ref if rate >= best_match.threshold_pct else "Non requis (Seuil non atteint)",
            "prevention": best_match.prevention,
            "diagramme": f"Cycle biologique de {best_match.name}"
        }
