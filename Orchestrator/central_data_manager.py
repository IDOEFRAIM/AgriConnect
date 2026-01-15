# orchestrator/central_data_manager.py

import logging
from typing import Dict, Any, TypedDict, Optional, List

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CentralDataManager")

# ======================================================================
# 1. ÉTAT GLOBAL DE L'ORCHESTRATEUR (Contrat de Données)
# ======================================================================

class OrchestratorState(TypedDict):
    """État global partagé entre tous les nœuds de l'orchestrateur."""
    zone_id: str
    user_query: str
    intent: str
    
    # Données injectées par le CentralDataManager
    meteo_data: Optional[Dict[str, Any]]
    culture_config: Optional[Dict[str, Any]]
    soil_config: Optional[Dict[str, Any]]
    user_profile: Optional[Dict[str, Any]]

    # Pipeline output
    agent_output: str
    status: str

# ======================================================================
# 2. GESTIONNAIRE DE DONNÉES CENTRALISÉ
# ======================================================================

class CentralDataManager:
    """
    Récupère et formate les données contextuelles pour les agents spécialisés.
    Fait le pont entre les bases de données (simulées ici) et les besoins des Tools.
    """

    # --- SIMULATION BASE DE DONNÉES UTILISATEUR ---
    _USER_DB = {
        "generic_user": {
            "crop_name": "Maïs",
            "sowing_date": "2024-06-15",
            "is_coop_member": True,
            "gender": "F",
            "soil_type": "sableux",
            "soil_ph": 6.0,
            "organic_level": "moyen",
            "budget": "bas" # Important pour recommend_p_source
        }
    }

    def retrieve_context(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Analyse l'intention et renvoie le dictionnaire de mise à jour de l'état.
        """
        intent = state.get("intent", "UNKNOWN").upper().strip()
        zone = state.get("zone_id", "Default_Zone")
        
        logger.info(f"🔍 Extraction du contexte pour l'intention: {intent} (Zone: {zone})")

        profile = self._USER_DB["generic_user"]
        context_update = {
            "meteo_data": None,
            "culture_config": None,
            "soil_config": None,
            "user_profile": None
        }

        # 1. LOGIQUE MÉTÉO / SANTÉ (METEO & HEALTH)
        if intent in ["METEO", "HEALTH"]:
            context_update["meteo_data"] = {
                "temp_max": 36.5,
                "temp_min": 24.0,
                "humidity": 45,
                "wind_speed_kmh": 18,
                "precip_mm": 0,
                "forecast_7d": "Sec"
            }
            context_update["culture_config"] = {
                "crop_name": profile["crop_name"],
                "stage": "Floraison",
                "vulnerability": "Haute"
            }

        # 2. LOGIQUE CULTURE (CROP)
        elif intent == "CROP":
            context_update["culture_config"] = {
                "crop_name": profile["crop_name"],
                "sowing_date": profile["sowing_date"],
                "stage": "Croissance",
                "days_since_sowing": 45
            }

        # 3. LOGIQUE SOL (SOIL) - Doit matcher SoilDoctorTool
        elif intent == "SOIL":
            context_update["soil_config"] = {
                "texture": profile["soil_type"],
                "ph": profile["soil_ph"],
                "organic_level": profile["organic_level"],
                "budget": profile["budget"],
                "history": "jachère de 2 ans",
                "crop": profile["crop_name"]
            }

        # 4. LOGIQUE SUBVENTION (SUBSIDY) - Doit matcher AgrimarketTool/GrantExpert
        elif intent == "SUBSIDY":
            context_update["user_profile"] = {
                "crop": profile["crop_name"],
                "zone": zone,
                "is_coop_member": profile["is_coop_member"],
                "gender": profile["gender"],
                "budget": profile["budget"],
                "documents_ok": ["CNI", "Titre Foncier"],
                "rga_registered": True
            }

        elif intent == "UNKNOWN":
            logger.warning("⚠️ Intention non reconnue, le contexte sera minimal.")

        return context_update

# ======================================================================
# 3. SCRIPT DE TEST (Validation du contrat)
# ======================================================================

if __name__ == "__main__":
    dm = CentralDataManager()
    
    def test_intent(test_name, intent_str):
        print(f"\n--- {test_name} ---")
        state: OrchestratorState = {
            "zone_id": "Koutiala",
            "user_query": "Test query",
            "intent": intent_str,
            "meteo_data": None, "culture_config": None, "soil_config": None, "user_profile": None,
            "agent_output": "", "status": ""
        }
        updates = dm.retrieve_context(state)
        # Simulation de la mise à jour de l'état dans LangGraph
        state.update(updates)
        
        # Vérification sélective
        for key in ["meteo_data", "culture_config", "soil_config", "user_profile"]:
            if state[key]:
                print(f"✅ {key} récupéré: {state[key]}")

    test_intent("TEST SOIL", "SOIL")
    test_intent("TEST SUBVENTION", "SUBSIDY")
    test_intent("TEST MÉTÉO", "METEO")