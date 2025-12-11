# orchestrator/central_data_manager.py

import logging
from typing import Dict, Any, TypedDict, Optional, List

logger = logging.getLogger("CentralDataManager")

# ======================================================================
# 1. ÉTAT GLOBAL DE L'ORCHESTRATEUR
# ======================================================================

class OrchestratorState(TypedDict):
    """État global partagé entre tous les nœuds de l'orchestrateur."""
    # Informations de base et d'intention
    zone_id: str
    user_query: str
    intent: str
    
    # Données spécifiques aux agents (récupérées par CentralDataManager)
    meteo_data: Optional[Dict]
    culture_config: Optional[Dict]
    soil_config: Optional[Dict]
    user_profile: Optional[Dict]

    # Sorties des agents
    agent_output: str
    status: str
    

class CentralDataManager:
    """
    Récupère les données nécessaires pour chaque agent.
    Cette version est statique mais peut être remplacée plus tard
    par une vraie base de données ou une API.
    """

    # La méthode reçoit l'état global (OrchestratorState)
    def retrieve_context(self, state: OrchestratorState) -> Dict:
        intent = state["intent"]
        zone = state["zone_id"]

        logger.info(f"Retrieving data for intent: {intent} in {zone}")

        # ✅ Profil utilisateur générique (Base de données simulée)
        user_profile = {
            "crop_name": "Maïs",
            "sowing_date": "2024-06-15",
            "is_coop_member": True, # Changement pour test
            "gender": "F",
            "soil_type": "sableux", # Ajout pour uniformité
            "soil_ph": 6.0
        }

        context_to_update = {}

        # ============================================================
        # 1. MÉTÉO (Utilisé par Agent Météo)
        # ============================================================
        if intent == "METEO":
            context_to_update["meteo_data"] = {
                "current": {
                    "temp_max": 36.5,
                    "temp_min": 24.0,
                    "humidity": 45,
                    "wind_speed_kmh": 18,
                    "precip_mm": 0
                }
            }
            context_to_update["culture_config"] = {
                "crop_name": user_profile["crop_name"],
                "stage": "Floraison" # Stade avancé pour le test
            }

        # ============================================================
        # 2. CULTURE (Utilisé par Agent Culture)
        # ============================================================
        elif intent == "CROP":
            context_to_update["culture_config"] = {
                "crop_name": user_profile["crop_name"],
                "sowing_date": user_profile["sowing_date"],
                "stage": "Croissance" 
            }

        # ============================================================
        # 3. SANTÉ DES CULTURES (Utilisé par Agent Santé)
        # ============================================================
        elif intent == "HEALTH":
             context_to_update["culture_config"] = {
                "crop_name": user_profile["crop_name"],
                "stage": "Floraison"
            }

        # ============================================================
        # 4. SOL (Utilisé par Agent Sol)
        # ============================================================
        elif intent == "SOIL":
            context_to_update["soil_config"] = {
                "texture": user_profile["soil_type"],
                "history": "jachère de 2 ans",
                "ph": user_profile["soil_ph"],
                "crop": user_profile["crop_name"]
            }
            

        # ============================================================
        # 5. SUBVENTIONS (Utilisé par Agent Subventions)
        # ============================================================
        elif intent == "SUBSIDY":
            # On passe le profil utilisateur pour que l'agent Subvention filtre les aides
            context_to_update["user_profile"] = {
                "crop": user_profile["crop_name"],
                "zone": zone,
                "is_coop_member": user_profile["is_coop_member"],
                "gender": user_profile["gender"],
                "documents_ok": ["CNI"]
            }

        # Retourne les données à fusionner avec l'état global
        return context_to_update

# ======================================================================
# 2. TEST AVEC if __main__
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_manager = CentralDataManager()

    # État initial simulé par l'orchestrateur après classification
    initial_state: OrchestratorState = {
        "zone_id": "Mopti",
        "user_query": "Il fait très chaud, puis-je traiter mon champ de maïs en floraison ?",
        "intent": "METEO",
        "meteo_data": None,
        "culture_config": None,
        "soil_config": None,
        "user_profile": None,
        "agent_output": "",
        "status": ""
    }
    
    # 1. Test MÉTÉO
    print("\n--- TEST 1 : Récupération MÉTÉO ---")
    retrieved_data_meteo = data_manager.retrieve_context(initial_state)
    final_state_meteo = {**initial_state, **retrieved_data_meteo}
    print(f"Intention: {final_state_meteo['intent']}")
    print(f"Données Météo : {final_state_meteo['meteo_data']}")
    print(f"Config Culture : {final_state_meteo['culture_config']}")
    
    # 2. Test SUBSIDY
    initial_state["intent"] = "SUBSIDY"
    print("\n--- TEST 2 : Récupération SUBVENTION ---")
    retrieved_data_subsidy = data_manager.retrieve_context(initial_state)
    final_state_subsidy = {**initial_state, **retrieved_data_subsidy}
    print(f"Intention: {final_state_subsidy['intent']}")
    print(f"Profil Utilisateur : {final_state_subsidy['user_profile']}")