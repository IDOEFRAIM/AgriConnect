from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import logging
from datetime import datetime

# Import du toolkit Agronomie
from Tools.crop.base_crop import CropManagerTool

# Configuration du logger
logger = logging.getLogger("agent.crop_management")

# ==============================================================================
# 1. DÉFINITION DE L'ÉTAT (INPUTS/OUTPUTS)
# ==============================================================================
class AgentState(TypedDict):
    zone_id: str
    user_query: str                  # Ex: "Quand mettre l'engrais ?"
    culture_config: Dict[str, Any]   # Ex: {"crop_name": "Maïs", "sowing_date": "2024-06-15"}
    
    technical_advice: Optional[str]  # Conseil technique brut
    final_response: str              # Réponse formatée
    status: str                      # SUCCESS / ERROR

# ==============================================================================
# 2. SERVICE DE GESTION DES CULTURES
# ==============================================================================
class CropManagementService:
    def __init__(self):
        self.name = "CropManagementService"
        self.agronomist = CropManagerTool()

    def _calculate_days_after_sowing(self, sowing_date_str: str) -> int:
        """Calcule l'âge de la culture en jours."""
        try:
            s_date = datetime.strptime(sowing_date_str, "%Y-%m-%d")
            today = datetime.now()
            delta = today - s_date
            return max(0, delta.days)
        except Exception:
            return -1

    def analyze_node(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "").lower()
        config = state.get("culture_config", {})
        crop_name = config.get("crop_name", "Inconnue")
        sowing_date = config.get("sowing_date")

        logger.info(f"[{self.name}] Analyse itinéraire pour {crop_name}")

        response_parts = []

        # --- 1. SEMIS & DENSITÉ ---
        if "semis" in query or "semer" in query or "densité" in query or "écartement" in query:
            advice = self.agronomist.get_seeding_advice(crop_name)
            response_parts.append(advice)

        # --- 2. FERTILISATION ---
        elif "engrais" in query or "npk" in query or "urée" in query:
            if not sowing_date:
                response_parts.append("Pour calculer la date d'engrais, j'ai besoin de votre date de semis.")
            else:
                das = self._calculate_days_after_sowing(sowing_date)
                if das == -1 and "simulated_day" in config:
                    das = config["simulated_day"]
                
                if das >= 0:
                    status = self.agronomist.check_fertilizer_status(crop_name, das)
                    response_parts.append(f"Stade de la culture : Jour {das}")
                    response_parts.append(status)
                else:
                    response_parts.append("Date de semis invalide (Format attendu: YYYY-MM-DD).")

        # --- 3. RÉCOLTE ---
        elif "récolte" in query or "couper" in query or "fin" in query:
            estimation = self.agronomist.estimate_harvest(crop_name, sowing_date)
            response_parts.append(f"Estimation de la récolte : {estimation}")

        # --- 4. CONSEIL GÉNÉRAL ---
        else:
            response_parts.append(f"Fiche Technique : {crop_name}")
            response_parts.append("Je peux vous conseiller sur :")
            response_parts.append("- Les densités de semis (écartements)")
            response_parts.append("- Le calendrier d'engrais (NPK/Urée)")
            response_parts.append("- Les dates de récolte")
            response_parts.append("Posez-moi une question précise sur ces sujets.")

        final_text = "\n\n".join(response_parts)

        return {
            **state,
            "final_response": final_text,
            "status": "SUCCESS"
        }

    # ==============================================================================
    # 3. WORKFLOW
    # ==============================================================================
    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("manage_crop", self.analyze_node)
        workflow.set_entry_point("manage_crop")
        workflow.add_edge("manage_crop", END)
        return workflow.compile()


if __name__ == "__main__":
    service = CropManagementService()
    graph = service.get_graph()

    test_state: AgentState = {
        "zone_id": "zone-001",
        "user_query": "Quand mettre l'engrais ?",
        "culture_config": {
            "crop_name": "Maïs",
            "sowing_date": "2024-06-15"
        },
        "technical_advice": None,
        "final_response": "",
        "status": ""
    }

    result = graph.invoke(test_state)
    print("=== Résultat de l'agent ===")
    print(result["final_response"])
    print("Status:", result["status"])
