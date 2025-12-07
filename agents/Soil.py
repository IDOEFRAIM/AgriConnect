from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import logging

# Import du toolkit Sol
from Tools.Soils.base_soil import SoilDoctorTool

# Configuration du logger
logger = logging.getLogger("agent.soil_management")

# ==============================================================================
# 1. DÉFINITION DE L'ÉTAT (INPUTS/OUTPUTS)
# ==============================================================================
class AgentState(TypedDict):
    zone_id: str
    user_query: str                  # Ex: "Mon champ est très sableux, que faire ?"
    soil_config: Dict[str, Any]      # Ex: {"texture": "sableux", "ph": 5.5, "history": "fatigué"}
    
    soil_advice: Optional[str]       # Conseil technique brut
    final_response: str              # Réponse formatée
    status: str                      # SUCCESS / ERROR

# ==============================================================================
# 2. SERVICE DE GESTION DES SOLS
# ==============================================================================
class SoilManagementService:
    def __init__(self):
        self.name = "SoilManagementService"
        self.pedologist = SoilDoctorTool()

    def analyze_node(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "").lower()
        config = state.get("soil_config", {})
        
        texture_input = config.get("texture", "")
        if "sable" in query: texture_input = "sableux"
        elif "argile" in query: texture_input = "argileux"
        elif "gravier" in query or "cailloux" in query: texture_input = "gravillonnaire"
        elif "dur" in query and "pluie" in query: texture_input = "limoneux"

        soil_condition = config.get("history", "normal")
        if "pauvre" in query or "rien ne pousse" in query: soil_condition = "pauvre"

        logger.info(f"[{self.name}] Analyse Sol | Texture: {texture_input} | État: {soil_condition}")

        response_parts = []
        has_advice = False

        # --- 1. TEXTURE & CONSERVATION ---
        if texture_input:
            analysis = self.pedologist.analyze_texture(texture_input)
            if analysis["found"]:
                response_parts.append(f"Diagnostic Sol : {analysis['profile']}")
                response_parts.append(f"- Rétention d'eau : {analysis['retention']}")
                response_parts.append(f"- Risques : {', '.join(analysis['risks'])}")
                response_parts.append(f"Technique conseillée : {analysis['recommendation_ces']}")
                has_advice = True
        else:
            response_parts.append("Pour vous conseiller, précisez si votre sol est sableux, argileux, gravillonnaire ou limoneux.")

        # --- 2. FERTILITÉ ---
        if "engrais" in query or "fumier" in query or "pauvre" in query or has_advice:
            phos_advice = self.pedologist.recommend_phosphorus()
            org_advice = self.pedologist.recommend_organic_amendment(soil_condition)
            response_parts.append("Conseils Fertilité :")
            response_parts.append(org_advice)
            response_parts.append(phos_advice)

        # --- 3. ACIDITÉ (pH) ---
        if "acide" in query or "ph" in query:
            ph_val = config.get("ph")
            if ph_val:
                ph_analysis = self.pedologist.analyze_ph(float(ph_val))
                response_parts.append(f"Analyse pH ({ph_val}) : {ph_analysis['status']}")
                response_parts.append(f"Solution : {ph_analysis['solution']}")
            else:
                response_parts.append("Je n'ai pas la mesure de votre pH. Si vous observez une baisse de rendement, apportez de la Dolomie par précaution.")

        # --- 4. CAS PAR DÉFAUT ---
        if not response_parts:
            response_parts.append("Docteur Sol à votre écoute.")
            response_parts.append("Je peux vous aider à restaurer vos terres.")
            response_parts.append("Exemples de questions :")
            response_parts.append("- Mon sol est sableux et l'eau fuit")
            response_parts.append("- Ma terre est fatiguée")
            response_parts.append("- J'ai des cailloux partout (sol gravillonnaire)")

        final_text = "\n".join(response_parts)
        
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
        workflow.add_node("manage_soil", self.analyze_node)
        workflow.set_entry_point("manage_soil")
        workflow.add_edge("manage_soil", END)
        return workflow.compile()
    

if __name__ == "__main__":
    service = SoilManagementService()
    graph = service.get_graph()

    test_state: AgentState = {
        "zone_id": "zone-003",
        "user_query": "Mon champ est très sableux, que faire ?",
        "soil_config": {
            "texture": "sableux",
            "ph": 5.5,
            "history": "fatigué"
        },
        "soil_advice": None,
        "final_response": "",
        "status": ""
    }

    result = graph.invoke(test_state)
    print("=== Résultat du Docteur Sol ===")
    print(result["final_response"])
    print("Status:", result["status"])
