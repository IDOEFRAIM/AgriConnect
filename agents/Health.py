from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import logging

# Import du toolkit Santé
from Tools.health.base_health import HealthDoctorTool

# Configuration du logger
logger = logging.getLogger("agent.plant_health")

# ==============================================================================
# 1. DÉFINITION DE L'ÉTAT (INPUTS/OUTPUTS)
# ==============================================================================
class AgentState(TypedDict):
    zone_id: str
    user_query: str                  # Ex: "Les feuilles de mon maïs sont déchiquetées"
    culture_config: Dict[str, Any]   # Ex: {"crop_name": "Maïs"}
    
    diagnosis: Optional[Dict]        # Résultat structuré du diagnostic
    final_response: str              # Réponse formatée
    status: str                      # SUCCESS / ERROR

# ==============================================================================
# 2. SERVICE DE SANTÉ VÉGÉTALE
# ==============================================================================
class HealthManagementService:
    def __init__(self):
        self.name = "HealthManagementService"
        self.doctor = HealthDoctorTool()

    def analyze_node(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "").lower()
        config = state.get("culture_config", {})
        crop_name = config.get("crop_name", "Inconnue")
        
        logger.info(f"[{self.name}] Diagnostic Santé pour {crop_name} | Symptômes: {query[:30]}...")

        response_parts = []
        has_diagnosis = False

        # --- 1. DIAGNOSTIC ---
        if any(w in query for w in ["maladie", "insecte", "trou", "feuille", "bête", "manger", "jaune", "rouge", "tache"]):
            diag_result = self.doctor.diagnose(crop_name, query)
            
            if diag_result["found"]:
                has_diagnosis = True
                disease = diag_result["disease"]
                advice = diag_result["advice"]
                
                response_parts.append(f"Diagnostic : {disease}")
                response_parts.append(f"Certitude : {diag_result['confidence']}")
                response_parts.append(f"Gravité : {diag_result['severity']}")
                
                response_parts.append("Ordonnance du Phyto-Docteur :")
                response_parts.append(f"- Solution Bio (accessible) : {advice['bio']}")
                response_parts.append(f"- Solution Chimique (cas grave) : {advice['chimique']}")
            else:
                response_parts.append(f"Je ne reconnais pas ces symptômes sur le {crop_name}.")
                response_parts.append(diag_result["message"])
                response_parts.append("Essayez de décrire plus précisément (ex: 'feuilles déchiquetées', 'taches rouges', 'chenilles vertes').")

        # --- 2. PRÉVENTION ---
        if "prévenir" in query or "protéger" in query or (not has_diagnosis and "maladie" not in query):
            prevention = self.doctor.get_prevention_plan(crop_name)
            response_parts.append(prevention)

        # --- 3. CAS PAR DÉFAUT ---
        if not response_parts:
            response_parts.append(f"Clinique des Plantes ({crop_name})")
            response_parts.append("Décrivez-moi ce que vous observez :")
            response_parts.append("- 'Il y a des chenilles dans le cornet'")
            response_parts.append("- 'Les feuilles jaunissent'")
            response_parts.append("- 'Une herbe parasite étouffe mon mil'")

        final_text = "\n".join(response_parts)
        
        return {
            **state,
            "diagnosis": diag_result if 'diag_result' in locals() else None,
            "final_response": final_text,
            "status": "SUCCESS"
        }

    # ==============================================================================
    # 3. WORKFLOW
    # ==============================================================================
    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("manage_health", self.analyze_node)
        workflow.set_entry_point("manage_health")
        workflow.add_edge("manage_health", END)
        return workflow.compile()
    

if __name__ == "__main__":
    service = HealthManagementService()
    graph = service.get_graph()

    test_state: AgentState = {
        "zone_id": "zone-002",
        "user_query": "Les feuilles de mon maïs sont déchiquetées",
        "culture_config": {"crop_name": "Maïs"},
        "diagnosis": None,
        "final_response": "",
        "status": ""
    }

    result = graph.invoke(test_state)
    print("=== Résultat du Phyto-Docteur ===")
    print(result["final_response"])
    print("Diagnosis structuré:", result["diagnosis"])
    print("Status:", result["status"])
