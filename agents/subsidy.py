from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import logging

# Import du toolkit Subvention
from Tools.subventions.base_subsidy import GrantExpertTool

# Configuration du logger
logger = logging.getLogger("agent.subsidy_finance")

# ==============================================================================
# 1. DÉFINITION DE L'ÉTAT (INPUTS/OUTPUTS)
# ==============================================================================
class AgentState(TypedDict):
    zone_id: str
    user_query: str                  # Ex: "J'ai reçu un SMS pour un fonds d'aide, c'est vrai ?"
    user_profile: Dict[str, Any]     # Ex: {"crop": "Maïs", "is_coop_member": False, "gender": "F"}
    
    opportunities: Optional[List]    # Liste des aides trouvées
    final_response: str              # Réponse formatée
    status: str                      # SUCCESS / ERROR / SCAM_DETECTED

# ==============================================================================
# 2. SERVICE DE GESTION DES AIDES
# ==============================================================================
class SubsidyManagementService:
    def __init__(self):
        self.name = "SubsidyManagementService"
        self.expert = GrantExpertTool()

    def analyze_node(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "").lower()
        profile = state.get("user_profile", {})
        
        logger.info(f"[{self.name}] Analyse demande financière | Profil: {profile.get('crop')}")

        response_parts = []
        status_code = "SUCCESS"

        # --- 1. SÉCURITÉ : DÉTECTION D'ARNAQUE ---
        if any(w in query for w in ["arnaque", "vrai", "frais", "envoyer", "sms", "dossier", "payer"]):
            scam_analysis = self.expert.check_scam(query)
            
            if scam_analysis["is_scam"]:
                status_code = "SCAM_DETECTED"
                response_parts.append(f"Alerte sécurité : {scam_analysis['warning']}")
                response_parts.append("Pourquoi c'est suspect :")
                for reason in scam_analysis["reasons"]:
                    response_parts.append(f"- {reason}")
                response_parts.append("Conseil : Ne payez jamais de frais de dossier par Orange Money/Mobicash avant d'avoir vu un agent officiel.")
                
                return {**state, "final_response": "\n".join(response_parts), "status": status_code}
            
            elif "vrai" in query:
                response_parts.append(scam_analysis["warning"])

        # --- 2. OPPORTUNITÉS : RECHERCHE D'AIDES ---
        if any(w in query for w in ["aide", "subvention", "argent", "financement", "projet", "banque"]):
            matches = self.expert.find_opportunities(profile)
            
            if matches:
                response_parts.append(f"Aides disponibles pour vous ({profile.get('crop', 'Divers')}) :")
                for m in matches:
                    status_icon = "Éligible" if m["status"] == "ÉLIGIBLE" else "Conditionnel"
                    response_parts.append(f"- Programme : {m['program_name']}")
                    response_parts.append(f"  Source : {m['provider']}")
                    response_parts.append(f"  Période : {m['deadline']}")
                    
                    if m["missing_documents"]:
                        response_parts.append(f"  Action requise : Il vous manque : {', '.join(m['missing_documents'])}")
            else:
                response_parts.append("Aucune subvention active correspondant à votre profil (Culture/Zone) pour le moment.")

        # --- 3. PROCÉDURE : COMMENT FAIRE ? ---
        if any(w in query for w in ["comment", "procédure", "papier", "document", "aller où"]):
            p_type = "irrigation" if any(w in query for w in ["pompe", "eau", "foncier"]) else "intrant"
            guide = self.expert.get_application_guide(p_type)
            response_parts.append(guide)

        # --- 4. CAS PAR DÉFAUT ---
        if not response_parts:
            response_parts.append("Guichet des Subventions")
            response_parts.append("Je peux vérifier les opportunités pour vous. Exemples de questions :")
            response_parts.append("- Y a-t-il une subvention pour le maïs ?")
            response_parts.append("- Comment obtenir une pompe solaire ?")
            response_parts.append("- J'ai reçu un message me demandant 5000F pour un dossier, est-ce une arnaque ?")

        final_text = "\n".join(response_parts)
        
        return {
            **state,
            "final_response": final_text,
            "status": status_code
        }

    # ==============================================================================
    # 3. WORKFLOW
    # ==============================================================================
    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("manage_subsidy", self.analyze_node)
        workflow.set_entry_point("manage_subsidy")
        workflow.add_edge("manage_subsidy", END)
        return workflow.compile()

if __name__ == "__main__":
    service = SubsidyManagementService()
    graph = service.get_graph()

    test_state: AgentState = {
        "zone_id": "zone-004",
        "user_query": "J'ai reçu un SMS pour un fonds d'aide, c'est vrai ?",
        "user_profile": {
            "crop": "Maïs",
            "is_coop_member": False,
            "gender": "F"
        },
        "opportunities": None,
        "final_response": "",
        "status": ""
    }

    result = graph.invoke(test_state)
    print("=== Résultat de l'Assistant Subvention ===")
    print(result["final_response"])
    print("Status:", result["status"])
    print("Opportunités:", result.get("opportunities"))
