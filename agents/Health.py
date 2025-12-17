import logging
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

# --- IMPORTATION DES OUTILS MÉTIERS ---
from Tools.health.base_health import HealthDoctorTool

logger = logging.getLogger("Agent.HealthSahel")

# ==============================================================================
# 1. DÉFINITION DE L'ÉTAT (STATE)
# ==============================================================================
class AgentState(TypedDict):
    user_query: str
    culture_config: Dict[str, Any]
    diagnosis_raw: Optional[Dict[str, Any]]
    technical_advice_text: str
    final_response: str
    status: str

# ==============================================================================
# 2. L'AGENT DE SANTÉ VÉGÉTALE
# ==============================================================================
class HealthManagementService:

    def __init__(self,ollama_host: str = "http://localhost:11434",llm_client=None,OLLAMA_MODEL:Optional[str] = "mistral" ):
        self.doctor = HealthDoctorTool() 
        self.model_name = OLLAMA_MODEL
        self.llm_client = self._initialize_llm(OLLAMA_MODEL,llm_client,ollama_host)

    def _initialize_llm(model_name,llm_client,self, ollama_host: str):
        try:
            return llm_client if llm_client else ChatOllama(model=model_name, base_url=ollama_host, temperature=0.1) # adapte la temperature

        except Exception as e:
            logger.error(f"Échec connexion LLM: {e}")
            return None

    # --- NŒUD 1 : ANALYSE (LOGIQUE MÉTIER) ---
    def analyze_node(self, state: AgentState) -> AgentState:
        """Utilise le HealthDoctorTool pour identifier la menace."""
        config = state.get("culture_config", {})
        crop_name = config.get("crop_name", "Culture inconnue")
        query = state.get("user_query", "")

        # Appel à l'outil métier importé
        diag = self.doctor.diagnose_and_prescribe(
            crop=crop_name, 
            user_obs=query
        )

        # Construction du rapport technique brut enrichi de supports visuels
        if diag.get("status") == "Trouvé":
            # On insère ici les tags visuels pour aider l'agriculteur à confirmer le diagnostic
            visual_aid = diag.get("visual_ref", "")
            prep_aid = diag.get("bio_step_by_step_tag", "")
            
            report = (
                f"PATHOLOGIE DÉTECTÉE : {diag['diagnostique']} ({diag.get('noms_locaux')})\n"
                f"IDENTIFICATION VISUELLE : {visual_aid}\n"
                f"NIVEAU DE RISQUE : {diag.get('niveau_alerte')}\n"
                f"RECETTE BIO : {diag.get('prescription_bio')}\n"
                f"GUIDE DE PRÉPARATION : {prep_aid}\n"
                f"MESURES PRÉVENTIVES : {diag.get('prevention')}"
            )
        else:
            report = f"ERREUR : {diag.get('message', 'Symptômes non reconnus.')}"

        return {
            **state,
            "diagnosis_raw": diag,
            "technical_advice_text": report,
            "status": diag.get("status")
        }

    # --- NŒUD 2 : FORMATAGE (LLM) ---
    def format_node(self, state: AgentState) -> AgentState:
        """Rend le diagnostic humain, bienveillant et structuré."""
        if self.llm_client is None or state["status"] != "Trouvé":
            return {**state, "final_response": state["technical_advice_text"]}

        system_prompt = (
            "Tu es un expert en santé végétale au Sahel. Ton ton est celui d'un conseiller agricole "
            "fraternel et expert. Structure ta réponse en 3 points : 1. Ce que nous avons identifié, "
            "2. Comment soigner naturellement (Bio), 3. Comment éviter cela à l'avenir.\n"
            "IMPORTANT : Recopie fidèlement les tags [Image of X] présents dans le rapport technique."
        )

        try:
            msg = self.llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Rapport Technique :\n{state['technical_advice_text']}")
            ])
            return {**state, "final_response": msg.content, "status": "COMPLETED"}
        except Exception:
            return {**state, "final_response": state["technical_advice_text"], "status": "FALLBACK"}

    # --- CONSTRUCTION DU WORKFLOW ---
    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("format", self.format_node)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "format")
        workflow.add_edge("format", END)
        
        return workflow.compile()