import logging
from typing import TypedDict, Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

# --- IMPORTATION DES OUTILS MÉTIERS ---
from tools.health.base_health import HealthDoctorTool

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
    status: str  # 'START', 'FOUND', 'UNKNOWN', 'ERROR'

# ==============================================================================
# 2. L'AGENT DE SANTÉ VÉGÉTALE
# ==============================================================================
class PlantHealthDoctor:
    def __init__(self, ollama_host: str = "http://localhost:11434", llm_client=None, model_name: str = "mistral"):
        self.doctor = HealthDoctorTool() 
        self.model_name = model_name
        self.llm = llm_client if llm_client else self._initialize_llm(model_name, ollama_host)

    def _initialize_llm(self, model_name, ollama_host):
        try:
            return ChatOllama(model=model_name, base_url=ollama_host, temperature=0.1)
        except Exception as e:
            logger.error(f"❌ Échec connexion LLM: {e}")
            return None

    def _identify_symptoms_semantically(self, user_text: str) -> str:
        """Détecte les mots-clés techniques à partir du langage naturel."""
        if not self.llm: 
            return user_text
        
        prompt = (
            "Tu es un expert en phytopathologie sahélienne.\n"
            "Analyse les symptômes décrits par l'agriculteur et extrais les termes techniques.\n"
            "Exemple : 'fleurs violettes' -> STRIGA WONGO.\n"
            f"Description : '{user_text}'\n"
            "Réponds UNIQUEMENT avec les mots-clés extraits, séparés par des virgules."
        )
        try:
            resp = self.llm.invoke([SystemMessage(content=prompt)])
            return f"{user_text}, {resp.content.upper()}"
        except Exception:
            return user_text

    # --- NŒUD 1 : ANALYSE ---
    def analyze_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- NODE: ANALYSE ---")
        config = state.get("culture_config", {})
        crop = config.get("crop_name", "Culture inconnue")
        query = state.get("user_query", "")
        
        # 1. Identification sémantique
        enhanced_query = self._identify_symptoms_semantically(query)

        # 2. Diagnostic via l'outil métier
        diag = self.doctor.diagnose_and_prescribe(crop=crop, user_obs=enhanced_query)

        if diag.get("status") == "Trouvé" or "diagnostique" in diag:
            # Extraction dynamique du tutoriel (ex: neem ou piment selon le diagnostic)
            target_bio = diag.get("target_pest", "neem")
            prep_aid = self.doctor.get_biopesticide_tutorial(target_bio)
            
            report = (
                f"🎯 PATHOLOGIE : {diag.get('diagnostique')}\n"
                f"⚠️ RISQUE : {diag.get('niveau_alerte')}\n"
                f"🌿 SOLUTION BIO : {diag.get('prescription_bio')}\n"
                f"📖 MÉTHODE : {prep_aid}\n"
                f"🧪 CHIMIE (Dernier recours) : {diag.get('conseil_chimique')}\n"
                f"🛡️ PRÉVENTION : {diag.get('prevention')}"
            )
            return {
                "diagnosis_raw": diag,
                "technical_advice_text": report,
                "status": "FOUND"
            }
        
        return {
            "technical_advice_text": "Désolé, je n'ai pas pu identifier la maladie. Veuillez consulter un agent de terrain.",
            "status": "UNKNOWN"
        }

    # --- NŒUD 2 : FORMATAGE ---
    def format_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- NODE: FORMATAGE ---")
        if not self.llm or state["status"] != "FOUND":
            return {"final_response": state["technical_advice_text"]}

        system_prompt = (
            "Tu es le Guérisseur des Plantes d'AgriConnect.\n"
            "TON : Bienveillant, expert, protecteur. Utilise des listes à puces.\n"
            "RÈGLE D'OR : Priorité absolue aux remèdes naturels (Bio)."
        )

        try:
            msg = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Transforme ce rapport en conseil amical :\n{state['technical_advice_text']}")
            ])
            return {"final_response": msg.content}
        except Exception:
            return {"final_response": state["technical_advice_text"]}

    # --- CONSTRUCTION DU GRAPH ---
    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("format", self.format_node)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "format")
        workflow.add_edge("format", END)
        
        return workflow.compile()