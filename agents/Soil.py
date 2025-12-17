import logging
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# --- Importations LangChain ---
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

from Tools.Soils.base_soil import SoilDoctorTool
# On suppose que ton outil SoilDoctorTool est déjà défini au-dessus ou importé.

# ======================================================================
# 1. DÉFINITION DE L'ÉTAT
# ======================================================================
class AgentState(TypedDict):
    user_query: str
    soil_config: Dict[str, Any]
    technical_advice_raw: Optional[str]
    final_response: str
    status: str

# ======================================================================
# 2. SERVICE DE GESTION DES SOLS (LangGraph)
# ======================================================================
class SoilManagementService:
    OLLAMA_MODEL = "mistral"

    def __init__(self, ollama_host: str = "http://localhost:11434",llm_client:str=None):
        self.name = "SoilService"
        self.pedologist = SoilDoctorTool()  # Ton outil réutilisé tel quel
        self.logger = logging.getLogger("agent.soil")
        
        try:
            self.llm_client = llm_client if llm_client else ChatOllama(model=self.OLLAMA_MODEL, base_url=ollama_host, temperature=0.1)
            # Test de connexion
            self.llm_client.invoke([SystemMessage(content="Ping")])
            self.logger.info("Ollama connecté.")
        except Exception as e:
            self.llm_client = None
            self.logger.warning(f"Ollama indisponible. Mode fallback activé. Erreur: {e}")

    # --- NŒUD 1 : ANALYSE TECHNIQUE (Appel du Tool) ---
    def analyze_node(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        config = state.get("soil_config", {})
        
        # Extraction des paramètres pour ton outil
        texture = config.get("texture", "sableux")
        ph = float(config.get("ph", 6.5))
        budget = config.get("budget", "bas")

        # Exécution de ton outil
        diagnosis = self.pedologist.get_full_diagnosis(texture=texture, obs_text=query, ph=ph)
        p_source = self.pedologist.recommend_p_source(budget=budget)

        # Construction de la sortie brute (RAW) pour le LLM
        # On y inclut les tags d'images contextuels pour les techniques CES
        raw_report = (
            f"TYPE DE SOL : {diagnosis['soil_type']}\n"
            f"ÉTAT HYDRIQUE : {diagnosis['moisture_status']}\n"
            f"BESOIN EN EAU : {diagnosis['water_to_add']}\n"
            f"ANALYSE CHIMIQUE : {diagnosis['ph_analysis']}\n"
            f"TECHNIQUE CES : {diagnosis['ces_recommendation']['technique']}\n"
            f"DÉTAILS CES : {diagnosis['ces_recommendation']['details']}\n"
            f"NUTRITION : {p_source}"
        )
        
        # Ajout dynamique d'un tag d'image selon la technique préconisée par ton tool
        if "Zaï" in diagnosis['ces_recommendation']['technique']:
            raw_report += "\nIMAGE_TAG: "
        elif "Cordons" in diagnosis['ces_recommendation']['technique']:
            raw_report += "\nIMAGE_TAG: "
        elif "Billonnage" in diagnosis['ces_recommendation']['technique']:
            raw_report += "\nIMAGE_TAG: "

        return {**state, "technical_advice_raw": raw_report, "status": "TECHNICAL_DONE"}

    # --- NŒUD 2 : FORMATAGE LLM ---
    def format_node(self, state: AgentState) -> AgentState:
        raw_advice = state.get("technical_advice_raw", "")
        
        if not self.llm_client:
            return {**state, "final_response": f"CONSEIL TECHNIQUE BRUT :\n{raw_advice}", "status": "FALLBACK"}

        system_prompt = (
            "Tu es le Docteur Sol, un expert pédologue sahélien. Ton ton est professionnel, "
            "pédagogique et encourageant. Transforme les données techniques en un plan d'action "
            "clair pour un paysan. Utilise des emojis (🌾, 💧, 🧱). "
            "Garde impérativement les tags  [Image of X] sans les modifier."
        )
        
        human_prompt = f"Voici les résultats de l'analyse :\n{raw_advice}"
        
        try:
            response = self.llm_client.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
            final_text = response.content
        except Exception:
            final_text = raw_advice

        return {**state, "final_response": final_text, "status": "SUCCESS"}

    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("format", self.format_node)
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "format")
        workflow.add_edge("format", END)
        return workflow.compile()

# ======================================================================
# 3. EXEMPLE D'UTILISATION
# ======================================================================
if __name__ == "__main__":
    service = SoilManagementService()
    graph = service.get_graph()

    test_input = {
        "user_query": "Mon sol est très sec et craquelé en surface, je ne sais pas quoi planter.",
        "soil_config": {
            "texture": "gravillonnaire",
            "ph": 5.1,
            "budget": "bas"
        },
        "technical_advice_raw": None, "final_response": "", "status": ""
    }

    result = graph.invoke(test_input)
    print("\n--- RÉPONSE DU DOCTEUR SOL ---")
    print(result["final_response"])