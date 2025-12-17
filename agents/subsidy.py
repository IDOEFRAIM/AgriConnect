import logging
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime

# --- Importations LangGraph & LangChain ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

# --- IMPORTATION DES OUTILS RÉELS ---
from Tools.subventions.base_subsidy import AgrimarketTool 

logger = logging.getLogger("agent.subsidy_finance")

# ======================================================================
# 1. ÉTAT DE L'AGENT
# ======================================================================

class AgentState(TypedDict):
    zone_id: str
    user_query: str
    user_profile: Dict[str, Any]
    technical_advice_raw: Optional[str]
    final_response: str
    status: str

# ======================================================================
# 2. SERVICE SUBSIDY CORRIGÉ
# ======================================================================

class SubsidyManagementService:
    OLLAMA_MODEL = "mistral"

    def __init__(self, ollama_host: str = "http://localhost:11434", llm_client=None):
        self.market_tool = AgrimarketTool() 
        self.llm_client = self._initialize_ollama(llm_client, ollama_host)

    def _initialize_ollama(self, llm_client, host: str):
        try:
            return llm_client if llm_client else ChatOllama(model=self.OLLAMA_MODEL, base_url=host, temperature=0.1)
        except Exception as e:
            logger.error(f"LLM non disponible: {e}")
            return None

    def analyze_node(self, state: AgentState) -> AgentState:
        """Analyse Sécurité, Marché et Subventions."""
        query = state.get("user_query", "").lower()
        profile = state.get("user_profile", {})
        crop = profile.get("crop", "Maïs")
        region = state.get("zone_id", "Centre")
        current_month = datetime.now().month
        
        response_parts = []
        status = "SUCCESS"

        # --- 1. SÉCURITÉ (Phishing) ---
        scam_triggers = ["payer", "orange money", "mobicash", "frais de dossier", "sms"]
        if any(t in query for t in scam_triggers):
            status = "SCAM_DETECTED"
            tag_scam = ""
            response_parts.append(f"🚨 **ALERTE SÉCURITÉ** {tag_scam}")
            response_parts.append("\nAttention : L'État ne demande jamais d'argent mobile pour les aides.")

        # --- 2. INTELLIGENCE MARCHÉ & AIDES ---
        else:
            # Info Marché
            market = self.market_tool.analyze_market_timing(crop, current_month)
            response_parts.append(f"📈 **MARCHÉ : {crop.upper()}**")
            tag_prices = ""
            response_parts.append(tag_prices)
            response_parts.append(f"- Statut : {market.get('etat_marche', 'N/A')}")
            response_parts.append(f"- Conseil : {market.get('conseil', 'N/A')}")

            # Warrantage (Si applicable)
            if market.get('opportunite_warrantage') == "CONSEILLÉ":
                tag_warr = ""
                response_parts.append(f"\n💡 **WARRANTAGE :** {tag_warr}")

            # Subventions (CORRECTION ICI : On ne fait pas .get() sur une string)
            sub_text = self.market_tool.get_subsidy_status(region)
            response_parts.append(f"\n💰 **AIDES RÉGIONALES :**")
            tag_docs = ""
            response_parts.append(tag_docs)
            response_parts.append(sub_text) # Ajout direct de la chaîne formatée

        raw_text = "\n".join(response_parts)
        return {**state, "technical_advice_raw": raw_text, "status": status}

    def format_node(self, state: AgentState) -> AgentState:
        """Mise en forme pédagogique via LLM."""
        if state["status"] == "SCAM_DETECTED" or not self.llm_client:
            return {**state, "final_response": state["technical_advice_raw"]}

        system_prompt = (
            "Tu es l'Expert Finance d'AgConnect. Rends ces données claires pour un paysan. "
            "Garde impérativement les tags [Image of X] ."
        )
        
        try:
            res = self.llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["technical_advice_raw"])
            ])
            return {**state, "final_response": res.content, "status": "COMPLETED"}
        except Exception:
            return {**state, "final_response": state["technical_advice_raw"], "status": "FALLBACK"}

    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("format", self.format_node)
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "format")
        workflow.add_edge("format", END)
        return workflow.compile()