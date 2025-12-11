from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional, Callable
import logging
from datetime import datetime

# Importations LangChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatOllama # <-- NOUVEAU

# Import du toolkit Agronomie
from Tools.crop.base_crop import CropManagerTool

logger = logging.getLogger("agent.crop_management")

# ==============================================================================
# 1. DÉFINITION DE L'ÉTAT (INCHANGÉ)
# ==============================================================================
class AgentState(TypedDict):
    """État du Graph pour l'Agent de Gestion des Cultures."""
    zone_id: str
    user_query: str
    culture_config: Dict[str, Any]
    
    technical_advice: Optional[str]
    final_response: str
    status: str

# ==============================================================================
# 2. SERVICE DE GESTION DES CULTURES (MISE À JOUR)
# ==============================================================================
class CropManagementService:
    """
    Service gérant le workflow de conseil agronomique.
    Utilise un outil technique et un LLM léger (ChatOllama) pour le formatage.
    """
    # Type de l'argument llm_client est maintenant Runnable (le type de ChatOllama)
    def __init__(self, llm_client: Optional[Runnable] = None): 
        self.name = "CropManagementService"
        self.agronomist = CropManagerTool()
        
        # Le client Ollama doit être passé à l'initialisation
        self.llm_client = llm_client
        if not self.llm_client:
            logger.error("Client ChatOllama non fourni. Le nœud LLM échouera.")

    # ----------------------------------------------------------------------
    # Fonctions Utilitaires (INCHANGÉES)
    # ----------------------------------------------------------------------

    def _calculate_days_after_sowing(self, sowing_date_str: str) -> int:
        try:
            s_date = datetime.strptime(sowing_date_str, "%Y-%m-%d")
            today = datetime.now()
            delta = today - s_date
            return max(0, delta.days)
        except Exception:
            return -1

    def analyze_node(self, state: AgentState) -> AgentState:
        """
        Nœud 1 : Détermine la catégorie de la requête et appelle l'outil Agronome.
        (Logique de routage inchangée)
        """
        # ... (La logique analyze_node reste la même que précédemment, elle produit technical_advice)
        
        query = (state.get("user_query") or "").lower()
        config = state.get("culture_config", {})

        crop_name = config.get("crop_name", "la culture")
        sowing_date = config.get("sowing_date")
        response_parts = []
        
        # Logique de routage (réutilisée pour la complétude)
        if any(w in query for w in ["semis", "semer", "densité", "écartement"]):
            advice = self.agronomist.get_seeding_advice(crop_name)
            response_parts.append(advice)
        elif any(w in query for w in ["engrais", "npk", "urée", "fertil"]):
            # ... (logique engrais)
            if not sowing_date:
                response_parts.append(
                    "Pour calculer la date d'engrais, j'ai besoin de votre date de semis (format : YYYY-MM-DD)."
                )
            else:
                das = self._calculate_days_after_sowing(sowing_date)
                if das >= 0:
                    status = self.agronomist.check_fertilizer_status(crop_name, das)
                    response_parts.append(f"📌 Stade de la culture : Jour {das}")
                    response_parts.append(status)
                else:
                    response_parts.append("Date de semis invalide (Format attendu : YYYY-MM-DD).")
        elif any(w in query for w in ["récolte", "couper", "fin", "maturité"]):
            # ... (logique récolte)
            if not sowing_date:
                response_parts.append("Pour estimer la récolte, j'ai besoin de votre date de semis.")
            else:
                estimation = self.agronomist.estimate_harvest(crop_name, sowing_date)
                response_parts.append(estimation)
        else:
            response_parts.append(f"📘 Fiche Technique – {crop_name}")
            response_parts.append("Je peux vous conseiller sur : Les densités de semis, le calendrier d'engrais et les dates de récolte.")
            response_parts.append("Posez-moi une question précise sur l'un de ces sujets.")

        technical_advice = "\n\n".join(response_parts)

        return {
            **state,
            "technical_advice": technical_advice,
            "status": "ADVICE_GENERATED"
        }


    def llm_formatter_node(self, state: AgentState) -> AgentState:
        """
        Nœud 2 : Utilise ChatOllama pour transformer le conseil technique 
        en une réponse conviviale pour l'utilisateur.
        """
        if not self.llm_client:
            raise ValueError("Le client LLM (ChatOllama) n'a pas été initialisé.")

        technical_advice = state.get("technical_advice", "Aucun conseil technique généré.")
        user_query = state.get("user_query", "")
        crop_name = state.get("culture_config", {}).get("crop_name", "votre culture")
        
        logger.info(f"[{self.name}] Début du formatage LLM avec ChatOllama pour {crop_name}.")

        # --- Définition du Prompt pour le LLM Léger ---
        system_prompt = (
            "Tu es un agronome professionnel, amical et facile à comprendre. "
            "Ta tâche est de transformer un conseil technique brut en une réponse naturelle "
            "et utile pour l'agriculteur. Ne donne pas de chiffres qui n'ont pas été "
            "fournis dans le conseil technique. Mets l'accent sur la clarté et l'action."
        )
        
        human_prompt = f"""
        **Contexte Agricole (Culture) :** {crop_name}
        **Question initiale de l'agriculteur :** "{user_query}"
        **Conseil Technique Brut (généré par l'outil) :** ---
        {technical_advice}
        ---
        
        Reformule ce conseil technique brut pour l'agriculteur.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        # Appel du client ChatOllama
        try:
            response = self.llm_client.invoke(messages)
            final_response = response.content
        except Exception as e:
            logger.error(f"Erreur lors de l'appel ChatOllama : {e}")
            final_response = f"Désolé, une erreur est survenue lors du formatage du conseil. Conseil brut : {technical_advice}"

        return {
            **state,
            "final_response": final_response,
            "status": "SUCCESS"
        }

    # ----------------------------------------------------------------------
    # 3. WORKFLOW LANGGRAPH (INCHANGÉ)
    # ----------------------------------------------------------------------
    def get_graph(self):
        """Construit et compile le Graph de l'Agent."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("manage_crop", self.analyze_node)       # Outil technique
        workflow.add_node("format_llm_response", self.llm_formatter_node) # LLM léger (post-traitement)
        
        workflow.set_entry_point("manage_crop")
        workflow.add_edge("manage_crop", "format_llm_response")
        workflow.add_edge("format_llm_response", END)
        
        return workflow.compile()
