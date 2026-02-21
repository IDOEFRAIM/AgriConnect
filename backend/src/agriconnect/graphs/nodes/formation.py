import json
import logging
from typing import Any, Dict, List, Optional, TypedDict,Annotated
import operator
from langgraph.graph import END, StateGraph

from backend.src.agriconnect.graphs.prompts import FORMATION_SYSTEM_TEMPLATE, FORMATION_USER_TEMPLATE, STYLE_GUIDANCE

# MCP Protocols
from backend.src.agriconnect.protocols.mcp import MCPRagServer, MCPContextServer

# AG-UI Protocols
from backend.src.agriconnect.protocols.ag_ui import (
    AgriResponse, AGUIComponent, ComponentType,
    TextBlock, ActionButton, ActionType, ListPicker,
)

from backend.src.agriconnect.tools.formation import FormationTool
from backend.src.agriconnect.tools.refine import RefineTool
logger = logging.getLogger("Agent.FormationCoach")


class FormationAgentState(TypedDict, total=False):
    user_query: str
    learner_profile: Dict[str, Any]
    intent: str
    urgency: str
    focus_topics: List[str]
    field_actions: List[str]
    safety_flags: List[str]
    optimized_query: str
    learning_modules: List[str]
    prerequisites: List[str]
    reasoning: str
    retrieved_context: str
    sources: Annotated[List[Dict[str, Any]], operator.add]
    answer_draft: str
    final_response: str
    evaluation: Dict[str, float]
    status: str
    is_relevant: Optional[bool]
    rejection_reason:str
    warnings: Annotated[List[str], operator.add] # Les warnings s'ajoutent
    critique_retry_count: int
    rewrited_retry_count: int
    agri_response: Optional[AgriResponse]

class FormationCoach:
    def __init__(
        self,
        llm_client=None,
        mcp_rag: Optional[MCPRagServer] = None,
        mcp_context: Optional[MCPContextServer] = None,
        # Keep old params for backward compatibility if needed, but they are deprecated
        retriever=None,
        evaluator=None,
    ):
        self.tool = FormationTool(llm=llm_client)
        self.refine = RefineTool(llm=llm_client)
        
        # Protocol Servers
        self.mcp_rag = mcp_rag
        self.mcp_context = mcp_context
        
        self.model_planner = "llama-3.1-8b-instant"
        self.model_answer = "llama-3.3-70b-versatile"

        try:
            from backend.src.agriconnect.rag.components import get_groq_sdk
            self.llm = llm_client if llm_client else get_groq_sdk()
        except Exception as exc:
            logger.error("Impossible d'initialiser le LLM : %s", exc)
            self.llm = None
        
        # Retro-compatibility wrapper if old retriever passed
        if retriever and not mcp_rag:
             logger.warning("Using legacy retriever. Please migrate to MCPRagServer.")
             # No simple wrapper possible without proper MCP init, assuming mcp_rag is passed by orchestrator

    # ------------------------------------------------------------------ #
    # Nœuds du graphe                                                    #
    # ------------------------------------------------------------------ #

    def analyze_node(self, state: FormationAgentState) -> FormationAgentState:
        """
        Analyse la question + Récupère le contexte via MCP Context.
        """
        query = state.get("user_query", "").strip()
        
        # MCP : Récupération du profil via le protocole
        profile = {}
        if self.mcp_context:
            user_id = state.get("user_profile", {}).get("id", "anonymous")
            try:
                profile = self.mcp_context.read_user_context(user_id)
            except Exception as e:
                 logger.warning(f"MCP Context read failed: {e}")
                 profile = state.get("learner_profile", {})
        else:
            profile = state.get("learner_profile", {})

        warnings = list(state.get("warnings", []))

        if not query:
            warnings.append("La question de formation est vide.")
            return {"status": "ERROR", "warnings": warnings}
        
        # On analyse la question
        analysis = self.tool._analyze_request(query, profile)

        intent = analysis.get("intent", "FORMATION")
        focus_topics = analysis.get("focus_topics", [])
        field_actions = analysis.get("field_actions", [])
        safety = analysis.get("safety_flags", [])
        urgency = analysis.get("urgency", "NORMAL")
        warnings.extend(analysis.get("warnings", []))
        is_relevant = analysis.get("is_relevant", True)
        rejection_reason = analysis.get("rejection_reason", "") 

        return {
            "intent": intent,
            "focus_topics": focus_topics,
            "field_actions": field_actions,
            "safety_flags": safety,
            "urgency": urgency,
            "warnings": warnings,
            "is_relevant": is_relevant,
            "rejection_reason": rejection_reason,
            "status": "ANALYZED",
        }

    def retrieve_node(self, state: FormationAgentState) -> FormationAgentState:
        """
        Recherche RAG via MCPRagServer.
        """
        warnings = list(state.get("warnings", []))
        query = state.get("user_query", "").strip()

        # Profil pour le niveau (via MCP Context si dispo, sinon state)
        profile = state.get("learner_profile", {})
        if self.mcp_context:
            try:
                profile = self.mcp_context.read_user_context(state.get("user_profile", {}).get("id"))
            except: 
                pass

        if not query:
            return {"status": "ERROR", "warnings": warnings}

        # GESTION DU RETRY
        if state.get("status") == "RETRY_SEARCH" and state.get("optimized_query"):
             # Déjà optimisé
             optimized_query = state.get("optimized_query")
        else:
             # Utilise l'outil interne juste pour la reformulation (planning)
             plan = self.tool._plan_retrieval(query, profile)
             optimized_query = plan.get("optimized_query") or query
        
        # ── APPEL MCP RAG ──
        if not self.mcp_rag:
            warnings.append("MCP RAG Server manquant.")
            return {"status": "NO_CONTEXT", "warnings": warnings}
            
        try:
            # MCP Search via call_tool abstraction
            search_level = profile.get("niveau", "debutant")
            args = {"query": optimized_query, "level": search_level, "top_k": 4}
            resp = self.mcp_rag.call_tool("search_agronomy_docs", args)

            if not resp or resp.get("status") != "ok":
                warnings.append("Aucun résultat MCP trouvé ou erreur MCP.")
                return {
                    "optimized_query": optimized_query,
                    "retrieved_context": "",
                    "sources": [],
                    "status": "NO_CONTEXT",
                    "warnings": warnings,
                }

            # Parse the returned content which is a JSON dump inside content[0]["text"]
            content_list = resp.get("content") or []
            if not content_list:
                return {
                    "optimized_query": optimized_query,
                    "retrieved_context": "",
                    "sources": [],
                    "status": "NO_CONTEXT",
                    "warnings": warnings + ["MCP response empty content"]
                }

            # Try to load JSON from the first content entry
            raw_text = content_list[0].get("text", "")
            try:
                parsed = json.loads(raw_text)
            except Exception:
                # If already a dict or unexpected formatting, try to handle gracefully
                parsed = raw_text if isinstance(raw_text, dict) else {}

            # Extract context and sources in a robust way
            context_text = parsed.get("context") if isinstance(parsed, dict) else str(parsed)
            sources = parsed.get("sources", []) if isinstance(parsed, dict) else []

            # Normalize sources to expected format
            norm_sources = []
            for s in sources:
                if isinstance(s, dict):
                    norm_sources.append({"title": s.get("source", s.get("title", "Doc")), "uri": s.get("uri")})
                else:
                    norm_sources.append({"title": str(s), "uri": None})

            return {
                "optimized_query": optimized_query,
                "retrieved_context": context_text or "",
                "sources": norm_sources,
                "status": "CONTEXT_FOUND",
                "warnings": warnings,
            }

        except Exception as e:
            logger.error(f"MCP RAG Error: {e}")
            warnings.append(f"Erreur MCP RAG: {e}")
            return {"status": "NO_CONTEXT", "warnings": warnings}


    def compose_node(self, state: FormationAgentState) -> FormationAgentState:
        warnings = list(state.get("warnings", []))
        
        # AG-UI Response Builder
        agri_response = AgriResponse()
        
        # Gestion du hors-sujet
        if state.get("is_relevant") is False:
            rejection = state.get("rejection_reason") or "Désolé, je ne peux répondre qu'aux questions agricoles."
            final_rejection = f"😊 **Bonjour !**\n\n{rejection}\n\nEn tant qu'expert AgriConnect, je suis à votre disposition pour toute question sur vos cultures."
            
            # AG-UI : Réponse simple pour le hors-sujet
            agri_response.add_text(final_rejection)
            
            return {
                "answer_draft": final_rejection,
                "final_response": final_rejection,
                "agri_response": agri_response,
                "status": "OFF_TOPIC",
                "warnings": warnings
            }

        # Contexte et Prompt Preparation
        context = state.get("retrieved_context", "").strip()
        query = state.get("user_query", "").strip()
        profile_text = self.tool._format_profile(state.get("learner_profile", {}))
        
        if not query:
            return {"warnings": warnings, "status": "ERROR"}

        # Génération de réponse via LLM
        final_answer = "Réponse technique indisponible."
        
        # ... Logique LLM existante ... (simplifiée pour AG-UI)
        try:
            # Construction du prompt (inchangée, utilise les templates)
            profile = state.get("learner_profile", {})
            level = str(profile.get("niveau", "standard")).lower()
            style_guidance = STYLE_GUIDANCE.get(level, STYLE_GUIDANCE["default"])
            
            system_content = FORMATION_SYSTEM_TEMPLATE.format(
                style_guidance=style_guidance,
                culture_context=f"Culture: {profile.get('culture_actuelle', 'N/A')}"
            )
            user_content = FORMATION_USER_TEMPLATE.format(
                query=query,
                feedback_hallucination=state.get("feedback_hallucination", ""),
                intent=state.get("intent", "FORMATION"),
                urgency=state.get("urgency", "NORMAL"),
                profile_text=profile_text,
                context=context
            )
            
            completion = self.llm.chat.completions.create(
                model=self.model_answer,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.35,
                max_tokens=900,
            )
            final_answer = completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            final_answer = "Désolé, je rencontre une difficulté technique pour formuler le conseil."
            warnings.append(str(e))

        # ── CONSTRUCTION AG-UI ──
        
        # 1. Texte principal (Markdown supporté par WhatsApp et Web)
        agri_response.add_text(final_answer)
        
        # 2. Boutons d'action (Quick Replies)
        # Suggérer des actions selon le contexte
        if "maladie" in query.lower() or "ravageur" in query.lower():
            agri_response.add(ListPicker(
                title="Que voulez-vous faire ?",
                items=[
                    {"id": "photo_upload", "label": "📷 Envoyer une photo"},
                    {"id": "call_expert", "label": "📞 Parler à un agent"}
                ]
            ))
        else:
            agri_response.add(ListPicker(
                title="Aller plus loin :",
                items=[
                    {"id": "more_details", "label": "🔍 Plus de détails"},
                    {"id": "related_topic", "label": "🌾 Sujet associé"}
                ]
            ))

        return {
            "answer_draft": final_answer,
            "final_response": final_answer,
            "agri_response": agri_response,
            "warnings": warnings,
            "status": "ANSWER_GENERATED",
        }

    def evaluate_node(self, state: FormationAgentState) -> FormationAgentState:
        warnings = list(state.get("warnings", []))

        if not getattr(self, "evaluator", None):
            warnings.append("Évaluation automatique indisponible.")
            return {"warnings": warnings}

        query = state.get("user_query", "")
        context = state.get("retrieved_context", "")
        answer = state.get("final_response", "")

        if not answer or not context:
            return {"warnings": warnings}

        try:
            scores = self.evaluator.evaluate_all(
                query=query,
                context=context,
                answer=answer,
            )
            return{
                "evaluation": scores,
                "warnings": warnings,
                "status": "EVALUATED",
            }
        except Exception as exc:
            warnings.append(f"Évaluation échouée : {exc}")
            return {"warnings": warnings}


    # ------------------------------------------------------------------ #
    # build                                                       
    # ------------------------------------------------------------------ #


    def build(self):
        workflow = StateGraph(FormationAgentState)
        
        # nœuds
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rewrite", self.refine.rewrite_query_node) 
        workflow.add_node("compose", self.compose_node)
        workflow.add_node("critique", self.refine.critique_node)  
        workflow.add_node("evaluate", self.evaluate_node)

        workflow.set_entry_point("analyze")

        # Logique de routage complexe
        workflow.add_conditional_edges("analyze", self.refine.route_after_analyze)
        
        workflow.add_conditional_edges(
            "retrieve", 
            self.refine.route_retrieval,
            {"compose": "compose", "rewrite": "rewrite"}
        )
        
        # Modif : Routage conditionnel après Rewrite pour éviter boucle infinie
        workflow.add_conditional_edges(
            "rewrite", 
            self.refine.route_after_rewrite,
            {"retrieve": "retrieve", "compose": "compose"}
        )
        
        workflow.add_edge("compose", "critique")
        
        workflow.add_conditional_edges(
            "critique",
            lambda x: "evaluate" if x["status"] == "VALIDATED" else "compose",
            {"evaluate": "evaluate", "compose": "compose"} # Re-rédiger si rejeté
        )
        
        workflow.add_edge("evaluate", END)
        return workflow.compile()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    agent = FormationCoach()
    workflow = agent.build()
    # Exemple d’état initial
    state = {
        "user_query": "Que sais tu sur la saison culture au burkina?",
        "learner_profile": {"niveau": "débutant", "région": "Boucle du Mouhoun"},
    }
    result = workflow.invoke(state)
    print(json.dumps(result, indent=2, ensure_ascii=False))