import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from backend.src.agriconnect.graphs.prompts import (
    SENTINELLE_USER_TEMPLATE,
    SENTINELLE_SYSTEM_TEMPLATE,
    STYLE_GUIDANCE,
)

from backend.src.agriconnect.rag.components import get_groq_sdk
from backend.src.agriconnect.rag.metric import RAGEvaluator
from backend.src.agriconnect.rag.retriever import AgileRetriever

from backend.src.agriconnect.tools.sentinelle import SentinelleTool
from backend.src.agriconnect.tools.refine import RefineTool
logger = logging.getLogger("Agent.ClimateSentinel")


class SentinelState(TypedDict, total=False):
    user_query: str
    location_profile: Dict[str, Any]
    user_level: str
    weather_snapshot: Dict[str, Any]
    satellite_signals: Dict[str, Any]
    raw_metrics: Dict[str, Any]
    flood_risk: Dict[str, Any]
    hazards: List[Dict[str, Any]]
    risk_summary: str
    optimized_query: str
    retrieved_context: str
    sources: List[Dict[str, Any]]
    final_response: str
    evaluation: Dict[str, float]
    status: str
    warnings: List[str]
    security_status: str
    security_reason: str
    critique_retry_count:int
    rewrited_retry_count:int


class ClimateSentinel:
    """Agent de veille climatique AgriConnect avec contrôle anti-fraude et diagnostics agro-météo."""

    _STYLE_GUIDANCE = STYLE_GUIDANCE

    def __init__(
        self,
        llm_client=None,
        retriever: Optional[AgileRetriever] = None,
        evaluator: Optional[RAGEvaluator] = None,
    ):
        self.refine  = RefineTool(llm=llm_client)
        self.model_planner = "llama-3.1-8b-instant"
        self.model_answer = "llama-3.3-70b-versatile"

        try:
            self.llm = llm_client if llm_client else get_groq_sdk()
        except Exception as exc:
            logger.error("Impossible d'initialiser le LLM : %s", exc)
            self.llm = None
            
        self.tools = SentinelleTool(llm_client=self.llm)

        try:
            self.retriever = retriever if retriever else AgileRetriever()
        except Exception as exc:
            logger.error("RAG indisponible : %s", exc)
            self.retriever = None

        try:
            self.evaluator = evaluator if evaluator else RAGEvaluator()
        except Exception as exc:
            logger.warning("Évaluateur indisponible : %s", exc)
            self.evaluator = None

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _build_context(self, nodes: List[Any]) -> str:
        """Construit le contexte RAG à partir des nœuds récupérés."""
        if not nodes:
            return ""
        
        context_parts = []
        for i, node_obj in enumerate(nodes):
            # Support LlamaIndex NodeWithScore or Dict
            if hasattr(node_obj, "node"):
                # It is a NodeWithScore
                text = node_obj.node.get_content()
                meta = node_obj.node.metadata
                source_type = meta.get("type", "Document")
                source_name = meta.get("source", "Inconnu")
            else:
                # It might be a dict (legacy)
                text = node_obj.get("text", "")
                meta = node_obj.get("metadata", {})
                source_type = meta.get("type", "Document")
                source_name = meta.get("source", "Inconnu")
            
            context_parts.append(
                f"--- DOCUMENT {i+1} ({source_type}: {source_name}) ---\n{text}\n"
            )
            
        return "\n".join(context_parts)

    def _serialize_sources(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        """Extrait les métadonnées pour la traçabilité."""
        sources = []
        for node_obj in nodes:
            if hasattr(node_obj, "node"):
                meta = node_obj.node.metadata
                score = node_obj.score
            else:
                meta = node_obj.get("metadata", {})
                score = node_obj.get("score", 0.0)

            sources.append({
                "source": meta.get("source", "Unknown"),
                "type": meta.get("type", "doc"),
                "similarity": score
            })
        return sources

    def _format_location(self, profile: Dict) -> str:
        """Formate le profil géographique."""
        if not profile:
            return "Zone inconnue (Burkina Faso)"
        
        parts = []
        if profile.get("village"): parts.append(profile["village"])
        if profile.get("zone"): parts.append(profile["zone"])
        return ", ".join(parts) if parts else "Burkina Faso"
        
    def _fallback_response(self, query, location, hazards, risk_summary, metrics, flood, sources) -> str:
        """Génère une réponse de secours en cas d'échec du LLM."""
        return (
            "Désolé, je rencontre des difficultés techniques pour analyser votre demande en détail. "
            "Cependant, voici les observations actuelles :\n\n"
            f"📍 {location}\n"
            f"⚠️ Risques : {risk_summary}\n"
            f"🌧️ Pluie du jour : {metrics.get('precip_mm', 0)} mm\n\n"
            "Veuillez réessayer dans quelques instants."
        )

    # ------------------------------------------------------------------ #
    # Nœuds du graphe                                                    #
    # ------------------------------------------------------------------ #

    def analyze_node(self, state: SentinelState) -> SentinelState:
        query = state.get("user_query", "").strip()
        warnings = list(state.get("warnings", []))
        location = state.get("location_profile", {}) or {}
        weather = state.get("weather_snapshot") or self.tools._fetch_real_weather(location)
        satellite = state.get("satellite_signals") or {}

        if not query:
            warnings.append("La requête de vigilance est vide.")
            state = dict(state)
            state.update({"warnings": warnings, "status": "ERROR"})
            return state

        moderation = self.tools._moderate_request(query)
        security_status = "SCAM_DETECTED" if moderation.get("is_scam") else "SAFE"
        security_reason = moderation.get("reason", "")
        if security_reason:
            warnings.append(security_reason)

        if security_status == "SCAM_DETECTED":
            state = dict(state)
            state.update({
                "warnings": warnings,
                "security_status": security_status,
                "security_reason": security_reason,
                "status": "SCAM_DETECTED",
            })
            return state

        metrics = self.tools._compute_metrics(weather, satellite)
        flood = self.tools._assess_flood_risk(weather, satellite, location)
        hazards = self.tools._derive_hazards(metrics, flood)

        if not hazards:
            warnings.append("Aucun risque majeur détecté (veille standard).")

        summary_lines = [
            f"- {hazard['label']} (niveau {hazard['severity']}): {hazard['explanation']}"
            for hazard in hazards
        ]
        risk_summary = "\n".join(summary_lines) or "Pas d'anomalie critique détectée."

        

        state = dict(state)
        state.update({
            "raw_metrics": metrics,
            "flood_risk": flood,
            "hazards": hazards,
            "risk_summary": risk_summary,
            "warnings": warnings,
            "security_status": security_status,
            "security_reason": security_reason,
            "status": "SIGNALS_READY",
        })
        return state

    def retrieve_node(self, state: SentinelState) -> SentinelState:
        warnings = list(state.get("warnings", []))
        security_status = state.get("security_status", "SAFE")

        if security_status == "SCAM_DETECTED":
            warnings.append("Requête bloquée par le module anti-fraude.")
            return {
                "warnings": warnings,
                "security_status": security_status,
                "security_reason": state.get("security_reason", ""),
                "status": "SCAM_DETECTED",
            }

        query = state.get("user_query", "")
        risk_summary = state.get("risk_summary", "")
        hazards = state.get("hazards", [])

        plan = self.tools._plan_retrieval(query, risk_summary, hazards)
        optimized_query = plan.get("optimized_query") or query
        warnings.extend(plan.get("warnings", []))

        if not self.retriever:
            warnings.append("Moteur RAG indisponible.")
            state = dict(state)
            state.update({
                "optimized_query": optimized_query,
                "retrieved_context": "",
                "sources": [],
                "warnings": warnings,
                "security_status": security_status,
                "security_reason": state.get("security_reason", ""),
                "status": "NO_CONTEXT",
            })
            return state

        nodes = self.retriever.search(
            optimized_query,
            user_level=state.get("user_level", "debutant"),
        )
        
        state = dict(state)
        if not nodes:
            warnings.append("Aucun document pertinent trouvé.")
            
            state.update({
                "optimized_query": optimized_query,
                "retrieved_context": "",
                "sources": [],
                "warnings": warnings,
                "security_status": security_status,
                "security_reason": state.get("security_reason", ""),
                "status": "NO_CONTEXT",
            })
            return state

        state.update({
            "optimized_query": optimized_query,
            "retrieved_context": self._build_context(nodes),
            "sources": self._serialize_sources(nodes),
            "warnings": warnings,
            "security_status": security_status,
            "security_reason": state.get("security_reason", ""),
            "status": "CONTEXT_READY",
        })
        return state

    def compose_node(self, state: SentinelState) -> SentinelState:
        warnings = list(state.get("warnings", []))
        security_status = state.get("security_status", "SAFE")

        if security_status == "SCAM_DETECTED":
            alert = self._build_scam_response(state)
            state = dict(state)
            state.update({
                "final_response": alert,
                "warnings": warnings,
                "status": "SCAM_DETECTED",
                "security_status": security_status,
                "security_reason": state.get("security_reason", ""),
            })
            return state

        context = state.get("retrieved_context", "")
        query = state.get("user_query", "")
        hazards = state.get("hazards", [])
        risk_summary = state.get("risk_summary", "")
        metrics = state.get("raw_metrics", {})
        flood = state.get("flood_risk", {})
        location = self._format_location(state.get("location_profile", {}))
        sources = state.get("sources", [])

        # Détection automatique de surface pour le calcul d'eau
        query_text = query.lower()
        surface_calc_info = ""
        et0 = metrics.get("et0_mm", 0.0)
        
        # Patterns robustes pour capturer hectares (ha) et mètres carrés (m², m2, mc)
        # Supporte : "1.5 ha", "1,5 hectares", "500 m2", "500 mètres carrés"
        ha_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ha|hectare|hectares)", query_text)
        m2_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mc|metres?\s*carres?|mètres?\s*carrés?)", query_text)
        
        if (ha_match or m2_match) and et0 > 0:
            try:
                if ha_match:
                    val = float(ha_match.group(1).replace(",", "."))
                    area_m2 = val * 10000
                    unit_str = f"{val} hectares"
                else:
                    val = float(m2_match.group(1).replace(",", "."))
                    area_m2 = val
                    unit_str = f"{val} m²"
                
                loss_liters = area_m2 * et0
                surface_calc_info = (
                    f"CALCUL AUTOMATIQUE EFFECTUÉ : Pour {unit_str} avec une ET0 de {et0}mm, "
                    f"la perte en eau est de {loss_liters:,.0f} litres AUJOURD'HUI. "
                    "Intègre ce chiffre IMPÉRATIVEMENT dans ta réponse."
                )
            except Exception:
                pass

        fallback = self._fallback_response(query, location, hazards, risk_summary, metrics, flood, sources)

        if not context:
            warnings.append("Réponse générée sans contexte vérifié.")

        if not self.llm:
            warnings.append("LLM indisponible (mode secours).")
            state = dict(state)
            state.update({
                "final_response": fallback,
                "warnings": warnings,
                "status": "LLM_DOWN",
            })
            return state

        hazard_json = json.dumps(hazards, ensure_ascii=False)
        metrics_json = json.dumps(metrics, ensure_ascii=False)
        current_date_str = "13 Février 2026 (Saison Sèche)"

        # 1. Détection du niveau utilisateur
        niveau = str(state.get("user_level", "debutant")).lower()
        style_guidance = self._STYLE_GUIDANCE.get(niveau, self._STYLE_GUIDANCE["default"])

        try:
            # 2. Prompt système adapté au niveau
            system_content = SENTINELLE_SYSTEM_TEMPLATE + f"\n\nCONSIGNE STYLE: {style_guidance}"
            # 3. Prompt utilisateur inchangé
            user_content = SENTINELLE_USER_TEMPLATE.format(
                current_date_str=current_date_str,
                query=query,
                location=location,
                risk_summary=risk_summary,
                metrics_json=json.dumps(metrics_json, ensure_ascii=False),
                flood_data=json.dumps(flood, ensure_ascii=False),
                hazard_json=json.dumps(hazard_json, ensure_ascii=False),
                context=context,
                surface_calc_info=surface_calc_info
            )

            # 4. Appel au LLM
            completion = self.llm.chat.completions.create(
                model=self.model_answer,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.35,
                max_tokens=650,
            )
            answer = completion.choices[0].message.content
            if not answer:
                raise ValueError("Réponse vide du LLM.")
            state = dict(state)
            state.update({
                "final_response": answer,
                "warnings": warnings,
                "status": "ANSWER_READY",
            })
            return state
        except Exception as exc:
            warnings.append(f"Erreur LLM: {exc}")
            state = dict(state)
            state.update({
                "final_response": fallback,
                "warnings": warnings,
                "status": "LLM_ERROR",
            })
            return state

    def evaluate_node(self, state: SentinelState) -> SentinelState:
        warnings = list(state.get("warnings", []))
        if state.get("security_status") == "SCAM_DETECTED":
            warnings.append("Évaluation ignorée : requête bloquée pour suspicion de fraude.")
            state = dict(state)
            state.update({
                "warnings": warnings,
                "status": "SCAM_DETECTED",
                "security_status": state.get("security_status", "SAFE"),
            })
            return state

        if not self.evaluator:
            warnings.append("Évaluateur RAG indisponible.")
            state = dict(state)
            state.update({"warnings": warnings})
            return state

        query = state.get("user_query", "")
        context = state.get("retrieved_context", "")
        answer = state.get("final_response", "")

        if not query or not context or not answer:
            state = dict(state)
            state.update({"warnings": warnings})
            return state

        try:
            scores = self.evaluator.evaluate_all(query=query, context=context, answer=answer)
            state = dict(state)
            state.update({"evaluation": scores, "warnings": warnings, "status": "EVALUATED"})
            return state
        except Exception as exc:
            warnings.append(f"Évaluation échouée: {exc}")
            state = dict(state)
            state.update({"warnings": warnings})
            return state

    def _build_scam_response(self, state: SentinelState) -> str:
        reason = state.get("security_reason") or "Demande suspecte détectée."
        return (
            "🚨 **ALERTE SÉCURITÉ**\n"
            f"{reason}\n\n"
            "AgriConnect ne demande jamais de paiement ni de code Orange/Moov Money. "
            "Ne partagez pas vos informations sensibles et contactez un conseiller officiel."
        )


    # ------------------------------------------------------------------ #
    # Graphe                                                             #
    # ------------------------------------------------------------------ #

    def build(self):
        workflow = StateGraph(SentinelState)
        
        # Ajout des nœuds
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rewrite", self.refine.rewrite_query_node)  # ✅ Ajout du nœud manquant
        workflow.add_node("compose", self.compose_node)
        workflow.add_node("critique", self.refine.critique_node)      # ✅ Ajout du nœud manquant
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
    import json

    logging.basicConfig(level=logging.INFO)
    agent = ClimateSentinel()
    workflow = agent.build()
    # Exemple d’état initial
    state = {
        "user_query": "Y a-t-il un risque alimentaire presentement au burkina ?",
        "location_profile": {"village": "Bobo-Dioulasso", "zone": "Hauts-Bassins", "country": "Burkina Faso"},
    }
    result = workflow.invoke(state)
    print(json.dumps(result, indent=2, ensure_ascii=False))