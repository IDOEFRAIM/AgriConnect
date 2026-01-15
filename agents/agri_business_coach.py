import logging
import json
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime

# --- Importations LangGraph & LangChain ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

# --- IMPORTATION DES OUTILS ---
# Assure-toi que le chemin correspond bien à l'emplacement de ton fichier Tool corrigé
from tools.subventions.base_subsidy import AgrimarketTool 

logger = logging.getLogger("agent.agri_business")

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
    metadata: Dict[str, Any]

# ======================================================================
# 2. SERVICE BUSINESS DU GRAND FRÈRE
# ======================================================================
class AgriBusinessCoach:
    OLLAMA_MODEL = "mistral"  # Ou "llama3:8b" selon ton installation

    def __init__(self, ollama_host: str = "http://localhost:11434", llm_client=None):
        self.market_tool = AgrimarketTool() 
        
        # Initialisation sécurisée avec Timeout pour éviter les déconnexions
        try:
            self.llm = llm_client if llm_client else ChatOllama(
                model=self.OLLAMA_MODEL, 
                base_url=ollama_host, 
                temperature=0,
                timeout=60,  # Augmenté à 60s pour éviter le "RemoteDisconnected"
                keep_alive="5m" # Garde le modèle en mémoire 5 minutes
            )
        except Exception as e:
            logger.error(f"Erreur init Ollama : {e}")
            self.llm = None

    def _analyze_intent_semantically(self, query: str) -> Dict[str, Any]:
        """
        Analyse sémantique pour détecter les arnaques et l'intention réelle.
        """
        if not self.llm:
            return {"is_scam": False, "intent": "INFO", "reason": "No LLM"}

        system_prompt = (
            "Tu es l'expert en sécurité d'AgriConnect Burkina. Analyse la requête.\n"
            "Réponds UNIQUEMENT au format JSON :\n"
            '{"is_scam": boolean, "intent": "VENTE" | "ACHAT" | "INFO", "reason": "string"}'
        )

        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
            resp = self.llm.invoke(messages)
            
            # Nettoyage robuste du JSON (au cas où le LLM met du Markdown autour)
            clean_content = resp.content.strip()
            if "```json" in clean_content:
                clean_content = clean_content.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_content:
                clean_content = clean_content.split("```")[1].strip()
            
            data = json.loads(clean_content)
            
            return {
                "is_scam": bool(data.get("is_scam", False)),
                "intent": data.get("intent", "INFO").upper(),
                "reason": data.get("reason", "")
            }
        except Exception as e:
            logger.error(f"Erreur analyse sémantique : {e}")
            # Fallback de sécurité basique
            is_scam_keyword = any(x in query.lower() for x in ["payer", "frais", "orange money", "code"])
            return {"is_scam": is_scam_keyword, "intent": "INFO", "reason": "fallback"}

    def analyze_node(self, state: AgentState) -> AgentState:
        """Nœud logique principal."""
        query = state["user_query"]
        profile = state.get("user_profile", {})
        crop = profile.get("crop", "Maïs")
        
        # 1. Analyse de l'intention
        analysis = self._analyze_intent_semantically(query)
        response_parts = []
        status = "SUCCESS"

        # 2. Gestion SCAM / ARNAQUE
        if analysis["is_scam"]:
            status = "SCAM_DETECTED"
            response_parts.append("🚨 **ALERTE SÉCURITÉ : TENTATIVE D'ARNAQUE DÉTECTÉE**")
            response_parts.append("\n⚠️ **STOP !** AgriConnect ne demande JAMAIS d'argent pour une subvention.")
            response_parts.append("Ne donnez jamais votre code Orange Money ou Moov Money.")
        
        else:
            intent = analysis["intent"]
            
            # --- CAS 1 : VENTE ---
            if intent == "VENTE":
                offers = self.market_tool.list_offers("ACHAT")
                response_parts.append("🏢 **OPPORTUNITÉS DE VENTE**")
                if offers:
                    for o in offers[:3]:
                        # Utilisation sécurisée des clés (.get)
                        prod = o.get('product', 'Produit')
                        price = o.get('price_per_kg', 'Prix N/C')
                        loc = o.get('location', 'Lieu N/C')
                        response_parts.append(f"✅ Acheteur : {prod} à {price} FCFA/kg ({loc})")
                else:
                    response_parts.append("Aucun acheteur enregistré pour le moment.")
                
                response_parts.append("\n💡 *Utilisez notre Tiers de Confiance pour sécuriser la transaction.*")

            # --- CAS 2 : ACHAT ---
            elif intent == "ACHAT":
                offers = self.market_tool.list_offers("VENTE")
                response_parts.append("🛒 **OFFRES DISPONIBLES**")
                if offers:
                    for o in offers[:3]:
                        prod = o.get('product', 'Produit')
                        price = o.get('price_per_kg', 'Prix N/C')
                        contact = o.get('contact', 'N/C')
                        response_parts.append(f"📦 {prod} : {price} FCFA/kg (Tel: {contact})")
                else:
                    response_parts.append("Aucune offre disponible pour le moment.")

            # --- CAS 3 : INFO / CONSEIL ---
            else:
                # CORRECTION MAJEURE ICI : Suppression de l'argument 'month'
                # L'outil calcule le mois tout seul via datetime.now()
                timing = self.market_tool.analyze_market_timing(crop)
                
                response_parts.append(f"📈 **INTELLIGENCE MARCHÉ : {crop.upper()}**")
                
                if timing.get("warrantage") == "CONSEILLÉ":
                    # Insertion du visuel Warrantage
                    response_parts.append("\n")
                    response_parts.append(f"💰 **CONSEIL OR :** {timing.get('conseil')}")
                else:
                    response_parts.append(f"ℹ️ **AVIS :** {timing.get('conseil')}")

        state["technical_advice_raw"] = "\n".join(response_parts)
        state["status"] = status
        state["metadata"] = analysis
        return state

    def format_node(self, state: AgentState) -> AgentState:
        """Mise en forme chaleureuse."""
        # Si c'est une arnaque ou si LLM cassé, on renvoie le brut
        if state["status"] == "SCAM_DETECTED" or not self.llm:
            state["final_response"] = state["technical_advice_raw"]
            return state

        system_prompt = (
            "Tu es le 'Grand Frère' d'AgriConnect Burkina. Ton ton est protecteur et expert.\n"
            "Tu ne changes PAS les données chiffrées.\n"
            "Tu gardes impérativement les balises  telles quelles.\n"
            "Sois concis et encourageant."
        )

        try:
            resp = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Voici les infos brutes : {state['technical_advice_raw']}")
            ])
            state["final_response"] = resp.content
        except Exception:
            state["final_response"] = state["technical_advice_raw"]
            
        return state

    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("format", self.format_node)
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "format")
        workflow.add_edge("format", END)
        return workflow.compile()