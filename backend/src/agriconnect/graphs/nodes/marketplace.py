"""
MarketplaceAgent — Agent conversationnel Agribusiness (LangGraph).

Gère les interactions agriculteurs via WhatsApp :
  • Identification par téléphone (auto-onboarding)
  • Gestion stock & dépenses
  • Mise en vente (produits)
  • Matching acheteur ↔ vendeur par zone
  • Commandes

Architecture : graphe LangGraph IDENTIFY → PARSE → EXECUTE → CONFIRM → MATCH.
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from backend.src.agriconnect.graphs.prompts import MARKETPLACE_SYSTEM_PROMPT
from backend.src.agriconnect.tools.marketplace import MarketplaceTool

# MCP & A2A & AG-UI Protocols
from backend.src.agriconnect.protocols.mcp import MCPDatabaseServer
from backend.src.agriconnect.protocols.a2a import A2ADiscovery, A2AMessage, MessageType, HandshakeStatus
from backend.src.agriconnect.protocols.ag_ui import (
    AgriResponse, AGUIComponent, ComponentType,
    TextBlock, Card, ActionButton, ActionType, ListPicker,
)

logger = logging.getLogger("Agent.Marketplace")


# ── État du graphe ──────────────────────────────────────────────
class MarketplaceState(TypedDict, total=False):
    # ...existing code...
    final_response: str
    agri_response: Optional[AgriResponse]
    status: str
    warnings: List[str]


# ── Intents reconnus ────────────────────────────────────────────
INTENTS = [
    "REGISTER_STOCK",
    "SELL_PRODUCT",
    "CHECK_STOCK",
    "CHECK_ORDERS",
    "UPDATE_STOCK",
    "FIND_BUYERS",
    "FIND_PRODUCTS",
    "CREATE_ORDER",
    "HELP",
]


class MarketplaceAgent:
    """Agent agribusiness — sous-graphe LangGraph appelé par l'orchestrateur."""

    def __init__(self, llm_client=None, mcp_db: MCPDatabaseServer = None, a2a: A2ADiscovery = None):
        self.model_planner = "llama-3.1-8b-instant"
        self.model_answer = "llama-3.3-70b-versatile"
        self.tool = MarketplaceTool() # Legacy tool, gradually migrate to MCP
        self.mcp_db = mcp_db
        self.a2a = a2a

        try:
            from backend.src.agriconnect.rag.components import get_groq_sdk
            self.llm = llm_client if llm_client else get_groq_sdk()
        except Exception as exc:
            logger.error("Impossible d'initialiser le LLM Marketplace : %s", exc)
            self.llm = None

    # ════════════════════════════════════════════════════════════════
    # NŒUDS DU GRAPHE
    # ════════════════════════════════════════════════════════════════

    def identify_user_node(self, state: MarketplaceState) -> MarketplaceState:
        """Identifie l'utilisateur par son numéro de téléphone."""
        state = dict(state)
        phone = state.get("user_phone", "")
        zone_id = state.get("zone_id")

        if not phone:
            state["status"] = "ERROR"
            state["final_response"] = (
                "Je n'ai pas pu identifier votre numéro. "
                "Envoyez votre message depuis WhatsApp pour que je vous reconnaisse."
            )
            return state

        try:
            user = self.tool.identify_or_create_user(phone, zone_id=zone_id)
            state["user_profile"] = user
            state["producer_id"] = user.get("producer_id")

            if user.get("is_new"):
                state["warnings"] = state.get("warnings", []) + ["NOUVEAU_UTILISATEUR"]
                logger.info("Nouvel agriculteur onboardé : %s", phone)

            # Récupérer la ferme
            if state["producer_id"]:
                farm = self.tool.get_or_create_farm(state["producer_id"], zone_id=zone_id)
                state["farm_id"] = farm["id"]

            state["status"] = "IDENTIFIED"
        except Exception as e:
            logger.error("Identification erreur : %s", e)
            state["status"] = "ERROR"
            state["final_response"] = "Erreur d'identification. Réessayez."

        return state

    def parse_intent_node(self, state: MarketplaceState) -> MarketplaceState:
        """Analyse l'intention et extrait les entités (produit, quantité, prix…)."""
        state = dict(state)
        if state.get("status") == "ERROR":
            return state

        query = state.get("user_query", "")
        if not query:
            state["intent"] = "HELP"
            state["parsed"] = {}
            return state

        prompt = (
            "Tu es un assistant agricole. Analyse le message de l'agriculteur.\n\n"
            f"Message : {query}\n\n"
            "Extrais les informations au format JSON strict :\n"
            "{{\n"
            '  "intent": "REGISTER_STOCK"|"SELL_PRODUCT"|"CHECK_STOCK"|"CHECK_ORDERS"|'
            '"UPDATE_STOCK"|"FIND_BUYERS"|"FIND_PRODUCTS"|"CREATE_ORDER"|"HELP",\n'
            '  "product": "nom du produit ou null",\n'
            '  "quantity": number ou null,\n'
            '  "unit": "sac"|"tine"|"plat"|"kg"|"tonne" ou null,\n'
            '  "price": number ou null,\n'
            '  "category": "Céréales"|"Légumineuses"|"Légumes"|"Fruits"|"Tubercules"|"Autres" ou null,\n'
            '  "description": "détails supplémentaires ou null"\n'
            "}}\n\n"
            "Exemples :\n"
            '- "J\'ai 10 sacs de maïs" → intent=REGISTER_STOCK, product=maïs, quantity=10, unit=sac\n'
            '- "Je veux vendre mon sorgho à 250 le kg" → intent=SELL_PRODUCT, product=sorgho, price=250\n'
            '- "Combien j\'ai en stock ?" → intent=CHECK_STOCK\n'
            '- "Qui cherche du mil ?" → intent=FIND_BUYERS, product=mil\n'
            '- "Je cherche du riz" → intent=FIND_PRODUCTS, product=riz\n'
        )

        try:
            resp = self.llm.chat.completions.create(
                model=self.model_planner,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning("Parse intent error : %s", e)
            parsed = {"intent": "HELP"}

        state["intent"] = parsed.get("intent", "HELP")
        state["parsed"] = parsed
        state["status"] = "PARSED"
        return state

    def execute_action_node(self, state: MarketplaceState) -> MarketplaceState:
        """Exécute l'action CRUD correspondant à l'intention."""
        state = dict(state)
        if state.get("status") == "ERROR":
            return state

        intent = state.get("intent", "HELP")
        parsed = state.get("parsed", {})
        producer_id = state.get("producer_id")
        farm_id = state.get("farm_id")
        zone_id = state.get("zone_id")
        phone = state.get("user_phone", "")
        result = {}

        try:
            if intent == "REGISTER_STOCK" and farm_id:
                product = parsed.get("product", "produit")
                quantity = parsed.get("quantity", 0)
                unit = parsed.get("unit", "kg")
                result = self.tool.add_stock(
                    farm_id=farm_id,
                    item_name=product,
                    quantity=quantity,
                    unit=unit,
                    reason=f"Déclaré via WhatsApp : {state.get('user_query', '')}",
                )

            elif intent == "SELL_PRODUCT" and producer_id:
                product = parsed.get("product", "produit")
                price = parsed.get("price", 0)
                quantity = parsed.get("quantity", 0)
                unit = parsed.get("unit", "kg")
                category = parsed.get("category", "Céréales")
                desc = parsed.get("description")

                result = self.tool.create_product(
                    producer_id=producer_id,
                    name=product,
                    price=price,
                    quantity_for_sale=quantity,
                    unit=unit,
                    category_label=category,
                    description=desc,
                )

            elif intent == "CHECK_STOCK" and farm_id:
                stocks = self.tool.get_stocks(farm_id)
                result = {"stocks": stocks, "count": len(stocks)}

            elif intent == "CHECK_ORDERS":
                products = self.tool.list_products(producer_id) if producer_id else []
                result = {"products": products}

            elif intent == "UPDATE_STOCK" and farm_id:
                product = parsed.get("product", "produit")
                quantity = parsed.get("quantity", 0)
                unit = parsed.get("unit", "kg")
                result = self.tool.add_stock(
                    farm_id=farm_id,
                    item_name=product,
                    quantity=quantity,
                    unit=unit,
                    reason="Mise à jour stock via WhatsApp",
                )

            elif intent == "FIND_BUYERS":
                product = parsed.get("product", "")
                result = {
                    "buyers": self.tool.find_buyers_for_product(product, zone_id),
                    "avg_price": self.tool.get_average_price(product, zone_id),
                }

            elif intent == "FIND_PRODUCTS":
                product = parsed.get("product", "")
                result = {
                    "products_available": self.tool.find_products_for_buyer(product, zone_id),
                }

            elif intent == "CREATE_ORDER":
                product_name = parsed.get("product", "")
                quantity = parsed.get("quantity", 1)
                # Chercher le produit correspondant
                available = self.tool.find_products_for_buyer(product_name, zone_id)
                if available:
                    best = available[0]  # moins cher
                    result = self.tool.create_order(
                        product_id=best["id"],
                        quantity=quantity,
                        buyer_phone=phone,
                        buyer_name=state.get("user_profile", {}).get("name"),
                        zone_id=zone_id,
                    )
                else:
                    result = {"error": f"Aucun {product_name} disponible dans votre zone."}
                    # Créer une alerte de recherche
                    alert = self.tool.create_market_alert(product_name, phone, zone_id)
                    result["alert"] = alert

            else:
                result = {"help": True}

        except Exception as e:
            logger.error("Execute action error (%s) : %s", intent, e)
            result = {"error": str(e)}

        state["action_result"] = result
        state["status"] = "EXECUTED"
        return state

    def confirm_node(self, state: MarketplaceState) -> MarketplaceState:
        """Génère la réponse conversationnelle (AG-UI) + Texte."""
        state = dict(state)
        agri_response = AgriResponse(agent="marketplace")
        
        if state.get("status") == "ERROR":
            return state

        intent = state.get("intent", "HELP")
        result = state.get("action_result", {})
        parsed = state.get("parsed", {})
        user = state.get("user_profile", {})
        is_new = "NOUVEAU_UTILISATEUR" in state.get("warnings", [])

        # Génération texte via LLM (inchangé)
        # ...existing code...
        context = {
            "intent": intent,
            "result": result,
            "parsed": parsed,
            "user_name": user.get("name", "Agriculteur"),
            "is_new_user": is_new,
        }

        prompt = (
            f"{MARKETPLACE_SYSTEM_PROMPT}\n\n"
            "CONTEXTE DE L'ACTION :\n"
            f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
            f"Message original de l'agriculteur : {state.get('user_query', '')}\n\n"
            "Génère une réponse claire et chaleureuse en français simple.\n"
            "Si c'est un nouvel utilisateur, souhaite-lui la bienvenue.\n"
            "Confirme l'action réalisée avec les détails importants.\n"
            "Utilise des emojis pertinents (🌾🛒✅💰📦).\n"
            "Propose une action suivante si pertinent."
        )

        try:
            resp = self.llm.chat.completions.create(
                model=self.model_answer,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            final_text = resp.choices[0].message.content
        except Exception as e:
            logger.warning("Confirm node error : %s", e)
            final_text = self._fallback_response(intent, result, is_new)
        
        state["final_response"] = final_text
        
        # ── Construction AG-UI ──
        response_obj = AgriResponse(agent="marketplace")
        response_obj.add_text(final_text)
        
        # Ajout de composants interactifs selon l'intent
        if intent == "SELL_PRODUCT":
             response_obj.add(ListPicker(
                title="Options de vente :",
                items=[
                    {"id": "view_offers", "label": "👀 Voir les offres"},
                    {"id": "modify_price", "label": "✏️ Modifier prix"}
                ]
            ))
        elif intent == "CHECK_STOCK":
             if result.get("stocks"):
                 # Créer un résumé structuré du stock en Card
                 stock_lines = "\n".join(
                     f"• {s.get('item_name')}: {s.get('quantity')} kg"
                     for s in result.get("stocks", [])
                 )
                 response_obj.add_card(
                     title="Votre Stock",
                     body=stock_lines,
                 )

        state["agri_response"] = response_obj
        state["status"] = "CONFIRMED"
        return state

    def match_check_node(self, state: MarketplaceState) -> MarketplaceState:
        """
        Vérifie les acheteurs (Local DB) ET Broadcast A2A pour la scalabilité.
        """
        state = dict(state)
        # ...existing code...
        intent = state.get("intent")

        if intent != "SELL_PRODUCT":
            return state

        result = state.get("action_result", {})
        product_name = state.get("parsed", {}).get("product", "")
        zone_id = state.get("zone_id")
        phone = state.get("user_phone", "")
        product_id = result.get("product_id")

        if product_name and product_id:
            # 1. Matching Local (Legacy)
            matches = self.tool.auto_match(product_name, zone_id, phone, product_id)
            if matches:
                state["matches"] = matches
                match_msg = (
                    f"\n\n🎯 **Bonne nouvelle !** {len(matches)} acheteur(s) "
                    f"cherchent du {product_name} dans votre zone :\n"
                )
                for m in matches:
                    match_msg += f"  • {m['buyer_phone']} ({m.get('zone_name', '')})\n"
                match_msg += "\nJe peux les mettre en contact avec vous. Voulez-vous ?"

                state["final_response"] = state.get("final_response", "") + match_msg
                
                # Mise à jour AG-UI
                if "agri_response" in state:
                    state["agri_response"].add_text(match_msg)
            
            # 2. Diffusion A2A (Nouveau Protocole)
            if self.a2a:
                try:
                    # On broadcast l'offre à tous les agents "ACHETEUR" ou "GROSSISTE"
                    offer_msg = A2AMessage(
                        message_type=MessageType.BROADCAST,
                        sender_id=f"agent_market_{state.get('user_phone')}",  # Agent éphémère pour l'utilisateur
                        intent="SELL_OFFER",
                        zone=zone_id or "global",
                        crop=product_name,
                        payload={
                            "product": product_name,
                            "quantity": state.get("parsed", {}).get("quantity"),
                            "price": state.get("parsed", {}).get("price"),
                            "product_id": product_id,
                            "seller_phone": phone # En production, on masque ça
                        }
                    )
                    # Broadcast sur le topic du produit
                    topic = f"SELL_{product_name.upper()}_{zone_id}"
                    count = self.a2a.broadcast_message(offer_msg, topic=topic)
                    logger.info(f"📢 A2A Broadcast: Offre {product_name} diffusée à {count} agents.")
                except Exception as e:
                    logger.warning(f"A2A Broadcast failed: {e}")

        state["status"] = "COMPLETED"
        return state

    # ════════════════════════════════════════════════════════════════
    # FALLBACK (sans LLM)
    # ════════════════════════════════════════════════════════════════

    def _fallback_response(
        self, intent: str, result: Dict, is_new: bool
    ) -> str:
        welcome = "🌾 Bienvenue sur AgriConnect ! Je suis votre assistant marketplace.\n\n" if is_new else ""

        if result.get("error"):
            return f"{welcome}⚠️ {result['error']}"

        if intent == "REGISTER_STOCK":
            return (
                f"{welcome}✅ Stock enregistré !\n"
                f"📦 {result.get('item_name')} : +{result.get('added_kg', 0):.0f} kg\n"
                f"Total en stock : {result.get('new_total_kg', 0):.0f} kg"
            )
        if intent == "SELL_PRODUCT":
            return (
                f"{welcome}🛒 Produit mis en vente !\n"
                f"📦 {result.get('name')} — {result.get('price_fcfa', 0):.0f} FCFA/kg\n"
                f"Code : {result.get('short_code')}"
            )
        if intent == "CHECK_STOCK":
            stocks = result.get("stocks", [])
            if not stocks:
                return f"{welcome}📦 Votre stock est vide. Dites-moi ce que vous avez récolté !"
            lines = [f"  • {s.get('item_name')} : {s.get('quantity', 0):.0f} kg" for s in stocks]
            return f"{welcome}📦 **Votre stock :**\n" + "\n".join(lines)
        if intent == "FIND_BUYERS":
            buyers = result.get("buyers", [])
            if not buyers:
                return f"{welcome}🔍 Aucun acheteur trouvé pour le moment. Je crée une alerte."
            lines = [f"  • {b.get('buyer_phone')} ({b.get('zone_name', '')})" for b in buyers]
            return f"{welcome}🎯 Acheteurs intéressés :\n" + "\n".join(lines)
        if intent == "HELP":
            return (
                f"{welcome}🌾 Je peux vous aider à :\n"
                "  📦 Enregistrer votre stock (ex: 'J'ai 10 sacs de maïs')\n"
                "  🛒 Mettre en vente (ex: 'Je vends du sorgho à 250 FCFA/kg')\n"
                "  🔍 Trouver des acheteurs (ex: 'Qui cherche du mil ?')\n"
                "  📊 Voir votre stock (ex: 'Mon stock')\n"
                "  🛍️ Commander (ex: 'Je cherche du riz')\n"
            )

        return f"{welcome}Action effectuée. Que souhaitez-vous faire ensuite ?"

    # ════════════════════════════════════════════════════════════════
    # BUILD — Compilation du graphe LangGraph
    # ════════════════════════════════════════════════════════════════

    def build(self):
        workflow = StateGraph(MarketplaceState)

        workflow.add_node("identify_user", self.identify_user_node)
        workflow.add_node("parse_intent", self.parse_intent_node)
        workflow.add_node("execute_action", self.execute_action_node)
        workflow.add_node("confirm", self.confirm_node)
        workflow.add_node("match_check", self.match_check_node)

        workflow.set_entry_point("identify_user")

        def route_after_identify(state):
            if state.get("status") == "ERROR":
                return END
            return "parse_intent"

        workflow.add_conditional_edges("identify_user", route_after_identify)
        workflow.add_edge("parse_intent", "execute_action")
        workflow.add_edge("execute_action", "confirm")
        workflow.add_edge("confirm", "match_check")
        workflow.add_edge("match_check", END)

        return workflow.compile()


# ── Test standalone ──────────────────────────────────────────────
if __name__ == "__main__":
    agent = MarketplaceAgent()
    app = agent.build()

    test_cases = [
        {
            "name": "ENREGISTRER STOCK",
            "query": "J'ai récolté 15 sacs de maïs",
            "phone": "+22670000001",
        },
        {
            "name": "METTRE EN VENTE",
            "query": "Je veux vendre mon sorgho à 250 le kilo, j'en ai 5 sacs",
            "phone": "+22670000001",
        },
        {
            "name": "CHERCHER ACHETEUR",
            "query": "Qui cherche du mil dans ma zone ?",
            "phone": "+22670000001",
        },
        {
            "name": "AIDE",
            "query": "Comment ça marche ?",
            "phone": "+22670000002",
        },
    ]

    print("🚀 Tests MarketplaceAgent...\n")
    for case in test_cases:
        print(f"--- {case['name']} ---")
        print(f"Query: {case['query']}")

        initial = {
            "user_query": case["query"],
            "user_phone": case["phone"],
            "zone_id": None,
            "warnings": [],
        }

        try:
            result = app.invoke(initial)
            print(f"Intent: {result.get('intent')}")
            print(f"Réponse:\n{result.get('final_response')}\n")
        except Exception as e:
            print(f"❌ Erreur : {e}\n")
        print("-" * 50)
