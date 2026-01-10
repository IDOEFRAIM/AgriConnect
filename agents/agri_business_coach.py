import logging
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime

# --- Importations LangGraph & LangChain ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

# --- IMPORTATION DES OUTILS RÉELS ---
from tools.subventions.base_subsidy import AgrimarketTool 

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
# 2. SERVICE BUSINESS DU GRAND FRERE
# ======================================================================

class AgriBusinessCoach:
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

    def _analyze_intent_semantically(self, query: str) -> Dict[str, Any]:
        """
        Remplace les mots-clés fragiles par une compréhension IA du contexte.
        Retourne : {'is_scam': bool, 'intent': 'VENTE'|'ACHAT'|'INFO'}
        """
        if not self.llm_client:
            # Fallback (Mode secours si pas d'IA)
            scam_words = ["payer", "frais", "code", "envoie"]
            is_scam = any(w in query.lower() for w in scam_words)
            intent = "INFO"
            if "vend" in query.lower() or "dispo" in query.lower(): intent = "VENTE"
            elif "ach" in query.lower() or "cherch" in query.lower(): intent = "ACHAT"
            return {"is_scam": is_scam, "intent": intent}

        prompt = (
            "Tu es le cerveau de sécurité d'AgriConnect. Analyse cette phrase paysanne.\n"
            f"Phrase : '{query}'\n\n"
            "TÂCHES :\n"
            "1. DETECTION ARNAQUE : L'utilisateur a-t-il reçu une demande suspecte d'argent/code ? (Attention: s'il veut payer un service légitime, ce n'est pas une arnaque).\n"
            "2. INTENTION : Veut-il VENDRE (proposer), ACHETER (chercher) ou s'INFORMER ?\n\n"
            "Réponds UNIQUEMENT sous ce format : 'SCAM=[OUI/NON] | INTENT=[VENTE/ACHAT/INFO]'"
        )
        
        try:
            resp = self.llm_client.invoke([SystemMessage(content=prompt), HumanMessage(content="Analyse ça.")])
            text = resp.content.upper()
            
            is_scam = "SCAM=OUI" in text
            if "INTENT=VENTE" in text: intent = "VENTE"
            elif "INTENT=ACHAT" in text: intent = "ACHAT"
            else: intent = "INFO"
            
            return {"is_scam": is_scam, "intent": intent}
        except Exception as e:
            logger.error(f"Erreur analyse sémantique: {e}")
            return {"is_scam": False, "intent": "INFO"}

    def analyze_node(self, state: AgentState) -> AgentState:
        """Analyse Sécurité, Marché et Subventions."""
        query = state.get("user_query", "").lower()
        profile = state.get("user_profile", {})
        crop = profile.get("crop", "Maïs")
        region = state.get("zone_id", "Centre")
        current_month = datetime.now().month
        
        # --- 0. ANALYSE SÉMANTIQUE (Le Cerveau) ---
        semantic = self._analyze_intent_semantically(query)
        
        response_parts = []
        status = "SUCCESS"

        # --- 1. SÉCURITÉ (Phishing) ---
        if semantic["is_scam"]:
            status = "SCAM_DETECTED"
            tag_scam = ""
            response_parts.append(f"🚨 **ALERTE SÉCURITÉ** {tag_scam}")
            response_parts.append("\n⚠️ **STOP !** Analyse IA : Cette demande ressemble à une arnaque.")
            response_parts.append("L'État et AgriConnect ne demandent JAMAIS de code ou de frais par message.")

        # --- 2. GRAND MARCHÉ NATIONAL ---
        elif semantic["intent"] in ["VENTE", "ACHAT"]:
            response_parts.append(f"🏢 **GRAND MARCHÉ NATIONAL**")
            
            if semantic["intent"] == "VENTE":
                # Simulation de publication (dans une vraie app, on demanderait les détails)
                # Ici on liste les offres d'achat existantes pour matcher
                offers = self.market_tool.list_offers("ACHAT")
                response_parts.append(f"Voici les acheteurs potentiels pour vos produits :")
                for o in offers:
                    response_parts.append(f"- {o['product']} : {o['quantity_kg']}kg à {o['price_per_kg']} FCFA/kg ({o['location']}) - 📞 {o['contact']}")
                
                response_parts.append("\n🛒 **VENDRE EN TOUTE SÉCURITÉ** :")
                response_parts.append("1. **TIERS DE CONFIANCE** : L'acheteur dépose l'argent sur AgriConnect. Vous êtes payé à la livraison. Zéro risque.")
                response_parts.append("2. **LOGISTIQUE** : Qui paie le transport ? (Cochez la case sur le bon de commande).")
                response_parts.append("Pour publier votre offre, dites 'Je veux vendre X kg de Y'.")
            
            elif semantic["intent"] == "ACHAT":
                offers = self.market_tool.list_offers("VENTE")
                response_parts.append(f"Voici les produits disponibles :")
                for o in offers:
                    response_parts.append(f"- {o['product']} : {o['quantity_kg']}kg à {o['price_per_kg']} FCFA/kg ({o['location']}) - 📞 {o['contact']}")
                response_parts.append("\n🔒 **ACHAT SÉCURISÉ** : Votre argent est protégé par le Tiers de Confiance AgriConnect jusqu'à réception.")

        # --- 3. INTELLIGENCE MARCHÉ & AIDES ---
        else:
            # Info Marché
            market = self.market_tool.analyze_market_timing(crop, current_month)
            
            # --- FEATURE : PRIX SONAGESS & REGRET VENDEUR ---
            sonagess_price = 1500 # Prix officiel simulé (Mock) pour l'exemple
            market_price_raw = market.get('prix_actuel_estime', 1000)
            try:
                current_price = int(str(market_price_raw).replace('F', '').replace('CFA', '').strip())
            except:
                current_price = 800

            response_parts.append(f"📈 **MARCHÉ : {crop.upper()}**")
            
            # Indicateur Visuel
            if current_price < sonagess_price:
                response_parts.append(f"🔴 **MAUVAISE VENTE**")
                response_parts.append(f"Prix marché ({current_price}F) < PRIX OFFICIEL SONAGESS ({sonagess_price}F).")
                response_parts.append(f"⚠️ Les 'Gens de l'Ombre' essaient de vous arnaquer.")
            else:
                 response_parts.append(f"🟢 **BONNE VENTE** (Prix marché {current_price}F > Officiel).")
            
            # --- FEATURE AJOUTÉE : STOCKAGE ANTI-REGRET (Si prix bas) ---
            if current_price < sonagess_price:
                 nearby_storage = self.market_tool.find_nearby_storage(region)
                 response_parts.append(f"\n🏚️ **MAGASIN DE STOCKAGE (Solution)**")
                 if nearby_storage:
                     w = nearby_storage[0]
                     response_parts.append(f"Ne bradez pas ! Stockez à **{w['name']}** ({w['ville']}).")
                     response_parts.append("En stockant 3 mois, vous vendrez au prix fort.")
                 else:
                     response_parts.append("Cherchez un magasin agréé pour faire du Warrantage.")

            # Le Regret du Vendeur
            predicted_price = int(current_price * 1.5) # +50% dans 3 mois
            response_parts.append(f"\n🔮 **MANQUE À GAGNER POTENTIEL**")
            response_parts.append(f"Si vous vendez aujourd'hui : {current_price}F/kg")
            response_parts.append(f"Si vous stockez 3 mois : {predicted_price}F/kg (Prévision)")
            response_parts.append(f"💰 **Vous perdez {predicted_price - current_price}F par kilo en vendant maintenant !**")

            # Warrantage (Si applicable)
            if market.get('opportunite_warrantage') == "CONSEILLÉ":
                tag_warr = ""
                response_parts.append(f"\n💡 **WARRANTAGE** : Stockez vos sacs et obtenez un crédit immédiat sans vendre.")

            # Subventions
            sub_text = self.market_tool.get_subsidy_status(region)
            response_parts.append(f"\n💰 **AIDES RÉGIONALES :**")
            tag_docs = ""
            response_parts.append(tag_docs)
            response_parts.append(sub_text)

        raw_text = "\n".join(response_parts)
        return {**state, "technical_advice_raw": raw_text, "status": status}

    def format_node(self, state: AgentState) -> AgentState:
        """Mise en forme pédagogique via LLM."""
        if state["status"] == "SCAM_DETECTED" or not self.llm_client:
            return {**state, "final_response": state["technical_advice_raw"]}

        system_prompt = (
            "Tu es le 'Grand Frère' d'AgriConnect. Ton but : protéger le revenu du paysan burkinabè.\n\n"
            
            "CONTEXTE : L'agriculteur a peur des arnaques ('ceux qui fouillent avec l'argent') et "
            "regrette souvent de vendre trop tôt (perte de 50% de gain).\n\n"
            
            "CONSIGNES DE RÉPONSE :\n"
            "1. ANALYSE PRIX : Compare le prix proposé au prix SONAGESS. Si < officiel, alerte en ROUGE.\n"
            "2. STRATÉGIE ANTI-REGRET : Si l'historique montre que le prix va doubler (ex: Oignons), "
            "propose le STOCKAGE au lieu de la vente immédiate.\n"
            "3. SÉCURITÉ : Rappelle que l'argent est bloqué par le 'Tiers de Confiance' AgriConnect "
            "jusqu'à ce que le taxi-moto livre la marchandise.\n"
            "4. TRANSPARENCE FRAIS : Précise que nous prenons 100 F/sac uniquement SI la vente réussit.\n\n"
            
            "STRUCTURE :\n"
            "💰 VERDICT PRIX : [Prix Marché] vs [Prix SONAGESS]. C'est une [Bonne/Mauvaise] affaire.\n"
            "📈 PRÉVISION : 'Si tu attends 3 mois, tu pourrais gagner X FCFA de plus.'\n"
            "🛡️ ACTION : 'Je bloque l'argent de l'acheteur maintenant. Qui paie le transport ?'\n"
            "📦 QUALITÉ : 'Envoie-moi une photo du sac pour que je confirme le deal.'"
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