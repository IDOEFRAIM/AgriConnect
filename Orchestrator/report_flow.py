import logging
import json
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOllama

from orchestrator.state import GlobalAgriState
from tools.meteo.flood_risk import FloodRiskTool
from tools.subventions.base_subsidy import AgrimarketTool

logger = logging.getLogger("ReportFlow")

class DailyReportFlow:
    """
    Générateur de bulletins quotidiens proactifs.
    Fusionne les risques environnementaux et les opportunités économiques.
    """
    def __init__(self, llm_client=None):
        self.flood_tool = FloodRiskTool()
        self.market_tool = AgrimarketTool()
        
        # Initialisation du LLM de synthèse (Mistral est excellent pour le Français)
        try:
            self.llm = llm_client if llm_client else ChatOllama(
                model="mistral", 
                base_url="http://localhost:11434", 
                temperature=0.3
            )
        except Exception as e:
            logger.warning(f"LLM non disponible pour la synthèse : {e}")
            self.llm = None

    def fetch_daily_data(self, state: GlobalAgriState):
        """Collecte les données fraîches du jour."""
        logger.info("--- NODE: FETCHING DAILY DATA ---")
        location = state.get("zone_id", "Ouagadougou")
        
        # 1. Analyse Risque Inondation (coordonnées moyennes simulées pour la zone)
        flood_risk = self.flood_tool.check_flood_risk(location, 12.37, -1.52)
        
        # 2. Analyse Prix du Marché
        market_prices = self.market_tool.get_market_prices()
        
        # 3. Prévisions Météo Générales
        weather_summary = {
            "condition": "Ensoleillé avec passages nuageux",
            "temp_max": 35, "temp_min": 24, "wind": "15 km/h"
        }
        
        return {
            "meteo_data": {"flood_risk": flood_risk, "forecast": weather_summary},
            "market_data": market_prices
        }

    def generate_report(self, state: GlobalAgriState):
        """Transforme les données en un flash info actionnable."""
        logger.info("--- NODE: GENERATING ACTIONABLE REPORT ---")
        
        context = {
            "zone": state.get("zone_id", "Ouagadougou"),
            "meteo": state.get("meteo_data", {}).get("forecast", {}),
            "risque_inondation": state.get("meteo_data", {}).get("flood_risk", {}),
            "marche": state.get("market_data", {})
        }
        
        if self.llm:
            system_prompt = (
                "Tu es le rédacteur en chef d'AgConnect Flash. Ton but est de protéger et d'enrichir l'agriculteur.\n"
                "RÈGLES DE RÉDACTION :\n"
                "1. TON : Fraternel, direct, expert.\n"
                "2. PRIORITÉ : Si le risque d'inondation est Élevé/Critique, commence par '🚨 ALERTE ROUGE'.\n"
                "3. ÉCONOMIE : Interprète les prix (ex: 'Le prix du maïs grimpe, attendez avant de vendre').\n"
                "4. ACTION : Donne un conseil pratique basé sur la météo du jour."
            )
            
            try:
                response = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Données brutes : {json.dumps(context, ensure_ascii=False)}")
                ])
                report_content = response.content
            except Exception as e:
                logger.error(f"Échec LLM : {e}")
                report_content = self._fallback_report(context)
        else:
            report_content = self._fallback_report(context)

        # Logique de priorité pour le canal d'envoi (SMS vs Notification)
        priority = "URGENT" if context["risque_inondation"].get("risk_level") in ["Élevé", "Critique"] else "NORMAL"

        return {
            "final_report": {"content": report_content, "priority": priority}
        }

    def _fallback_report(self, context: Dict) -> str:
        """Template de secours en cas de panne du LLM."""
        return f"📢 FLASH AGCONNECT - {context['zone']}\n\nMétéo: {context['meteo']['condition']} ({context['meteo']['temp_max']}°C)\nRisque Inondation: {context['risque_inondation']['risk_level']}\nMarché: {len(context['marche'])} cultures suivies."

    def build_graph(self):
        """Compile le workflow LangGraph."""
        workflow = StateGraph(GlobalAgriState)
        workflow.add_node("fetch_data", self.fetch_daily_data)
        workflow.add_node("generate_report", self.generate_report)
        
        workflow.set_entry_point("fetch_data")
        workflow.add_edge("fetch_data", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()