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
    def __init__(self, llm_client=None):
        self.flood_tool = FloodRiskTool()
        self.market_tool = AgrimarketTool()
        
        # Initialize LLM for report synthesis
        try:
            self.llm = llm_client if llm_client else ChatOllama(model="mistral", base_url="http://localhost:11434", temperature=0.3)
        except Exception as e:
            logger.warning(f"LLM not available for report generation: {e}")
            self.llm = None

    def fetch_daily_data(self, state: GlobalAgriState):
        """Fetches fresh data for the report."""
        logger.info("Fetching daily data...")
        
        # 1. Flood Data
        location = state.get("zone_id", "Ouagadougou")
        flood_risk = self.flood_tool.check_flood_risk(location, 12.37, -1.52)
        
        # 2. Market Data
        market_prices = self.market_tool.get_market_prices()
        
        # 3. General Weather (Simulated or fetched via another tool if available)
        # For now, we infer general weather context from flood risk or use a placeholder
        # In a real integration, we would call ClimateVigilance's tool here.
        weather_summary = {
            "condition": "Ensoleillé avec passages nuageux",
            "temp_max": 35,
            "temp_min": 24,
            "wind": "Modéré (15 km/h)"
        }
        
        return {
            "meteo_data": {"flood_risk": flood_risk, "forecast": weather_summary},
            "market_data": market_prices,
            "execution_path": ["Daily Data Fetched"]
        }

    def generate_report(self, state: GlobalAgriState):
        """Synthesizes the data into a readable report using LLM."""
        logger.info("Generating report...")
        
        flood_risk = state.get("meteo_data", {}).get("flood_risk", {})
        forecast = state.get("meteo_data", {}).get("forecast", {})
        market_data = state.get("market_data", {})
        zone = state.get("zone_id", "Ouagadougou")
        
        # Prepare context for LLM
        context = {
            "zone": zone,
            "meteo": forecast,
            "risque_inondation": flood_risk,
            "marche": market_data
        }
        
        if self.llm:
            system_prompt = (
                "Tu es l'Assistant Intelligent d'AgConnect. Ta tâche est de rédiger le 'Flash Quotidien' pour un agriculteur sahélien.\n"
                "Ton rapport doit être concis, motivant et utile.\n\n"
                "STRUCTURE DU RAPPORT :\n"
                "1. 🌤️ MÉTÉO & ACTION : Ne donne pas juste la température. Interprète-la ! (ex: 'Il fait très chaud (38°C), évitez les traitements en plein soleil et arrosez ce soir').\n"
                "2. ⚠️ VIGILANCE : Si risque inondation (Élevé/Critique), mets une ALERTE ROUGE. Sinon, dis que tout est calme.\n"
                "3. 💰 LE GRAND MARCHÉ : Affiche le PRIX JUSTE vs PRIX MARCHÉ. Si l'écart est grand, dis 'Attention aux spéculateurs !'. Conseille de vendre ou stocker.\n"
                "4. 💡 CONSEIL DU JOUR : Une phrase d'encouragement ou technique (ex: 'Bonne journée pour le sarclage').\n\n"
                "Ton : Proche, respectueux, direct, orienté ACTION."
            )
            
            try:
                response = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Données du jour : {json.dumps(context, ensure_ascii=False)}")
                ])
                report_content = response.content
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                report_content = self._fallback_report(context)
        else:
            report_content = self._fallback_report(context)

        # Determine priority based on risk
        priority = "HIGH" if flood_risk.get("risk_level") in ["Élevé", "Critique"] else "NORMAL"

        return {
            "final_report": {"content": report_content, "priority": priority},
            "execution_path": ["Report Generated"]
        }

    def _fallback_report(self, context: Dict) -> str:
        """Template-based report if LLM fails."""
        zone = context['zone']
        flood = context['risque_inondation']
        market = context['marche']
        
        report = f"📅 FLASH INFO - {zone}\n\n"
        
        # Meteo
        report += f"🌤️ MÉTÉO : {context['meteo']['condition']}, Max {context['meteo']['temp_max']}°C\n\n"
        
        # Risk
        level = flood.get("risk_level", "Inconnu")
        if level in ["Élevé", "Critique"]:
            report += f"⚠️ ALERTE INONDATION : {level} ! {flood.get('alert_message')}\n\n"
        else:
            report += f"✅ RISQUES : {level} (Situation normale)\n\n"
            
        # Market
        report += "💰 PRIX DU JOUR :\n"
        for crop, price in market.items():
            report += f"- {crop} : {price}\n"
            
        return report

    def build_graph(self):
        graph = StateGraph(GlobalAgriState)
        
        graph.add_node("fetch_data", self.fetch_daily_data)
        graph.add_node("generate_report", self.generate_report)
        
        graph.set_entry_point("fetch_data")
        graph.add_edge("fetch_data", "generate_report")
        graph.add_edge("generate_report", END)
        
        return graph.compile()
