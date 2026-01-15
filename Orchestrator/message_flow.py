import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from orchestrator.state import GlobalAgriState
from services.utils.cache import StorageManager
from agents.climate_vigilance import ClimateVigilance
from agents.agri_business_coach import AgriBusinessCoach

logger = logging.getLogger("MessageFlow")

class MessageResponseFlow:
    """
    Orchestrateur LangGraph pour fusionner l'intelligence climatique 
    et économique (Agri-Business).
    """
    def __init__(self, llm_client=None):
        self.storage = StorageManager() 
        
        # Configuration du LLM de synthèse (Llama 3)
        try:
            self.llm = llm_client if llm_client else ChatOllama(model="llama3:8b", keep_alive=-1)
        except Exception as e:
            logger.warning(f"Orchestrator LLM init failed: {e}")
            self.llm = None

        # Agents spécialisés
        self.meteo_agent = ClimateVigilance(llm_client=llm_client)
        self.market_agent = AgriBusinessCoach(llm_client=llm_client)

    def run_meteo(self, state: GlobalAgriState) -> Dict[str, Any]:
        """Exécute l'analyse climatique et hydrique."""
        logger.info("--- NODE: CLIMATE VIGILANCE ---")
        zone_id = state.get("zone_id", "Centre")
        
        # Simulation/Récupération des données locales
        weather_data = {"t_min": 25, "t_max": 35, "rh": 40, "precip": 0} 
        try:
            raw = self.storage.get_raw_data(zone_id=zone_id, category="METEO_VECTOR", limit=1)
            if raw: weather_data.update(raw[0])
        except Exception as e:
            logger.error(f"Erreur cache météo: {e}")

        agent_state = {
            "user_query": state.get("requete_utilisateur", ""),
            "weather_data": weather_data,
            "culture_info": {"crop_name": state.get("crop", "Maïs"), "location": zone_id},
            "final_response": ""
        }
        
        # Cycle interne de l'agent météo
        res = self.meteo_agent.validate_and_calculate(agent_state)
        agent_state.update(res)
        final = self.meteo_agent.generate_expert_response(agent_state)
        
        return {"meteo_info": final["final_response"]}

    def run_market(self, state: GlobalAgriState) -> Dict[str, Any]:
        """Exécute l'analyse de rentabilité et d'opportunité de marché."""
        logger.info("--- NODE: AGRI-BUSINESS COACH ---")
        
        agent_state = {
            "zone_id": state.get("zone_id", "Centre"),
            "user_query": state.get("requete_utilisateur", ""),
            "user_profile": {"crop": state.get("crop", "Maïs")},
            "final_response": ""
        }
        
        # Cycle interne de l'agent marché
        res = self.market_agent.analyze_node(agent_state)
        agent_state.update(res)
        final = self.market_agent.format_node(agent_state)
        
        return {"market_info": final["final_response"]}

    def synthesize_answer(self, state: GlobalAgriState) -> Dict[str, Any]:
        """
        Nœud de décision final : Fusionne les risques climatiques et les gains économiques.
        """
        query = state.get("requete_utilisateur", "")
        meteo_resp = state.get("meteo_info", "Alerte : Données météo manquantes.")
        market_resp = state.get("market_info", "Alerte : Données marché manquantes.")

        if not self.llm:
            return {"final_response": f"{meteo_resp}\n\n---\n\n{market_resp}"}
            
        system_prompt = (
            "Tu es le Superviseur AgriConnect Expert. Ton rôle est de fournir un conseil "
            "stratégique cross-domaine (Climat + Marché) pour un agriculteur au Burkina Faso.\n\n"
            "DIRECTIVES :\n"
            "1. L'action doit être PRIORITAIRE (ex: Ne pas épandre d'engrais s'il va pleuvoir, même si le prix est bon).\n"
            "2. Calcule l'impact financier du risque climatique si possible.\n"
            "3. Utilise un ton de 'grand frère' expert, pragmatique et rassurant.\n"
            "4. Structure la réponse : 1. Action Immédiate | 2. Analyse Risque/Gain | 3. Conseil Marché."
        )
        
        try:
            human_msg = (
                f"REQUÊTE : '{query}'\n\n"
                f"CONTEXTE CLIMATIQUE : {meteo_resp}\n\n"
                f"CONTEXTE ÉCONOMIQUE : {market_resp}\n\n"
                "CONSEIL STRATÉGIQUE :"
            )
            res = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_msg)])
            return {"final_response": res.content}
        except Exception as e:
            logger.error(f"Echec synthèse: {e}")
            return {"final_response": f"Stratégie Climat: {meteo_resp}\nStratégie Marché: {market_resp}"}

    def build_graph(self):
        """Compile le workflow de décision."""
        workflow = StateGraph(GlobalAgriState)
        
        # Définition des nœuds d'expertise
        workflow.add_node("meteo_node", self.run_meteo)
        workflow.add_node("market_node", self.run_market)
        workflow.add_node("synthesizer_node", self.synthesize_answer)
        
        # Flux de données : Météo -> Marché -> Synthèse
        # Cette séquence permet à l'agent marché d'avoir potentiellement accès aux infos météo dans l'état
        workflow.set_entry_point("meteo_node")
        workflow.add_edge("meteo_node", "market_node")
        workflow.add_edge("market_node", "synthesizer_node")
        workflow.add_edge("synthesizer_node", END)
        
        return workflow.compile()