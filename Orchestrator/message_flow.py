import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from orchestrator.state import GlobalAgriState
from orchestrator.intention import IntentClassifier
from services.utils.cache import StorageManager
from agents.climate_vigilance import ClimateVigilance
from agents.production_expert import ProductionExpert
from agents.soil_doctor import SoilDoctor
from agents.plant_doctor import PlantHealthDoctor
from agents.agri_business_coach import AgriBusinessCoach

logger = logging.getLogger("MessageFlow")

class MessageResponseFlow:
    def __init__(self, llm_client=None):
        self.classifier = IntentClassifier()
        self.storage = StorageManager() # Injected Storage
        
        # Initialize Orchestrator LLM for checks/synthesis
        try:
            self.llm = llm_client if llm_client else ChatOllama(model="llama3:8b", keep_alive=-1)
        except Exception as e:
            logger.warning(f"Orchestrator LLM init failed: {e}")
            self.llm = None

        # Initialize Agents
        self.meteo_agent = ClimateVigilance(llm_client=llm_client)

        self.crop_agent = ProductionExpert(llm_client=llm_client)
        self.soil_agent = SoilDoctor(llm_client=llm_client)
        self.health_agent = PlantHealthDoctor(llm_client=llm_client)
        self.subsidy_agent = AgriBusinessCoach(llm_client=llm_client)

    def classify_intent(self, state: GlobalAgriState):
        query = state.get("requete_utilisateur", "")
        intent = self.classifier.predict(query)
        logger.info(f"Intent detected: {intent}")
        return {"execution_path": [f"Intent: {intent}"]}

    def route_request(self, state: GlobalAgriState):
        # Logic to determine the next node based on the last execution path entry
        last_step = state["execution_path"][-1]
        intent = last_step.split(": ")[1]
        
        if intent == "METEO":
            return "meteo_agent"
        elif intent == "CROP":
            return "crop_agent"
        elif intent == "SOIL":
            return "soil_agent"
        elif intent == "HEALTH":
            return "health_agent"
        elif intent == "SUBSIDY":
            return "subsidy_agent"
        else:
            return "unknown_handler"

    def run_meteo(self, state: GlobalAgriState):
        logger.info("Executing Meteo Node")
        user_query = state.get("requete_utilisateur", "")
        zone_id = state.get("zone_id", "Centre")

        # 1. Fetch meteo data from Cache/RAG
        # We try to get the raw data ingested by the scheduler
        weather_data = {"t_min": 25, "t_max": 35, "rh": 40, "precip": 0} # Fallback
        
        try:
            # Look for recent raw data in cache (category=METEO)
            raw_meteo_list = self.storage.get_raw_data(zone_id=zone_id, category="METEO_VECTOR", limit=1)
            if raw_meteo_list:
                latest_meteo = raw_meteo_list[0]
                # Assuming the ingested data structure matches what we need or gives us clues.
                # In a real scenario, we might need a parser here.
                # For now, let's try to extract from 'metadata' if present or use the content.
                logger.info(f"Found cached meteo data from {latest_meteo.get('effective_date')}")
                
                # Mock extraction logic similar to RAG Textualizer
                # Ideally, we should parse the JSON content field
                if "t_min" in latest_meteo: weather_data["t_min"] = latest_meteo["t_min"]
                if "t_max" in latest_meteo: weather_data["t_max"] = latest_meteo["t_max"]
        except Exception as e:
            logger.warning(f"Failed to fetch meteo from cache: {e}")

        # Adapt Global State to Agent State
        agent_state = {
            "user_query": user_query,
            "weather_data": weather_data,
            "culture_info": {"crop_name": "Maïs", "location": zone_id}, 
            "raw_diagnosis": None,
            "flood_risk": None,
            "final_response": "",
            "error_log": []
        }
        
        # Run Agent Logic
        res1 = self.meteo_agent.validate_and_calculate(agent_state)
        # Attempt fallback or notify user if critical data is still missing
        if res1.get("error_log"):
             logger.warning(f"Meteo Agent Errors: {res1['error_log']}")
             
        agent_state.update(res1)
        res2 = self.meteo_agent.generate_expert_response(agent_state)
        
        return {"final_response": res2["final_response"], "execution_path": ["Meteo Agent Executed"]}

    def run_crop(self, state: GlobalAgriState):
        # Adapt Global State to Agent State
        agent_state = {
            "user_query": state.get("requete_utilisateur", ""),
            "crop_name": "Maïs", # Should be extracted
            "location_zone": state.get("zone_id", "Centre"),
            "surface_ha": 1.0,
            "technical_data": None,
            "final_response": "",
            "errors": []
        }
        
        # Run Agent Logic
        res1 = self.crop_agent.process_technical_node(agent_state)
        agent_state.update(res1)
        res2 = self.crop_agent.expert_response_node(agent_state)
        
        return {"final_response": res2["final_response"], "execution_path": ["Crop Agent Executed"]}

    def run_soil(self, state: GlobalAgriState):
        # Adapt Global State to Agent State
        agent_state = {
            "user_query": state.get("requete_utilisateur", ""),
            "soil_config": {"texture": "sableux", "ph": 6.5, "budget": "moyen"}, # Defaults
            "technical_advice_raw": None,
            "final_response": "",
            "status": "INIT"
        }
        
        # Run Agent Logic
        res1 = self.soil_agent.analyze_node(agent_state)
        agent_state.update(res1)
        res2 = self.soil_agent.format_node(agent_state)
        
        return {"final_response": res2["final_response"], "execution_path": ["Soil Agent Executed"]}

    def run_health(self, state: GlobalAgriState):
        # Adapt Global State to Agent State
        agent_state = {
            "user_query": state.get("requete_utilisateur", ""),
            "culture_config": {"crop_name": "Maïs"}, # Default
            "diagnosis_raw": None,
            "technical_advice_text": "",
            "final_response": "",
            "status": "INIT"
        }
        
        # Run Agent Logic
        res1 = self.health_agent.analyze_node(agent_state)
        agent_state.update(res1)
        res2 = self.health_agent.format_node(agent_state)
        
        return {"final_response": res2["final_response"], "execution_path": ["Health Agent Executed"]}

    def run_subsidy(self, state: GlobalAgriState):
        # Adapt Global State to Agent State
        agent_state = {
            "zone_id": state.get("zone_id", "Centre"),
            "user_query": state.get("requete_utilisateur", ""),
            "user_profile": {"crop": "Maïs"},
            "technical_advice_raw": None,
            "final_response": "",
            "status": "INIT"
        }
        
        # Run Agent Logic
        res1 = self.subsidy_agent.analyze_node(agent_state)
        agent_state.update(res1)
        res2 = self.subsidy_agent.format_node(agent_state)
        
        return {"final_response": res2["final_response"], "execution_path": ["Subsidy Agent Executed"]}

    def synthesize_answer(self, state: GlobalAgriState):
        """Global Synthesis Step: Ensures the final response is decision-driven and focused."""
        query = state.get("requete_utilisateur", "")
        agent_resp = state.get("final_response", "")

        # If LLM unavailable or response is empty, pass through
        if not self.llm or not agent_resp:
            return {"final_response": agent_resp}
            
        system_prompt = (
            "Tu es le Superviseur Intelligent du système AgConnect. "
            "Tu reçois une réponse technique d'un agent spécialisé (Météo, Sol, Culture...)."
            "Ton but : T'assurer que l'utilisateur reçoit une réponse UTILE, DIRECTE et CLAIRE.\n\n"
            "RÈGLES :\n"
            "1. SI la question est simple (Oui/Non), la réponse DOIT commencer par la réponse (Pas de 'Bonjour...').\n"
            "2. SI la réponse de l'agent est trop longue/bavarde, résume l'essentiel (Problème -> Solution).\n"
            "3. GARDE les chiffres clés (Températures, mm de pluie, dosages).\n"
            "4. TON : Proche du terrain, pragmatique."
        )
        
        try:
            human_msg = f"CONTEXTE:\nQuestion: '{query}'\nRéponse Brute Agent: '{agent_resp}'\n\nTACHE: Reformule pour l'agriculteur (Si c'est déjà bon, renvoie tel quel) :"
            res = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_msg)])
            return {"final_response": res.content}
        except Exception as e:
            logger.warning(f"Synthesis failed, using raw response: {e}")
            return {"final_response": agent_resp}

    def handle_unknown(self, state: GlobalAgriState):
        return {"final_response": "Désolé, je ne comprends pas votre demande. Je peux vous aider sur la météo, les cultures, le sol, la santé des plantes ou les subventions.", "execution_path": ["Unknown Intent"]}

    def build_graph(self):
        graph = StateGraph(GlobalAgriState)
        
        graph.add_node("classifier", self.classify_intent)
        graph.add_node("meteo_agent", self.run_meteo)
        graph.add_node("crop_agent", self.run_crop)
        graph.add_node("soil_agent", self.run_soil)
        graph.add_node("health_agent", self.run_health)
        graph.add_node("subsidy_agent", self.run_subsidy)
        graph.add_node("unknown_handler", self.handle_unknown)
        graph.add_node("synthesizer", self.synthesize_answer) # Added Synthesis Node
        
        graph.set_entry_point("classifier")
        
        graph.add_conditional_edges(
            "classifier",
            self.route_request,
            {
                "meteo_agent": "meteo_agent",
                "crop_agent": "crop_agent",
                "soil_agent": "soil_agent",
                "health_agent": "health_agent",
                "subsidy_agent": "subsidy_agent",
                "unknown_handler": "unknown_handler"
            }
        )
        
        # All agents go to Synthesizer instead of END
        graph.add_edge("meteo_agent", "synthesizer")
        graph.add_edge("crop_agent", "synthesizer")
        graph.add_edge("soil_agent", "synthesizer")
        graph.add_edge("health_agent", "synthesizer")
        graph.add_edge("subsidy_agent", "synthesizer")
        
        graph.add_edge("unknown_handler", END)
        graph.add_edge("synthesizer", END)
        
        return graph.compile()
