from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import logging
from agents import CropManagementService, HealthManagementService, MeteoAgent, SoilManagementService, SubsidyManagementService
from .intention import IntentClassifier

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestrator")

# ==============================================================================
# 1. ÉTAT GLOBAL
# ==============================================================================
class OrchestratorState(TypedDict):
    user_id: str
    zone_id: str
    user_query: str
    
    intent: str
    context_data: Dict
    
    final_response: str
    execution_trace: List[str]

# ==============================================================================
# 2. CENTRAL DATA MANAGER
# ==============================================================================
class CentralDataManager:
    def retrieve_context(self, state: OrchestratorState) -> Dict:
        intent = state["intent"]
        zone = state["zone_id"]
        
        logger.info(f"Retrieving data for intent: {intent} in {zone}")
        
        context = {}
        user_profile = {
            "crop_name": "Maïs", 
            "sowing_date": "2024-06-15",
            "is_coop_member": False,
            "gender": "M"
        }
        
        if intent == "METEO":
            context["meteo_data"] = {
                "current": {"temp_max": 36.5, "temp_min": 24.0, "humidity": 45, "wind_speed_kmh": 18, "precip_mm": 0}
            }
            context["culture_config"] = {"crop_name": user_profile["crop_name"], "stage": "Croissance"}
            
        elif intent == "CROP":
            context["culture_config"] = {
                "crop_name": user_profile["crop_name"],
                "sowing_date": user_profile["sowing_date"]
            }
            
        elif intent == "HEALTH":
            context["culture_config"] = {"crop_name": user_profile["crop_name"]}
            
        elif intent == "SOIL":
            context["soil_config"] = {"texture": "sableux", "history": "jachère de 2 ans", "ph": 6.0}
            
        elif intent == "SUBSIDY":
            context["user_profile"] = {
                "crop": user_profile["crop_name"], 
                "zone": zone,
                "is_coop_member": user_profile["is_coop_member"],
                "gender": user_profile["gender"]
            }
            
        return context

# ==============================================================================
# 3. ORCHESTRATOR
# ==============================================================================
class AgriculturalOrchestrator:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.data_manager = CentralDataManager()
        self.agents = {
            "METEO": MeteoAgent(),
            "CROP": CropManagementService(),
            "SOIL": SoilManagementService(),
            "HEALTH": HealthManagementService(),
            "SUBSIDY": SubsidyManagementService()
        }

    def classify_node(self, state: OrchestratorState) -> OrchestratorState:
        intent = self.classifier.predict(state["user_query"])
        logger.info(f"Intent detected: {intent}")
        return {**state, "intent": intent, "execution_trace": [f"Intent: {intent}"]}

    def retrieve_node(self, state: OrchestratorState) -> OrchestratorState:
        if state["intent"] == "UNKNOWN":
            return state
        context = self.data_manager.retrieve_context(state)
        return {**state, "context_data": context, "execution_trace": state["execution_trace"] + ["Data retrieved"]}

    def dispatch_node(self, state: OrchestratorState) -> OrchestratorState:
        intent = state["intent"]
        query = state["user_query"]
        zone = state["zone_id"]
        data = state["context_data"]
        
        trace = state["execution_trace"]
        response = ""
        
        try:
            if intent == "UNKNOWN":
                response = "Je n'ai pas compris. Essayez de préciser si vous parlez de météo, culture, sol, santé ou subvention."
            
            elif intent in self.agents:
                service = self.agents[intent]
                logger.info(f"Calling Agent: {service.name}")
                
                input_payload = {"zone_id": zone, "user_query": query}
                input_payload.update(data) 
                
                agent_graph = service.get_graph()
                result = agent_graph.invoke(input_payload)
                
                response = result["final_response"]
                trace.append(f"Execution: {intent} agent terminé avec succès")
                
            else:
                response = "Erreur système : Agent non trouvé."
                trace.append("Error: Agent Missing")

        except Exception as e:
            logger.error(f"Error during dispatch: {e}", exc_info=True)
            response = "Une erreur technique est survenue lors du traitement."
            trace.append(f"Error: {str(e)}")

        return {**state, "final_response": response, "execution_trace": trace}

    def get_graph(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("classify", self.classify_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("dispatch", self.dispatch_node)
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "dispatch")
        workflow.add_edge("dispatch", END)
        return workflow.compile()

# ==============================================================================
# 4. DEMO
# ==============================================================================
if __name__ == "__main__":
    orchestrator = AgriculturalOrchestrator()
    main_graph = orchestrator.get_graph()
    
    print("\n>>> DÉMARRAGE DU SYSTÈME AGRI-COPILOTE (BURKINA) <<<\n")
    
    test_queries = [
        ("Meteo", "Est-ce qu'il va pleuvoir à Koudougou pour semer ?"),
        ("Health", "J'ai des chenilles dans le cornet de mon maïs, aidez-moi !"),
        ("Subsidy", "J'ai reçu un SMS pour payer 5000F pour le fonds ONU, c'est vrai ?"),
        ("Crop", "Quand est-ce que je dois mettre l'engrais NPK ?"),
        ("Soil", "Ma terre est sableuse et ne retient pas l'eau.")
    ]
    
    for category, query in test_queries:
        print(f"\nUSER ({category}): {query}")
        print("-" * 50)
        
        result = main_graph.invoke({
            "user_id": "U12345",
            "zone_id": "Koudougou", 
            "user_query": query,
            "intent": "",
            "context_data": {},
            "final_response": "",
            "execution_trace": []
        })
        
        print(f"TRACE: { ' -> '.join(result['execution_trace']) }")
        print("\nRÉPONSE FINALE :")
        print(result["final_response"])
        print("=" * 60)
