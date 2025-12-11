# orchestrator/orchestrator.py
import os
import sys

# Ajoute le dossier parent (AgConnect) au sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# NOUVELLE IMPORTATION pour le LLM (remplace l'import de base)
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import logging

# Assurez-vous que ces modules sont accessibles
from orchestrator.intention import IntentClassifier
from orchestrator.central_data_manager import CentralDataManager, OrchestratorState as DataManagerState # Renommé pour éviter le conflit

# Agents (NOTE: Assurez-vous que ces classes sont définies et importables)
from agents.Crop import CropManagementService
from agents.Health import HealthManagementService
from agents.Meteo import MeteoAgent
from agents.Soil import SoilManagementService
from agents.subsidy import SubsidyManagementService


logger = logging.getLogger("Orchestrator")


# ======================================================================
# 1. ÉTAT DE L'ORCHESTRATEUR (RÉALIGNEMENT)
# ======================================================================
class OrchestratorState(TypedDict):
    """État global du graphe."""
    user_id: str
    zone_id: str
    user_query: str
    intent: str
    context_data: Dict # Données récupérées par le Data Manager
    final_response: str
    execution_trace: List[str]

    # Pour les agents: Les champs spécifiques aux agents ne sont pas nécessaires ici
    # car ils sont passés via context_data, mais on peut les ajouter pour la lisibilité
    meteo_data: Optional[Dict]
    culture_config: Optional[Dict]
    soil_config: Optional[Dict]
    user_profile: Optional[Dict]


# ======================================================================
# 2. ORCHESTRATEUR PRINCIPAL
# ======================================================================
class AgriculturalOrchestrator:
    def __init__(self):
        # Initialisation du client LLM pour les agents
        self.ollama_client = ChatOllama(model="mistral", temperature=0.1) 
        
        # Initialisation des composants centraux
        self.classifier = IntentClassifier()
        self.data_manager = CentralDataManager()

        # Initialisation des agents (on passe le LLM là où il est nécessaire)
        self.agents = {
            "METEO": MeteoAgent(llm_client=self.ollama_client),
            "CROP": CropManagementService(llm_client=self.ollama_client),
            "SOIL": SoilManagementService(llm_client=self.ollama_client),
            "HEALTH": HealthManagementService(llm_client=self.ollama_client),
            "SUBSIDY": SubsidyManagementService() # Agent Subsidy n'a pas besoin du LLM en init
        }
    
    # Nœud LLM de Fallback (pour les questions UNKNOWN)
    def llm_fallback_node(self, state: OrchestratorState) -> OrchestratorState:
        """Utilise le LLM pour répondre directement si l'intention est inconnue."""
        logger.warning(f"Intention inconnue. Utilisation du Fallback LLM.")
        trace = state["execution_trace"] + ["Execution: LLM Fallback (UNKNOWN)"]
        
        try:
             response = self.ollama_client.invoke(
                 f"Répondez de manière amicale à la question suivante en expliquant que vous êtes un expert agricole mais que cette question ne concerne pas directement l'agriculture : {state['user_query']}"
             )
             final_response = response.content
        except Exception as e:
             final_response = f"Désolé, une erreur de connexion est survenue. (Erreur LLM: {e})"
        
        return {**state, "final_response": final_response, "execution_trace": trace}


    def classify_node(self, state: OrchestratorState) -> OrchestratorState:
        """Classifie la requête utilisateur."""
        intent = self.classifier.predict(state["user_query"])
        logger.info(f"Intent detected: {intent}")
        return {**state, "intent": intent, "execution_trace": state["execution_trace"] + [f"Intent: {intent}"]}

    def retrieve_node(self, state: OrchestratorState) -> OrchestratorState:
        """Récupère les données contextuelles nécessaires pour l'Agent ciblé."""
        if state["intent"] == "UNKNOWN":
             # Le Data Manager ne doit pas être appelé pour UNKNOWN
            return state
        
        # Le Data Manager retourne un Dict qui est MERGÉ dans l'état
        retrieved_data = self.data_manager.retrieve_context(state)
        
        # Fusionner les données récupérées dans l'état (nécessaire pour les agents)
        merged_state = {**state, **retrieved_data}
        
        return {**merged_state, "execution_trace": state["execution_trace"] + ["Data retrieved"]}

    def dispatch_node(self, state: OrchestratorState) -> OrchestratorState:
        """Distribue la requête et l'état à l'Agent spécialisé."""
        intent = state["intent"]
        trace = state["execution_trace"]
        response = ""
        
        try:
            # L'agent est déjà dans self.agents
            service = self.agents[intent]
            logger.info(f"Calling Agent: {service.name}")

            # L'état de l'orchestrateur contient déjà toutes les clés spécifiques
            # (zone_id, user_query, meteo_data, culture_config, etc.) 
            # grâce à retrieve_node (via fusion des données)
            
            agent_graph = service.get_graph()
            
            # L'état passé à l'agent est le state de l'orchestrateur complet, 
            # mais seuls les champs pertinents pour l'agent sont utilisés.
            result = agent_graph.invoke(state)

            response = result["final_response"]
            trace.append(f"Execution: {intent} agent terminé (Status: {result.get('status', 'N/A')})")

        except Exception as e:
            logger.error(f"Error during dispatch of {intent} agent: {e}", exc_info=True)
            response = f"Une erreur technique est survenue lors de l'appel de l'agent {intent}."
            trace.append(f"Error: {str(e)}")

        return {**state, "final_response": response, "execution_trace": trace}

    # ======================================================================
    # 3. WORKFLOW LANGGRAPH (ROUTAGE CONDITIONNEL)
    # ======================================================================
    def get_graph(self):
        workflow = StateGraph(OrchestratorState)
        
        # Nœuds
        workflow.add_node("classify", self.classify_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("dispatch", self.dispatch_node)
        workflow.add_node("llm_fallback", self.llm_fallback_node)

        # 1. Point d'entrée
        workflow.set_entry_point("classify")
        
        # 2. Routage après classification
        def route_by_intent(state: OrchestratorState) -> str:
            """Fonction routeur qui décide du prochain nœud."""
            intent = state.get("intent")
            if intent == "UNKNOWN":
                return "fallback"
            else:
                return "retrieve" # Aller chercher les données

        workflow.add_conditional_edges(
            "classify",
            route_by_intent,
            {
                "fallback": "llm_fallback",
                "retrieve": "retrieve",
            }
        )
        
        # 3. Flux principal
        workflow.add_edge("retrieve", "dispatch")
        
        # 4. Points de sortie
        workflow.add_edge("dispatch", END)
        workflow.add_edge("llm_fallback", END)

        # Voici une représentation visuelle du Graphe :
        

        return workflow.compile()


# ======================================================================
# 4. BLOC D'EXÉCUTION DIRECTE (TESTS)
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    orchestrator = AgriculturalOrchestrator()
    graph = orchestrator.get_graph()

    def run_test(query: str, expected_intent: str):
        print(f"\n==============================================")
        print(f"TEST RUN: '{query}' (Attendu: {expected_intent})")
        print("==============================================")
        test_state: OrchestratorState = {
            "user_id": "test_user",
            "zone_id": "Mopti",
            "user_query": query,
            "intent": "",
            "context_data": {},
            "final_response": "",
            "execution_trace": [],
            "meteo_data": None, "culture_config": None, "soil_config": None, "user_profile": None
        }

        result = graph.invoke(test_state)
        print("\n=== RÉSULTAT FINAL ===")
        print(result["final_response"])
        print("\n=== TRACE D'EXÉCUTION ===")
        for step in result["execution_trace"]:
            print(f" - {step}")
        print(f"--- Fin du test ---")
        return result

    # 1. Test : Intention connue (ex: SOIL)
    run_test("Mon sol est très sableux et fatigué. Que puis-je faire ?", "SOIL")

    # 2. Test : Intention UNKNOWN (doit passer par le fallback LLM)
    run_test("Quel temps fera-t-il à Paris demain ?", "UNKNOWN")
    
    # 3. Test : Intention nécessitant des données Météo (ex: METEO)
    run_test("Puis-je traiter mes cultures cet après-midi ? Il fait chaud.", "METEO")