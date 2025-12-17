# orchestrator/orchestrator.py
import os
import sys
import logging
from typing import TypedDict, List, Dict, Any, Optional

# --- Configuration du chemin pour les imports locaux ---
# Permet d'importer les modules depuis le dossier parent
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# --- Imports LangChain/LangGraph ---
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# --- Imports Modules Locaux (Architecture Modulaire) ---
from orchestrator.intention import IntentClassifier
from orchestrator.central_data_manager import CentralDataManager
# Assurez-vous que ces fichiers existent ou commentez les imports si test partiel
try:
    from agents.Meteo import MeteoAgent  # Suppose que MeteoAgent a une méthode get_graph()
    from agents.Crop import BurkinaCropAgent
    from agents.Soil import SoilManagementService
    from agents.Health import HealthManagementService
    from agents.subsidy import SubsidyManagementService
except ImportError as e:
    logging.warning(f"⚠️ Certains agents n'ont pas pu être importés : {e}")

# --- Configuration Logging ---
logger = logging.getLogger("Orchestrator")
logger.setLevel(logging.INFO)

# ======================================================================
# 1. DÉFINITION DE L'ÉTAT GLOBAL (Le "Bus" de données)
# ======================================================================
class OrchestratorState(TypedDict):
    """
    État partagé qui circule entre tous les nœuds du graphe.
    Contient l'intention, les données contextuelles et la réponse finale.
    """
    user_id: str
    zone_id: str
    user_query: str
    intent: str
    
    # Context Data (Injecté par CentralDataManager)
    meteo_data: Optional[Dict]
    culture_config: Optional[Dict]
    soil_config: Optional[Dict]
    user_profile: Optional[Dict]

    # Sorties et Traçabilité
    final_response: str
    execution_trace: List[str] # Pour le debugging et l'explicabilité


# ======================================================================
# 2. ORCHESTRATEUR PRINCIPAL ("The Boss")
# ======================================================================
class AgriculturalOrchestrator:
    
    OLLAMA_MODEL = "mistral"
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        
        # 1. Initialisation du Client LLM Central
        # On partage ce client avec les agents pour économiser les ressources
        self.ollama_client = self._init_llm_client()
        
        # 2. Initialisation des Services Core
        self.classifier = IntentClassifier(model_name=self.OLLAMA_MODEL)
        self.data_manager = CentralDataManager()

        # 3. Chargement des Agents Spécialisés
        # Chaque agent est une "boîte noire" autonome
        self.agents = {}
        self._load_agents()

    def _init_llm_client(self):
        """Tente de connecter Ollama avec une gestion d'erreur robuste."""
        try:
            client = ChatOllama(
                model=self.OLLAMA_MODEL, 
                base_url=self.ollama_url,
                temperature=0.1 # Basse température pour la précision technique
            )
            # Ping test
            client.invoke("Hi")
            logger.info(f"✅ Master Orchestrator connecté à Ollama ({self.OLLAMA_MODEL})")
            return client
        except Exception as e:
            logger.error(f"❌ CRITIQUE : Ollama indisponible. Le mode dégradé sera activé. Erreur: {e}")
            return None

    def _load_agents(self):
        """Charge dynamiquement les agents disponibles."""
        # On passe le client LLM aux agents pour éviter qu'ils ne réinstancient chacun une connexion
        try:
            self.agents["METEO"] = MeteoAgent()
            self.agents["CROP"] = BurkinaCropAgent()
            self.agents["SOIL"] = SoilManagementService()
            self.agents["HEALTH"] = HealthManagementService()
            self.agents["SUBSIDY"] = SubsidyManagementService() # Subsidy gère son client différemment dans ton code précédent
            logger.info(f"✅ {len(self.agents)} Agents chargés avec succès.")
        except NameError:
            logger.warning("⚠️ Certains agents ne sont pas définis (NameError). Vérifiez les imports.")

    # ============================================================
    # NODES (Les étapes du processus)
    # ============================================================

    def classify_node(self, state: OrchestratorState) -> OrchestratorState:
        """Étape 1 : Comprendre ce que veut l'utilisateur."""
        query = state.get("user_query", "")
        
        # Utilisation du Classifier Robuste (LLM + Fallback Regex)
        intent = self.classifier.predict(query)
        
        trace = state.get("execution_trace", []) + [f"Intent Detected: {intent}"]
        logger.info(f"🧠 Classification: {intent}")
        
        return {**state, "intent": intent, "execution_trace": trace}

    def retrieve_node(self, state: OrchestratorState) -> OrchestratorState:
        """Étape 2 : Récupérer les munitions (données) pour l'agent."""
        if state["intent"] == "UNKNOWN":
            return state

        # Le DataManager sait quelles données aller chercher selon l'intention
        context_data = self.data_manager.retrieve_context(state)
        
        # On fusionne les nouvelles données dans l'état
        new_state = {**state, **context_data}
        
        keys_found = [k for k, v in context_data.items() if v is not None]
        trace = state["execution_trace"] + [f"Context Loaded: {keys_found}"]
        
        return {**new_state, "execution_trace": trace}

    def dispatch_node(self, state: OrchestratorState) -> OrchestratorState:
        """Étape 3 : Déléguer à l'Expert (Agent)."""
        intent = state["intent"]
        trace = state["execution_trace"]
        
        agent_service = self.agents.get(intent)
        
        if not agent_service:
            logger.error(f"Agent {intent} not found in registry.")
            return {
                **state, 
                "final_response": "Désolé, le service demandé est momentanément indisponible.",
                "execution_trace": trace + ["Error: Agent missing"]
            }

        logger.info(f"🚀 Dispatching to Agent: {intent}")
        
        try:
            # Invocation du graphe de l'agent
            # L'agent reçoit tout l'état, fait son travail, et retourne son état local
            agent_result = agent_service.get_graph().invoke(state)
            
            # Extraction de la réponse finale de l'agent
            response = agent_result.get("final_response", "L'agent n'a pas retourné de réponse.")
            status = agent_result.get("status", "UNKNOWN_STATUS")
            
            return {
                **state,
                "final_response": response,
                "execution_trace": trace + [f"Agent Execution: SUCCESS ({status})"]
            }

        except Exception as e:
            logger.error(f"💥 Error inside Agent {intent}: {e}", exc_info=True)
            return {
                **state,
                "final_response": f"Une erreur technique est survenue lors de l'analyse ({intent}). Veuillez réessayer.",
                "execution_trace": trace + [f"Agent Crash: {str(e)}"]
            }

    def fallback_node(self, state: OrchestratorState) -> OrchestratorState:
        """Étape Secours : Si l'intention est inconnue."""
        query = state["user_query"]
        trace = state["execution_trace"] + ["Fallback: General LLM"]
        
        if not self.ollama_client:
            return {**state, "final_response": "Je suis hors ligne. Veuillez vérifier ma connexion.", "execution_trace": trace}

        # Prompt optimisé pour être utile même en cas d'incompréhension, avec gestion des DIAGRAMMES
        system_prompt = (
            "Tu es un assistant agricole intelligent. L'utilisateur a posé une question qui ne correspond "
            "pas à nos catégories standards (Météo, Sol, Culture, Santé, Subventions). "
            "1. Réponds poliment et essaie d'aider si le sujet reste agricole (ex: machinerie, élevage). "
            "2. Si la question est hors-sujet, redirige-le vers l'agriculture. "
            "3. Si l'explication bénéficie d'un schéma visuel (ex: anatomie d'une vache, pièce de tracteur), "
            "utilise le tag. Sois économe avec les images, utilise-les seulement si instructif."
        )

        try:
            response = self.ollama_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ])
            return {**state, "final_response": response.content, "execution_trace": trace}
        except Exception as e:
            return {**state, "final_response": "Je n'ai pas compris votre demande.", "execution_trace": trace + ["Fallback Error"]}

    # ============================================================
    # 3. CONSTRUCTION DU GRAPHE (ROUTAGE)
    # ============================================================
    
    def get_graph(self):
        workflow = StateGraph(OrchestratorState)

        # Ajout des nœuds
        workflow.add_node("classify", self.classify_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("dispatch", self.dispatch_node)
        workflow.add_node("fallback", self.fallback_node)

        # Point d'entrée
        workflow.set_entry_point("classify")

        # Logique de branchement conditionnel
        def route_intent(state):
            intent = state.get("intent", "UNKNOWN")
            if intent in ["METEO", "CROP", "SOIL", "HEALTH", "SUBSIDY"]:
                return "retrieve"
            return "fallback"

        workflow.add_conditional_edges(
            "classify",
            route_intent,
            {
                "retrieve": "retrieve",
                "fallback": "fallback"
            }
        )

        # Flux linéaire pour les cas connus
        workflow.add_edge("retrieve", "dispatch")
        workflow.add_edge("dispatch", END)
        workflow.add_edge("fallback", END)

        return workflow.compile()

# ======================================================================
# 4. EXÉCUTION DE TEST (SIMULATION)
# ======================================================================
if __name__ == "__main__":
    # Setup pour le visuel console
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
    
    print("\n🚜 INITIALISATION DE L'ORCHESTRATEUR AGRICOLE...")
    orchestrator = AgriculturalOrchestrator()
    app = orchestrator.get_graph()

    def run_simulation(query: str, zone: str = "Koudougou"):
        print(f"\n{'='*60}")
        print(f"👤 USER ({zone}): {query}")
        print(f"{'='*60}")
        
        initial_state = {
            "user_id": "sim_user_01",
            "zone_id": zone,
            "user_query": query,
            # Le reste est initialisé à vide ou None
            "intent": "", "final_response": "", "execution_trace": [],
            "meteo_data": None, "culture_config": None, "soil_config": None, "user_profile": None
        }
        
        result = app.invoke(initial_state)
        
        print(f"\n🤖 BOT RESPONSE:\n{result['final_response']}")
        print(f"\n🔍 TRACE: {' -> '.join(result['execution_trace'])}")

    # TEST 1 : Cas Complexe (Santé + Météo implicite via DataManager)
    run_simulation("Les feuilles de mon maïs jaunissent et il y a des taches. Que faire ?")

    # TEST 2 : Cas Subvention (Avec visuel attendu dans l'agent)
    run_simulation("C'est quoi la procédure pour avoir l'engrais subventionné ?")
    
    # TEST 3 : Cas Fallback (Machinerie - doit déclencher le LLM généraliste + Image potentielle)
    run_simulation("Comment fonctionne un moteur diesel de tracteur ?")