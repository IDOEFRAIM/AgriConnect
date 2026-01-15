import logging
import time
from typing import Dict, Any, Union
from orchestrator.state import GlobalAgriState
from orchestrator.message_flow import MessageResponseFlow
from orchestrator.report_flow import DailyReportFlow

# Configuration du logging centralisé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MainOrchestrator")

class MainOrchestrator:
    """
    Point d'entrée unique de la plateforme AgriConnect.
    Aiguille les requêtes vers les graphes de décision LangGraph appropriés.
    """
    def __init__(self):
        # Compilation des graphes au démarrage pour optimiser la latence
        try:
            self.message_flow = MessageResponseFlow().build_graph()
            self.report_flow = DailyReportFlow().build_graph()
            logger.info("✅ Graphes de flux compilés avec succès.")
        except Exception as e:
            logger.critical(f"❌ Échec de compilation des flux : {e}")
            raise

    def run(self, initial_state: GlobalAgriState) -> Dict[str, Any]:
        """
        Exécute le flux demandé (MESSAGE ou REPORT).
        """
        start_time = time.time()
        flow_type = initial_state.get("flow_type", "MESSAGE").upper()
        
        logger.info(f"🚀 Orchestrateur : Démarrage du flux [{flow_type}]")

        try:
            if flow_type == "MESSAGE":
                # Flux interactif (Question -> Analyse -> Réponse)
                result = self.message_flow.invoke(initial_state)
                
            elif flow_type == "REPORT":
                # Flux proactif (Data Gathering -> Bulletin -> Envoi)
                result = self.report_flow.invoke(initial_state)
                
            else:
                logger.error(f"Type de flux inconnu : {flow_type}")
                return {"error": f"Le type de flux '{flow_type}' n'est pas supporté."}

            # Calcul des performances
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"⏱️ Fin d'exécution : {execution_time:.2f} ms")
            
            # Injection des métadonnées de performance
            if isinstance(result, dict):
                result["metadata"] = {
                    "execution_time_ms": round(execution_time, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "flow_executed": flow_type
                }
            
            return result

        except Exception as e:
            logger.error(f"💥 Erreur critique durant l'orchestration : {e}", exc_info=True)
            return {"error": "Une erreur interne a interrompu le traitement."}