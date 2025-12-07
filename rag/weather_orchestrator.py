import logging
import os
import sys
import time
from typing import Dict, Any, List, Callable
# from rag.retriever import AgentRetriever # Non utilisé
from services.utils.indexer import UniversalIndexer
from services.meteo import DocumentScraper, WeatherForecastService
# Remarque : Les services importés (DocumentScraper, ClimatScraper, UniversalIndexer) 
# doivent être disponibles dans le chemin d'exécution.

# --- Configuration du Logging ---
LOG_FILE = "scrape_orchestrator.log"
# S'assurer que le fichier de log est écrit dans le répertoire courant ou un répertoire accessible
os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True) 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout) # Affichage des messages importants dans la console
    ]
)
logger = logging.getLogger("Orchestrator")

# ==============================================================================
# --- CONFIGURATION SIMULÉE ---
# Simule l'import d'un fichier de configuration
# ==============================================================================

SCRAPER_CONFIG = {
    'HEADLESS_MODE': True,  # Mode sans tête par défaut
    'INDEX_TIMEOUT_S': 10
}
CONFIG = SCRAPER_CONFIG 


# ==============================================================================
# --- ORCHESTRATEUR PRINCIPAL ---
# ==============================================================================

class ScrapeOrchestrator:
    """
    Orchestre l'exécution de tous les services de scraping et gère l'envoi 
    des données collectées (fichiers, métadonnées) au service d'indexation d'embeddings.
    """

    def __init__(self):
        # Initialisation des instances de service
        headless_mode = CONFIG.get('HEADLESS_MODE', True)
        
        # UNIQUEMENT les deux services demandés
        self.bulletin_scraper = DocumentScraper()
        self.forecast_scraper = WeatherForecastService()
        
        # Initialisation du service d'indexation
        self.indexer = UniversalIndexer()

        # Dictionnaire des services à exécuter
        self.services: Dict[str, Callable[[], Dict[str, Any]]] = {
            "agronomic_bulletins": self.bulletin_scraper.scrape_bulletins,
            "weather_forecast": self.forecast_scraper.scrape_forecast,
        }

    def _execute_service(self, name: str, service_function: Callable) -> Dict[str, Any]:
        """Exécute un service de scraping, gère le chronométrage et la journalisation des erreurs."""
        start_time = time.time()
        logger.info(f"======== Démarrage du service : {name.upper()} ========")
        
        try:
            # L'appel à la fonction de scraping réelle
            result = service_function()
            end_time = time.time()
            duration = end_time - start_time
            
            status = result.get('status', 'UNKNOWN')
            
            logger.info(f"Résultat pour {name.upper()}: Statut={status}, Message={result.get('message', 'N/A')}")
            logger.info(f"Durée d'exécution: {duration:.2f} secondes.")
            
            return result
        except Exception as e:
            # Journalisation d'une erreur critique (exception non gérée par le scraper lui-même)
            logger.error(f"Erreur critique lors de l'exécution du service {name.upper()}: {e}", exc_info=True)
            return {"status": "CRITICAL_ERROR", "message": str(e)}

    def _index_data(self, service_name: str, result: Dict[str, Any]):
        """
        Transfère les données collectées au service d'indexation des embeddings.
        """
        logger.info(f"Préparation des données pour l'indexation : {service_name}")

        status = result.get('status', 'ERROR')
        if status != "SUCCESS":
            logger.warning(f"Indexation ignorée: Le scraping de {service_name} n'a pas réussi.")
            return

        data_to_index: List[Dict[str, Any]] = []

        if service_name == "agronomic_bulletins":
            # Les bulletins sont attendus comme une liste de documents (dict)
            data_to_index = result.get('results', [])
        
        elif service_name == "weather_forecast":
            if result.get('results'):
                # Cas 1: Le scraper retourne une liste de résultats structurés
                data_to_index = result.get('results', [])
            elif result.get('data_path'):
                # Cas 2 (corrigé): Le scraper ne retourne qu'un seul élément structuré.
                # Nous le formatons pour qu'il corresponde à ce qu'attend l'indexer (qui attend une liste).
                # NOTE: Correction des clés 'path' en 'data_path' et ajout de 'city'
                data_to_index.append({
                    "data_path": result.get('data_path'), 
                    "city": result.get('city', 'Location Inconnue'),
                    "content_preview": result.get('preview_content_snippet', '')
                })

        # Appel réel au service d'embedding
        if data_to_index:
            try:
                index_result = self.indexer.index_data(data_to_index, service_name)
                # Assurez-vous que l'indexer retourne un dictionnaire comme dans la version corrigée
                logger.info(f"Résultat de l'indexation: {index_result.get('message', 'Succès sans message spécifique.')}")
            except Exception as e:
                # Correction: Log simple de l'erreur
                logger.error(f"Échec de l'indexation pour {service_name}: {e}")
        else:
            logger.warning(f"Aucune donnée valide à indexer pour {service_name}.")


    def run_all(self):
        """
        Méthode principale exécutant la boucle d'orchestration pour les services actifs.
        """
        orchestration_start_time = time.time()
        logger.info("=====================================================")
        logger.info("====== DÉMARRAGE DE L'ORCHESTRATEUR DE SCRAPING (2 Services) =======")
        logger.info("=====================================================")

        all_results = {}
        
        # Exécution séquentielle des services
        for name, service_function in self.services.items():
            result = self._execute_service(name, service_function)
            all_results[name] = result
            
            # Indexation immédiate après la collecte
            self._index_data(name, result)
        
        orchestration_end_time = time.time()
        total_duration = orchestration_end_time - orchestration_start_time
        
        logger.info("=====================================================")
        logger.info("======== ORCHESTRATION TERMINÉE =========")
        logger.info(f"Durée totale du cycle: {total_duration:.2f} secondes.")
        logger.info(f"Rapport de tous les résultats stocké dans '{LOG_FILE}'.")
        logger.info("=====================================================")
        
        return all_results

# --- Point d'entrée ---
if __name__ == "__main__":
    try:
        orchestrator = ScrapeOrchestrator()
        final_report = orchestrator.run_all()
        
        print("\n--- RÉSULTATS COMPLETS (PRÊTS À L'UTILISATION) ---")
        for name, result in final_report.items():
            status = result.get('status', 'ERROR')
            msg = result.get('message', 'N/A')
            print(f"[{status:<10}] {name}: {msg}")
            
    except KeyboardInterrupt:
        logger.info("Orchestration interrompue par l'utilisateur.")
    except Exception as e:
        logger.critical(f"Erreur inattendue au point d'entrée: {e}", exc_info=True)