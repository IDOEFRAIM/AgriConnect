import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
# Importation du StorageManager corrigé
from services.utils import StorageManager 
# Utilisation de l'ancienne classe Logger pour ne pas recréer la config de logging
# logger = logging.getLogger("scraper.orchestrator") 
# logging.basicConfig(level=logging.INFO)

# --- Configuration du Logging (Simplifiée) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ScraperOrchestrator")


# --- Stubs pour les composants RAG manquants (pour que le script tourne) ---
class VectorStoreHandler:
    def search(self, query_vector: list, k: int, source_filter: str, vector_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return [{"content": f"Résultat vectoriel pour {source_filter}", "score": 0.9}]

    def index_data(self, category: str, data: Dict[str, Any]):
        logging.getLogger("VectorStoreHandler").debug(f"Indexation simulée pour {category}.")
        pass

class Reranker:
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logging.getLogger("Reranker").debug("Reranking simulé.")
        return results

class EmbeddingService:
    def embed_query(self, query: str) -> list:
        logging.getLogger("EmbeddingService").debug("Embedding de requête simulé.")
        return [0.1] * 128


# --- Interface d'Agent (Pour le découplage) ---
class ScraperAgent:
    def __init__(self, category: str):
        self.category = category 

    def run(self, zone_id: str) -> List[Dict[str, Any]]:
        """Simule l'exécution de l'agent pour une zone spécifique."""
        logger.info(f"⚙️ Exécution de l'agent [{self.category}] pour la zone {zone_id}...")
        
        # --- Simule la collecte de données variées (robustesse) ---
        if self.category == 'METEO':
            # La météo change à chaque heure, donc le hash changera
            now = datetime.now()
            return [{
                "time_prevision": (now + timedelta(hours=i)).isoformat(),
                "temp_c": 20.0 + (i * 0.1) + (now.minute / 100), # Change légèrement pour tester le hash
                "description": "Bulletin Météorologique pour la journée",
                "source_url": f"http://meteo.com/bulletin/{zone_id}"
            } for i in range(2)]
        elif self.category == 'SUBVENTION':
            # La subvention reste stable (même hash) pour la déduplication
            return [{
                "grant_id": "S999",
                "title": "Aide Agricole Urgente",
                "deadline": (datetime.now() + timedelta(days=60)).isoformat(),
                "amount_eur": 50000.0,
                "eligible_zones": [zone_id],
                "source_url": "http://subventions.gouv/agri"
            }]
        elif self.category == 'ALERTE_INONDATION':
            # Alerte stable pour la démo
            return [{
                "level": "Rouge",
                "risk_area": zone_id,
                "timestamp": datetime.now().isoformat(),
                "details": "Niveau de crue critique sur le fleuve.",
                "source_url": "http://alertes.gouv/inondation"
            }]
        else:
            return []

class ScraperOrchestrator:
    """Gère la chaîne d'exécution des agents de scraping et collecte les résultats bruts."""

    def __init__(self, agents: Dict[str, ScraperAgent], zones: List[str]):
        self.agents = agents 
        self.zones = zones 
        logger.info(f"🌐 Orchestrateur initialisé. {len(self.agents)} agents pour {len(self.zones)} zones.")
    
    def run_agent_and_collect(self, category: str, agent: ScraperAgent) -> List[Dict[str, Any]]:
        """Exécute un agent pour toutes les zones et retourne la liste brute des résultats."""
        collected_data = []
        logger.info(f"\n--- Démarrage de l'agent : {category} ---")
        
        for zone_id in self.zones:
            try:
                raw_results = agent.run(zone_id)
                for result in raw_results:
                    # Enrichir la donnée brute avec les métadonnées de l'exécution
                    collected_data.append({
                        "category": category,
                        "zone_id": zone_id,
                        "data": result,
                        "acquisition_time": time.time(), 
                    })
                
                logger.info(f"✅ Collecte réussie pour {zone_id} : {len(raw_results)} enregistrements.")

            except Exception as e:
                logger.error(f"❌ Erreur critique de l'agent {category} pour {zone_id}: {e}")

        return collected_data

    def run_pipeline(self) -> List[Dict[str, Any]]:
        """Lance l'exécution de tous les agents et retourne tous les résultats collectés."""
        all_collected_data = []
        for category, agent in self.agents.items():
            results = self.run_agent_and_collect(category, agent)
            all_collected_data.extend(results)
        return all_collected_data


# --- DÉMONSTRATION DE LA PIPELINE DÉCOUPLÉE ---
if __name__ == '__main__':
    # Initialisation des composants
    DB_PATH = "data/orchestrator_final.db"
    
    # 1. PRÉPARATION DES OUTILS (StorageManager pour la persistance)
    try:
        # Assurez-vous que le StorageManager est bien instancié après la correction
        storage = StorageManager(db_path=DB_PATH)
        store = VectorStoreHandler()
        embedder = EmbeddingService()
        reranker = Reranker()
    
    except Exception as e:
        logger.error(f"Erreur fatale d'initialisation des services: {e}. Vérifiez storage_manager.py.")
        exit()

    # 2. DÉFINITION DE LA TÂCHE
    agents_map = {
        "METEO": ScraperAgent("METEO"),
        "SUBVENTION": ScraperAgent("SUBVENTION"),
        "ALERTE_INONDATION": ScraperAgent("ALERTE_INONDATION"),
    }
    zones_list = ["Paris", "Lyon", "Marseille"]

    # 3. L'ORCHESTRATEUR (Exécution pure)
    orchestrator = ScraperOrchestrator(agents_map, zones_list)
    
    print("\n[Étape 1] 🚀 Lancement de l'Orchestrateur pour collecter les données...")
    final_collected_data = orchestrator.run_pipeline() 
    
    print(f"\n[Étape 1 Terminé] Total des enregistrements collectés par l'Orchestrateur : {len(final_collected_data)}")

    # 4. LE FLUX DE TRAITEMENT AVAL (Persistance, Caching, Vectorisation)
    print("\n[Étape 2] 💾 Démarrage du Traitement Aval (Persistence et Caching/Déduplication)...")
    
    processed_count = 0
    for item in final_collected_data:
        # Stockage de la donnée brute dans la table dynamique
        is_new = storage.save_raw_data(
            zone_id=item["zone_id"],
            category=item["category"],
            data=item["data"],
            effective_date=item["acquisition_time"],
            source_url=item["data"].get("source_url")
        )
        
        # Le même 'item' serait envoyé à un Vector Store SEULEMENT s'il est nouveau ou modifié
        if is_new:
            store.index_data(item["category"], item["data"]) 
            processed_count += 1

    print(f"\n[Étape 2 Terminé] Total des NOUVEAUX enregistrements persistés (après déduplication) : {processed_count}")

    # 5. Testons la robustesse/déduplication en relançant l'Orchestrateur
    print("\n[Étape 3] 🔄 Relance de la pipeline (pour tester la déduplication)...")
    second_run_data = orchestrator.run_pipeline()
    processed_count_second = 0
    for item in second_run_data:
        is_new = storage.save_raw_data(
            zone_id=item["zone_id"],
            category=item["category"],
            data=item["data"],
            effective_date=item["acquisition_time"],
            source_url=item["data"].get("source_url")
        )
        if is_new: 
            processed_count_second += 1

    # Attente pour s'assurer que les logs de la base de données sont clairs
    time.sleep(0.5) 

    print("\n[Étape 3 Terminé] Analyse des résultats :")
    print(f"- Total collecté (2e exécution) : {len(second_run_data)}")
    print(f"- Total des NOUVEAUX enregistrements persistés (doit être bas) : {processed_count_second}")

    # Testons le 'retrieve facile' du cache pour les données critiques
    print("\n[Test Retrieve Facile] 🔍 Récupération des données d'alerte à Lyon via Cache...")
    alertes_lyon = storage.get_raw_data(zone_id="Lyon", category="ALERTE_INONDATION", limit=1)
    if alertes_lyon:
        print(f"-> Résultat du Cache (Alerte Lyon) : Niveau '{alertes_lyon[0].get('level')}'")
    else:
        print("-> Aucune alerte trouvée dans le cache.")

    storage.close()