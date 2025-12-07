import sqlite3
import json
import logging
import hashlib
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ScraperOrchestrator")

# ======================================================================================
# 1. GESTIONNAIRE DE STOCKAGE (StorageManager) - Intégration de la version finalisée
#    Utilisé comme gestionnaire de contexte (via 'with') pour la sécurité de la DB.
# ======================================================================================

class StorageManager:
    """
    Gestionnaire de stockage/cache centralisé utilisant SQLite.
    Assure la persistance des données brutes (raw_agent_data) et des caches.
    Implémente le protocole de gestionnaire de contexte.
    """

    def __init__(self, db_path: str = "data/agconnect_cache.db"):
        self.conn: Optional[sqlite3.Connection] = None
        self.db_path = db_path
        
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.conn = self._connect() 
            self._init_schema()
            
            logging.getLogger("StorageManager").info("StorageManager initialisé avec succès.")
        except Exception as e:
            logging.getLogger("StorageManager").error(f"❌ Erreur critique lors de l'initialisation : {e}")
            if self.conn:
                self.conn.close()
            self.conn = None

    def _connect(self) -> sqlite3.Connection:
        """Établit la connexion et définit la row_factory pour l'accès par nom de colonne."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row 
        return conn

    def _init_schema(self):
        """Initialise les tables de la base de données."""
        if not self.conn: return

        cursor = self.conn.cursor()
        try:
            # Table agent_cache (Cache des résultats intermédiaires)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    category TEXT NOT NULL, 
                    zone_id TEXT NOT NULL,
                    payload TEXT NOT NULL, 
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                    expires_at TIMESTAMP,
                    source TEXT DEFAULT 'system', 
                    UNIQUE(category, zone_id)
                )
            """)
            
            # Table embeddings (Cache des vecteurs) - Schéma non modifié
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, text_hash TEXT NOT NULL, model_name TEXT NOT NULL,
                    vector TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, expires_at TIMESTAMP,
                    UNIQUE(text_hash, model_name)
                )
            """)

            # Table raw_agent_data (Données brutes)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_agent_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    agent_category TEXT NOT NULL, 
                    zone_id TEXT NOT NULL, 
                    effective_date REAL, 
                    source_url TEXT, 
                    data_hash TEXT NOT NULL, 
                    content_json TEXT NOT NULL, 
                    UNIQUE(zone_id, agent_category, data_hash)
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_lookup ON raw_agent_data(agent_category, zone_id, effective_date DESC);")
            self.conn.commit()
        except sqlite3.Error as e:
            logging.getLogger("StorageManager").error(f"❌ Erreur lors de l'initialisation du schéma : {e}")
            self.conn.rollback()

    def _hash_data(self, data: Dict[str, Any]) -> str:
        """Crée un hash stable du contenu JSON pour la déduplication."""
        json_string = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_string.encode('utf-8')).hexdigest()

    # Context Manager
    def __enter__(self) -> 'StorageManager':
        if self.conn is None:
            raise RuntimeError("StorageManager n'a pas pu s'initialiser correctement.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Ferme la connexion à la base de données."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
            except sqlite3.Error:
                pass # Ignorer l'erreur de fermeture

    def save_raw_data(self, zone_id: str, category: str, data: Dict[str, Any], effective_date: float, source_url: Optional[str] = None) -> bool:
        """Sauvegarde les données brutes avec déduplication (INSERT OR IGNORE)."""
        if not self.conn: return False

        try:
            cursor = self.conn.cursor()
            json_content = json.dumps(data, ensure_ascii=False)
            data_hash = self._hash_data(data)
            
            cursor.execute("""
                INSERT OR IGNORE INTO raw_agent_data (zone_id, agent_category, content_json, effective_date, source_url, data_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (zone_id, category, json_content, effective_date, source_url, data_hash))
            
            rows_inserted = cursor.rowcount
            self.conn.commit()
            
            if rows_inserted == 0:
                 logging.getLogger("StorageManager").debug(f"Données ignorées (doublon): {category}/{zone_id} ({data_hash[:8]})")
            
            return rows_inserted > 0
                
        except sqlite3.Error as e:
            logging.getLogger("StorageManager").error(f"❌ Erreur SQLite lors de la sauvegarde ({category}/{zone_id}): {e}")
            self.conn.rollback()
            return False

    def get_raw_data(self, zone_id: str, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les données brutes triées par date d'effet la plus récente."""
        if not self.conn: return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT content_json, effective_date, source_url 
                FROM raw_agent_data 
                WHERE zone_id = ? AND agent_category = ?
                ORDER BY effective_date DESC 
                LIMIT ?
            """, (zone_id, category, limit))
            
            results = []
            for row in cursor.fetchall():
                data = json.loads(row['content_json'])
                data['effective_date'] = row['effective_date']
                data['source_url'] = row['source_url']
                results.append(data)
                
            return results
        except sqlite3.Error as e:
            logging.getLogger("StorageManager").error(f"❌ Erreur SQLite lors de la récupération ({category}/{zone_id}): {e}")
            return []

    # Méthodes de cache (set_cache/get_cache) non incluses ici pour se concentrer sur le raw_data flow

# ======================================================================================
# 2. SIMULATIONS D'INTERFACES DES SERVICES AVAL (Pour la démonstration architecturale)
#    Ces classes représentent des dépendances externes (Vector DB, Embedding Service) 
#    et sont simulées pour illustrer le découplage de l'Orchestrateur.
# ======================================================================================

class VectorStoreHandler:
    def search(self, query_vector: list, k: int, source_filter: str, vector_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        logging.getLogger("VectorStoreHandler").debug("Recherche vectorielle simulée.")
        return [{"content": f"Résultat vectoriel simulé pour {source_filter}", "score": 0.9}]

    def index_data(self, category: str, data: Dict[str, Any]):
        """Simule l'indexation dans une base de données vectorielle."""
        logging.getLogger("VectorStoreHandler").info(f"💾 Indexation (simulée) pour {category}: Nouvelle donnée détectée.")
        pass

class Reranker:
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logging.getLogger("Reranker").debug("Reranking simulé.")
        return results

class EmbeddingService:
    def embed_query(self, query: str) -> list:
        logging.getLogger("EmbeddingService").debug("Embedding de requête simulé.")
        return [0.1] * 128

# ======================================================================================
# 3. AGENT DE SCRAPING (Interface et Implémentation Simulé)
# ======================================================================================

class ScraperAgent:
    def __init__(self, category: str):
        self.category = category 

    def run(self, zone_id: str) -> List[Dict[str, Any]]:
        """Simule l'exécution de l'agent pour une zone spécifique et génère des données."""
        logger.info(f"⚙️ Exécution de l'agent [{self.category}] pour la zone {zone_id}...")
        
        # --- Simule la collecte de données variées (robustesse) ---
        if self.category == 'METEO':
            # Météo: change à chaque heure (pour tester que le hash change)
            now = datetime.now()
            return [{
                "time_prevision": (now + timedelta(hours=i)).isoformat(),
                "temp_c": 20.0 + (i * 0.1) + (now.minute / 100), # Change légèrement pour éviter la déduplication au second run
                "description": "Bulletin Météorologique pour la journée",
                "source_url": f"http://meteo.com/bulletin/{zone_id}"
            } for i in range(2)]
        elif self.category == 'SUBVENTION':
            # Subvention: reste stable (pour tester la déduplication)
            return [{
                "grant_id": "S999",
                "title": "Aide Agricole Urgente",
                "deadline": (datetime.now() + timedelta(days=60)).isoformat(),
                "amount_eur": 50000.0,
                "eligible_zones": [zone_id],
                "source_url": "http://subventions.gouv/agri"
            }]
        elif self.category == 'ALERTE_INONDATION':
            # Alerte: Reste stable (pour tester la déduplication)
            return [{
                "level": "Rouge",
                "risk_area": zone_id,
                "timestamp": datetime.now().isoformat(),
                "details": "Niveau de crue critique sur le fleuve.",
                "source_url": "http://alertes.gouv/inondation"
            }]
        else:
            return []

# ======================================================================================
# 4. ORCHESTRATEUR PRINCIPAL
# ======================================================================================

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


# ======================================================================================
# 5. DÉMONSTRATION DE LA PIPELINE COMPLÈTE
# ======================================================================================

if __name__ == '__main__':
    
    DB_PATH = "data/orchestrator_final.db"
    
    # 1. PRÉPARATION DES OUTILS (Instanciation des services en aval)
    # Note : Le StorageManager est instancié plus bas dans un bloc 'with' pour la robustesse.
    store = VectorStoreHandler()
    embedder = EmbeddingService()
    reranker = Reranker()
    
    # 2. DÉFINITION DE LA TÂCHE
    agents_map = {
        "METEO": ScraperAgent("METEO"),
        "SUBVENTION": ScraperAgent("SUBVENTION"),
        "ALERTE_INONDATION": ScraperAgent("ALERTE_INONDATION"),
    }
    zones_list = ["Paris", "Lyon", "Marseille"]

    # 3. L'ORCHESTRATEUR (Exécution pure)
    orchestrator = ScraperOrchestrator(agents_map, zones_list)
    
    print("\n" + "="*50)
    print("[Étape 1] 🚀 Lancement de l'Orchestrateur pour collecter les données...")
    print("="*50)
    final_collected_data = orchestrator.run_pipeline() 
    
    print(f"\n[Étape 1 Terminé] Total des enregistrements collectés : {len(final_collected_data)}")

    # 4. LE FLUX DE TRAITEMENT AVAL (Persistance, Caching, Vectorisation)
    print("\n" + "="*50)
    print("[Étape 2] 💾 Démarrage du Traitement Aval (Persistence & Déduplication)...")
    print("="*50)
    
    # Utilisation du StorageManager comme Context Manager pour une fermeture garantie
    with StorageManager(db_path=DB_PATH) as storage:
        processed_count = 0
        for item in final_collected_data:
            # Stockage de la donnée brute. La méthode retourne True si la donnée est nouvelle.
            is_new = storage.save_raw_data(
                zone_id=item["zone_id"],
                category=item["category"],
                data=item["data"],
                effective_date=item["acquisition_time"],
                source_url=item["data"].get("source_url")
            )
            
            # Indexation dans le Vector Store SEULEMENT si c'est une donnée nouvelle
            if is_new:
                store.index_data(item["category"], item["data"]) 
                processed_count += 1

        print(f"\n[Étape 2 Terminé] Total des NOUVEAUX enregistrements persistés : {processed_count}")

        # 5. Testons la robustesse/déduplication en relançant l'Orchestrateur
        print("\n" + "="*50)
        print("[Étape 3] 🔄 Relance de la pipeline (pour tester la déduplication)...")
        print("="*50)
        
        second_run_data = orchestrator.run_pipeline()
        processed_count_second = 0
        
        # Nouvelle boucle de persistance
        for item in second_run_data:
            is_new = storage.save_raw_data(
                zone_id=item["zone_id"],
                category=item["category"],
                data=item["data"],
                effective_date=item["acquisition_time"],
                source_url=item["data"].get("source_url")
            )
            if is_new: 
                store.index_data(item["category"], item["data"])
                processed_count_second += 1

        print("\n[Étape 3 Terminé] Analyse des résultats :")
        print(f"- Total collecté (2e exécution) : {len(second_run_data)} (identique au 1er run)")
        # On s'attend à ce que 'METEO' change à chaque exécution (temp_c) mais 'SUBVENTION' et 'ALERTE_INONDATION' soient ignorées.
        # Total attendu : 2 météo * 3 zones = 6 nouveaux enregistrements
        print(f"- Total des NOUVEAUX enregistrements persistés (après 2ème déduplication) : {processed_count_second}")

        # 6. Testons le 'retrieve facile' du cache
        print("\n" + "="*50)
        print("[Test Retrieve Facile] 🔍 Récupération des données d'alerte à Lyon...")
        print("="*50)
        alertes_lyon = storage.get_raw_data(zone_id="Lyon", category="ALERTE_INONDATION", limit=1)
        if alertes_lyon:
            print(f"-> Résultat du Cache (Alerte Lyon) : Niveau '{alertes_lyon[0].get('level')}'")
            print(f"-> Détails : {alertes_lyon[0].get('details')}")
        else:
            print("-> Aucune alerte trouvée dans le cache.")

# Le StorageManager ferme automatiquement la connexion grâce au 'with'