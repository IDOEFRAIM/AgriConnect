import logging
from typing import List, Dict, Any, Optional
# Simuler l'importation de vos classes Milvus/HNSW
# from vector_store_handler import VectorStoreHandler
# from embedder import Embedder, Reranker

logger = logging.getLogger("AgentRetriever")
# Initialisation du logger
logging.basicConfig(level=logging.INFO)

class AgentRetriever:
    """
    Récupère des chunks pour un agent spécifique en utilisant un store vectoriel, 
    un embedder et un reranker.
    """
    
    def __init__(self, store: Any, embedder: Any, reranker: Any, storage: Any):
        """
        Initialisation avec les dépendances (le store serait votre VectorStoreHandler pour Milvus/HNSW).
        """
        self.store = store # VectorStoreHandler (Milvus/HNSW)
        self.embedder = embedder # Modèle d'embedding
        self.reranker = reranker # Modèle de reranking
        self.storage = storage # StorageManager (Couche Cache/SQLite)
        logger.info("AgentRetriever initialisé (store, embedder, reranker, storage).")


    # CORRECTION : Ajout de l'argument 'filters' dans la signature.
    def retrieve_for_agent(self, query: str, agent_role: str, zone_id: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Récupère les documents pertinents pour une requête et un rôle d'agent donné.

        :param query: La requête de l'utilisateur.
        :param agent_role: Le rôle de l'agent ('CROP', 'METEO', etc.) utilisé pour filtrer.
        :param zone_id: L'ID de la zone géographique pour le filtre structurel.
        :param limit: Le nombre maximal de documents à retourner.
        :param filters: Filtres additionnels passés par l'évaluateur (par exemple, pour des tests plus granulaires).
        :return: Une liste de documents récupérés et classés.
        """
        
        # --- RÉPONSE À VOTRE QUESTION : Utilisation du cache pour le "retrieve facile" ---
        # Si la requête concerne une donnée structurelle très récente (alerte, date limite),
        # il est plus rapide et plus fiable d'interroger la table raw_agent_data (le cache)
        # plutôt que le Vector Store.
        if agent_role in ["ALERTE_INONDATION", "SUBVENTION", "METEO"]:
            # On vérifie si la donnée la plus récente suffit (ex: "Quelle est l'alerte actuelle à Lyon ?")
            logger.info(f"Vérification du cache (SQLite) pour les données structurelles de {agent_role}...")
            
            # Utilisation de la méthode de récupération du StorageManager
            latest_data = self.storage.get_raw_data(zone_id=zone_id, category=agent_role, limit=3)
            
            if latest_data:
                # Si l'alerte est trouvée, on la retourne immédiatement (retrieve très facile)
                if agent_role == "ALERTE_INONDATION":
                    logger.info("Alerte critique trouvée dans le cache. Skipping vector search.")
                    return latest_data 
        # ---------------------------------------------------------------------------------
        
        try:
            # 1. Embed la requête (Simulé car l'embedder n'est pas fourni)
            # query_vector = self.embedder.embed_query(query)
            query_vector = [0.1] * 128 # Simuler le vecteur

            # 2. Recherche initiale dans le store vectoriel (Milvus/HNSW)
            # Utilise agent_role comme filtre de source
            initial_results = self.store.search(
                query_vector=query_vector, 
                k=limit * 2, 
                source_filter=agent_role,
                # Le filtre Milvus/HNSW peut combiner l'agent_role et zone_id
                vector_filters={"zone_id": zone_id, **(filters or {})}
            )
            
            if not initial_results:
                logger.warning(f"Aucun résultat vectoriel trouvé pour le rôle {agent_role}.")
                return []

            # 3. Reranking des résultats (Simulé car le reranker n'est pas fourni)
            # reranked_results = self.reranker.rerank(query, initial_results)
            reranked_results = initial_results # Pas de rerank dans la simulation
            
            # 4. Limiter au nombre final demandé
            final_results = reranked_results[:limit]
            
            logger.debug(f"Récupération vectorielle réussie pour {agent_role}. Résultats finaux: {len(final_results)}")
            return final_results

        except Exception as e:
            # Lancement de l'erreur pour que l'évaluation puisse la capter et logguer
            logger.error(f"Erreur retrieval pour {agent_role}: {e}")
            raise