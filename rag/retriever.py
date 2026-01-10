import logging
import json
from typing import List, Dict, Any, Optional

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag.Retriever")

class AgentRetriever:
    """
    Système de récupération de contextes (Retrieval) optimisé.
    Stratégie Hybride : 
    1. Cache/SQL pour les données structurées chaudes (Alertes, Prix actuels).
    2. Vector Search (FAISS) pures pour la connaissance non-structurée (Bulletins, PDFs).
    """
    
    def __init__(self, store: Any, embedder: Any, storage: Any = None):
        self.store = store      # FAISS VectorStoreHandler
        self.embedder = embedder # SBERT EmbeddingService
        self.storage = storage   # (Optionnel) Accès SQL/CacheDirect
        
        logger.info("📡 AgentRetriever prêt (FAISS + Embedding).")

    def retrieve_for_agent(self, query: str, agent_role: str, zone_id: str, limit: int = 4, cache_ttl_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Point d'entrée principal pour les agents.
        :param query: La question de l'utilisateur (Ex: "Quel est le prix du maïs ?").
        :param agent_role: Le contexte métier (MARKET_REPORT, METEO_VECTOR, AGRI_REPORT).
        :param zone_id: La zone géographique (Ex: "Boucle du Mouhoun").
        :param cache_ttl_minutes: Durée de vie du cache en minutes (default: 60).
        """
        logger.info(f"🔎 Retrieval demandé par {agent_role} (Zone: {zone_id}) : '{query}'")

        # --- 1. CACHE CHECK (HOT PATH) ---
        # Check if we have a cached result for this exact query/agent/zone combo
        if self.storage is not None:
            try:
                cached_results = self.storage.get_agent_cache(query, agent_role, zone_id, ttl_minutes=cache_ttl_minutes)
                if cached_results is not None:
                    logger.info(f"⚡ CACHE HIT for {agent_role}/{zone_id} - Skipping embedding computation")
                    return cached_results
            except Exception as e:
                logger.warning(f"⚠️ Cache lookup failed (non-blocking): {e}")

        # --- 2. VECTOR SEARCH (DEEP PATH) ---
        try:
            # A. Vectorisation de la requête (only if cache miss)
            logger.debug(f"💾 CACHE MISS - Computing embedding for query")
            query_vector = self.embedder.model.encode([query])[0].tolist()

            # B. Recherche FAISS avec filtres
            source_filter = self._map_role_to_source(agent_role)
            
            results = self.store.search(
                query_vector=query_vector, 
                k=limit, 
                source_filter=source_filter
            )
            
            # C. Post-Processing (Log & Format)
            formatted_results = []
            for res in results:
                formatted_results.append({
                    "content": res.get("content"),
                    "score": res.get("score"),
                    "source": res.get("metadata", {}).get("title"),
                    "date": res.get("metadata", {}).get("created_at")
                })
            
            logger.info(f"✅ {len(formatted_results)} documents trouvés pour {agent_role}.")

            # --- 3. CACHE STORE (WARM UP FOR NEXT REQUEST) ---
            if self.storage is not None and formatted_results:
                try:
                    self.storage.set_agent_cache(query, agent_role, zone_id, formatted_results)
                    logger.debug(f"💾 Results cached for future requests")
                except Exception as e:
                    logger.warning(f"⚠️ Cache store failed (non-blocking): {e}")

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Erreur critique lors du retrieval : {e}")
            return []

    def _map_role_to_source(self, agent_role: str) -> Optional[str]:
        """Convertit le rôle de l'agent en type de source documentaire."""
        mapping = {
            "SUBSIDY": "MARKET_REPORT", # L'agent business lit les rapports de marché
            "MARKET": "MARKET_REPORT",
            "METEO": "METEO_VECTOR",
            "CLIMATE": "METEO_VECTOR",
            "CROP": "AGRI_REPORT",
            "HEALTH": "AGRI_REPORT"
        }
        # Retourne None si pas de mapping (recherche globale)
        return mapping.get(agent_role, None)