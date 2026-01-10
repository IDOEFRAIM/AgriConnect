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
    
    def __init__(self, store: Any, embedder: Any, reranker: Any = None, storage: Any = None):
        self.store = store      # FAISS VectorStoreHandler
        self.embedder = embedder # SBERT EmbeddingService
        self.reranker = reranker # CrossEncoder Reranker
        self.storage = storage   # (Optionnel) Accès SQL/CacheDirect
        
        logger.info("📡 AgentRetriever prêt (FAISS + Embedding + Reranking).")

    def retrieve_for_agent(self, query: str, agent_role: str, zone_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Point d'entrée principal pour les agents.
        :param query: La question de l'utilisateur (Ex: "Quel est le prix du maïs ?").
        :param agent_role: Le contexte métier (MARKET_REPORT, METEO_VECTOR, AGRI_REPORT).
        :param zone_id: La zone géographique (Ex: "Boucle du Mouhoun").
        """
        logger.info(f"🔎 Retrieval demandé par {agent_role} (Zone: {zone_id}) : '{query}'")

        # --- 1. SMART CACHING CHECK ---
        # Si un storage manager est configuré, on vérifie le cache d'abord
        if self.storage:
            cached_results = self.storage.get_agent_cache(query, agent_role, zone_id)
            if cached_results:
                logger.info("🚀 Réponse servie depuis le cache (Fast Path).")
                return cached_results

        # --- 2. VECTOR SEARCH (DEEP PATH) ---
        try:
            # A. Vectorisation
            query_vector = self.embedder.model.encode([query])[0].tolist()

            # B. Recherche FAISS (Large Retrieval k*3 pour le reranking)
            source_filter = self._map_role_to_source(agent_role)
            initial_k = limit * 3 if self.reranker else limit
            
            raw_results = self.store.search(
                query_vector=query_vector, 
                k=initial_k, 
                source_filter=source_filter
            )
            
            if not raw_results:
                return []

            # C. Reranking (Si disponible)
            if self.reranker:
                logger.info(f"⚖️ Reranking de {len(raw_results)} documents...")
                reranked_results = self.reranker.rerank(
                    documents=raw_results,
                    agent_role=agent_role,
                    original_query=query
                )
                final_results = reranked_results[:limit]
            else:
                final_results = raw_results[:limit]
            
            # D. Formatage
            formatted_results = []
            for res in final_results:
                formatted_results.append({
                    "content": res.get("content") or res.get("text_content"),
                    "score": res.get("score") or res.get("cross_score"), # Score du reranker si dispo
                    "source": res.get("metadata", {}).get("title"),
                    "date": res.get("metadata", {}).get("created_at")
                })
            
            # --- 3. CACHE SAVING ---
            if self.storage and formatted_results:
                self.storage.set_agent_cache(query, agent_role, zone_id, formatted_results)
            
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Erreur critique lors du retrieval : {e}")
            return []

    # ... (Reste de la classe inchangé) ...

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