from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List
import random

# Configuration de base pour l'observabilité
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_logger = logging.getLogger("RAGReferencePipeline")

# ==============================================================================
# 1. Structures de Données
# ==============================================================================

@dataclass
class QueryResult:
    """Représente un morceau de contexte récupéré avec son score."""
    text: str
    source: str
    score: float
    chunk_id: str

@dataclass
class ReferenceResponse:
    """Contient les documents de référence finaux et les métriques de performance."""
    query: str
    context_references: List[QueryResult]
    full_latency_ms: float
    retrieval_latency_ms: float
    reranking_latency_ms: float
    
# ==============================================================================
# 2. Composants Utilisés (Simulations des classes que vous avez codées)
# Ces classes simulent l'API que l'Orchestrator va appeler.
# ==============================================================================

class EmbedderComponent:
    """Simule votre classe Embedder (e.g., pour BGE ou E5)."""
    def embed_query(self, query: str) -> List[float]:
        """Génère un vecteur factice pour la requête."""
        time.sleep(0.01) # Petite latence pour l'embedding
        # Simule un vecteur de 64 dimensions
        return [random.random() for _ in range(64)]

class VectorStoreComponent:
    """Simule votre Indexer (e.g., Milvus Indexer) pour la recherche vectorielle."""
    
    def __init__(self):
        # Données factices similaires à un index
        self._data = [
            {"chunk_id": "doc1_c0", "text": "L'expansion du cloud a augmenté de 15% au T1, dépassant les attentes.", "source": "Rapport 2024", "relevance": 0.9},
            {"chunk_id": "doc1_c1", "text": "Le projet R&D a dépassé le budget de 5M EUR, nécessitant un réalignement.", "source": "Rapport 2024", "relevance": 0.5},
            {"chunk_id": "doc1_c2", "text": "L'IA générative sera déployée au T3 pour les processus d'automatisation.", "source": "Rapport 2024", "relevance": 0.85},
            {"chunk_id": "doc2_c0", "text": "Changer le mot de passe tous les 90 jours est une politique de sécurité.", "source": "Sécurité PDF", "relevance": 0.3},
            {"chunk_id": "doc3_c0", "text": "Les microservices sont finalisés en Mars avec la stack Kubernetes.", "source": "Tech Blog", "relevance": 0.7},
            {"chunk_id": "doc4_c0", "text": "Les tendances du marché indiquent une hausse des prix des matières premières.", "source": "Analyse Marché", "relevance": 0.4},
            {"chunk_id": "doc5_c0", "text": "L'adoption de l'open source pour la stack de développement est prioritaire.", "source": "Note Interne", "relevance": 0.6},
        ]
        
    def search(self, query_vector: List[float], top_k: int) -> List[QueryResult]:
        """Simule la recherche dans le Vector Store et retourne les Top-K."""
        
        # Simule une latence d'API Milvus
        time.sleep(0.05) 
        
        results = []
        # Le score est simulé ici, basé sur la pertinence initiale
        for item in self._data:
            initial_score = item['relevance'] + random.uniform(-0.1, 0.1) 
            results.append(QueryResult(
                text=item["text"],
                source=item["source"],
                score=max(0.0, initial_score), 
                chunk_id=item["chunk_id"]
            ))
            
        # Tri et limitation (le "Retriever")
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        return sorted_results

class RerankerComponent:
    """Simule votre Reranker (Cross-Encoder) pour la précision."""
    
    def rerank(self, query: str, results: List[QueryResult], top_n: int) -> List[QueryResult]:
        """
        Réévalue le score pour une pertinence plus fine.
        """
        time.sleep(0.02) # Latence du modèle Cross-Encoder
        
        reranked_results = []
        # Le Reranker affine le score en fonction des mots-clés et du contexte réel
        for res in results:
            if "cloud" in res.text.lower() or "ia générative" in res.text.lower():
                 new_score = res.score * 1.3  # Boost pour les thèmes clés
            elif "mot de passe" in res.text.lower() and "sécurité" in query.lower():
                 new_score = res.score * 1.5 # Simule une bonne correspondance
            else:
                 new_score = res.score * 0.8  # Pénalité légère pour les résultats moins pertinents
            
            reranked_results.append(QueryResult(
                text=res.text,
                source=res.source,
                score=max(0.0, new_score),
                chunk_id=res.chunk_id
            ))
            
        # Sélection des N meilleurs (Top-N)
        final_results = sorted(reranked_results, key=lambda x: x.score, reverse=True)[:top_n]
        return final_results

# ==============================================================================
# 3. Le Chef d'Orchestre (Mode Références Uniquement)
# ==============================================================================

class RAGReferenceOrchestrator:
    """
    Orchestre le flux RAG jusqu'à l'étape du Reranking pour fournir
    la liste des documents de référence (contexte) à un LLM externe (Ollama).
    """
    def __init__(self, embedder: EmbedderComponent, vector_store: VectorStoreComponent, 
                 reranker: RerankerComponent):
        
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        
        # Hyperparamètres de contrôle
        self.RETRIEVAL_TOP_K = 10  # Top-K pour le rappel (Recall)
        self.RERANK_TOP_N = 3      # Top-N pour la précision (Precision)
        
        _logger.info("RAGReferenceOrchestrator initialisé (mode Références Uniquement).")

    def get_references(self, query: str) -> ReferenceResponse:
        """Exécute la séquence de Récupération et de Reranking."""
        
        full_start_time = time.time()
        
        # --- 1. Embedding de la requête ---
        _logger.debug("Étape 1: Embedding de la requête.")
        query_vector = self.embedder.embed_query(query)

        # --- 2. Récupération (Retriever) ---
        retrieval_start = time.time()
        _logger.debug(f"Étape 2: Recherche dans le Vector Store (Top-K={self.RETRIEVAL_TOP_K}).")
        initial_results = self.vector_store.search(query_vector, top_k=self.RETRIEVAL_TOP_K)
        retrieval_latency_ms = (time.time() - retrieval_start) * 1000
        
        # --- 3. Classement (Reranker) ---
        reranking_start = time.time()
        _logger.debug(f"Étape 3: Reranking des résultats (Top-N={self.RERANK_TOP_N}).")
        context_references = self.reranker.rerank(query, initial_results, top_n=self.RERANK_TOP_N)
        reranking_latency_ms = (time.time() - reranking_start) * 1000
        
        full_latency_ms = (time.time() - full_start_time) * 1000

        _logger.info(f"Références récupérées et classées en {full_latency_ms:.2f} ms.")
        
        return ReferenceResponse(
            query=query,
            context_references=context_references,
            full_latency_ms=full_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            reranking_latency_ms=reranking_latency_ms
        )

# ==============================================================================
# 4. Démonstration du Pipeline de Références
# ==============================================================================
if __name__ == "__main__":
    _logger.setLevel(logging.INFO) 
    print("--- DÉMARRAGE DU PIPELINE RAG (MODE RÉFÉRENCES UNIQUEMENT) ---")

    # 1. Instanciation des composants (Injection de Dépendances)
    embedder = EmbedderComponent()
    vector_store = VectorStoreComponent()
    reranker = RerankerComponent()

    # 2. Instanciation de l'Orchestrateur
    orchestrator = RAGReferenceOrchestrator(embedder, vector_store, reranker)

    # 3. Requête Utilisateur
    user_query = "Donnez les mises à jour importantes concernant la performance cloud et l'IA générative."
    print(f"\n[UTILISATEUR] : {user_query}")

    # 4. Exécution du pipeline pour obtenir les références
    response = orchestrator.get_references(user_query)

    # 5. Affichage des références finales (pour Ollama) et des métriques
    print("\n--- RÉFÉRENCES FINALEMENT SÉLECTIONNÉES (Contexte pour Ollama) ---")
    if response.context_references:
        for i, res in enumerate(response.context_references):
            print(f"  {i+1}. Score de Pertinence: {res.score:.3f} | Source: {res.source}")
            print(f"     -> Texte: {res.text}")
    else:
        print("  Aucun document de référence pertinent n'a été trouvé après le classement.")
    
    print("\n--- MÉTRIQUES DE PERFORMANCE ---")
    print(f"Latence Totale (Embedding + Récupération + Reranking): {response.full_latency_ms:.2f} ms")
    print(f"  > Latence Récupération (Vector Store): {response.retrieval_latency_ms:.2f} ms")
    print(f"  > Latence Classement (Reranker): {response.reranking_latency_ms:.2f} ms")
    
    print("\n--- FIN DE LA DÉMONSTRATION ---")