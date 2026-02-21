import logging
from llama_index.core import VectorStoreIndex, QueryBundle, load_index_from_storage
from llama_index.core.schema import NodeWithScore
from typing import List, Optional
from .components import init_settings, get_storage_context, get_groq_sdk
from .config import TOP_K_RETRIEVAL, TOP_K_RERANK, get_rag_profile, RAGProfile

logger = logging.getLogger(__name__)

# Optional: Load Reranker if available
RERANKER = None
try:
    from sentence_transformers import CrossEncoder
    # Initialize light reranker
    RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except ImportError:
    pass

class AgileRetriever:
    def __init__(self):
        init_settings()
        storage_context = get_storage_context()
        # Load from persistence
        try:
            structs = storage_context.index_store.index_structs()
            if not structs:
                raise ValueError("No index found in storage.")
            
            if hasattr(storage_context.vector_store, "client"):
                logger.info("FAISS #total before load: %d", storage_context.vector_store.client.ntotal)

            # Default to the first index found
            target_index_id = structs[0].index_id
            
            self.index = load_index_from_storage(storage_context, index_id=target_index_id)
            if hasattr(self.index, "_vector_store") and hasattr(self.index._vector_store, "client"):
                logger.info("FAISS ntotal loaded: %d", self.index._vector_store.client.ntotal)
            # Default retriever (overridden per-search)
            self.vector_retriever = self.index.as_retriever(similarity_top_k=TOP_K_RETRIEVAL)
        except Exception as e:
            logger.warning("Error loading index: %s", e)
            self.index = None
            self.vector_retriever = None
            
        self.llm = get_groq_sdk()

    def generate_hyde_doc(self, query_str: str, tone: str = "standard") -> str:
        """
        HyDe (Hypothetical Document Embeddings):
        Génère un faux document qui répond à la question, puis cherche ce document.
        Le ton s'adapte au profil utilisateur.
        """
        if tone == "technique":
            persona = (
                "Tu es un agronome chercheur spécialisé Sahel/Burkina Faso. "
                "Rédige un paragraphe technique détaillé avec termes scientifiques, "
                "noms latins des pathogènes, doses précises, et références."
            )
        elif tone == "simple":
            persona = (
                "Tu es un conseiller agricole de village. "
                "Rédige un court paragraphe avec des mots simples et des exemples concrets."
            )
        else:
            persona = (
                "Tu es un expert agricole. "
                "Rédige un paragraphe technique qui répond à cette question."
            )

        try:
            prompt = f"{persona}\nQuestion : '{query_str}'"
            chat_completion = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
            )
            hypothetical_doc = chat_completion.choices[0].message.content or query_str
            return hypothetical_doc
        except Exception as e:
            logger.warning("HyDe generation warning: %s", e)
            return query_str

    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: int = TOP_K_RERANK) -> List[NodeWithScore]:
        """
        Re-rank retrieved nodes using a CrossEncoder for higher precision.
        Le nombre de résultats dépend du profil (debutant=3, expert=8).
        """
        if not RERANKER or not nodes:
            return nodes[:top_k]
            
        texts = [n.node.get_content() for n in nodes]
        inputs = [[query, text] for text in texts]
        scores = RERANKER.predict(inputs)
        
        for i, node in enumerate(nodes):
            node.score = float(scores[i])
            
        # Re-sort by relevance
        nodes.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        return nodes[:top_k]

    def search(
        self,
        query_str: str,
        user_level: str = "debutant",
        use_hyde: Optional[bool] = None,
    ) -> List[NodeWithScore]:
        """
        Recherche adaptative selon le profil utilisateur.

        - debutant  : pas de HyDe (rapide ~0.5s), top_k=5, rerank_k=3
        - intermediaire : HyDe activé, top_k=10, rerank_k=5
        - expert    : HyDe technique, top_k=20, rerank_k=8 (précision max)
        """
        if not self.vector_retriever:
            logger.warning("Index not initialized.")
            return []

        # Charger le profil adapté
        profile = get_rag_profile(user_level)
        should_hyde = use_hyde if use_hyde is not None else profile.use_hyde

        logger.info(
            "[RAG] Profil=%s | top_k=%d | rerank_k=%d | hyde=%s",
            user_level, profile.top_k, profile.rerank_k, should_hyde,
        )

        # Adapter le retriever au top_k du profil
        retriever = self.index.as_retriever(similarity_top_k=profile.top_k)

        search_query = query_str
        if should_hyde:
            hypo_doc = self.generate_hyde_doc(query_str, tone=profile.tone)
            logger.info("[HyDe] Generated hypothetical doc (%d chars, tone=%s)", len(hypo_doc), profile.tone)
            search_query = f"{query_str}\n{hypo_doc}"
            
        nodes = retriever.retrieve(search_query)
        logger.info("[Retriever] Found %d raw nodes.", len(nodes))
        
        # Rerank using ORIGINAL query, with profile-specific top_k
        final_nodes = self.rerank(query_str, nodes, top_k=profile.rerank_k)
        return final_nodes
