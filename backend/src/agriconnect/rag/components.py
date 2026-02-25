import logging
import os

logger = logging.getLogger(__name__)

try:
    import faiss
    from llama_index.core import Settings, VectorStoreIndex, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore

    # ── LLM : délégué au provider abstrait (Groq/Azure/Bedrock) ──
    from agriconnect.services.llm_clients import get_chat_client, get_sdk_client

    from agriconnect.core.settings import settings as app_settings
    from .config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, DB_DIR

    # Dimension for sentence-transformers/all-MiniLM-L6-v2
    EMBEDDING_DIM = 384
    INDEX_FILE = os.path.join(DB_DIR, "faiss_index.bin")

    def get_embedding_model():
        return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    # ── Aliases rétro-compatibles pour tous les imports existants ──
    def get_llm_client():
        """Retourne un client LangChain Chat (provider-agnostic)."""
        return get_chat_client()

    def get_groq_sdk():
        """Retourne un SDK client brut (provider-agnostic)."""
        return get_sdk_client()

    def init_settings():
        Settings.embed_model = get_embedding_model()
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP

    def get_vector_store():
        """
        Returns a FaissVectorStore.
        If the index file exists, loads it. Otherwise, creates a new HNSW index.
        """
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR)

        if os.path.exists(INDEX_FILE):
            try:
                faiss_index = faiss.read_index(INDEX_FILE)
                return FaissVectorStore(faiss_index=faiss_index)
            except Exception as e:
                logger.warning("Could not load existing index: %s. Creating new one.", e)

        faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32, faiss.METRIC_INNER_PRODUCT)
        return FaissVectorStore(faiss_index=faiss_index)

    def get_storage_context():
        vector_store = get_vector_store()
        # Check if docstore.json exists to decide whether to load or create new
        docstore_path = os.path.join(DB_DIR, "docstore.json")
        if os.path.exists(docstore_path):
            return StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(DB_DIR))
        else:
            return StorageContext.from_defaults(vector_store=vector_store)

    def save_index(index):
        if hasattr(index.vector_store, "client"):
            faiss.write_index(index.vector_store.client, INDEX_FILE)

except Exception:
    # Fallbacks when heavy ML dependencies are not installed (tests / CI lightweight runs)

    def get_llm_client():
        class DummyLLM:
            def chat(self, *args, **kwargs):
                return {"content": "dummy response"}

            def generate(self, *args, **kwargs):
                return {"content": "dummy response"}

        return DummyLLM()

    def get_groq_sdk():
        return None

    def init_settings():
        logger.debug("ML dependencies missing: init_settings is a no-op in fallback mode.")

    def get_vector_store():
        raise RuntimeError("Faiss / llama_index not available in fallback mode")

    def get_storage_context():
        raise RuntimeError("Faiss / llama_index not available in fallback mode")

    def save_index(index):
        logger.debug("save_index called in fallback mode; skipping.")

