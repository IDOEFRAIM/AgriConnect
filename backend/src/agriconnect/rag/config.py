from pathlib import Path
from typing import NamedTuple
from agriconnect.core.settings import settings

# Paths
# settings.BASE_DIR points to backend/src — adjust to repository layout
BASE_DIR = settings.BASE_DIR
# Raw data lives in backend/sources/raw_data (one level up from src)
RAW_DATA_DIR = BASE_DIR.parent / "sources" / "raw_data"
# RAG DB folder in backend/rag_db (one level up from src)
DB_DIR = BASE_DIR.parent / "rag_db"

# Model Config — single source of truth from settings
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM Config
LLM_MODEL_NAME = settings.LLM_MODEL

# RAG Parameters — defaults (backward-compatible)
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
TOP_K_RETRIEVAL = settings.TOP_K_RETRIEVAL
TOP_K_RERANK = settings.TOP_K_RERANK


# ── Profils RAG adaptatifs ────────────────────────────────────
class RAGProfile(NamedTuple):
    """Paramètres de retrieval adaptés au niveau de l'utilisateur."""
    top_k: int          # Nombre de documents à récupérer
    rerank_k: int       # Nombre de documents après reranking
    use_hyde: bool      # Activer HyDe (latence +1s, précision +++)
    tone: str           # "simple" | "standard" | "technique"


RAG_PROFILES = {
    "debutant": RAGProfile(
        top_k=settings.RAG_DEBUTANT_TOP_K,
        rerank_k=settings.RAG_DEBUTANT_RERANK_K,
        use_hyde=settings.RAG_DEBUTANT_USE_HYDE,
        tone="simple",
    ),
    "intermediaire": RAGProfile(
        top_k=settings.RAG_INTER_TOP_K,
        rerank_k=settings.RAG_INTER_RERANK_K,
        use_hyde=settings.RAG_INTER_USE_HYDE,
        tone="standard",
    ),
    "expert": RAGProfile(
        top_k=settings.RAG_EXPERT_TOP_K,
        rerank_k=settings.RAG_EXPERT_RERANK_K,
        use_hyde=settings.RAG_EXPERT_USE_HYDE,
        tone="technique",
    ),
}


def get_rag_profile(user_level: str = "debutant") -> RAGProfile:
    """Retourne le profil RAG adapté au niveau utilisateur."""
    return RAG_PROFILES.get(user_level, RAG_PROFILES["debutant"])
