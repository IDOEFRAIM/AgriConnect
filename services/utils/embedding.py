import logging
import hashlib
import time
from typing import List, Optional

import config
from services.utils.cache import StorageManager
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag.embedding")

class EmbeddingService:
    """
    Service d'embedding :
    - Focalisé sur embeddings (SBERT)
    - Cache via StorageManager (get_embedding/save_embedding)
    - Batching, monitoring, fallback neutre
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or getattr(config, 'EMBEDDING_MODEL', "paraphrase-multilingual-MiniLM-L12-v2")
        self.cache = StorageManager()
        try:
            logger.info(f"🔄 Chargement du modèle d'embedding : {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Modèle chargé. Dimension vectorielle : {self.dimension}")
        except Exception as e:
            logger.critical(f"❌ Erreur chargement modèle : {e}")
            raise RuntimeError("Impossible de charger le modèle d'embedding.") from e

    def _cache_key(self, text: str) -> str:
        h = hashlib.md5(text.encode()).hexdigest()
        return f"{self.model_name}:{h}"

    def embed_query(self, text: str, ttl_hours: Optional[int] = 24) -> List[float]:
        clean = text.replace("\n", " ").strip()
        if not clean:
            return [0.0] * self.dimension
        key = self._cache_key(clean)
        cached = self.cache.get_embedding(key, self.model_name)
        if cached:
            return cached
        try:
            vec = self.model.encode(clean).tolist()
            self.cache.save_embedding(key, self.model_name, vec, ttl_hours=ttl_hours)
            return vec
        except Exception as e:
            logger.error(f"Erreur inférence modèle: {e}")
            return [0.0] * self.dimension

    def embed_documents(self, texts: List[str], ttl_hours: Optional[int] = 24) -> List[List[float]]:
        clean_texts = [t.replace("\n", " ").strip() for t in texts if t.strip()]
        results = []
        to_compute = []
        keys = []
        # check cache
        for t in clean_texts:
            key = self._cache_key(t)
            keys.append(key)
            cached = self.cache.get_embedding(key, self.model_name)
            if cached:
                results.append(cached)
            else:
                to_compute.append(t)
        # compute missing
        if to_compute:
            start = time.time()
            try:
                new_vecs = self.model.encode(to_compute, batch_size=32, convert_to_numpy=True)
                elapsed = time.time() - start
                logger.info(f"⏱️ Embedding batch {len(to_compute)} textes en {elapsed:.2f}s")
                # save in cache and append
                j = 0
                for t in clean_texts:
                    key = self._cache_key(t)
                    cached = self.cache.get_embedding(key, self.model_name)
                    if cached:
                        continue
                    v = new_vecs[j].tolist()
                    self.cache.save_embedding(key, self.model_name, v, ttl_hours=ttl_hours)
                    j += 1
            except Exception as e:
                logger.error(f"Erreur inférence batch: {e}")
                # fallback: zeros for missing
                for t in to_compute:
                    key = self._cache_key(t)
                    self.cache.save_embedding(key, self.model_name, [0.0]*self.dimension, ttl_hours=ttl_hours)
        # final gather in original order
        final = []
        for t in clean_texts:
            key = self._cache_key(t)
            final.append(self.cache.get_embedding(key, self.model_name) or [0.0]*self.dimension)
        return final