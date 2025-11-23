# embedder.py
import os
import requests
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class Embedder:
    """
    Utilise Ollama en local pour obtenir des embeddings.
    Variables d'environnement :
      OLLAMA_EMBED_URL (ex: http://localhost:11434/api/embeddings)  # configurable
      OLLAMA_EMBED_MODEL (ex: mistral)  # nom du modèle sur Ollama
    Fallback : sentence-transformers local model (all-MiniLM-L6-v2).
    """
    def __init__(self, embed_model_local: str = "all-MiniLM-L6-v2"):
        self.ollama_url = os.environ.get("OLLAMA_EMBED_URL")  # ex: http://localhost:11434/api/embeddings
        self.ollama_model = os.environ.get("OLLAMA_EMBED_MODEL", "mistral")
        self.local_model_name = embed_model_local
        self.local_model = None
        if SentenceTransformer is not None:
            try:
                self.local_model = SentenceTransformer(self.local_model_name)
            except Exception:
                self.local_model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Try Ollama if configured
        if self.ollama_url:
            try:
                payload = {
                    "model": self.ollama_model,
                    "input": texts
                }
                resp = requests.post(self.ollama_url, json=payload, timeout=60)
                resp.raise_for_status()
                j = resp.json()
                # Support multiple response shapes
                if isinstance(j, dict):
                    if "embeddings" in j and isinstance(j["embeddings"], list):
                        return j["embeddings"]
                    if "data" in j and isinstance(j["data"], list):
                        # data: [{ "embedding": [...] }, ...]
                        out = []
                        for item in j["data"]:
                            if isinstance(item, dict) and "embedding" in item:
                                out.append(item["embedding"])
                        if out:
                            return out
                if isinstance(j, list) and isinstance(j[0], list):
                    return j
                # fallback to local if response unexpected
                print("⚠️ Ollama: format de réponse inattendu pour embeddings, fallback local.")
            except Exception as e:
                print(f"⚠️ Ollama embeddings failed: {e} — fallback local if available.")

        # Fallback local
        if self.local_model is not None:
            emb = self.local_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return emb.tolist()

        raise RuntimeError("Aucun embedder disponible : configure OLLAMA_EMBED_URL ou installe sentence-transformers.")