# indexer.py
import os
os.environ["OLLAMA_HOST"] = "http://localhost:11434"
import json
import uuid
from pathlib import Path
from typing import List, Dict, Callable, Optional, Any
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

try:
    from sentence_transformers import SentenceTransformer
    from transformers import BitsAndBytesConfig
except Exception:
    SentenceTransformer = None
    BitsAndBytesConfig = None

class GenericIndexer:
    def __init__(
        self,
        persist_dir: str = "chroma_persist",
        collection_name: str = "generic_collection",
        embedding_model: str = "mistral:latest",
        local_fallback_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        mistral_endpoint: Optional[str] = None,
        mistral_api_key: Optional[str] = None
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.local_fallback_model = local_fallback_model
        self.batch_size = batch_size
        self.mistral_endpoint = mistral_endpoint or os.environ.get("MISTRAL_EMBEDDING_URL")
        self.mistral_api_key = mistral_api_key or os.environ.get("MISTRAL_API_KEY")

    def index_records(
        self,
        records: List[Dict[str, Any]],
        text_builder: Callable[[Dict[str, Any]], str],
        metadata_builder: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ):
        if not records:
            print("Aucun enregistrement à indexer.")
            return

        texts = [text_builder(r) for r in records]
        embeddings = self._embed_documents(texts)
        ids = [r.get("id") or str(uuid.uuid4()) for r in records]
        raw_metadatas = [
            metadata_builder(r) if metadata_builder else {k: v for k, v in r.items() if k != "text"}
            for r in records
        ]
        metadatas = [_sanitize_metadata_record(m) for m in raw_metadatas]

        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise RuntimeError("Incohérence de tailles avant ajout.")

        client = self._get_chroma_client()
        collection = self._get_or_create_collection(client, self.collection_name)

        for i, m in enumerate(metadatas):
            if not m:
                print(f"⚠️ metadata vide pour id {ids[i]} (index {i})")

        try:
            collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            try:
                client.persist()
            except Exception:
                pass
            print(f"Indexation terminée dans ChromaDB ({len(records)} documents).")
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'ajout des documents : {e}")

    def _get_chroma_client(self):
        try:
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_dir,
                chroma_segment_cache_policy="LRU",
                chroma_memory_limit_bytes=2_000_000_000
            )
            client = chromadb.Client(settings=settings)
            print("✅ Chroma: Client avec cache LRU initialisé.")
            return client
        except Exception as e:
            print("⚠️ Échec Client(Settings(...)) :", e)

        try:
            client = chromadb.PersistentClient(path=self.persist_dir)
            print("✅ Chroma: PersistentClient initialisé.")
            return client
        except Exception as e:
            print("ℹ️ PersistentClient non disponible :", e)

        try:
            client = chromadb.Client()
            print("⚠️ Fallback: Chroma en mémoire.")
            return client
        except Exception as e:
            raise RuntimeError("Impossible d'initialiser ChromaDB : " + str(e))

    def _get_or_create_collection(self, client, name: str):
        try:
            return client.get_or_create_collection(name=name)
        except Exception:
            try:
                return client.get_collection(name=name)
            except Exception:
                try:
                    return client.create_collection(name=name)
                except Exception as e:
                    raise RuntimeError(f"Impossible de créer la collection '{name}': {e}")

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.mistral_endpoint:
            return self._embed_remote(texts)
        else:
            return self._embed_local(texts)

    def _embed_remote(self, texts: List[str]) -> List[List[float]]:
        print("Utilisation du endpoint distant :", self.mistral_endpoint)
        import requests
        headers = {"Content-Type": "application/json"}
        if self.mistral_api_key:
            headers["Authorization"] = f"Bearer {self.mistral_api_key}"        
        embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding remote"):
            batch = texts[i:i + self.batch_size]
            payload = {"model": self.embedding_model, "input": batch}
            try:
                resp = requests.post(self.mistral_endpoint, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                j = resp.json()
                if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
                    embeddings.extend([d.get("embedding") for d in j["data"]])
                elif isinstance(j, dict) and "embeddings" in j:
                    embeddings.extend(j["embeddings"])
                elif isinstance(j, list) and isinstance(j[0], list):
                    embeddings.extend(j)
                else:
                    raise ValueError("Format de réponse inattendu")
            except Exception as e:
                print("Erreur embedding distant :", e)
                return self._embed_local(texts)
        return embeddings

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers non disponible.")
        print("Utilisation du modèle local :", self.local_fallback_model)

        try:
            if BitsAndBytesConfig:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                model = SentenceTransformer(self.local_fallback_model, quantization_config=quant_config)
                print("✅ Quantization 8-bit activée.")
            else:
                model = SentenceTransformer(self.local_fallback_model)
        except Exception as e:
            print("⚠️ Quantization échouée, fallback standard :", e)
            model = SentenceTransformer(self.local_fallback_model)

        embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding local"):
            batch = texts[i:i + self.batch_size]
            emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings.extend(emb.tolist())
        return embeddings

def _sanitize_value(v):
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    try:
        return str(v)
    except Exception:
        return None

def _sanitize_metadata_record(meta: dict) -> dict:
    out = {}
    if not meta:
        return out
    for k, v in (meta or {}).items():
        sv = _sanitize_value(v)
        if sv is not None:
            out[k] = sv
    return out