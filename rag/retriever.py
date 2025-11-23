# semantic_retriever.py
import os
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
import chromadb

class Retriever:
    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = "meteo_fanfar",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.persist_dir = persist_dir or os.path.join(script_dir, "chroma_persist")
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        # Initialisation du client Chroma (compatible avec plusieurs versions)
        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        except Exception:
            from chromadb.config import Settings
            self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_dir))

        # get_or_create_collection pour éviter NotFoundError
        try:
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception:
            # fallback compatibilité
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(name=self.collection_name)

        # modèle d'embeddings local
        self.embed_model = SentenceTransformer(self.embedding_model_name)

    def retrieve(self, query: str, n_results: int = 5, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Recherche les documents les plus proches sémantiquement d'une requête.

        Accepts:
            query (str): La requête utilisateur.
            n_results (int): Nombre de résultats par défaut.
            top_k (Optional[int]): Compatibilité avec d'autres appels (ex: pipeline).
            **kwargs: tolère des arguments additionnels sans erreur.

        Returns:
            List[Dict]: Liste de résultats structurés.
        """
        # Compatibilité : top_k prend le pas si fourni
        if top_k is not None:
            n_results = top_k

        if not query or not query.strip():
            raise ValueError("La requête est vide.")

        # Encodage de la requête
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)[0].tolist()

        # Requête Chroma
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({
                "source_file": (meta.get("source_file") if isinstance(meta, dict) else None) or "inconnu",
                "distance": dist,
                "excerpt": (doc or "")[:2000],
                "metadata": meta if isinstance(meta, dict) else {}
            })
        return out