from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from chromadb import PersistentClient

import chromadb 
#from chromadb.config import Settings
from typing import List, Optional
import logging
import os

from src.data_processing import load_documents_from_corpus , split_corpus_data

logger = logging.getLogger(__name__)

"""
Nous utilisons Ollama Embeddings pour creer une base de donnees vectorielle Chroma a partir du
 corpus extrait.Cloudfare est utilise pour creer un tunnel securise vers le serveur Ollama local.
"""

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://made-tanks-tissue-accuracy.trycloudflare.com")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "mistral:7b-instruct-q4_K_M")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
TOP_K = os.getenv("TOP_K", 5)

CORPUS_DIR = os.getenv("CORPUS_DIR", "data/corpus.json")

# Creation de la base de donnees vectorielle Chroma a partir des nodes

def create_chroma_vector_db(nodes: List[Document]) -> Optional[Chroma]:
    try:
        embedding_model = OllamaEmbeddings(
            model=EMBEDDING_MODEL_ID,
            base_url=OLLAMA_BASE_URL
        )

        vector_db = Chroma(
            embedding_function=embedding_model,
            collection_name="corpus",
            persist_directory=CHROMA_DB_PATH
        )

        texts = [doc.page_content for doc in nodes]
        metadatas = [doc.metadata for doc in nodes]

        vector_db.add_texts(texts=texts, metadatas=metadatas)

        logger.info("Base vectorielle Chroma créée avec succès")
        return vector_db

    except Exception as e:
        logger.error(f"Erreur dans create_chroma_vector_db: {e}")
        return None
    
# La fonction:verify_embeddings sert a savoir si l'embedding d'u documents (representation numerique) est valide

def verify_embeddings(vector_db: Chroma) -> bool:
    try:
        count = vector_db._collection.count()
        if count == 0:
            logger.warning("Aucun embedding trouvé.")
            return False
        logger.info(f"{count} embeddings trouvés.")
        return True
    except Exception as e:
        logger.error(f"Erreur dans verify_embeddings: {e}")
        return False
    

def run_vectorstore() -> Optional[Chroma]:
    documents = load_documents_from_corpus(CORPUS_DIR)
    if not documents:
        logger.error(f"Corpus introuvable dans {CORPUS_DIR}")
        return None

    nodes = split_corpus_data(documents)
    if not nodes:
        logger.error("Échec du découpage du corpus.")
        return None

    vector_db = create_chroma_vector_db(nodes)
    if not vector_db:
        logger.error("Échec de la création de la base vectorielle.")
        return None

    if verify_embeddings(vector_db):
        retriever = vector_db.as_retriever(search_kwargs={"k": int(TOP_K)})
        logger.info("Retriever créé avec succès.")
        return retriever
    else:
        logger.warning("Embeddings invalides.")
        return None
    