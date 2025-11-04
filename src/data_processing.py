import re, unicodedata
import os
import json,logging
from typing import Dict,List

from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

CORPUS_DIR = os.getenv("CORPUS_DIR", "data/corpus.json")

# Fonction pour charger les documents a partir d'un fichier json
def document_to_dict(doc):
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

# Fonction pour diviser le corpus en segments plus petits
def split_corpus_data(corpus_data: List[Document]) -> List[Document]:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        nodes = splitter.split_documents(corpus_data)
        logger.info(f"Nous avons cree {len(nodes)} nodes a partir du corpus data")
        return nodes
    except Exception as e:
        logger.error(f"La fonction split_corpus_data rencontre ce probleme: {e}")
        return []

# Fonctions pour nettoyer le texte et le normaliser 
def clean_text(text: str) -> str:
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)
    return text.strip()

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', r'\3-\2-\1', text)
    return text

def preprocess_corpus(corpus: List[Dict]) -> List[Dict]:
    for doc in corpus:
        brut = doc["text"]
        nettoye = clean_text(brut)
        normalise = normalize_text(nettoye)
        doc["text"] = normalise
    return corpus


def load_documents_from_corpus(path: str = CORPUS_DIR) -> List[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier corpus n est pas trouve a:{path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return [
        Document(
            page_content=item["text"], 
            metadata=item.get("metadata", {})
            )
        for item in raw_data
    ]