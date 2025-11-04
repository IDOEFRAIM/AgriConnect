
"""
Les packages ou modules de AgriBot

Ce package contient:
- data_extraction: Nous permet d'extraire les documents du web et d'autres documents dans data/localDocuments
- data_processing: Nous permet de gerer tout ce qui est traitement de dossier ou nettoyage
- data_vectordb: Nous permet de creer notre base de donne vectorielle
- api/py : Notre backend
"""

from src.data_extraction import run_extraction
from src.data_processing import split_corpus_data, clean_text, preprocess_corpus, load_documents_from_corpus
from src.data_extraction import run_extraction
