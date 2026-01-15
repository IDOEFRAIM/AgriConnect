import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rag.utils.indexer_init import MilvusIndexer
from rag.utils.embedder import Embedder

from pathlib import Path
import sys




indexer = MilvusIndexer()
embedder = Embedder()

# Cette ligne supprime automatiquement l'ancienne collection
indexer.drop_collection("sample_collection")

# Puis réindexe avec le nouveau format
result = indexer.init_collection_from_folder(
    collection="sample_collection",
    embedder=embedder,
    folder="bulletins_json"
)
print(result)