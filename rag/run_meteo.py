# run_meteo.py
from ragPipeline import RAGPipeline
from retriever import Retriever
from embedder import Embedder
from reRank import ReRanker
from promptBuilder import PromptBuilder
from generator import UniversalGenerator
from evaluator import Evaluator
from logBuilder import Logger

from ingestionAdapter import IngestionRouter
from documentLoader import DocumentLoader
from indexer import GenericIndexer
from buildText import build_text, build_meta

# === Configuration ===
COLLECTION_NAME = "meteo_fanfar"
DATA_PATH = "bulletins_json/"  # dossier contenant tes fichiers
PERSIST_DIR = "chroma_persist"

# === Ingestion et normalisation (pré-indexation explicite) ===
router = IngestionRouter()
loader = DocumentLoader(DATA_PATH)

# Charger les documents (avec noms de fichiers)
docs = loader.load_all(with_filenames=True)

# Router + normaliser -> records
records = []
for doc, fname in docs:
    recs = router.route(doc, fname)
    if recs:
        records.extend(recs)

print(f"Records produits pour indexation: {len(records)}")

# === Indexation ===
embedder = Embedder()  # stub ou implémentation réelle
indexer = GenericIndexer(collection_name=COLLECTION_NAME, persist_dir=PERSIST_DIR)

indexer.index_records(records, text_builder=build_text, metadata_builder=build_meta)

# === Composants RAG (après indexation) ===
retriever = Retriever(persist_dir=PERSIST_DIR, collection_name=COLLECTION_NAME)
reranker = ReRanker()
prompt_builder = PromptBuilder()
generator = UniversalGenerator()
evaluator = Evaluator()
logger = Logger("logs/meteo_rag.jsonl")

# === Pipeline complet ===
pipeline = RAGPipeline(
    retriever=retriever,
    generator=generator,
    reranker=reranker,
    prompt_builder=prompt_builder,
    evaluator=evaluator,
    logger=logger
)

# === Requête utilisateur ===
query = "Quel est le climat typique à Bobo-Dioulasso en août ?"
result = pipeline.answer(query)

# === Affichage ===
print("🧠 Réponse :", result.get("response"))
print("📚 Sources :", [s.get("excerpt")[:120] for s in result.get("sources", [])])
print("📊 Scores :", result.get("scores"))