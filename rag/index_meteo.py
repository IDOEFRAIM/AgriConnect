from ingestionAdapter import IngestionRouter
from documentLoader import DocumentLoader
from normalizer import Normalizer
from indexer import GenericIndexer
from build_text import build_text, build_meta

router = IngestionRouter()
loader = DocumentLoader("data/")
raw_docs = loader.load_all(with_filenames=True)

records = []
for doc, fname in raw_docs:
    records.extend(router.route(doc, fname))

normalizer = Normalizer(adapter=lambda d: d)
indexer = GenericIndexer(collection_name="meteo_fanfar", persist_dir="chroma_persist")
indexer.index_records(records, text_builder=build_text, metadata_builder=build_meta)