#!/usr/bin/env python3
# meteo_indexer_all_in_one.py
"""
Script tout-en-un :
- Normalise JSON météo (ville -> mois -> valeurs)
- Sauvegarde JSON / JSONL / CSV normalisés
- Calcule embeddings (priorité mistral:latest via endpoint si configuré)
- Indexe dans ChromaDB (duckdb+parquet) et persiste
"""

import os
import glob
import json
import csv
import uuid
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any

# Embedding fallback
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

# ChromaDB
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# ---------- Configuration par défaut ----------
DEFAULT_DATA_DIR = "json/climat.json"         # dossier contenant fichiers JSON bruts
OUT_DIR = "meteo_out"                  # dossier de sortie pour normalisés
PERSIST_DIR = "chroma_persist"         # dossier de persistance ChromaDB
COLLECTION_NAME = "meteo_mistral"      # nom collection ChromaDB
EMBEDDING_MODEL_NAME = "mistral:latest"
LOCAL_FALLBACK_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
# ---------------------------------------------

def load_json_files(folder: str) -> List[Dict]:
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    objs = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                objs.append((fp, json.load(f)))
        except Exception as e:
            print(f"Warning: impossible de lire {fp} -> {e}")
    return objs

MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def normalize_raw_jsons(json_files: List[tuple]) -> List[Dict]:
    """
    Attendu : chaque fichier contient un dict { "City": { "Température minimale": {...}, "Température maximale": {...}, "Précipitations": {...} } }
    Retourne : liste d'enregistrements plats par ville x mois
    """
    records = []
    for fp, data in json_files:
        source = os.path.basename(fp)
        if not isinstance(data, dict):
            continue
        for city, payload in data.items():
            if not isinstance(payload, dict):
                continue
            mins = payload.get("Température minimale", {}) or {}
            maxs = payload.get("Température maximale", {}) or {}
            precs = payload.get("Précipitations", {}) or {}
            months = sorted(set(list(mins.keys()) + list(maxs.keys()) + list(precs.keys())),
                            key=lambda m: MONTH_ORDER.index(m) if m in MONTH_ORDER else 999)
            for m in months:
                rec = {
                    "id": str(uuid.uuid4()),
                    "city": city,
                    "month": m,
                    "temp_min": _to_float_or_none(mins.get(m)),
                    "temp_max": _to_float_or_none(maxs.get(m)),
                    "precipitation": _to_float_or_none(precs.get(m)),
                    "source_file": source,
                    "embedding_model": EMBEDDING_MODEL_NAME
                }
                records.append(rec)
    return records

def _to_float_or_none(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None

def save_normalized(records: List[Dict], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(out_dir, "meteo_normalized.json")
    jsonl_path = os.path.join(out_dir, "meteo_normalized.jsonl")
    csv_path = os.path.join(out_dir, "meteo_normalized.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    keys = ["id","city","month","temp_min","temp_max","precipitation","source_file","embedding_model"]
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k,"") for k in keys})
    print(f"Saved normalized: {json_path}, {jsonl_path}, {csv_path}")

# ---------- Embeddings handling ----------
def get_mistral_embeddings(texts: List[str], model: str, endpoint: str, api_key: str = None) -> List[List[float]]:
    """
    Tentative d'appel batch à un endpoint d'embeddings Mistral.
    Format attendu du endpoint : POST JSON { "model": model, "input": [texts...] } -> réponse JSON contenant embeddings.
    Adapte selon ton endpoint réel.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "input": texts}
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    # heuristiques pour extraire embeddings
    if isinstance(j, dict):
        if "data" in j and isinstance(j["data"], list) and "embedding" in j["data"][0]:
            return [d["embedding"] for d in j["data"]]
        if "embeddings" in j and isinstance(j["embeddings"], list):
            return j["embeddings"]
        if "embedding" in j and isinstance(j["embedding"][0], list):
            return j["embedding"]
    # fallback: try top-level list of vectors
    if isinstance(j, list) and isinstance(j[0], list):
        return j
    raise ValueError("Format de réponse embeddings inattendu: " + str(j)[:200])

def embed_documents(records: List[Dict], batch_size: int = 64) -> List[List[float]]:
    """
    Priorité : MISTRAL_EMBEDDING_URL env var -> appel remote batch.
    Sinon fallback local sentence-transformers.
    """
    texts = [build_doc_text(r) for r in records]
    endpoint = os.environ.get("MISTRAL_EMBEDDING_URL")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if endpoint:
        print("Using remote Mistral endpoint for embeddings:", endpoint)
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding remote batches"):
            batch = texts[i:i+batch_size]
            try:
                emb_batch = get_mistral_embeddings(batch, EMBEDDING_MODEL_NAME, endpoint, api_key)
                embeddings.extend(emb_batch)
            except Exception as e:
                print("Remote embedding error:", e)
                print("Falling back to local model for remaining documents.")
                return _embed_local(texts, batch_size)
        return embeddings
    else:
        return _embed_local(texts, batch_size)

def _embed_local(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers non installé et aucun endpoint Mistral configuré.")
    print("Using local SentenceTransformer model:", LOCAL_FALLBACK_MODEL)
    model = SentenceTransformer(LOCAL_FALLBACK_MODEL)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding local batches"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.extend(emb.tolist())
    return embeddings

def build_doc_text(rec: Dict[str, Any]) -> str:
    parts = [
        rec.get("city",""),
        rec.get("month",""),
        f"temp_min {rec.get('temp_min')}" if rec.get('temp_min') is not None else "",
        f"temp_max {rec.get('temp_max')}" if rec.get('temp_max') is not None else "",
        f"precip {rec.get('precipitation')}" if rec.get('precipitation') is not None else ""
    ]
    return " | ".join([p for p in parts if p != ""])

# ---------- ChromaDB indexing ----------
def index_to_chromadb(records: List[Dict], embeddings: List[List[float]], persist_dir: str, collection_name: str):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
    collection = client.get_or_create_collection(name=collection_name)
    ids = [r["id"] for r in records]
    docs = [build_doc_text(r) for r in records]
    metadatas = [{"city": r["city"], "month": r["month"], "source_file": r["source_file"], "embedding_model": r["embedding_model"]} for r in records]
    # ensure embeddings are lists
    emb_list = [list(e) for e in embeddings]
    print(f"Adding {len(ids)} items to ChromaDB collection '{collection_name}'")
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=emb_list)
    client.persist()
    print("ChromaDB persisted to", persist_dir)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Normalise JSON météo et indexe dans ChromaDB")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Dossier contenant fichiers JSON bruts")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Dossier de sortie pour normalisés")
    parser.add_argument("--persist_dir", default=PERSIST_DIR, help="Dossier de persistance ChromaDB")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Nom de la collection ChromaDB")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Taille de batch pour embeddings")
    args = parser.parse_args()

    json_files = load_json_files(args.data_dir)
    if not json_files:
        print("Aucun fichier JSON trouvé dans", args.data_dir)
        return

    records = normalize_raw_jsons(json_files)
    if not records:
        print("Aucun enregistrement normalisé généré.")
        return

    save_normalized(records, args.out_dir)

    # embeddings
    embeddings = embed_documents(records, batch_size=args.batch_size)
    if not embeddings or len(embeddings) != len(records):
        print("Erreur: embeddings manquants ou taille différente. Abandon.")
        return

    # index
    index_to_chromadb(records, embeddings, args.persist_dir, args.collection)
    print("Terminé. Enregistrements indexés:", len(records))

if __name__ == "__main__":
    main()