# rag/evaluator.py
from __future__ import annotations
import math
import time
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict

_logger = logging.getLogger("rag.evaluator")
_logger.addHandler(logging.NullHandler())

# --- utilitaires métriques ---
def reciprocal_rank(ranked_ids: List[str], relevant: Set[str]) -> float:
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

def hit_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> int:
    return 1 if any(d in relevant for d in ranked_ids[:k]) else 0

def precision_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    topk = ranked_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for d in topk if d in relevant) / len(topk)

def dcg_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def idcg_at_k(relevant_count: int, k: int) -> float:
    # ideal DCG when all top positions are relevant
    idcg = 0.0
    for i in range(1, min(relevant_count, k) + 1):
        idcg += (2**1 - 1) / math.log2(i + 1)
    return idcg

def ndcg_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    dcg = dcg_at_k(ranked_ids, relevant, k)
    idcg = idcg_at_k(len(relevant), k)
    return dcg / idcg if idcg > 0 else 0.0

# --- évaluation principale ---
def evaluate(
    run_results: Iterable[Dict[str, Any]],
    qrels: Dict[str, Set[str]],
    ks: List[int] = (1, 5, 10),
    return_per_query: bool = False
) -> Dict[str, Any]:
    """
    run_results: iterable de dicts {"q_id": str, "candidates": [{"id":...}, ...], "timings": {...}}
    qrels: mapping q_id -> set of relevant doc ids
    ks: list of k pour metrics
    """
    ks = sorted(set(ks))
    stats = {k: {"hits": 0, "precision_sum": 0.0, "ndcg_sum": 0.0} for k in ks}
    mrr_sum = 0.0
    count = 0
    latencies = defaultdict(list)
    per_query = {}

    for res in run_results:
        qid = res.get("q_id")
        if qid is None:
            _logger.warning("run result missing q_id, skipping")
            continue
        ranked = [c["id"] for c in res.get("candidates", [])]
        relevant = qrels.get(qid, set())
        # metrics
        mrr = reciprocal_rank(ranked, relevant)
        mrr_sum += mrr
        for k in ks:
            stats[k]["hits"] += hit_at_k(ranked, relevant, k)
            stats[k]["precision_sum"] += precision_at_k(ranked, relevant, k)
            stats[k]["ndcg_sum"] += ndcg_at_k(ranked, relevant, k)
        # timings
        t = res.get("timings") or {}
        for step, val in t.items():
            try:
                latencies[step].append(float(val))
            except Exception:
                pass
        count += 1
        if return_per_query:
            per_query[qid] = {
                "mrr": mrr,
                "relevant": list(relevant),
                "ranked": ranked[: max(ks)],
                "hits": {k: hit_at_k(ranked, relevant, k) for k in ks}
            }

    if count == 0:
        return {"error": "no queries evaluated"}

    result = {
        "count": count,
        "mrr": mrr_sum / count,
        "metrics": {},
        "latency": {},
    }
    for k in ks:
        result["metrics"][f"recall@{k}"] = stats[k]["hits"] / count
        result["metrics"][f"precision@{k}"] = stats[k]["precision_sum"] / count
        result["metrics"][f"ndcg@{k}"] = stats[k]["ndcg_sum"] / count

    # latency aggregates
    for step, vals in latencies.items():
        vals_sorted = sorted(vals)
        def pct(p):
            if not vals_sorted:
                return None
            idx = min(len(vals_sorted)-1, max(0, int(len(vals_sorted)*p/100)))
            return vals_sorted[idx]
        result["latency"][step] = {
            "p50": pct(50),
            "p95": pct(95),
            "p99": pct(99),
            "avg": sum(vals_sorted)/len(vals_sorted) if vals_sorted else None
        }

    if return_per_query:
        result["per_query"] = per_query
    return result

# --- helpers IO pour formats courants ---
def load_qrels(path: str) -> Dict[str, Set[str]]:
    """
    Charge un fichier JSONL ou JSON contenant qrels.
    JSONL: une ligne par query dict {"q_id":..., "relevant_doc_ids":[...]}
    JSON: list de tels dicts
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return {}
        if text.lstrip().startswith("["):
            data = json.loads(text)
        else:
            # JSONL
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
    qrels = {}
    for item in data:
        qid = item["q_id"]
        rels = set(item.get("relevant_doc_ids", []))
        qrels[qid] = rels
    return qrels

def load_run(path: str) -> List[Dict[str, Any]]:
    """
    Charge run results JSONL or JSON list.
    Each item must contain q_id and candidates list with id fields.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        if text.lstrip().startswith("["):
            data = json.loads(text)
        else:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data