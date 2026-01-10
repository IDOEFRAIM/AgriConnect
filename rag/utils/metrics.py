import logging
import re
import math
from collections import Counter
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool, cpu_count

logger = logging.getLogger("rag.metrics")

# Module-level compiled regex for tokenization (faster)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Optional spaCy for better tokenization/lemmatization (if installed)
try:
    import spacy
    _SPACY_NLP = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
except Exception:
    _SPACY_NLP = None


class RAGMetrics:
    """
    Métriques RAG optimisées :
    - Tokenize cached
    - Batch helpers
    - Optional spaCy lemmatization
    """

    @staticmethod
    @lru_cache(maxsize=8192)
    def _tokenize_cached(text: str, use_lemma: bool = False) -> Tuple[str, ...]:
        """Tokenize and optionally lemmatize; cached for repeated calls."""
        if not text:
            return tuple()
        text = text.lower()
        if _SPACY_NLP and use_lemma:
            doc = _SPACY_NLP(text)
            return tuple(tok.lemma_ for tok in doc if tok.is_alpha)
        # fallback fast regex tokenization
        return tuple(_TOKEN_RE.findall(text))

    # -------------------------
    # Single-pair metrics
    # -------------------------
    @staticmethod
    def calculate_keyword_recall(expected_keywords: List[str], retrieved_text: str) -> float:
        if not expected_keywords:
            return 0.0
        text_lower = retrieved_text.lower()
        matches = 0
        for kw in expected_keywords:
            if not kw:
                continue
            if kw.lower() in text_lower:
                matches += 1
        return matches / len(expected_keywords)

    @staticmethod
    def calculate_f1(reference: str, candidate: str, use_lemma: bool = False) -> float:
        ref_tokens = RAGMetrics._tokenize_cached(reference, use_lemma)
        cand_tokens = RAGMetrics._tokenize_cached(candidate, use_lemma)
        if not ref_tokens or not cand_tokens:
            return 0.0
        ref_counts = Counter(ref_tokens)
        cand_counts = Counter(cand_tokens)
        shared = sum((cand_counts & ref_counts).values())
        if shared == 0:
            return 0.0
        precision = shared / len(cand_tokens)
        recall = shared / len(ref_tokens)
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_jaccard(reference: str, candidate: str, use_lemma: bool = False) -> float:
        ref_set = set(RAGMetrics._tokenize_cached(reference, use_lemma))
        cand_set = set(RAGMetrics._tokenize_cached(candidate, use_lemma))
        if not ref_set or not cand_set:
            return 0.0
        inter = ref_set & cand_set
        union = ref_set | cand_set
        return len(inter) / len(union)

    @staticmethod
    def is_perfect_hit(recall_score: float) -> bool:
        return recall_score >= 1.0

    @staticmethod
    def calculate_mrr(rank_of_relevant_doc: int) -> float:
        return 1.0 / rank_of_relevant_doc if rank_of_relevant_doc > 0 else 0.0

    @staticmethod
    def calculate_bleu(reference: str, candidate: str, n: int = 1, use_lemma: bool = False) -> float:
        """
        BLEU-n simplified. Default n=1 (unigram). Use small smoothing to avoid zeros.
        Kept simple and efficient for batch use.
        """
        ref_tokens = RAGMetrics._tokenize_cached(reference, use_lemma)
        cand_tokens = RAGMetrics._tokenize_cached(candidate, use_lemma)
        if not cand_tokens:
            return 0.0
        if n == 1:
            ref_counts = Counter(ref_tokens)
            cand_counts = Counter(cand_tokens)
            overlap = sum(min(cand_counts[t], ref_counts[t]) for t in cand_counts)
            precision = overlap / len(cand_tokens)
            bp = math.exp(1 - (len(ref_tokens) / len(cand_tokens))) if len(cand_tokens) < len(ref_tokens) else 1.0
            return bp * precision
        # For n>1 compute n-gram precisions
        def ngrams(tokens, k):
            return [tuple(tokens[i:i+k]) for i in range(len(tokens)-k+1)] if len(tokens) >= k else []
        precisions = []
        for k in range(1, n+1):
            ref_ngrams = Counter(ngrams(ref_tokens, k))
            cand_ngrams = Counter(ngrams(cand_tokens, k))
            if not cand_ngrams:
                precisions.append(1e-9)
                continue
            overlap = sum(min(cand_ngrams[g], ref_ngrams[g]) for g in cand_ngrams)
            precisions.append(overlap / max(1, sum(cand_ngrams.values())))
        # geometric mean with smoothing
        log_sum = sum(math.log(p + 1e-12) for p in precisions) / n
        bleu_prec = math.exp(log_sum)
        bp = math.exp(1 - (len(ref_tokens) / len(cand_tokens))) if len(cand_tokens) < len(ref_tokens) else 1.0
        return bp * bleu_prec

    @staticmethod
    def calculate_rouge(reference: str, candidate: str, use_lemma: bool = False) -> float:
        ref_tokens = RAGMetrics._tokenize_cached(reference, use_lemma)
        cand_tokens = RAGMetrics._tokenize_cached(candidate, use_lemma)
        if not ref_tokens:
            return 0.0
        ref_counts = Counter(ref_tokens)
        cand_counts = Counter(cand_tokens)
        overlap = sum(min(ref_counts[t], cand_counts[t]) for t in ref_counts)
        return overlap / len(ref_tokens)

    @staticmethod
    def calculate_rouge_l(reference: str, candidate: str, use_lemma: bool = False) -> float:
        ref = RAGMetrics._tokenize_cached(reference, use_lemma)
        cand = RAGMetrics._tokenize_cached(candidate, use_lemma)
        if not ref or not cand:
            return 0.0
        # LCS dynamic programming (optimized memory)
        m, n = len(ref), len(cand)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            ri = ref[i - 1]
            for j in range(1, n + 1):
                if ri == cand[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
            prev = curr
        lcs = prev[-1]
        return lcs / m

    @staticmethod
    def calculate_entity_recall(expected_entities: List[str], candidate: str) -> float:
        if not expected_entities:
            return 0.0
        cand = candidate.lower()
        found = sum(1 for e in expected_entities if e and e.lower() in cand)
        return found / len(expected_entities)

    @staticmethod
    def bertscore_placeholder(reference: str, candidate: str, use_lemma: bool = False) -> float:
        # Lightweight semantic proxy: cosine of token overlap ratio
        ref_set = set(RAGMetrics._tokenize_cached(reference, use_lemma))
        cand_set = set(RAGMetrics._tokenize_cached(candidate, use_lemma))
        if not ref_set or not cand_set:
            return 0.0
        inter = len(ref_set & cand_set)
        denom = math.sqrt(len(ref_set) * len(cand_set))
        return inter / denom if denom > 0 else 0.0

    # -------------------------
    # Batch helpers (parallel)
    # -------------------------
    @staticmethod
    def _pair_worker(args: Tuple[str, str, bool]) -> Dict[str, float]:
        ref, cand, use_lemma = args
        return {
            "f1": RAGMetrics.calculate_f1(ref, cand, use_lemma),
            "jaccard": RAGMetrics.calculate_jaccard(ref, cand, use_lemma),
            "rouge_l": RAGMetrics.calculate_rouge_l(ref, cand, use_lemma),
            "bleu": RAGMetrics.calculate_bleu(ref, cand, n=1, use_lemma=use_lemma)
        }

    @staticmethod
    def batch_evaluate(pairs: List[Tuple[str, str]], use_lemma: bool = False, processes: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Evaluate a list of (reference, candidate) pairs in parallel.
        Returns list of metric dicts in same order.
        """
        if not pairs:
            return []
        procs = processes or max(1, min(cpu_count() - 1, 4))
        args = [(r, c, use_lemma) for r, c in pairs]
        if procs == 1:
            return [RAGMetrics._pair_worker(a) for a in args]
        with Pool(processes=procs) as p:
            return p.map(RAGMetrics._pair_worker, args)