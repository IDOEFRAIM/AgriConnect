# rag/text_builder.py
from __future__ import annotations
import re
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("rag.text_builder")
_logger.addHandler(logging.NullHandler())

# -------------------------
# Configuration dataclass
# -------------------------
@dataclass
class TBConfig:
    chunk_size_tokens: int = 200
    chunk_overlap_tokens: int = 40
    min_chunk_length: int = 40
    min_tokens_per_chunk: int = 8
    chunk_id_prefix: str = "chunk"
    chunk_metadata_fields: List[str] = field(default_factory=lambda: ["source", "raw_id"])
    lowercase: bool = False
    strip_html: bool = True
    preserve_paragraphs: bool = True
    max_chunks_per_doc: Optional[int] = None

# -------------------------
# Tokenizer class
# -------------------------
class SimpleTokenizer:
    """
    Tokenizer léger et encapsulé. Instanciable pour permettre configuration future.
    """
    def __init__(self, token_pattern: str = r"\w+|[^\w\s]"):
        self.token_re = re.compile(token_pattern, flags=re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return self.token_re.findall(text)

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

# -------------------------
# Normalisation et split
# -------------------------
def _strip_html(text: str) -> str:
    # enlève scripts/styles puis balises
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r'\n{2,}|\r\n{2,}', text)
    return [p.strip() for p in parts if p.strip()]

def _split_sentences(text: str) -> List[str]:
    # séparation naïve mais robuste pour chunking
    parts = re.split(r'(?<=[.!?…])\s+', text)
    return [p.strip() for p in parts if p.strip()]

# -------------------------
# TextBuilder principal
# -------------------------
class TextBuilder:
    """
    TextBuilder robuste et testable.
    - Instancie son tokenizer dans __init__
    - Expose preprocess_text, chunk_text, build_records, inspect_raw_doc, debug_build_records
    """

    def __init__(self, cfg: Optional[TBConfig] = None, tokenizer: Optional[SimpleTokenizer] = None):
        self.cfg = cfg or TBConfig()
        self.tokenizer = tokenizer or SimpleTokenizer()
        # validation simple des paramètres
        if self.cfg.chunk_size_tokens < 1:
            raise ValueError("chunk_size_tokens must be >= 1")
        if self.cfg.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be >= 0")

    # -------------------------
    # Prétraitement
    # -------------------------
    def preprocess_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        t = text
        if self.cfg.strip_html:
            t = _strip_html(t)
        t = _normalize_whitespace(t)
        if self.cfg.lowercase:
            t = t.lower()
        return t

    # -------------------------
    # Chunking
    # -------------------------
    def chunk_text(self, text: str) -> List[str]:
        """
        Découpe en chunks basés sur paragraphes/sentences puis fenêtres de tokens.
        Retourne liste de chaînes (chunks).
        """
        if not text:
            return []

        cleaned = self.preprocess_text(text)
        if not cleaned:
            return []

        pieces: List[str] = []
        if self.cfg.preserve_paragraphs:
            paras = _split_paragraphs(cleaned)
            for p in paras:
                sents = _split_sentences(p)
                if len(sents) <= 1:
                    pieces.append(p)
                else:
                    # regroupe des phrases pour approcher chunk_size_tokens
                    current: List[str] = []
                    current_tokens = 0
                    for s in sents:
                        tcount = self.tokenizer.count(s)
                        if current and (current_tokens + tcount) > self.cfg.chunk_size_tokens:
                            pieces.append(" ".join(current))
                            current = [s]
                            current_tokens = tcount
                        else:
                            current.append(s)
                            current_tokens += tcount
                    if current:
                        pieces.append(" ".join(current))
        else:
            pieces = [cleaned]

        chunks: List[str] = []
        chunk_size = max(1, int(self.cfg.chunk_size_tokens))
        overlap = max(0, int(min(self.cfg.chunk_overlap_tokens, chunk_size - 1)))

        for piece in pieces:
            tokens = self.tokenizer.tokenize(piece)
            n = len(tokens)
            if n == 0:
                continue
            start = 0
            while start < n:
                end = start + chunk_size
                window = tokens[start:end]
                chunk_text = " ".join(window).strip()
                if len(chunk_text) >= self.cfg.min_chunk_length and len(window) >= self.cfg.min_tokens_per_chunk:
                    chunks.append(chunk_text)
                    if self.cfg.max_chunks_per_doc and len(chunks) >= self.cfg.max_chunks_per_doc:
                        return chunks
                if end >= n:
                    break
                start = end - overlap

        # fallback si aucun chunk produit
        if not chunks:
            fallback = cleaned.strip()
            if len(fallback) >= self.cfg.min_chunk_length:
                chunks.append(fallback)
        return chunks

    # -------------------------
    # Construction des enregistrements (records)
    # -------------------------
    def build_records(self, raw_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        raw_doc attendu: {raw_id, source, text, meta}
        Retourne: liste de records {chunk_id, doc_id, text, meta}
        """
        raw_id = raw_doc.get("raw_id") or raw_doc.get("id") or str(uuid.uuid4())
        source = raw_doc.get("source") or (raw_doc.get("meta") or {}).get("source") or ""
        meta = dict(raw_doc.get("meta") or {})

        text_candidate = raw_doc.get("text") or meta.get("text") or meta.get("content") or ""
        text = self.preprocess_text(text_candidate)
        if not text:
            return []

        chunks = self.chunk_text(text)
        records: List[Dict[str, Any]] = []
        for i, c in enumerate(chunks):
            chunk_id = f"{self.cfg.chunk_id_prefix}_{raw_id}_{i}"
            rec_meta = dict(meta)
            rec_meta.setdefault("source", source)
            rec_meta.setdefault("raw_id", raw_id)
            rec_meta.setdefault("chunk_index", i)
            rec = {
                "chunk_id": chunk_id,
                "doc_id": raw_id,
                "text": c,
                "meta": rec_meta,
            }
            records.append(rec)
        return records

    # -------------------------
    # Diagnostics
    # -------------------------
    def inspect_raw_doc(self, raw_doc: Dict[str, Any], max_preview_chars: int = 800) -> Dict[str, Any]:
        meta = raw_doc.get("meta") or {}
        candidates = {
            "raw_text": raw_doc.get("text"),
            "meta_text": meta.get("text"),
            "meta_content": meta.get("content"),
        }
        chosen = raw_doc.get("text") or meta.get("text") or meta.get("content") or ""
        cleaned = self.preprocess_text(chosen)
        token_count = self.tokenizer.count(cleaned)
        chunks = self.chunk_text(cleaned)
        diag = {
            "raw_id": raw_doc.get("raw_id") or raw_doc.get("id"),
            "source": raw_doc.get("source") or meta.get("source"),
            "meta_keys": list(meta.keys()),
            "text_candidates": {k: (v[:max_preview_chars] + "..." if isinstance(v, str) and len(v) > max_preview_chars else v) for k, v in candidates.items()},
            "chosen_preview": (chosen[:max_preview_chars] + "...") if isinstance(chosen, str) and len(chosen) > max_preview_chars else chosen,
            "cleaned_preview": (cleaned[:max_preview_chars] + "...") if len(cleaned) > max_preview_chars else cleaned,
            "cleaned_length": len(cleaned),
            "token_count": token_count,
            "chunks_produced": len(chunks),
            "chunks_preview": [c[:max_preview_chars] for c in chunks[:5]],
            "tb_config": asdict(self.cfg),
        }
        return diag

    def debug_build_records(self, raw_doc: Dict[str, Any], verbose: bool = True) -> List[Dict[str, Any]]:
        diag = self.inspect_raw_doc(raw_doc)
        if verbose:
            print("=== TextBuilder.inspect_raw_doc ===")
            for k, v in diag.items():
                if k in ("text_candidates", "chunks_preview"):
                    print(f"{k}:")
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            print(f"  - {kk}: {vv}")
                    else:
                        for idx, vv in enumerate(v):
                            print(f"  - {idx}: {vv}")
                else:
                    print(f"{k}: {v}")
            print("=== End inspect ===\n")
        recs = list(self.build_records(raw_doc))
        if verbose:
            print(f"Records produced: {len(recs)}")
            for r in recs[:5]:
                print(f"- {r['chunk_id']} (len {len(r['text'])} chars)")
        return recs