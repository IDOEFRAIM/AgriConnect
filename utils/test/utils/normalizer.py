# rag/normalizers.py
from __future__ import annotations
import re
import unicodedata
import html
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

_logger = logging.getLogger("rag.normalizers")
_logger.addHandler(logging.NullHandler())

# --- Base interface ---
class NormalizerBase:
    """
    Base class for normalizers. Implement `normalize` to return normalized text.
    Optionally implement `diagnostics` to return metadata about the operation.
    """
    def normalize(self, text: str) -> str:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, Any]:
        return {}

# --- Simple normalizers ---
class UnicodeNormalizer(NormalizerBase):
    """Normalize unicode to NFC form."""
    def normalize(self, text: str) -> str:
        try:
            return unicodedata.normalize("NFC", text)
        except Exception:
            return text

class WhitespaceNormalizer(NormalizerBase):
    """Collapse whitespace and trim."""
    def __init__(self, collapse_newlines: bool = True):
        self.collapse_newlines = collapse_newlines
        self._ops = []

    def normalize(self, text: str) -> str:
        t = text
        if self.collapse_newlines:
            t = re.sub(r"\s+", " ", t)
            self._ops.append("collapse_whitespace")
        else:
            t = re.sub(r"[ \t]+", " ", t)
            self._ops.append("collapse_spaces")
        return t.strip()

    def diagnostics(self) -> Dict[str, Any]:
        return {"ops": list(self._ops)}

class ControlCharStripper(NormalizerBase):
    """Remove control characters except common whitespace."""
    def normalize(self, text: str) -> str:
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", text)

class HtmlStripNormalizer(NormalizerBase):
    """Very small HTML stripper. For production use bleach or lxml."""
    SCRIPT_RE = re.compile(r"<script.*?>.*?</script>", flags=re.S | re.I)
    TAG_RE = re.compile(r"<[^>]+>")
    ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")

    def normalize(self, text: str) -> str:
        t = text
        t = self.SCRIPT_RE.sub(" ", t)
        t = self.TAG_RE.sub(" ", t)
        # unescape HTML entities
        try:
            t = html.unescape(t)
        except Exception:
            pass
        t = re.sub(r"\s+", " ", t).strip()
        return t

class PunctuationNormalizer(NormalizerBase):
    """Normalize repeated punctuation and fix common unicode punctuation to ASCII."""
    PUNCT_MAP = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "..."
    }
    def normalize(self, text: str) -> str:
        t = text
        for k, v in self.PUNCT_MAP.items():
            t = t.replace(k, v)
        # collapse repeated punctuation like "!!!" -> "!"
        t = re.sub(r"([!?.,;:])\1{1,}", r"\1", t)
        return t

# --- Language-aware / advanced normalizers (optional dependencies) ---
class SentenceSplitterNormalizer(NormalizerBase):
    """
    Sentence-aware normalizer that preserves sentence boundaries.
    Uses regex fallback; if spaCy is available and language provided, it will use spaCy.
    """
    def __init__(self, language: Optional[str] = None):
        self.language = language
        self._use_spacy = False
        self._nlp = None
        if language:
            try:
                import spacy
                # attempt to load small model for language if available
                try:
                    self._nlp = spacy.load(language)
                    self._use_spacy = True
                except Exception:
                    # try generic multi-language model
                    try:
                        self._nlp = spacy.load("xx_sent_ud_sm")
                        self._use_spacy = True
                    except Exception:
                        self._use_spacy = False
            except Exception:
                self._use_spacy = False

    def normalize(self, text: str) -> str:
        if self._use_spacy and self._nlp:
            doc = self._nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return " ".join(sentences)
        # fallback: simple sentence split on punctuation followed by space + capital letter or newline
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
        parts = [p.strip() for p in parts if p.strip()]
        return " ".join(parts)

class LanguageAwareNormalizer(NormalizerBase):
    """
    Placeholder for language-specific normalization pipelines.
    Inject language-specific tokenizers or rules as needed.
    """
    def __init__(self, language: Optional[str] = None):
        self.language = language

    def normalize(self, text: str) -> str:
        # Minimal behavior: lowercasing for languages where appropriate
        if self.language and self.language.lower().startswith("fr"):
            return text  # keep case for French by default
        return text.lower()

# --- PII redaction ---
class PIIRedactor(NormalizerBase):
    """
    Redact common PII patterns: emails, phone numbers, simple IDs.
    This is conservative and configurable.
    """
    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")
    ID_RE = re.compile(r"\b[A-Z0-9]{6,}\b")

    def __init__(self, redact_email: bool = True, redact_phone: bool = True, redact_ids: bool = False, placeholder: str = "[REDACTED]"):
        self.redact_email = redact_email
        self.redact_phone = redact_phone
        self.redact_ids = redact_ids
        self.placeholder = placeholder

    def normalize(self, text: str) -> str:
        t = text
        if self.redact_email:
            t = self.EMAIL_RE.sub(self.placeholder, t)
        if self.redact_phone:
            t = self.PHONE_RE.sub(self.placeholder, t)
        if self.redact_ids:
            t = self.ID_RE.sub(self.placeholder, t)
        return t

# --- Composite normalizer to chain steps ---
class CompositeNormalizer(NormalizerBase):
    """
    Compose multiple NormalizerBase instances into a pipeline.
    Each step receives the output of the previous step.
    """
    def __init__(self, steps: Iterable[NormalizerBase]):
        self.steps: List[NormalizerBase] = list(steps)

    def normalize(self, text: str) -> str:
        t = text
        for step in self.steps:
            try:
                t = step.normalize(t)
            except Exception:
                _logger.exception("normalizer step failed: %s", type(step).__name__)
        return t

    def diagnostics(self) -> Dict[str, Any]:
        return {"steps": [type(s).__name__ for s in self.steps]}

# --- Factory helpers ---
def default_normalizer_for_agent(agent_name: Optional[str] = None) -> NormalizerBase:
    """
    Return a sensible default CompositeNormalizer for common agent types.
    """
    if agent_name and agent_name.startswith("agent_bulletin"):
        return CompositeNormalizer([
            UnicodeNormalizer(),
            HtmlStripNormalizer(),
            WhitespaceNormalizer(collapse_newlines=False),
            PunctuationNormalizer(),
            PIIRedactor(redact_email=True, redact_phone=True, redact_ids=False),
        ])
    if agent_name and agent_name.startswith("agent_forecast"):
        return CompositeNormalizer([
            UnicodeNormalizer(),
            HtmlStripNormalizer(),
            WhitespaceNormalizer(collapse_newlines=True),
            SentenceSplitterNormalizer(language="fr_core_news_sm" if agent_name.endswith("_fr") else None),
            PunctuationNormalizer(),
        ])
    # generic default
    return CompositeNormalizer([
        UnicodeNormalizer(),
        HtmlStripNormalizer(),
        ControlCharStripper(),
        WhitespaceNormalizer(),
        PunctuationNormalizer(),
    ])

# --- Small utilities for testing and quick use ---
def normalize_text(text: str, normalizer: Optional[NormalizerBase] = None) -> str:
    normalizer = normalizer or default_normalizer_for_agent(None)
    return normalizer.normalize(text)

# --- End of module ---