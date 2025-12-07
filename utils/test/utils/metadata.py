# rag/metadata.py
from __future__ import annotations
import re
import json
import uuid
import time
import logging
import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

_logger = logging.getLogger("rag.metadata")
_logger.addHandler(logging.NullHandler())

# --- Types ---
Meta = Dict[str, Any]

# --- Helpers ---
ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[Tt ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?)?$"
)

def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return str(v)
    except Exception:
        return json.dumps(v, ensure_ascii=False)

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def generate_id(prefix: Optional[str] = None) -> str:
    uid = uuid.uuid4().hex
    return f"{prefix}_{uid}" if prefix else uid

# --- Canonical metadata dataclass ---
@dataclass
class CanonicalMeta:
    # Core fields commonly used across agents
    id: str
    source: str
    source_type: Optional[str] = None
    collection: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    published_at: Optional[str] = None  # ISO8601
    retrieved_at: str = field(default_factory=now_iso)
    region: Optional[str] = None
    forecast_time: Optional[str] = None  # ISO8601 for weather forecasts
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    raw_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Meta:
        return asdict(self)

# --- Validation and normalization functions ---
def parse_iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.replace(microsecond=0).isoformat() + "Z"
    s = safe_str(value).strip()
    if ISO_DATE_RE.match(s):
        # try to normalize timezone-less dates to full ISO
        try:
            if "T" not in s and " " not in s:
                # date only
                dt = datetime.datetime.fromisoformat(s)
                return dt.replace(microsecond=0).isoformat() + "Z"
            # parse with fromisoformat (Python 3.11+ handles offsets)
            dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            pass
    # fallback: try common formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            continue
    return None

def normalize_tags(tags: Optional[Iterable[Any]]) -> List[str]:
    if not tags:
        return []
    out = []
    for t in tags:
        s = normalize_whitespace(safe_str(t)).lower()
        if s:
            out.append(s)
    # dedupe while preserving order
    seen = set()
    res = []
    for t in out:
        if t not in seen:
            seen.add(t)
            res.append(t)
    return res

def normalize_region(region: Any) -> Optional[str]:
    if not region:
        return None
    s = normalize_whitespace(safe_str(region))
    return s

# --- Provenance utilities ---
def build_provenance(source: str, extractor: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = {
        "source": source,
        "extracted_at": now_iso(),
        "extractor": extractor or "ingestor",
    }
    if extra:
        p.update(extra)
    return p

# --- Merge and canonicalize raw metadata into CanonicalMeta ---
def canonicalize_meta(raw_meta: Meta, source: str, prefer: Optional[Dict[str, Any]] = None) -> CanonicalMeta:
    prefer = prefer or {}
    # id
    raw_id = raw_meta.get("raw_id") or raw_meta.get("id") or generate_id("doc")
    # title heuristics
    title = raw_meta.get("title") or raw_meta.get("headline") or raw_meta.get("name")
    if title:
        title = normalize_whitespace(safe_str(title))
    # author
    author = raw_meta.get("author") or raw_meta.get("byline")
    if author:
        author = normalize_whitespace(safe_str(author))
    # language
    language = raw_meta.get("language") or raw_meta.get("lang")
    if language:
        language = safe_str(language).lower()
    # dates
    published = parse_iso_date(raw_meta.get("published_at") or raw_meta.get("date") or raw_meta.get("pub_date"))
    retrieved = parse_iso_date(raw_meta.get("retrieved_at")) or now_iso()
    forecast_time = parse_iso_date(raw_meta.get("forecast_time"))
    valid_from = parse_iso_date(raw_meta.get("valid_from"))
    valid_to = parse_iso_date(raw_meta.get("valid_to"))
    # tags and region
    tags = normalize_tags(raw_meta.get("tags") or raw_meta.get("keywords"))
    region = normalize_region(raw_meta.get("region") or raw_meta.get("area") or raw_meta.get("location"))
    # collection
    collection = raw_meta.get("collection") or prefer.get("collection")
    # provenance
    prov = raw_meta.get("provenance") or build_provenance(source, raw_meta.get("extractor"))
    # merge raw_meta copy
    raw_copy = dict(raw_meta)
    # build canonical
    cm = CanonicalMeta(
        id=safe_str(raw_id),
        source=source,
        source_type=safe_str(raw_meta.get("source_type") or raw_meta.get("type") or ""),
        collection=safe_str(collection) if collection else None,
        title=title,
        author=author,
        language=language,
        published_at=published,
        retrieved_at=retrieved,
        region=region,
        forecast_time=forecast_time,
        valid_from=valid_from,
        valid_to=valid_to,
        tags=tags,
        provenance=prov,
        raw_meta=raw_copy,
    )
    return cm

# --- Field mapping for agents (6 agents support) ---
# Provide mapping templates per agent to extract or rename fields for downstream use.
AGENT_FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "agent_forecast": {
        "id": "id",
        "source": "source",
        "forecast_time": "forecast_time",
        "valid_from": "valid_from",
        "valid_to": "valid_to",
        "region": "region",
        "title": "title",
        "tags": "tags",
    },
    "agent_bulletin": {
        "id": "id",
        "source": "source",
        "published_at": "published_at",
        "title": "title",
        "author": "author",
        "tags": "tags",
    },
    "agent_climatology": {
        "id": "id",
        "source": "source",
        "published_at": "published_at",
        "region": "region",
        "tags": "tags",
    },
    "agent_observation": {
        "id": "id",
        "source": "source",
        "published_at": "published_at",
        "region": "region",
        "raw_meta": "raw_meta",
    },
    "agent_alerts": {
        "id": "id",
        "source": "source",
        "valid_from": "valid_from",
        "valid_to": "valid_to",
        "region": "region",
        "tags": "tags",
    },
    "agent_general": {
        "id": "id",
        "source": "source",
        "title": "title",
        "author": "author",
        "published_at": "published_at",
        "tags": "tags",
    },
}

def map_for_agent(cm: CanonicalMeta, agent_name: str) -> Meta:
    mapping = AGENT_FIELD_MAPS.get(agent_name, AGENT_FIELD_MAPS["agent_general"])
    out: Meta = {}
    d = cm.to_dict()
    for out_key, cm_key in mapping.items():
        out[out_key] = d.get(cm_key)
    # always include provenance and raw_meta minimally
    out.setdefault("provenance", cm.provenance)
    out.setdefault("raw_meta", cm.raw_meta)
    return out

# --- Validation utilities ---
def validate_canonical_meta(cm: CanonicalMeta) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not cm.id:
        errors.append("missing id")
    if not cm.source:
        errors.append("missing source")
    # dates sanity
    if cm.published_at and not parse_iso_date(cm.published_at):
        errors.append("published_at not ISO")
    if cm.valid_from and not parse_iso_date(cm.valid_from):
        errors.append("valid_from not ISO")
    if cm.valid_to and not parse_iso_date(cm.valid_to):
        errors.append("valid_to not ISO")
    # forecast_time should be ISO if present
    if cm.forecast_time and not parse_iso_date(cm.forecast_time):
        errors.append("forecast_time not ISO")
    return (len(errors) == 0, errors)

# --- Merge strategy for metadata updates (idempotent) ---
def merge_meta(existing: Meta, incoming: Meta, prefer_incoming: bool = True) -> Meta:
    """
    Merge two metadata dicts. If prefer_incoming True, incoming values override existing when present.
    - Lists are merged and deduped.
    - Dicts are merged recursively shallowly.
    """
    out = dict(existing or {})
    for k, v in (incoming or {}).items():
        if v is None:
            continue
        if k not in out:
            out[k] = v
            continue
        # both present
        ev = out[k]
        if isinstance(ev, list) and isinstance(v, list):
            # merge lists dedup preserving order
            seen = set(ev)
            merged = list(ev)
            for item in v:
                if item not in seen:
                    merged.append(item)
                    seen.add(item)
            out[k] = merged
        elif isinstance(ev, dict) and isinstance(v, dict):
            # shallow merge
            merged = dict(ev)
            for kk, vv in v.items():
                if prefer_incoming or kk not in merged:
                    merged[kk] = vv
            out[k] = merged
        else:
            out[k] = v if prefer_incoming else ev
    return out

# --- Export helpers for indexer / storage ---
def to_index_meta(cm: CanonicalMeta, minimal: bool = True) -> Meta:
    """
    Convert canonical meta to a flat dict suitable for indexing.
    If minimal True, include only essential fields to reduce index size.
    """
    d = cm.to_dict()
    if minimal:
        return {
            "id": d["id"],
            "source": d["source"],
            "collection": d.get("collection"),
            "title": d.get("title"),
            "language": d.get("language"),
            "published_at": d.get("published_at"),
            "retrieved_at": d.get("retrieved_at"),
            "region": d.get("region"),
            "tags": d.get("tags"),
            "provenance": d.get("provenance"),
        }
    # full export
    return d

# --- Utilities for geolocation parsing (simple) ---
COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)[,;\s]+(-?\d+(?:\.\d+)?)")

def parse_coords(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    s = safe_str(value)
    m = COORD_RE.search(s)
    if not m:
        return None
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
        return (lat, lon)
    except Exception:
        return None

# --- Small utilities for debugging / pretty print ---
def pretty_meta(meta: Meta, max_keys: int = 12) -> str:
    keys = list(meta.keys())[:max_keys]
    return json.dumps({k: meta[k] for k in keys}, ensure_ascii=False, indent=2)

# --- Example pipeline helper ---
def enrich_and_canonicalize(raw_doc: Dict[str, Any], source: str, agent_hint: Optional[str] = None, prefer: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience helper used by Ingestor/TextBuilder pipeline:
    - canonicalize raw_doc['meta']
    - validate canonical meta
    - map for agent if agent_hint provided
    Returns dict with canonical_meta, valid flag, errors, mapped_meta
    """
    raw_meta = raw_doc.get("meta", {}) or {}
    cm = canonicalize_meta(raw_meta, source=source, prefer=prefer)
    valid, errors = validate_canonical_meta(cm)
    mapped = map_for_agent(cm, agent_hint or "agent_general")
    return {"canonical": cm.to_dict(), "valid": valid, "errors": errors, "mapped": mapped}

# --- End of module ---