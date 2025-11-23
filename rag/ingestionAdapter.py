# ingestionAdapter.py
from typing import List, Dict, Any, Tuple

def _to_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

class IngestionRouter:
    def __init__(self):
        self.adapters = {
            "climat": self._climat_adapter,
            "bulletin": self._bulletin_adapter,
            "bulletins": self._bulletin_adapter
        }

    def route(self, doc: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
        fname = (filename or "").lower()
        for key, adapter in self.adapters.items():
            if key in fname:
                return adapter(doc, filename)
        # heuristique par structure
        if isinstance(doc, list):
            # probablement bulletin (liste de pages)
            return self._bulletin_adapter(doc, filename)
        if isinstance(doc, dict) and any(isinstance(v, dict) for v in doc.values()):
            return self._climat_adapter(doc, filename)
        return []

    def _climat_adapter(self, doc: Dict[str, Any], filename: str = None) -> List[Dict[str, Any]]:
        records = []
        for city, payload in doc.items():
            if not isinstance(payload, dict):
                continue
            mins = payload.get("Temp. min") or payload.get("Temp. min.") or payload.get("Température minimale") or {}
            maxs = payload.get("Temp. max") or payload.get("Temp. max.") or payload.get("Température maximale") or {}
            precs = payload.get("Précipitations") or payload.get("Precipitations") or {}
            months = set(list(mins.keys()) + list(maxs.keys()) + list(precs.keys()))
            for m in months:
                records.append({
                    "id": f"{city}_{m}",
                    "source_file": filename,
                    "source": "climat",
                    "city": city,
                    "month": m,
                    "temp_min": _to_float(mins.get(m)),
                    "temp_max": _to_float(maxs.get(m)),
                    "precipitation": _to_float(precs.get(m))
                })
        return records

    def _bulletin_adapter(self, doc: Any, filename: str = None) -> List[Dict[str, Any]]:
        """
        Gère deux cas :
        - doc est une liste de pages (comme l'exemple fourni)
        - doc est un dict unique représentant une page/bulletin
        Chaque page devient un record avec : id, source_file, source, page, text, images, short_text
        """
        records = []

        def make_short(text: str, max_chars: int = 500) -> str:
            t = text.strip().replace("\n\n", "\n")
            return t[:max_chars]

        # cas liste de pages
        if isinstance(doc, list):
            for item in doc:
                if not isinstance(item, dict):
                    continue
                page = item.get("page")
                text = item.get("text", "")
                images = item.get("images", []) or []
                records.append({
                    "id": f"{filename}_page_{page}" if filename else f"page_{page}",
                    "source_file": filename,
                    "source": "bulletin",
                    "page": page,
                    "text": text,
                    "short_text": make_short(text, 1000),
                    "images": images
                })
            return records

        # cas dict unique (une page)
        if isinstance(doc, dict):
            page = doc.get("page")
            text = doc.get("text", "") or doc.get("content", "")
            images = doc.get("images", []) or []
            records.append({
                "id": f"{filename}_page_{page}" if filename else f"page_{page}",
                "source_file": filename,
                "source": "bulletin",
                "page": page,
                "text": text,
                "short_text": make_short(text, 1000),
                "images": images
            })
            return records

        return records