# evaluator.py
from typing import List, Dict
import difflib

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, query: str, response: str, sources: List[Dict]) -> Dict:
        """
        Évalue la qualité d'une réponse RAG.

        Args:
            query: la question posée
            response: la réponse générée
            sources: liste de documents récupérés (avec 'excerpt' ou 'document')

        Returns:
            Dict avec scores : faithfulness, relevance, coverage
        """
        source_texts = [s.get("excerpt") or s.get("document", "") for s in sources]
        combined_sources = "\n".join(source_texts).lower()
        response_lower = response.lower()

        return {
            "faithfulness": self._score_faithfulness(response_lower, combined_sources),
            "source_coverage": self._score_coverage(response_lower, source_texts),
            "relevance": self._score_relevance(query.lower(), response_lower)
        }

    def _score_faithfulness(self, response: str, sources: str) -> float:
        """Mesure si la réponse s'appuie sur les sources (0 à 1)."""
        overlap = difflib.SequenceMatcher(None, response, sources).ratio()
        return round(overlap, 3)

    def _score_coverage(self, response: str, source_texts: List[str]) -> float:
        """Mesure combien de sources ont été partiellement utilisées."""
        used = 0
        for src in source_texts:
            if any(word in response for word in src.split()[:10]):
                used += 1
        return round(used / len(source_texts), 3) if source_texts else 0.0

    def _score_relevance(self, query: str, response: str) -> float:
        """Mesure la similarité entre la question et la réponse."""
        return round(difflib.SequenceMatcher(None, query, response).ratio(), 3)