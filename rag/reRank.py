# reranker.py
from typing import List, Dict
import difflib

class ReRanker:
    def __init__(self, method: str = "overlap"):
        """
        Args:
            method: stratégie de reranking ('overlap' ou 'sequence_match')
        """
        self.method = method

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Reclasse les documents selon leur pertinence par rapport à la requête.

        Args:
            query: la requête utilisateur
            docs: liste de documents avec champ 'excerpt' ou 'document'
            top_k: nombre de documents à retourner après reranking

        Returns:
            Liste de documents rerankés
        """
        scored = []
        for doc in docs:
            text = doc.get("excerpt") or doc.get("document", "")
            score = self._score(query.lower(), text.lower())
            scored.append((score, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [d for _, d in scored[:top_k]]

    def _score(self, query: str, text: str) -> float:
        if self.method == "sequence_match":
            return difflib.SequenceMatcher(None, query, text).ratio()
        else:  # default: word overlap
            q_words = set(query.split())
            t_words = set(text.split())
            if not t_words: return 0.0
            return len(q_words & t_words) / len(q_words)