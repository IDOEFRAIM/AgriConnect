import unittest
from unittest.mock import MagicMock

from rag.components.re_ranker import Reranker

class TestReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = Reranker()
        # Mock internal encoder to avoid downloading models or slow inference
        self.reranker.encoder.predict = MagicMock(return_value=[0.8, 0.2])

    def test_rerank_logic(self):
        docs = [
            {"text": "Relevant Doc", "score": 5.0}, # Vector score (L2 Distance -> Lower is better)
            {"text": "Irrelevant Doc", "score": 25.0} # High distance -> Bad match
        ]
        
        # Test METEO profile
        reranked = self.reranker.rerank(docs, "METEO", "Pluie prévue ?")
        
        self.assertEqual(len(reranked), 2)
        # We expect normalization and reordering.
        # Since encoder returns [0.8, 0.2] (Simulated), the first doc is semantic winner.
        # And with score 5.0 vs 25.0, it is also the vector winner.
        # So it MUST win.
        self.assertTrue(reranked[0]["text"] == "Relevant Doc")
        
        # Verify normalization (score between 0 and 1)
        score = reranked[0]["score"]
        self.assertTrue(0.0 <= score <= 1.0, f"Score {score} out of range")

    def test_unknown_role(self):
        docs = [{"score": 10}, {"score": 20}]
        reranked = self.reranker.rerank(docs, "UNKNOWN_ROLE", "Query")
        # Should fallback to sort by vector score desc
        self.assertEqual(reranked[0]["score"], 20)

if __name__ == '__main__':
    unittest.main()
