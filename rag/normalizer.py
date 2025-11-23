# normalizer.py
from typing import List, Dict, Callable, Any
import uuid

class Normalizer:
    def __init__(self, adapter: Callable[[Dict[str, Any]], List[Dict[str, Any]]]):
        """
        Args:
            adapter: fonction qui prend un document brut (dict) et retourne une liste d'enregistrements plats.
        """
        self.adapter = adapter

    def normalize_batch(self, raw_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applique l'adaptateur à une liste de documents bruts.

        Args:
            raw_documents: liste de dicts bruts (ex: JSON, CSV row)

        Returns:
            Liste d'enregistrements plats avec champs standardisés.
        """
        normalized = []
        for doc in raw_documents:
            try:
                records = self.adapter(doc)
                for r in records:
                    if "id" not in r:
                        r["id"] = str(uuid.uuid4())
                    normalized.append(r)
            except Exception as e:
                print("Erreur normalisation:", e)
        return normalized