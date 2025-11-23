# log_builder.py
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

class Logger:
    def __init__(self, log_path: str = "logs/rag_log.jsonl"):
        """
        Args:
            log_path: chemin du fichier de log (JSON Lines)
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        query: str,
        response: str,
        sources: List[Dict],
        scores: Optional[Dict] = None,
        prompt: Optional[str] = None,
        agent: Optional[str] = None
    ):
        """
        Enregistre une interaction RAG dans le fichier de log.

        Args:
            query: question utilisateur
            response: réponse générée
            sources: documents utilisés
            scores: dict d’évaluation (facultatif)
            prompt: prompt utilisé (facultatif)
            agent: nom de l’agent ou du domaine (facultatif)
        """
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "response": response,
            "sources": sources,
            "scores": scores,
            "prompt": prompt,
            "agent": agent
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")