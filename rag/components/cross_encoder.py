import math
import logging
from typing import List, Tuple

from sentence_transformers.cross_encoder import CrossEncoder as STCrossEncoder

logger = logging.getLogger("rag.cross_encoder")

class CrossEncoder:
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", batch_size: int = 32, temperature: float = 1.0):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.temperature = float(temperature)
        try:
            self.model = STCrossEncoder(self.model_name, device=self.device)
            logger.info(f"✅ CrossEncoder chargé : {self.model_name} sur {self.device}")
        except Exception as e:
            logger.critical(f"❌ Erreur lors du chargement du CrossEncoder: {e}")
            raise RuntimeError("CrossEncoder model could not be loaded.") from e

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x / max(1e-6, self.temperature)))

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        logits = self.model.predict(pairs, show_progress_bar=False, batch_size=self.batch_size)
        return [self._sigmoid(float(l)) for l in logits]

    def predict_score(self, query: str, context: str) -> float:
        return self.predict_batch([(query, context)])[0]