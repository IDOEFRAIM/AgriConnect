"""Parallel executor helper for MessageResponseFlow.

Encapsulates threadpool execution and task composition for experts.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from backend.src.agriconnect.graphs.message_flow_helpers import ExpertInvoker
from backend.src.agriconnect.graphs.state import ExpertResponse
import logging

logger = logging.getLogger(__name__)


class ParallelExecutor:
    def __init__(self, invoker: ExpertInvoker):
        self.invoker = invoker

    def run(
        self,
        needs: Dict[str, bool],
        query: str,
        zone: str,
        crop: str,
        user_level: str,
    ) -> List[ExpertResponse]:
        """Run requested experts in parallel and return ExpertResponse list."""
        # Determine leader
        if needs.get("needs_formation"):
            lead = "formation"
        elif needs.get("needs_market"):
            lead = "market"
        else:
            lead = "sentinelle"

        tasks = {}
        if needs.get("needs_sentinelle", True):
            tasks["sentinelle"] = lambda: self.invoker.call_sentinelle(query, zone, user_level)
        if needs.get("needs_formation"):
            tasks["formation"] = lambda: self.invoker.call_formation(query, crop, user_level)
        if needs.get("needs_market"):
            tasks["market"] = lambda: self.invoker.call_market(query, zone, user_level)

        if not tasks:
            tasks["sentinelle"] = lambda: self.invoker.call_sentinelle(query, zone, user_level)

        expert_responses: List[ExpertResponse] = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result(timeout=30)
                    resp_text = result.get("final_response", "")
                    has_alerts = bool(result.get("hazards"))
                    expert_responses.append(
                        ExpertResponse(
                            expert=name,
                            response=resp_text,
                            is_lead=(name == lead),
                            has_alerts=has_alerts,
                        )
                    )
                except Exception as e:
                    logger.warning("Expert %s failed: %s", name, e)
                    expert_responses.append(
                        ExpertResponse(
                            expert=name,
                            response=f"[{name}] Données indisponibles.",
                            is_lead=(name == lead),
                            has_alerts=False,
                        )
                    )

        return expert_responses
