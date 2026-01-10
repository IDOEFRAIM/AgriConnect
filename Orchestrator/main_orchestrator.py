import logging
import time
from typing import Dict, Any
from orchestrator.state import GlobalAgriState
from orchestrator.message_flow import MessageResponseFlow
from orchestrator.report_flow import DailyReportFlow

logger = logging.getLogger("MainOrchestrator")

class MainOrchestrator:
    def __init__(self):
        self.message_flow = MessageResponseFlow().build_graph()
        self.report_flow = DailyReportFlow().build_graph()

    def run(self, initial_state: GlobalAgriState) -> Dict[str, Any]:
        """
        Main entry point. Routes execution based on flow_type.
        """
        start_time = time.time()
        flow_type = initial_state.get("flow_type", "MESSAGE")
        logger.info(f"🚀 Starting Orchestrator with flow: {flow_type}")

        result = {}
        if flow_type == "MESSAGE":
            # Run the interactive message flow
            result = self.message_flow.invoke(initial_state)
            
        elif flow_type == "REPORT":
            # Run the proactive report flow
            result = self.report_flow.invoke(initial_state)
            
        else:
            logger.error(f"Unknown flow type: {flow_type}")
            result = {"error": "Unknown flow type"}
            
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000 # ms
        logger.info(f"⏱️ Execution time: {execution_time:.2f} ms")
        
        # Add execution time to result if it's a dict
        if isinstance(result, dict):
            result["execution_time_ms"] = execution_time
            
        return result

