import unittest
import os
import sys

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from orchestrator.main_orchestrator import MainOrchestrator

class TestOrchestratorIntegration(unittest.TestCase):
    def setUp(self):
        self.orchestrator = MainOrchestrator()

    def test_message_flow_structure(self):
        """Test if the message flow returns a properly structured dictionary."""
        msg_state_meteo = {
            "user_id": "test_mid_1",
            "zone_id": "Dédougou",
            "requete_utilisateur": "Va-t-il pleuvoir demain ?",
            "flow_type": "MESSAGE",
            "history": [],
            "meteo_data": {},
            "soil_data": {},
            "health_data": {},
            "market_data": {},
            "global_alerts": [],
            "execution_path": []
        }
        
        result = self.orchestrator.run(msg_state_meteo)
        
        self.assertIsInstance(result, dict)
        self.assertIn("execution_time_ms", result)
        # We expect a final_response or at least an updated state
        self.assertIn("execution_path", result)

    def test_report_flow_structure(self):
        """Test if the report flow generates a report structure."""
        report_state = {
            "user_id": "test_rid_1",
            "zone_id": "Ouagadougou",
            "flow_type": "REPORT",
            "global_alerts": [],
            "execution_path": [],
            "meteo_data": {},
            "market_data": {}
        }
        
        result = self.orchestrator.run(report_state)
        
        self.assertIsInstance(result, dict)
        self.assertIn("final_report", result)
        report = result["final_report"]
        self.assertIn("content", report)
        self.assertIn("priority", report)

if __name__ == '__main__':
    unittest.main()
