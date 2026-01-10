import unittest
import os
import sys

sys.path.append(os.getcwd())

from tools.health.base_health import HealthDoctorTool, SahelPathologyDB

class TestHealthTool(unittest.TestCase):
    def setUp(self):
        # Create a mock for the database if needed, but integration test with simple JSON file is cleaner if file is small.
        # Check if diseases_data.json exists. If not, test might fail.
        # But user wants "solid bricks", which includes data loading.
        pass

    def test_db_loading(self):
        db = SahelPathologyDB()
        # It's possible the JSON data file is missing or empty in a fresh environment
        # If it fails, we know data is missing.
        self.assertIsNotNone(db._DATA)

    def test_diagnosis_logic(self):
        tool = HealthDoctorTool()
        # Mocking data manually using DiseaseProfile objects
        from tools.health.base_health import DiseaseProfile
        
        profile = DiseaseProfile(
            name="Chenille Légionnaire", 
            local_names=["Sossé"], 
            symptoms_keywords=["chenille", "feuille", "trou"],
            risk_level="CRITIQUE",
            threshold_pct=10,
            bio_recipe="Neem",
            chemical_ref="Emamectine",
            prevention="Piège"
        )
        
        tool.db._DATA = {
            "maïs": [profile]
        }
        
        # Test detection
        result = tool.diagnose_and_prescribe("Maïs", "Il y a des chenilles qui mangent les feuilles", 15.0)
        self.assertIn("diagnostique", result)
        self.assertEqual(result["diagnostique"], "Chenille Légionnaire")
        self.assertEqual(result["niveau_alerte"], "CRITIQUE")

if __name__ == '__main__':
    unittest.main()
