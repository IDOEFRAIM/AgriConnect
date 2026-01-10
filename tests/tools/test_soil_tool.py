import unittest
import os
import sys

sys.path.append(os.getcwd())

from tools.soils.base_soil import SoilDoctorTool

class TestSoilTool(unittest.TestCase):
    def test_soil_loading(self):
        tool = SoilDoctorTool()
        # Allows checking empty if file missing, but logic should hold
        self.assertIsNotNone(tool._SOILS)

    def test_diagnosis(self):
        tool = SoilDoctorTool()
        # Mocking data
        from tools.soils.base_soil import SahelSoilProfile
        tool._SOILS = {
            "argileux": SahelSoilProfile(
                name="Sol Argileux",
                local_term="Gravillon",
                description="Lourd",
                water_retention="High",
                pwp_fc=(10, 20),
                ces_technique="Zaï",
                ces_description="Digues"
            )
        }
        
        result = tool.get_full_diagnosis("argileux", "Mon sol est dur")
        
        # Check dictionary keys based on implementation in base_soil.py
        self.assertIn("soil_type", result)
        self.assertIn("ces_recommendation", result)
        self.assertIn("Sol Argileux", result["soil_type"])
        self.assertEqual(result["ces_recommendation"]["technique"], "Zaï")

if __name__ == '__main__':
    unittest.main()
