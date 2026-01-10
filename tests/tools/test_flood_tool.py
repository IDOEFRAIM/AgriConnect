import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from tools.meteo.flood_risk import FloodRiskTool

class TestFloodRiskTool(unittest.TestCase):
    def setUp(self):
        self.tool = FloodRiskTool()

    @patch('tools.meteo.flood_risk.FloodRiskTool._load_latest_data')
    def test_check_flood_risk_no_data(self, mock_load):
        """Test behavior when no flood data is available."""
        mock_load.return_value = []
        result = self.tool.check_flood_risk("Ouagadougou", 12.0, -1.5)
        self.assertEqual(result['risk_level'], "Faible")
        self.assertIn("Aucun risque majeur", result['alert_message'])

    @patch('tools.meteo.flood_risk.FloodRiskTool._load_latest_data')
    def test_check_flood_risk_with_data_safe(self, mock_load):
        """Test behavior when data exists but point is far."""
        # Mock feature far away
        mock_load.return_value = [{
            "geometry": {
                "coordinates": [[[10.0, 10.0], [10.1, 10.1], [10.0, 10.1], [10.0, 10.0]]]
            },
            "properties": {"risk_level": 3}
        }]
        result = self.tool.check_flood_risk("Ouagadougou", 12.0, -1.5)
        self.assertEqual(result['risk_level'], "Faible") 

    # Note: Testing the actual "Hit" logic requires reproducing the point_in_polygon logic or similar 
    # which depends on how check_flood_risk is implemented. 
    # Assuming it checks for location string match or proximity if simpler.

if __name__ == '__main__':
    unittest.main()
