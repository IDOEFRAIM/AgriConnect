import unittest
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from tools.subventions.base_subsidy import AgrimarketTool

class TestMarketTool(unittest.TestCase):
    def setUp(self):
        self.tool = AgrimarketTool()

    def test_load_market_data(self):
        """Test if market data is loaded correctly."""
        self.assertTrue(len(self.tool._MARKET_DATA) > 0, "Market data should not be empty")

    def test_get_market_prices(self):
        """Test fetching market prices."""
        prices = self.tool.get_market_prices()
        self.assertIsInstance(prices, dict)
        # Check if we have at least one crop pricing info
        self.assertTrue(len(prices) > 0)
        first_key = list(prices.keys())[0]
        self.assertIn("price", prices[first_key])
        self.assertIn("fair_price", prices[first_key])

if __name__ == '__main__':
    unittest.main()
