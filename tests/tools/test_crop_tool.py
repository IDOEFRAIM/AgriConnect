import unittest
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from tools.crop.base_crop import BurkinaCropTool

class TestCropTool(unittest.TestCase):
    def setUp(self):
        self.tool = BurkinaCropTool()

    def test_load_crops(self):
        """Test if crops data is loaded correctly."""
        self.assertTrue(len(self.tool._CROPS) > 0, "Crops database should not be empty")
        self.assertIn("maïs", self.tool._CROPS, "Maïs should be in the crop database")

    def test_get_technical_sheet_found(self):
        """Test retrieving a valid technical sheet."""
        sheet = self.tool.get_technical_sheet("Maïs", "Centre")
        self.assertIn("FICHE TECHNIQUE EXPERT : MAÏS (CENTRE)", sheet)
        self.assertIn("SR21", sheet) # "SR21" is in the Centre variety list based on error output

    def test_get_technical_sheet_not_found(self):
        """Test retrieving an invalid crop."""
        sheet = self.tool.get_technical_sheet("Pomme", "Centre")
        self.assertIn("Culture 'Pomme' non répertoriée", sheet)

if __name__ == '__main__':
    unittest.main()
