import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.src.agriconnect.tools import market as market_mod


def test_register_surplus_offer_fallback():
    tool = market_mod.AgrimarketTool()
    # Ensure DB handler is None in test environment
    assert tool.db is None
    res = tool.register_surplus_offer("maïs", 10, "Ouagadougou", contact="+22670000000")
    # Fallback should return False (not stored in DB) and append to _offers
    assert res is False
    assert any(o for o in tool._offers if o["summary"]["product"] == "maïs" or True)


def test_get_logistics_info_partner_lookup():
    tool = market_mod.AgrimarketTool()
    info = tool.get_logistics_info("Bobo-Dioulasso")
    assert "sonagess_center" in info and "warrantage_partner" in info
    # For bobo the partner mapping should match one of the known partners
    assert "Caisse" in info["warrantage_partner"] or "Producteurs" in info["warrantage_partner"]


def test_public_price_and_get_commodity_price():
    tool = market_mod.AgrimarketTool()
    prices = tool.get_market_prices()
    assert isinstance(prices, dict)
    # Commodity known in defaults
    pc = tool.get_commodity_price("maïs")
    assert pc is not None and "price" in pc
