import logging
from datetime import date
from typing import Dict, Any, List,Optional

logger = logging.getLogger("services.market")

class MarketDataService:
    """
    Collecte les données de marché et les annonces gouvernementales.
    Nécessite une mise à jour régulière (scraping ou API) pour les prix.
    """
    
    # Simulation des données de prix (instables)
    CURRENT_MARKET_PRICES_FCFA = {
        "coton": 320,  # Prix d'achat garanti par coopérative
        "mais": 150,   # Prix au kg, marché local
        "niebe": 250,
        "arachide": 350
    }
    
    # Simulation des règles de subventions (institutionnel)
    GOVT_SUBSIDY_RULES = {
        "date_mise_a_jour": "2025-11-20",
        "engrais_subv": {
            "active": True,
            "max_ha": 5, 
            "requires_coop": True,
            "deadline": "2026-01-30"
        },
        "semences_resist": {
            "active": True,
            "target_regions": ["sahel", "nord", "est"]
        }
    }

    def get_price(self, product_name: str) -> Optional[float]:
        """Retourne le prix marché actuel pour un produit."""
        return self.CURRENT_MARKET_PRICES_FCFA.get(product_name.lower())

    def get_subsidy_rules(self) -> Dict[str, Any]:
        """Retourne les règles de subvention en vigueur."""
        # Dans un cas réel, ceci ferait un appel à un fichier JSON mis à jour.
        return self.GOVT_SUBSIDY_RULES

    def get_credit_opportunities(self, farmer_status: Dict[str, Any]) -> List[str]:
        """Simule la vérification d'opportunités de crédit."""
        opportunities = []
        if farmer_status.get("previous_loan_status", "clean") == "clean":
            opportunities.append("Micro-crédit saisonnier (Crédit Rural)")
        return opportunities