from typing import Dict, Optional, List,Any

class AgriFinanceLib:
    """
    Règles et calculs pour l'analyse économique et la subvention.
    """

    @staticmethod
    def calculate_break_even_point(production_kg: float, total_costs: float) -> float:
        """Calcule le prix minimum (en FCFA/kg) pour couvrir les coûts."""
        if production_kg <= 0:
            return float('inf')
        # Point mort = Coûts totaux / Volume de production
        return round(total_costs / production_kg, 2)

    @staticmethod
    def evaluate_rentability(market_price: float, break_even_price: float) -> str:
        """Évalue si la vente est rentable au prix actuel."""
        if market_price > break_even_price * 1.2:
            return "HAUTE_RENTABILITE"
        elif market_price > break_even_price:
            return "RENTABILITE_MODEREE"
        else:
            return "VENTE_A_PERTE"

    @staticmethod
    def check_subsidy_eligibility(farmer_status: Dict[str, Any], subsidy_rules: Dict[str, Any]) -> List[str]:
        """Vérifie l'éligibilité du paysan aux subventions."""
        eligible_subsidies = []
        
        # Exemple de vérification pour la subvention "Engrais subventionné"
        if subsidy_rules.get("engrais_subv", {}).get("active"):
            if farmer_status.get("surface_ha", 0) <= subsidy_rules["engrais_subv"]["max_ha"] and \
               farmer_status.get("is_coop_member", False) == subsidy_rules["engrais_subv"]["requires_coop"]:
                eligible_subsidies.append("Engrais Subventionné (Burkina)")
                
        # Exemple pour les "Semences Résistantes"
        if subsidy_rules.get("semences_resist", {}).get("active"):
            if farmer_status.get("region", "unknown") in subsidy_rules["semences_resist"]["target_regions"]:
                eligible_subsidies.append("Semences Résistantes (Zone Sèche)")
                
        return eligible_subsidies