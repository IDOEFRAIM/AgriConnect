import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

# --- CONFIGURATION MARCHÉ SAHÉLIEN ---

@dataclass
class MarketPrice:
    crop: str
    base_price_harvest: int  # Prix au champ (récolte) en FCFA/kg
    peak_price_soudure: int  # Prix pendant la période de soudure (juin-août)
    local_demand: str
    export_potential: str
    is_regulated: bool       # Si le prix est fixé par l'État (ex: Coton)

class AgrimarketTool:
    """
    Outil d'intelligence économique pour le producteur burkinabè.
    Gère les prix, les opportunités de stockage (Warrantage) et les intrants.
    """

    _MARKET_DATA: Dict[str, MarketPrice] = {
        "maïs": MarketPrice("Maïs grain", 140, 275, "Très Forte", "Régional", False),
        "niébé": MarketPrice("Niébé", 250, 500, "Forte", "International", False),
        "sorgho": MarketPrice("Sorgho", 130, 250, "Moyenne (base)", "Nulle", False),
        "coton": MarketPrice("Coton graine", 300, 300, "Monopole", "Mondial", True),
        "anacarde": MarketPrice("Noix de Cajou", 310, 450, "Export", "Très Fort", False)
    }

    def analyze_market_timing(self, crop_name: str, current_month: int) -> Dict[str, Any]:
        """
        Analyse s'il faut vendre maintenant ou stocker (stratégie de Warrantage).
        
        Args:
            crop_name: Nom de la culture.
            current_month: Mois actuel (1-12).
        """
        crop = crop_name.lower().strip()
        data = self._MARKET_DATA.get(crop)
        
        if not data:
            return {"status": "Erreur", "message": "Données non disponibles."}

        # Simulation du prix actuel selon la saisonnalité sahélienne
        # La période de "Soudure" (Mois 6, 7, 8) voit les prix doubler
        if 6 <= current_month <= 8:
            current_price = data.peak_price_soudure - random.randint(0, 30)
            market_state = "Période de Soudure (Prix Hauts)"
        elif 9 <= current_month <= 11:
            current_price = data.base_price_harvest + random.randint(0, 20)
            market_state = "Période de Récolte (Prix Bas)"
        else:
            current_price = (data.base_price_harvest + data.peak_price_soudure) // 1.5
            market_state = "Transition"

        # Conseil de stockage (Warrantage)
        can_warrant = not data.is_regulated and current_month >= 9
        
        return {
            "culture": data.crop,
            "prix_actuel_estime": f"{current_price} FCFA/kg",
            "etat_marche": market_state,
            "opportunite_warrantage": "CONSEILLÉ" if can_warrant else "NON APPLICABLE",
            "conseil": "Stockez dans un magasin agréé pour obtenir un crédit et vendre en période de soudure." if can_warrant else "Vente immédiate recommandée."
        }

    def calculate_profitability(self, crop_name: str, surface_ha: float, yield_kg_ha: float, total_costs: float) -> Dict[str, Any]:
        """
        Calcule le point mort (break-even) et le bénéfice net estimé.
        """
        data = self._MARKET_DATA.get(crop_name.lower())
        if not data: return {"error": "Inconnu"}

        total_production = surface_ha * yield_kg_ha
        # On utilise le prix moyen
        avg_price = (data.base_price_harvest + data.peak_price_soudure) / 2
        
        gross_revenue = total_production * avg_price
        net_profit = gross_revenue - total_costs
        break_even_price = total_costs / total_production if total_production > 0 else 0

        return {
            "production_totale_kg": total_production,
            "chiffre_affaires_estime": f"{int(gross_revenue)} FCFA",
            "benefice_net": f"{int(net_profit)} FCFA",
            "prix_de_revient_kg": f"{int(break_even_price)} FCFA/kg",
            "rentabilite": "OUI" if net_profit > 0 else "NON"
        }

    def get_subsidy_status(self, region: str) -> str:
        """
        Simule les alertes de subvention gouvernementale (MAAH) au Burkina Faso.
        """
        subsidized_npk = 12000 # Prix subventionné cible
        market_npk = 28000
        
        return (
            f"📢 **ALERTE SUBVENTION ({region})**\n"
            f"- Le sac de NPK subventionné est à **{subsidized_npk} FCFA** contre {market_npk} au marché.\n"
            f"- **Condition :** Être recensé dans la base RGA (Recensement Général de l'Agriculture).\n"
            f"- **Lieu :** Direction Régionale de l'Agriculture."
        )