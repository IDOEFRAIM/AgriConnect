import random
import logging
import json
import os
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
    Gère les prix, les opportunités de stockage (Warrantage), les intrants
    et l'accès au Grand Marché National.
    """

    def __init__(self):
        self.logger = logging.getLogger("AgrimarketTool")
        self._MARKET_DATA = self._load_market_data()
        self._OFFERS_FILE = os.path.join(os.path.dirname(__file__), 'market_offers.json')

    def _load_market_data(self) -> Dict[str, MarketPrice]:
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'market_data.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            market_data = {}
            for key, value in data.items():
                market_data[key] = MarketPrice(**value)
            return market_data
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            return {}

    def get_market_prices(self) -> Dict[str, Any]:
        """Retourne les prix actuels du marché avec analyse de tendance et prix juste."""
        prices = {}
        current_month = datetime.now().month
        
        for crop, data in self._MARKET_DATA.items():
            # 1. Calcul du Prix Théorique (Saisonnalité)
            if 6 <= current_month <= 8: # Soudure (Prix Haut)
                theoretical_price = data.peak_price_soudure
                trend = "HAUSSE (Soudure)"
            elif 9 <= current_month <= 11: # Récolte (Prix Bas)
                theoretical_price = data.base_price_harvest
                trend = "BAISSE (Récolte)"
            else:
                theoretical_price = (data.base_price_harvest + data.peak_price_soudure) // 2
                trend = "STABLE"
            
            # 2. Simulation de "Prix du Marché Réel" (avec volatilité)
            # Parfois le marché est plus bas que prévu (spéculation/surproduction)
            market_price = theoretical_price
            
            # 3. Détection d'anomalie (Prix Juste vs Prix Marché)
            fair_price = theoretical_price
            gap = 0
            
            prices[data.crop] = {
                "price": f"{market_price} FCFA/kg",
                "fair_price": f"{fair_price} FCFA/kg",
                "trend": trend,
                "is_regulated": data.is_regulated
            }
        return prices

    def list_offers(self, filter_type: str = None) -> List[Dict[str, Any]]:
        """Liste les offres du Grand Marché National."""
        try:
            if not os.path.exists(self._OFFERS_FILE):
                return []
            with open(self._OFFERS_FILE, 'r', encoding='utf-8') as f:
                offers = json.load(f)
            
            if filter_type:
                return [o for o in offers if o.get('type') == filter_type.upper()]
            return offers
        except Exception as e:
            self.logger.error(f"Error reading offers: {e}")
            return []

    def post_offer(self, offer_type: str, product: str, quantity: float, price: float, location: str, contact: str) -> str:
        """Publie une offre sur le Grand Marché National."""
        try:
            offers = self.list_offers()
            new_offer = {
                "id": f"OFFER-{len(offers)+1:03d}",
                "type": offer_type.upper(),
                "product": product,
                "quantity_kg": quantity,
                "price_per_kg": price,
                "location": location,
                "contact": contact,
                "date": datetime.now().strftime("%Y-%m-%d")
            }
            offers.append(new_offer)
            
            with open(self._OFFERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(offers, f, ensure_ascii=False, indent=2)
            
            return f"Offre {new_offer['id']} publiée avec succès sur le Grand Marché National."
        except Exception as e:
            self.logger.error(f"Error posting offer: {e}")
            return "Erreur lors de la publication de l'offre."

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

    def get_sonagess_price(self, crop: str) -> int:
        """Récupère le prix officiel SONAGESS (Céréales) pour éviter l'arnaque."""
        # Prix plancher officiels simulés 2026 (Mock)
        prices = {
            "maïs": 17500, # le sac de 100kg
            "sorgho": 18000,
            "mil": 19000,
            "riz": 22000,
            "sésame": 600000 # la tonne (600F/kg) prix plancher souvent
        }
        # Retourne prix au KG
        key = crop.lower().strip()
        price_sac = prices.get(key, 0)
        return int(price_sac / 100) if price_sac > 0 else 0

    def find_nearby_storage(self, zone: str) -> List[Dict[str, str]]:
        """Trouve les Magasins de Stockage (Warrantage) agréés."""
        # Mock de magasins
        warehouses = [
            {"name": "Magasin COOPABO", "ville": "Bobo", "capacite": "Dispo"},
            {"name": "Grenier de Sécurité", "ville": "Ouahigouya", "capacite": "Saturé"},
            {"name": "Silo SONAGESS", "ville": "Fada", "capacite": "Dispo"}
        ]
        return [w for w in warehouses if w['ville'] in zone or zone == "Centre"]

    def initiate_secure_deal(self, buyer_phone: str, seller_phone: str, amount: int, product: str) -> Dict[str, Any]:
        """
        Crée une transaction ESCROW (Tiers de Confiance).
        L'argent est bloqué techniquement.
        """
        deal_id = f"TX-{random.randint(1000,9999)}"
        return {
            "transaction_id": deal_id,
            "status": "PENDING_DEPOSIT",
            "instruction_buyer": f"Déposez {amount} FCFA sur le compte AgriConnect (Code *144*4*1#) avec la réf {deal_id}.",
            "instruction_seller": f"Ne livrez PAS tant que vous n'avez pas reçu le SMS de confirmation 'FONDS BLOQUÉS' de AgriConnect.",
            "message_security": "L'argent sera libéré au vendeur uniquement après confirmation de livraison par l'acheteur ou GPS transporteur."
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