from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
try:
    from .db_handler import DBHandler
except ImportError:
    # Fallback pour éviter crash si db_handler a un souci
    DBHandler = None

DEFAULT_DATASET = {
    "prices": {
        "mais": {
            "aliases": ["maïs", "mais", "maize"],
            "price": 260,
            "fair_price": 250,
            "volatility": 0.18,
            "last_update": "2023-04-09",
            "source": "SONAGESS BHI S14 2023",
        },
        "mil": {
            "aliases": ["mil", "mil local", "mil rouge"],
            "price": 330,
            "fair_price": 325,
            "volatility": 0.21,
            "last_update": "2023-04-09",
            "source": "SONAGESS BHI S14 2023",
        },
        "sorgho": {
            "aliases": ["sorgho", "sorgho blanc"],
            "price": 269,
            "fair_price": 260,
            "volatility": 0.24,
            "last_update": "2023-04-09",
            "source": "SONAGESS BHI S14 2023",
        },
        "riz": {
            "aliases": ["riz décortiqué", "riz local"],
            "price": 415,
            "fair_price": 400,
            "volatility": 0.16,
            "last_update": "2023-04-09",
            "source": "SONAGESS tableau hebdomadaire S14 2023",
        },
    },
    "offers": [
        {
            "side": "VENTE",
            "commodity": "maïs",
            "location": "Kaya",
            "price_per_kg": 245,
            "contact": "+226 70 00 00 01",
            "expires": "2023-05-15",
        },
        {
            "side": "VENTE",
            "commodity": "sorgho",
            "location": "Bobo-Dioulasso",
            "price_per_kg": 255,
            "contact": "+226 70 00 00 02",
            "expires": "2023-05-12",
        },
        {
            "side": "ACHAT",
            "commodity": "maïs",
            "location": "Ouagadougou",
            "price_per_kg": 270,
            "contact": "+226 70 00 00 03",
            "expires": "2023-05-20",
        },
        {
            "side": "ACHAT",
            "commodity": "mil",
            "location": "Dori",
            "price_per_kg": 340,
            "contact": "+226 70 00 00 04",
            "expires": "2023-05-25",
        },
    ],
    "seasonal": {
        "mais": {
            "harvest_window": ["septembre", "octobre", "novembre"],
            "storage_gain_ratio": 0.08,
        },
        "mil": {
            "harvest_window": ["octobre", "novembre", "décembre"],
            "storage_gain_ratio": 0.06,
        },
        "sorgho": {
            "harvest_window": ["octobre", "novembre", "décembre"],
            "storage_gain_ratio": 0.07,
        },
        "riz": {
            "harvest_window": ["octobre", "novembre", "décembre"],
            "storage_gain_ratio": 0.05,
        },
    },
}


class AgrimarketTool:
    """Fournit les données marché et les heuristiques utilisées par MarketCoach et AgriBusinessCoach."""

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self._prices: Dict[str, Dict[str, Any]] = {}
        self._offers: List[Dict[str, Any]] = []
        self._seasonal: Dict[str, Dict[str, Any]] = {}
        
        # Initialisation de la connexion DB
        self.db = DBHandler() if DBHandler else None
        
        self._load_defaults()
        if self.dataset_path:
            self._load_external_dataset(self.dataset_path)

    # ------------------------------------------------------------------ #
    # API publique consommée par les agents                              #
    # ------------------------------------------------------------------ #

    def register_surplus_offer(self, commodity: str, quantity: float, location: str, contact: str = "TBD") -> bool:
        """
        Enregistre un surplus déclaré par l'utilisateur.
        Retourne True si enregistré en DB, False sinon.
        """
        # Tentative DB
        if self.db:
            success = self.db.register_surplus(commodity, quantity, location, contact)
            if success:
                return True
        
        # Fallback mémoire (Simulation Mode Hors-Ligne)
        self._offers.append({
            "side": "VENTE",
            "commodity": self._normalize(commodity),
            "quantity": quantity,
            "location": location,
            "contact": contact,
            "expires": "SYNC_PENDING", # Marqué pour synchro future
            "source": "USER_OFFLINE_CACHE"
        })
        return False

    def get_logistics_info(self, location: str) -> Dict[str, str]:
        """Retourne les infos logistiques (SONAGESS + Warrantage) pour une localité."""
        loc = self._normalize(location)
        
        # 1. Recherche dynamique SONAGESS via DB
        center_addr = None
        if self.db:
            center_addr = self.db.get_logistics_center(loc)
        
        # Fallback générique
        if not center_addr:
             center_addr = "Centre Régional le plus proche (consultez votre mairie)"
        
        # Base de données simulée des partenaires financiers (Warrantage)
        partners = {
            "nouna": "Caisse Populaire de Nouna ou Coopérative 'Union'",
            "dedougou": "RCPB Dédougou ou Coris Bank",
            "bobo": "Caisse Populaire ou Réseau des Producteurs de Coton",
        }
        
        partner_addr = "Caisse Populaire locale"
        for city, partner in partners.items():
            if city in loc:
                partner_addr = partner
                break
                
        return {
            "sonagess_center": center_addr,
            "warrantage_partner": partner_addr
        }

    def get_market_prices(self) -> Dict[str, Dict[str, Any]]:
        payload: Dict[str, Dict[str, Any]] = {}
        for record in self._prices.values():
            payload[record["primary_alias"]] = self._public_price(record)
            for alias in record["aliases"]:
                if alias == record["primary_alias"]:
                    continue
                payload[alias] = self._public_price(record)
        return payload

    def get_commodity_price(self, commodity: str) -> Optional[Dict[str, Any]]:
        """Récupère le prix spécifique d'une denrée (Utilisé par MarketAgent)."""
        record = self._get_price_record(commodity)
        if record:
            return self._public_price(record)
        return None

    def analyze_market_trends(self, commodity: str) -> Dict[str, Any]:
        """Analyse les tendances et le timing de vente (Alias pour MarketAgent)."""
        return self.analyze_market_timing(commodity)

    def list_offers(
        self,
        side: str,
        commodity: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        normalized_side = self._normalize(side)
        target = self._normalize(commodity) if commodity else None
        matches = []
        for offer in self._offers:
            if self._normalize(offer["side"]) != normalized_side:
                continue
            if target and target not in offer["normalized_aliases"]:
                continue
            matches.append(deepcopy(offer["summary"]))
            if len(matches) >= limit:
                break
        return matches

    def analyze_market_timing(
        self,
        commodity: str,
        quantity_kg: float = 1000.0,
    ) -> Dict[str, Any]:
        record = self._get_price_record(commodity)
        seasonal = self._get_seasonal_record(commodity)

        if not record:
            return {
                "commodity": commodity,
                "status": "INCONNU",
                "conseil": "Collectez des prix supplémentaires avant décision.",
            }

        current_price = record["price"]
        fair_price = record["fair_price"]
        spread = current_price - fair_price
        seasonal_gain = seasonal.get("storage_gain_ratio", 0.0) if seasonal else 0.0

        verdict = "VENDRE"
        conseil = "Profitez du marché actuel, qui reste aligné sur la référence."
        gain_potentiel = 0.0
        horizon = 0

        if seasonal_gain > 0.0:
            gain_potentiel = round(current_price * seasonal_gain * max(quantity_kg, 0), 2)
            horizon = 180  # ~6 mois
            if spread < -5:
                verdict = "CONSERVER"
                conseil = (
                    "Prix actuel bas. Stockez ! 💡 **Le Warrantage** : Déposez vos sacs dans un magasin sécurisé "
                    "pour obtenir un prêt immédiat à la banque, puis vendez quand les prix montent."
                )
            elif spread <= 10:
                verdict = "NEUTRE"
                conseil = "Prix stable. Vendez une partie pour vos besoins immédiats et stockez le reste."
            else:
                verdict = "VENDRE"
                conseil = "Prix haut ! C'est le bon moment pour vendre et encaisser le bénéfice."

        return {
            "commodity": record["primary_alias"],
            "verdict": verdict,
            "conseil": conseil,
            "market_price": current_price,
            "market_ref_price": fair_price,
            "warrantage": "CONSEILLÉ" if gain_potentiel > 0 else "OPTIONNEL",
            "gain_potentiel_stockage": gain_potentiel,
            "horizon_jours": horizon,
            "confiance": 1.0 - min(record["volatility"], 0.9),
            "source": record["source"],
            "derniere_mise_a_jour": record["last_update"],
        }

    def check_price_fairness(self, commodity: str, offered_price: float) -> Dict[str, Any]:
        record = self._get_price_record(commodity)
        if not record:
            return {
                "commodity": commodity,
                "status": "INCONNU",
                "reason": "Aucune donnée SONAGESS disponible.",
            }

        ref = record["fair_price"]
        diff = offered_price - ref
        absolute_diff = abs(diff)

        if offered_price <= 0:
            status = "INVALID"
            reason = "Le prix proposé doit être strictement positif."
        elif absolute_diff <= 0.05 * ref:
            status = "ALIGNÉ"
            reason = "Prix raisonnable : négociez surtout sur les volumes et les délais."
        elif diff > 0:
            status = "SURCOTÉ"
            reason = "Négociez à la baisse : argumentez avec la référence SONAGESS."
        else:
            status = "SOUS-COTÉ"
            reason = "Prix trop bas : refusez ou demandez une révision en mentionnant la référence SONAGESS."

        return {
            "commodity": record["primary_alias"],
            "status": status,
            "reason": reason,
            "market_price": record["price"],
            "market_ref_price": ref,
            "difference": diff,
            "volatility": record["volatility"],
            "source": record["source"],
            "last_update": record["last_update"],
        }

    @staticmethod
    def utcnow_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # ------------------------------------------------------------------ #
    # Chargement / normalisation                                         #
    # ------------------------------------------------------------------ #

    def _load_defaults(self) -> None:
        self._prices = {}
        for key, payload in DEFAULT_DATASET["prices"].items():
            primary_alias = payload["aliases"][0]
            self._prices[self._normalize(key)] = {
                "key": self._normalize(key),
                "primary_alias": primary_alias,
                "aliases": payload["aliases"],
                "price": float(payload["price"]),
                "fair_price": float(payload["fair_price"]),
                "volatility": float(payload.get("volatility", 0.0)),
                "last_update": payload.get("last_update", "1970-01-01"),
                "source": payload.get("source", "SONAGESS"),
            }

        self._seasonal = {
            self._normalize(key): value.copy()
            for key, value in DEFAULT_DATASET.get("seasonal", {}).items()
        }

        self._offers = []
        for offer in DEFAULT_DATASET["offers"]:
            self._offers.append(self._normalize_offer(offer))

    def _load_external_dataset(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Fichier dataset introuvable: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        prices = payload.get("prices") or {}
        for key, value in prices.items():
            aliases = value.get("aliases") or [key]
            primary_alias = aliases[0]
            self._prices[self._normalize(key)] = {
                "key": self._normalize(key),
                "primary_alias": primary_alias,
                "aliases": aliases,
                "price": float(value["price"]),
                "fair_price": float(value.get("fair_price", value["price"])),
                "volatility": float(value.get("volatility", 0.0)),
                "last_update": value.get("last_update", self.utcnow_iso()),
                "source": value.get("source", "SONAGESS"),
            }

        offers = payload.get("offers") or []
        if offers:
            self._offers = [self._normalize_offer(item) for item in offers]

        seasonal = payload.get("seasonal") or {}
        if seasonal:
            self._seasonal = {
                self._normalize(key): value.copy() for key, value in seasonal.items()
            }

    def _normalize_offer(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        aliases = self._collect_aliases(offer.get("commodity"))
        summary = {
            "side": offer.get("side", "").upper(),
            "product": aliases[0] if aliases else offer.get("commodity", ""),
            "location": offer.get("location", ""),
            "price_per_kg": float(offer.get("price_per_kg", 0)),
            "contact": offer.get("contact", "N/C"),
            "expires": offer.get("expires", ""),
        }
        return {
            "side": offer.get("side", "").upper(),
            "commodity": offer.get("commodity", ""),
            "normalized_aliases": aliases,
            "summary": summary,
        }

    def _collect_aliases(self, commodity: Optional[str]) -> List[str]:
        if not commodity:
            return []
        record = self._get_price_record(commodity)
        if record:
            return record["aliases"]
        normalized = self._normalize(commodity)
        return [commodity, normalized]

    def _public_price(self, record: Dict[str, Any]) -> Dict[str, Any]:
        spread = record["price"] - record["fair_price"]
        return {
            "price": record["price"],
            "fair_price": record["fair_price"],
            "spread": spread,
            "volatility": record["volatility"],
            "last_update": record["last_update"],
            "source": record["source"],
        }

    def _get_price_record(self, commodity: Optional[str]) -> Optional[Dict[str, Any]]:
        if not commodity:
            return None
        key = self._normalize(commodity)
        if key in self._prices:
            return self._prices[key]
        # Recherche dans les alias si nom non normalisé
        for record in self._prices.values():
            aliases = [self._normalize(alias) for alias in record["aliases"]]
            if key in aliases:
                return record
        return None

    def _get_seasonal_record(self, commodity: Optional[str]) -> Dict[str, Any]:
        if not commodity:
            return {}
        key = self._normalize(commodity)
        return self._seasonal.get(key, {})

    @staticmethod
    def _normalize(value: Optional[str]) -> str:
        if not value:
            return ""
        value = value.lower().strip()
        value = unicodedata.normalize("NFKD", value)
        return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
