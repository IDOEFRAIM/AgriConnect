from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable, Set
from datetime import datetime
import hashlib
import json
import re

# --- 1. Normalisation et Helpers (Stateless) ---

class Normalizers:
    """
    Fonctions statiques pour le nettoyage et la standardisation des valeurs.
    Garantit des fonctions pures et testables.
    """

    @staticmethod
    def safe_str(x: Any) -> Optional[str]:
        """Convertit en chaîne ou None, gère les exceptions."""
        if x is None:
            return None
        try:
            return str(x)
        except Exception:
            # Idéalement, on loggerait l'erreur ici
            return None

    @staticmethod
    def normalize_date(value: Any) -> Optional[str]:
        """Convertit une date en format ISO standard (YYYY-MM-DD)."""
        s = Normalizers.safe_str(value)
        if not s:
            return None
        s = s.strip()
        # Formats courants à essayer
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y.%m.%d"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except Exception:
                pass
        # Tentative d'extraction avec regex si la conversion stricte échoue
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        return m.group(0) if m else None

    @staticmethod
    def normalize_lang(s: Optional[str]) -> Optional[str]:
        """Normalise les codes de langue en format ISO 639-1 (e.g., 'fr', 'en')."""
        if not s:
            return None
        s = s.lower().strip()
        # Mappage pour consolidation
        table = {"fra": "fr", "fr-fr": "fr", "eng": "en", "en-us": "en"}
        # Retourne le mapping ou le code original si non trouvé
        return table.get(s, s)

    @staticmethod
    def first_non_none(*vals: Any) -> Any:
        """Retourne la première valeur non-None."""
        for v in vals:
            if v is not None:
                return v
        return None

# --- 2. Configuration (Data Class) ---

@dataclass
class MetadataBuilderConfig:
    # Champs canoniques à inclure par défaut
    include_keys: List[str] = field(default_factory=lambda: [
        "region", "crop", "date", "source", "language",
        "filename", "url", "title", "source_type", "agent_profile"
    ])

    # Profil (contrôle la sélection des champs supplémentaires)
    agent_profile: str = "generic"

    # Champs supplémentaires par profil
    profile_fields: Dict[str, List[str]] = field(default_factory=lambda: {
        "weather": ["station", "variables", "forecast_horizon", "resolution"],
        "irrigation": ["system", "flow_rate", "schedule", "soil_moisture"],
        "trends": ["time_window", "indicator", "region_aggregation"],
        "agronomy": ["variety", "soil_type", "phenology_stage"],
        "opportunities": ["program", "eligibility", "deadline", "benefit"],
        "cleaner": ["issues", "fixes", "confidence"],
        "generic": []
    })

    # Options pour l'ajout de métadonnées calculées
    add_quality_flags: bool = True
    add_text_hash: bool = True
    add_length_hints: bool = True

    # Option pour la compaction finale des listes/dicts en JSON
    compact_complex_values: bool = True

    # Enrichers (stratégies d'enrichissement)
    enrichers: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = field(default_factory=list)


# --- 3. Modules de Construction Spécifiques (Granularité / Testabilité) ---

class MetadataPruner:
    """Logique pour la sélection finale des champs de métadonnées."""
    def __init__(self, cfg: MetadataBuilderConfig):
        self.cfg = cfg

    def _required_computed_keys(self) -> Set[str]:
        keys = set()
        if self.cfg.add_text_hash:
            keys.add("text_hash")
        if self.cfg.add_length_hints:
            keys.update({"text_len", "is_too_short", "is_too_long"})
        if self.cfg.add_quality_flags:
            keys.update({"quality_completeness", "has_provenance"})
        return keys

    def prune(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Filtre les métadonnées pour ne garder que les clés configurées."""
        profile_keys = self.cfg.profile_fields.get(self.cfg.agent_profile, [])
        keep = set(self.cfg.include_keys) | set(profile_keys) | self._required_computed_keys()
        
        # Le merge des clés restantes doit être fait avant le pruning
        # Pour le moment, on garde tout ce qui est dans `keep`
        return {k: v for k, v in meta.items() if k in keep}


class FeatureAdder:
    """Logique pour l'ajout des champs calculés."""
    def __init__(self, cfg: MetadataBuilderConfig):
        self.cfg = cfg

    def _hash_text(self, s: Optional[str]) -> Optional[str]:
        """Calcule le hash SHA256 tronqué pour l'unicité."""
        if not s:
            return None
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def _quality(self, meta: Dict[str, Any]) -> float:
        """Score de complétude minimal sur les champs clés."""
        # Un score de qualité plus granulaire pourrait être ajouté ici
        keys = ["region", "date", "source"]
        present = sum(1 for k in keys if meta.get(k))
        return round(present / len(keys), 3)

    def add_features(self, record: Dict[str, Any], meta: Dict[str, Any]) -> None:
        """Ajoute les champs calculés directement au dictionnaire de métadonnées."""
        text = Normalizers.safe_str(record.get("text"))

        if self.cfg.add_text_hash:
            meta["text_hash"] = self._hash_text(text)

        if self.cfg.add_length_hints:
            ln = len(text or "")
            meta["text_len"] = ln
            meta["is_too_short"] = ln < 40
            meta["is_too_long"] = ln > 5000

        if self.cfg.add_quality_flags:
            meta["quality_completeness"] = self._quality(meta)
            # Utilise 'filename' ou 'url' déjà normalisés dans 'meta'
            meta["has_provenance"] = bool(meta.get("filename") or meta.get("url"))


class ValueCompactor:
    """Logique de compaction des objets complexes en chaînes JSON."""

    @staticmethod
    def compact(v: Any) -> Any:
        """Convertit list/dict en chaîne JSON compacte."""
        if isinstance(v, (list, dict)):
            try:
                # Compact, sans espace, assure_ascii=False pour Unicode
                return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                # En cas d'erreur de sérialisation, on retourne None ou une valeur par défaut
                return None 
        return v

    @staticmethod
    def compact_all(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Applique la compaction à toutes les valeurs du dictionnaire."""
        return {k: ValueCompactor.compact(v) for k, v in meta.items()}

# --- 4. Le Builder Central (Orchestration) ---

class MetadataBuilder:
    """
    Orchestrateur de la construction des métadonnées.
    S'appuie sur des classes granulaires pour chaque étape.
    """
    def __init__(self, cfg: MetadataBuilderConfig):
        self.cfg = cfg
        self.pruner = MetadataPruner(cfg)
        self.feature_adder = FeatureAdder(cfg)

    # ---------------- Public API ---------------- #

    def build(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construit un dictionnaire de métadonnées canonique, 
        profil-aware et prêt pour le filtrage.
        """
        meta_in = (record.get("metadata") or {}).copy()
        out: Dict[str, Any] = {}

        # 1. Champs Canoniques et Normalisation
        self._build_canonical_fields(record, meta_in, out)

        # 2. Champs Spécifiques au Profil
        self._build_profile_fields(record, meta_in, out)

        # 3. Champs Calculés (Features)
        self.feature_adder.add_features(record, out)

        # 4. Merge des Métadonnées Restantes (non destructif)
        self._merge_remaining_metadata(meta_in, out)

        # 5. Pruning selon le schéma configuré
        pruned = self.pruner.prune(out)

        # 6. Enrichissement
        pruned = self._apply_enrichers(pruned)
        
        # 7. Compaction des valeurs complexes
        if self.cfg.compact_complex_values:
            pruned = ValueCompactor.compact_all(pruned)
            
        # 8. Validation/Finalisation (si nécessaire)
        self._validate(pruned) 

        return pruned

    # ---------------- Internal Steps ---------------- #

    def _build_canonical_fields(self, record: Dict[str, Any], meta_in: Dict[str, Any], out: Dict[str, Any]):
        """Construit les champs de base requis avec normalisation."""
        safe = Normalizers.safe_str
        
        # Priorité : metadata > record > None
        out["region"] = safe(Normalizers.first_non_none(meta_in.get("region"), record.get("region")))
        out["crop"] = safe(Normalizers.first_non_none(meta_in.get("crop"), record.get("crop")))
        out["date"] = Normalizers.normalize_date(Normalizers.first_non_none(meta_in.get("date"), record.get("date")))
        out["source"] = safe(record.get("source"))
        out["language"] = Normalizers.normalize_lang(record.get("language"))
        out["filename"] = safe(record.get("filename"))
        out["url"] = safe(record.get("url"))
        out["title"] = safe(record.get("title"))
        out["source_type"] = safe(Normalizers.first_non_none(meta_in.get("source_type"), record.get("source_type")))
        out["agent_profile"] = self.cfg.agent_profile

    def _build_profile_fields(self, record: Dict[str, Any], meta_in: Dict[str, Any], out: Dict[str, Any]):
        """Ajoute les champs spécifiques au profil de l'agent."""
        profile_fields = self.cfg.profile_fields.get(self.cfg.agent_profile, [])
        for f in profile_fields:
            val = Normalizers.first_non_none(meta_in.get(f), record.get(f))
            out[f] = val # Pas de safe_str ici, car ces champs peuvent être des listes/dicts

    def _merge_remaining_metadata(self, meta_in: Dict[str, Any], out: Dict[str, Any]):
        """Ajoute les clés restantes de meta_in qui ne sont pas encore dans out ou qui sont None."""
        for k, v in meta_in.items():
            if k not in out or out[k] is None:
                out[k] = v

    def _apply_enrichers(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Applique les fonctions d'enrichissement définies dans la configuration."""
        out = meta.copy()
        for enrich in self.cfg.enrichers:
            try:
                # Passe une copie pour éviter que l'enricher ne modifie in-place
                extra = enrich(out.copy())
                if extra:
                    out.update(extra)
            except Exception as e:
                # Logging essentiel ici pour l'usage industriel
                # print(f"Warning: Enricher failed: {e}") 
                pass 
        return out

    def _validate(self, meta: Dict[str, Any]) -> None:
        """Dernière vérification et normalisation des champs critiques."""
        # Note: La validation est légère car la normalisation est déjà faite
        
        # Re-normalisation de la date si nécessaire (e.g. après enrichissement)
        d = meta.get("date")
        if d and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(d)):
            meta["date"] = Normalizers.normalize_date(d)
            
        # Normalisation du code langue
        meta["language"] = Normalizers.normalize_lang(meta.get("language"))

        'ig'