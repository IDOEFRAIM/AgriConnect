"""
normalizer.py - Text Normalization & Chunking Module
Améliorations:
- Déduplication robuste avec cache de hashes
- Chunking par phrases pour meilleure cohérence sémantique
- Validation et nettoyage avancés
- Statistiques détaillées
- Support de multiple stratégies de chunking
- Gestion améliorée des métadonnées
"""
from __future__ import annotations
import re
import uuid
import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Generator, Set
from collections import defaultdict
from enum import Enum

_logger = logging.getLogger("normalizer")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class ChunkingStrategy(Enum):
    """Stratégies de découpage disponibles"""
    TOKEN_BASED = "token"  # Découpage par tokens (mots)
    SENTENCE_BASED = "sentence"  # Découpage par phrases
    PARAGRAPH_BASED = "paragraph"  # Découpage par paragraphes


@dataclass
class NormalizerConfig:
    """Configuration pour le normalizer"""
    # Chunking
    chunk_size_tokens: int = 200
    chunk_overlap_tokens: int = 50
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.TOKEN_BASED
    
    # Validation
    min_chunk_length: int = 20
    max_chunk_length: int = 2000
    min_tokens_per_chunk: int = 5
    
    # Nettoyage
    normalize_whitespace: bool = True
    lowercase: bool = False
    remove_urls: bool = True
    remove_emails: bool = False
    remove_special_chars: bool = False
    
    # Métadonnées
    keep_metadata_fields: Tuple[str, ...] = ("source", "date", "region", "language", "author")
    id_prefix: str = "doc"
    
    # Fonctionnalités avancées
    language_detection: bool = False
    enable_deduplication: bool = True
    compute_statistics: bool = True
    
    # Validation
    validate_utf8: bool = True
    max_doc_size_chars: int = 1_000_000


# ==============================================================================
# NORMALIZER
# ==============================================================================

class Normalizer:
    """
    Normalizer: convertit des documents bruts en chunks indexables.
    
    Responsabilités:
      - Nettoyage et validation du texte
      - Détection de langue (optionnel)
      - Découpage intelligent en chunks avec overlap
      - Déduplication basée sur hash
      - Production de records structurés
    """

    def __init__(self, cfg: NormalizerConfig):
        self.cfg = cfg
        
        # Cache pour déduplication
        self._seen_hashes: Set[str] = set() if cfg.enable_deduplication else None
        
        # Statistiques
        self._stats = {
            "docs_processed": 0,
            "docs_skipped": 0,
            "chunks_created": 0,
            "chunks_deduplicated": 0,
            "errors": 0,
            "total_chars": 0,
            "total_tokens": 0
        }
        
        # Patterns regex compilés pour performance
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self._sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])'
        )

    def normalize_stream(
        self,
        raw_items: Iterable[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Traite un flux d'éléments bruts et produit un flux de chunks"""
        for raw in raw_items:
            try:
                records = self.normalize_one(raw)
                for rec in records:
                    yield rec
            except Exception as e:
                self._stats["errors"] += 1
                _logger.exception(
                    f"normalize_stream failed for item {raw.get('raw_id')}: {e}"
                )

    def normalize_one(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalise un document brut en liste de chunks.
        
        Input: { raw_id, source, text, metadata }
        Output: [{ id, chunk_id, chunk_index, text, meta, ... }]
        """
        # Validation initiale
        if not self._validate_raw_item(raw):
            self._stats["docs_skipped"] += 1
            return []
        
        text = raw.get("text", "")
        
        # Validation de la taille
        if len(text) > self.cfg.max_doc_size_chars:
            _logger.warning(
                f"Document trop grand ({len(text)} chars), tronqué à {self.cfg.max_doc_size_chars}"
            )
            text = text[:self.cfg.max_doc_size_chars]
        
        # Validation UTF-8
        if self.cfg.validate_utf8:
            text = self._ensure_utf8(text)
        
        # Nettoyage
        text = self._clean_text(text)
        
        if not text or len(text) < self.cfg.min_chunk_length:
            _logger.info(f"Text too short after cleaning: {raw.get('raw_id')}")
            self._stats["docs_skipped"] += 1
            return []
        
        # Métadonnées
        metadata = self._extract_metadata(raw)
        
        # Détection de langue
        if self.cfg.language_detection and "language" not in metadata:
            metadata["language"] = self._detect_language(text)
        
        # Génération ID document
        doc_id = self._generate_doc_id(raw)
        
        # Chunking selon la stratégie
        chunks = self._create_chunks(text)
        
        # Construction des records
        records = self._build_records(doc_id, chunks, metadata, raw)
        
        # Statistiques
        self._stats["docs_processed"] += 1
        self._stats["chunks_created"] += len(records)
        self._stats["total_chars"] += len(text)
        
        return records

    def _validate_raw_item(self, raw: Dict[str, Any]) -> bool:
        """Valide un élément brut"""
        if not isinstance(raw, dict):
            _logger.warning("Raw item is not a dict")
            return False
        
        if "text" not in raw or not raw["text"]:
            _logger.debug(f"Empty text for item {raw.get('raw_id')}")
            return False
        
        if not isinstance(raw["text"], str):
            _logger.warning(f"Text is not a string: {type(raw['text'])}")
            return False
        
        return True

    def _ensure_utf8(self, text: str) -> str:
        """Assure que le texte est valide UTF-8"""
        try:
            # Encode puis decode pour nettoyer les caractères invalides
            return text.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception as e:
            _logger.warning(f"UTF-8 validation failed: {e}")
            return text

    def _clean_text(self, text: str) -> str:
        """
        Applique les étapes de nettoyage configurable.
        """
        t = text
        
        # Suppression des URLs
        if self.cfg.remove_urls:
            t = self._url_pattern.sub(' [URL] ', t)
        
        # Suppression des emails
        if self.cfg.remove_emails:
            t = self._email_pattern.sub(' [EMAIL] ', t)
        
        # Suppression des caractères de contrôle
        t = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]+', ' ', t)
        
        # Normalisation des espaces
        if self.cfg.normalize_whitespace:
            # Remplace multiples espaces/newlines par un seul espace
            t = re.sub(r'\s+', ' ', t).strip()
        
        # Minuscules
        if self.cfg.lowercase:
            t = t.lower()
        
        # Suppression caractères spéciaux (optionnel)
        if self.cfg.remove_special_chars:
            # Garde lettres, chiffres, espaces et ponctuation de base
            t = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
        
        return t

    def _create_chunks(self, text: str) -> List[str]:
        """Crée les chunks selon la stratégie configurée"""
        strategy = self.cfg.chunking_strategy
        
        if strategy == ChunkingStrategy.SENTENCE_BASED:
            return self._chunk_by_sentences(text)
        elif strategy == ChunkingStrategy.PARAGRAPH_BASED:
            return self._chunk_by_paragraphs(text)
        else:  # TOKEN_BASED
            return self._chunk_by_tokens(text)

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Découpage par tokens (mots) avec overlap"""
        tokens = text.split()
        
        if not tokens:
            return []
        
        chunks = []
        win_size = max(1, self.cfg.chunk_size_tokens)
        step = max(1, win_size - self.cfg.chunk_overlap_tokens)
        
        for start in range(0, len(tokens), step):
            end = min(len(tokens), start + win_size)
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens).strip()
            
            if self._is_valid_chunk(chunk_text, chunk_tokens):
                chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
        
        return chunks

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Découpage par phrases avec overlap intelligent"""
        # Split en phrases
        sentences = self._sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_sentences = max(1, self.cfg.chunk_overlap_tokens // 50)  # ~1-2 phrases
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            # Si ajouter cette phrase dépasse la taille max
            if current_tokens + sentence_tokens > self.cfg.chunk_size_tokens and current_chunk:
                # Sauvegarder le chunk actuel
                chunk_text = " ".join(current_chunk)
                if self._is_valid_chunk(chunk_text, chunk_text.split()):
                    chunks.append(chunk_text)
                
                # Garder les dernières phrases pour overlap
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_tokens = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Dernier chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if self._is_valid_chunk(chunk_text, chunk_text.split()):
                chunks.append(chunk_text)
        
        return chunks

    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """Découpage par paragraphes"""
        # Split sur double newline ou plus
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        
        for para in paragraphs:
            para_tokens = para.split()
            
            # Si le paragraphe est trop grand, le découper en tokens
            if len(para_tokens) > self.cfg.chunk_size_tokens:
                sub_chunks = self._chunk_by_tokens(para)
                chunks.extend(sub_chunks)
            elif self._is_valid_chunk(para, para_tokens):
                chunks.append(para)
        
        return chunks

    def _is_valid_chunk(self, chunk_text: str, tokens: List[str]) -> bool:
        """Valide qu'un chunk respecte les contraintes"""
        # Longueur minimale
        if len(chunk_text) < self.cfg.min_chunk_length:
            return False
        
        # Longueur maximale
        if len(chunk_text) > self.cfg.max_chunk_length:
            return False
        
        # Nombre minimum de tokens
        if len(tokens) < self.cfg.min_tokens_per_chunk:
            return False
        
        return True

    def _build_records(
        self,
        doc_id: str,
        chunks: List[str],
        metadata: Dict[str, Any],
        raw: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Construit les records finaux à partir des chunks"""
        records = []
        
        for chunk_index, chunk_text in enumerate(chunks):
            # Hash pour déduplication
            text_hash = self._hash_text(chunk_text)
            
            # Vérification déduplication
            if self._seen_hashes is not None:
                if text_hash in self._seen_hashes:
                    self._stats["chunks_deduplicated"] += 1
                    _logger.debug(f"Chunk dupliqué ignoré: {text_hash[:16]}...")
                    continue
                self._seen_hashes.add(text_hash)
            
            # Construction du record
            chunk_id = f"{doc_id}_c{chunk_index}"
            tokens = chunk_text.split()
            
            record = {
                "id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "meta": metadata.copy(),
                "text_hash": text_hash,
                "length_chars": len(chunk_text),
                "length_tokens": len(tokens),
                "source": raw.get("source", "unknown")
            }
            
            # Statistiques du chunk (optionnel)
            if self.cfg.compute_statistics:
                record["stats"] = self._compute_chunk_stats(chunk_text, tokens)
            
            records.append(record)
            self._stats["total_tokens"] += len(tokens)
        
        return records

    def _compute_chunk_stats(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """Calcule des statistiques sur le chunk"""
        return {
            "avg_word_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            "sentence_count": len(self._sentence_pattern.split(text)),
            "unique_tokens": len(set(tokens)),
            "lexical_diversity": len(set(tokens)) / len(tokens) if tokens else 0
        }

    def _extract_metadata(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait et filtre les métadonnées"""
        raw_meta = raw.get("metadata", {})
        metadata = {}
        
        for field in self.cfg.keep_metadata_fields:
            if field in raw_meta:
                metadata[field] = raw_meta[field]
            elif field in raw:
                metadata[field] = raw[field]
        
        return metadata

    def _generate_doc_id(self, raw: Dict[str, Any]) -> str:
        """Génère un ID unique pour le document"""
        raw_id = raw.get('raw_id') or raw.get('id')
        
        if raw_id is None:
            raw_id = uuid.uuid4()
        
        return f"{self.cfg.id_prefix}_{raw_id}"

    def _hash_text(self, text: str) -> str:
        """Calcule un hash SHA256 du texte"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _detect_language(self, text: str) -> str:
        """Détecte la langue du texte (nécessite langdetect)"""
        try:
            import langdetect
            # Utilise un sample pour performance
            sample = text[:1000] if len(text) > 1000 else text
            return langdetect.detect(sample)
        except ImportError:
            _logger.warning("langdetect not installed. Install: pip install langdetect")
            return "und"
        except Exception as e:
            _logger.debug(f"Language detection failed: {e}")
            return "und"

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de normalisation"""
        stats = self._stats.copy()
        
        # Calculs dérivés
        if stats["docs_processed"] > 0:
            stats["avg_chunks_per_doc"] = round(
                stats["chunks_created"] / stats["docs_processed"], 2
            )
            stats["avg_chars_per_doc"] = round(
                stats["total_chars"] / stats["docs_processed"], 2
            )
        
        if stats["chunks_created"] > 0:
            stats["avg_tokens_per_chunk"] = round(
                stats["total_tokens"] / stats["chunks_created"], 2
            )
        
        if self._seen_hashes is not None:
            stats["unique_chunks"] = len(self._seen_hashes)
        
        return stats

    def reset_stats(self) -> None:
        """Réinitialise les statistiques"""
        self._stats = {
            "docs_processed": 0,
            "docs_skipped": 0,
            "chunks_created": 0,
            "chunks_deduplicated": 0,
            "errors": 0,
            "total_chars": 0,
            "total_tokens": 0
        }
        
        if self._seen_hashes is not None:
            self._seen_hashes.clear()

    def clear_deduplication_cache(self) -> int:
        """Vide le cache de déduplication"""
        if self._seen_hashes is None:
            return 0
        
        count = len(self._seen_hashes)
        self._seen_hashes.clear()
        return count


# ==============================================================================
# EXEMPLE D'UTILISATION
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = NormalizerConfig(
        chunk_size_tokens=50,
        chunk_overlap_tokens=10,
        chunking_strategy=ChunkingStrategy.SENTENCE_BASED,
        min_chunk_length=20,
        normalize_whitespace=True,
        remove_urls=True,
        enable_deduplication=True,
        language_detection=True
    )
    
    # Données de test
    test_data = [
        {
            "raw_id": "test_1",
            "source": "exemple.txt",
            "text": """
                La normalisation de texte est une étape cruciale dans le traitement du langage naturel.
                Elle permet de standardiser les données avant l'analyse. Les techniques incluent la 
                tokenisation, le nettoyage et la segmentation. Pour plus d'infos: https://example.com
                
                Le chunking intelligent améliore la qualité des embeddings. Il faut trouver le bon 
                équilibre entre la taille des chunks et leur cohérence sémantique. C'est un défi 
                important pour les systèmes RAG.
            """,
            "metadata": {
                "date": "2024-01-15",
                "author": "Jean Dupont",
                "region": "FR"
            }
        }
    ]
    
    # Normalisation
    normalizer = Normalizer(config)
    
    print("=== NORMALISATION EN COURS ===\n")
    
    for record in normalizer.normalize_stream(test_data):
        print(f"Chunk ID: {record['chunk_id']}")
        print(f"Texte: {record['text'][:80]}...")
        print(f"Tokens: {record['length_tokens']}")
        print(f"Hash: {record['text_hash'][:16]}...")
        print("-" * 60)
    
    # Statistiques
    stats = normalizer.get_stats()
    print("\n=== STATISTIQUES ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

        'i'