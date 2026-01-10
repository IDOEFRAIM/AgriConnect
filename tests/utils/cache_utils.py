"""
Utilitaires pour la gestion robuste du cache d'embeddings.
Fichier: rag/utils/cache_utils.py
"""

import hashlib
import unicodedata
import re
from typing import List


def normalize_cache_key(text: str, max_length: int = 500) -> str:
    """
    Normalise un texte pour l'utiliser comme clé de cache.
    
    Args:
        text: Texte brut à normaliser
        max_length: Longueur max avant hashing (défaut: 500 car)
    
    Returns:
        Clé normalisée stable et déterministe
    
    Normalisation appliquée:
    - Unicode NFC (canonical composition)
    - Suppression espaces multiples/leading/trailing
    - Suppression caractères de contrôle
    - Hashing si trop long (MD5 pour perf)
    """
    # 1. Normalisation Unicode (é vs e+´)
    normalized = unicodedata.normalize('NFC', text)
    
    # 2. Suppression caractères de contrôle (sauf \n \t)
    normalized = ''.join(
        c for c in normalized 
        if unicodedata.category(c)[0] != 'C' or c in '\n\t'
    )
    
    # 3. Collapse espaces multiples
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 4. Strip leading/trailing
    normalized = normalized.strip()
    
    # 5. Hash si trop long (évite clés géantes dans JSON)
    if len(normalized) > max_length:
        # Garder début + hash pour debuggabilité
        prefix = normalized[:100]
        hash_suffix = hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]
        return f"{prefix}__hash_{hash_suffix}"
    
    return normalized


def normalize_cache_keys_batch(texts: List[str]) -> List[str]:
    """Normalise un batch de textes."""
    return [normalize_cache_key(t) for t in texts]


def compute_cache_stats(hits: int, misses: int) -> dict:
    """
    Calcule les stats de cache de manière robuste.
    
    Args:
        hits: Nombre de hits
        misses: Nombre de misses
    
    Returns:
        Dict avec hit_rate, total_requests, etc.
    """
    total = hits + misses
    
    return {
        'hits': hits,
        'misses': misses,
        'total_requests': total,
        'hit_rate': (hits / total) if total > 0 else 0.0,
        'hit_rate_pct': f"{(hits / total * 100) if total > 0 else 0.0:.1f}%"
    }