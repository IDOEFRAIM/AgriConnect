"""
Système de tracking des métriques de performance pour la pipeline RAG.
"""

import time
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class MetricsTracker:
    """Collecte et agrège les métriques de performance."""
    
    # Configuration
    max_history: int = 1000  # Garder les N dernières requêtes
    
    # Compteurs
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Latences (en secondes)
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Timestamps
    start_time: float = field(default_factory=time.perf_counter)
    
    def record_query(self, latency_s: float, success: bool = True):
        """
        Enregistre une requête.
        
        Args:
            latency_s: Temps de réponse en secondes
            success: Si la requête a réussi
        """
        self.total_queries += 1
        
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        self.latencies.append(latency_s)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne un snapshot des métriques actuelles.
        
        Returns:
            Dict avec toutes les métriques agrégées
        """
        if not self.latencies:
            return {
                'total_queries': self.total_queries,
                'successful_queries': self.successful_queries,
                'failed_queries': self.failed_queries,
                'success_rate': 0.0,
                'avg_latency_s': 0.0,
                'min_latency_s': 0.0,
                'max_latency_s': 0.0,
                'p50_latency_s': 0.0,
                'p95_latency_s': 0.0,
                'p99_latency_s': 0.0,
                'uptime_s': time.perf_counter() - self.start_time
            }
        
        latencies_list = list(self.latencies)
        latencies_sorted = sorted(latencies_list)
        
        return {
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'success_rate': (
                self.successful_queries / self.total_queries 
                if self.total_queries > 0 else 0.0
            ),
            'avg_latency_s': statistics.mean(latencies_list),
            'min_latency_s': min(latencies_list),
            'max_latency_s': max(latencies_list),
            'p50_latency_s': self._percentile(latencies_sorted, 0.50),
            'p95_latency_s': self._percentile(latencies_sorted, 0.95),
            'p99_latency_s': self._percentile(latencies_sorted, 0.99),
            'uptime_s': time.perf_counter() - self.start_time
        }
    
    @staticmethod
    def _percentile(sorted_values: List[float], p: float) -> float:
        """Calcule le percentile p d'une liste triée."""
        if not sorted_values:
            return 0.0
        
        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = f + 1
        
        if c >= len(sorted_values):
            return sorted_values[-1]
        
        # Interpolation linéaire
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])
    
    def reset(self):
        """Reset tous les compteurs."""
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.latencies.clear()
        self.start_time = time.perf_counter()