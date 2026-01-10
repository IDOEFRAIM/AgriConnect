# rag/benchmark/performance_tests.py
"""
Benchmarks et tests de performance pour la pipeline d'augmentation RAG.
Mesure les performances sous différentes charges et configurations.
"""

from __future__ import annotations
import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



@dataclass
class BenchmarkResult:
    """Résultats d'un benchmark."""
    name: str
    total_queries: int
    successful: int
    failed: int
    timeouts: int
    
    # Latency metrics
    min_latency: float
    max_latency: float
    avg_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    
    # Throughput
    queries_per_second: float
    total_duration: float
    
    # Quality metrics
    avg_snippets: float
    avg_tokens: int
    cache_hit_rate: float
    
    # Resource usage
    avg_cache_size: float
    peak_cache_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "queries": {
                "total": self.total_queries,
                "successful": self.successful,
                "failed": self.failed,
                "timeouts": self.timeouts,
                "success_rate": round(self.successful / max(1, self.total_queries), 3)
            },
            "latency": {
                "min": round(self.min_latency, 3),
                "max": round(self.max_latency, 3),
                "avg": round(self.avg_latency, 3),
                "median": round(self.median_latency, 3),
                "p95": round(self.p95_latency, 3),
                "p99": round(self.p99_latency, 3)
            },
            "throughput": {
                "qps": round(self.queries_per_second, 2),
                "total_duration": round(self.total_duration, 3)
            },
            "quality": {
                "avg_snippets": round(self.avg_snippets, 1),
                "avg_tokens": self.avg_tokens,
                "cache_hit_rate": round(self.cache_hit_rate, 3)
            },
            "resources": {
                "avg_cache_size": round(self.avg_cache_size, 1),
                "peak_cache_size": self.peak_cache_size
            }
        }
    
    def print_summary(self):
        """Affiche un résumé formaté des résultats."""
        print(f"\n{'=' * 70}")
        print(f"BENCHMARK: {self.name}")
        print(f"{'=' * 70}")
        
        print(f"\n📊 Queries:")
        print(f"  Total: {self.total_queries}")
        print(f"  Success: {self.successful} ({self.successful/max(1,self.total_queries)*100:.1f}%)")
        print(f"  Failed: {self.failed}")
        print(f"  Timeouts: {self.timeouts}")
        
        print(f"\n⏱️  Latency:")
        print(f"  Min: {self.min_latency:.3f}s")
        print(f"  Avg: {self.avg_latency:.3f}s")
        print(f"  Median: {self.median_latency:.3f}s")
        print(f"  P95: {self.p95_latency:.3f}s")
        print(f"  P99: {self.p99_latency:.3f}s")
        print(f"  Max: {self.max_latency:.3f}s")
        
        print(f"\n🚀 Throughput:")
        print(f"  QPS: {self.queries_per_second:.2f}")
        print(f"  Duration: {self.total_duration:.3f}s")
        
        print(f"\n✨ Quality:")
        print(f"  Avg snippets: {self.avg_snippets:.1f}")
        print(f"  Avg tokens: {self.avg_tokens}")
        print(f"  Cache hit rate: {self.cache_hit_rate*100:.1f}%")
        
        print(f"\n💾 Resources:")
        print(f"  Avg cache: {self.avg_cache_size:.1f} entries")
        print(f"  Peak cache: {self.peak_cache_size} entries")


class PipelineBenchmark:
    """
    Framework de benchmark pour la pipeline d'augmentation.
    """
    
    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.results: List[BenchmarkResult] = []
    
    async def run_single_query_benchmark(
        self,
        query: str,
        iterations: int = 100
    ) -> BenchmarkResult:
        """Benchmark d'une seule requête répétée."""
        print(f"\nRunning single query benchmark ({iterations} iterations)...")
        
        latencies: List[float] = []
        snippets_counts: List[int] = []
        token_counts: List[int] = []
        cache_sizes: List[int] = []
        successful = 0
        failed = 0
        timeouts = 0
        
        start_time = time.time()
        
        for i in range(iterations):
            try:
                iter_start = time.time()
                response = await self.pipeline.augment(query)
                latency = time.time() - iter_start
                
                if response.context.is_timeout:
                    timeouts += 1
                elif response.context.snippets:
                    successful += 1
                    latencies.append(latency)
                    snippets_counts.append(len(response.context.snippets))
                    token_counts.append(response.context.total_tokens)
                else:
                    failed += 1
                
                # Track cache size
                diag = response.diagnostics.to_dict()
                if "cache" in diag:
                    cache_sizes.append(diag["cache"].get("cache_size", 0))
                
                # Progress
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{iterations}")
                
            except Exception as e:
                failed += 1
                print(f"  Error on iteration {i+1}: {e}")
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        if latencies:
            sorted_latencies = sorted(latencies)
            result = BenchmarkResult(
                name=f"Single Query ({iterations} iterations)",
                total_queries=iterations,
                successful=successful,
                failed=failed,
                timeouts=timeouts,
                min_latency=min(latencies),
                max_latency=max(latencies),
                avg_latency=statistics.mean(latencies),
                median_latency=statistics.median(latencies),
                p95_latency=sorted_latencies[int(len(sorted_latencies) * 0.95)],
                p99_latency=sorted_latencies[int(len(sorted_latencies) * 0.99)],
                queries_per_second=successful / total_duration,
                total_duration=total_duration,
                avg_snippets=statistics.mean(snippets_counts) if snippets_counts else 0,
                avg_tokens=int(statistics.mean(token_counts)) if token_counts else 0,
                cache_hit_rate=0.0,  # Will be calculated from pipeline metrics
                avg_cache_size=statistics.mean(cache_sizes) if cache_sizes else 0,
                peak_cache_size=max(cache_sizes) if cache_sizes else 0
            )
        else:
            result = BenchmarkResult(
                name=f"Single Query ({iterations} iterations)",
                total_queries=iterations,
                successful=0,
                failed=failed,
                timeouts=timeouts,
                min_latency=0, max_latency=0, avg_latency=0,
                median_latency=0, p95_latency=0, p99_latency=0,
                queries_per_second=0,
                total_duration=total_duration,
                avg_snippets=0, avg_tokens=0, cache_hit_rate=0,
                avg_cache_size=0, peak_cache_size=0
            )
        
        self.results.append(result)
        return result
    
    async def run_concurrent_benchmark(
        self,
        queries: List[str],
        concurrency: int = 10
    ) -> BenchmarkResult:
        """Benchmark avec requêtes concurrentes."""
        print(f"\nRunning concurrent benchmark ({len(queries)} queries, concurrency={concurrency})...")
        
        latencies: List[float] = []
        snippets_counts: List[int] = []
        token_counts: List[int] = []
        successful = 0
        failed = 0
        timeouts = 0
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_query(q: str):
            nonlocal successful, failed, timeouts
            
            async with semaphore:
                try:
                    start = time.time()
                    response = await self.pipeline.augment(q)
                    latency = time.time() - start
                    
                    if response.context.is_timeout:
                        timeouts += 1
                    elif response.context.snippets:
                        successful += 1
                        latencies.append(latency)
                        snippets_counts.append(len(response.context.snippets))
                        token_counts.append(response.context.total_tokens)
                    else:
                        failed += 1
                        
                except Exception:
                    failed += 1
        
        start_time = time.time()
        tasks = [process_query(q) for q in queries]
        await asyncio.gather(*tasks)
        total_duration = time.time() - start_time
        
        # Calculate statistics
        if latencies:
            sorted_latencies = sorted(latencies)
            result = BenchmarkResult(
                name=f"Concurrent ({len(queries)} queries, concurrency={concurrency})",
                total_queries=len(queries),
                successful=successful,
                failed=failed,
                timeouts=timeouts,
                min_latency=min(latencies),
                max_latency=max(latencies),
                avg_latency=statistics.mean(latencies),
                median_latency=statistics.median(latencies),
                p95_latency=sorted_latencies[int(len(sorted_latencies) * 0.95)],
                p99_latency=sorted_latencies[int(len(sorted_latencies) * 0.99)],
                queries_per_second=successful / total_duration,
                total_duration=total_duration,
                avg_snippets=statistics.mean(snippets_counts) if snippets_counts else 0,
                avg_tokens=int(statistics.mean(token_counts)) if token_counts else 0,
                cache_hit_rate=0.0,
                avg_cache_size=0,
                peak_cache_size=0
            )
        else:
            result = BenchmarkResult(
                name=f"Concurrent ({len(queries)} queries, concurrency={concurrency})",
                total_queries=len(queries),
                successful=0,
                failed=failed,
                timeouts=timeouts,
                min_latency=0, max_latency=0, avg_latency=0,
                median_latency=0, p95_latency=0, p99_latency=0,
                queries_per_second=0,
                total_duration=total_duration,
                avg_snippets=0, avg_tokens=0, cache_hit_rate=0,
                avg_cache_size=0, peak_cache_size=0
            )
        
        self.results.append(result)
        return result
    
    async def run_stress_test(
        self,
        query: str,
        duration_seconds: int = 60,
        target_qps: int = 10
    ) -> BenchmarkResult:
        """Test de stress pendant une durée donnée."""
        print(f"\nRunning stress test ({duration_seconds}s, target {target_qps} QPS)...")
        
        latencies: List[float] = []
        successful = 0
        failed = 0
        timeouts = 0
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        interval = 1.0 / target_qps
        
        async def worker():
            nonlocal successful, failed, timeouts
            
            while time.time() < end_time:
                try:
                    iter_start = time.time()
                    response = await self.pipeline.augment(query)
                    latency = time.time() - iter_start
                    
                    if response.context.is_timeout:
                        timeouts += 1
                    elif response.context.snippets:
                        successful += 1
                        latencies.append(latency)
                    else:
                        failed += 1
                    
                    # Rate limiting
                    await asyncio.sleep(max(0, interval - (time.time() - iter_start)))
                    
                except Exception:
                    failed += 1
        
        # Start workers
        num_workers = min(target_qps, 20)
        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
        
        # Wait for completion
        await asyncio.gather(*workers)
        
        total_duration = time.time() - start_time
        total_queries = successful + failed + timeouts
        
        if latencies:
            sorted_latencies = sorted(latencies)
            result = BenchmarkResult(
                name=f"Stress Test ({duration_seconds}s, {target_qps} QPS)",
                total_queries=total_queries,
                successful=successful,
                failed=failed,
                timeouts=timeouts,
                min_latency=min(latencies),
                max_latency=max(latencies),
                avg_latency=statistics.mean(latencies),
                median_latency=statistics.median(latencies),
                p95_latency=sorted_latencies[int(len(sorted_latencies) * 0.95)],
                p99_latency=sorted_latencies[int(len(sorted_latencies) * 0.99)],
                queries_per_second=successful / total_duration,
                total_duration=total_duration,
                avg_snippets=0,
                avg_tokens=0,
                cache_hit_rate=0,
                avg_cache_size=0,
                peak_cache_size=0
            )
        else:
            result = BenchmarkResult(
                name=f"Stress Test ({duration_seconds}s, {target_qps} QPS)",
                total_queries=total_queries,
                successful=0,
                failed=failed,
                timeouts=timeouts,
                min_latency=0, max_latency=0, avg_latency=0,
                median_latency=0, p95_latency=0, p99_latency=0,
                queries_per_second=0,
                total_duration=total_duration,
                avg_snippets=0, avg_tokens=0, cache_hit_rate=0,
                avg_cache_size=0, peak_cache_size=0
            )
        
        self.results.append(result)
        return result
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Génère un rapport complet de tous les benchmarks."""
        report = {
            "timestamp": time.time(),
            "total_benchmarks": len(self.results),
            "benchmarks": [r.to_dict() for r in self.results],
            "summary": {
                "total_queries": sum(r.total_queries for r in self.results),
                "total_successful": sum(r.successful for r in self.results),
                "overall_success_rate": sum(r.successful for r in self.results) / 
                                       max(1, sum(r.total_queries for r in self.results)),
                "avg_latency": statistics.mean([r.avg_latency for r in self.results if r.avg_latency > 0]),
                "best_qps": max([r.queries_per_second for r in self.results], default=0)
            }
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Report saved to {output_path}")
        
        return report


# =========================================================================
# Test Scenarios
# =========================================================================

async def scenario_baseline():
    """Scenario 1: Baseline performance avec configuration standard."""
    from rag.contextaugmentedv9 import AugmentationPipeline, AugmentationConfig
    
    # Setup pipeline (adapter selon votre setup)
    # pipeline = setup_pipeline()
    # benchmark = PipelineBenchmark(pipeline)
    
    print("\n" + "=" * 70)
    print("SCENARIO 1: Baseline Performance")
    print("=" * 70)
    
    # Single query repeated
    # result = await benchmark.run_single_query_benchmark(
    #     "Quelle est la météo pour demain ?",
    #     iterations=50
    # )
    # result.print_summary()


async def scenario_concurrent_load():
    """Scenario 2: Charge concurrente."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Concurrent Load")
    print("=" * 70)
    
    queries = [
        "Prévisions de pluie pour octobre 2025",
        "Température moyenne en septembre",
        "Alertes météo en cours",
        "Cumul de précipitations depuis avril",
        "Tendance des vents pour la semaine",
    ] * 10  # 50 queries total
    
    # benchmark = PipelineBenchmark(pipeline)
    # result = await benchmark.run_concurrent_benchmark(queries, concurrency=10)
    # result.print_summary()


async def scenario_stress_test():
    """Scenario 3: Test de stress prolongé."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Stress Test")
    print("=" * 70)
    
    # benchmark = PipelineBenchmark(pipeline)
    # result = await benchmark.run_stress_test(
    #     "Quelle est la météo ?",
    #     duration_seconds=30,
    #     target_qps=5
    # )
    # result.print_summary()


async def main():
    """Exécute tous les scénarios de benchmark."""
    print("\n" + "=" * 70)
    print("PIPELINE BENCHMARK SUITE")
    print("=" * 70)
    
    await scenario_baseline()
    await scenario_concurrent_load()
    await scenario_stress_test()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())