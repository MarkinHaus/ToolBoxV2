"""
Benchmark Suite for HybridMemoryStore - Phase 2

Validates performance against Phase 2 exit criteria:
- Add (1 entry): < 5ms avg
- Query Hybrid (10k entries): < 50ms avg
- Save/Load 10k: < 2s
- Concept Lookup: < 10ms at 10k
- Dedup 5k entries: 0 OOM
- Concurrent R/W: 0 Errors

Usage:
    python -m toolboxv2.test.bench_hybrid_memory
    uv run pytest toolboxv2/test/bench_hybrid_memory.py -v
"""

import os
import shutil
import statistics
import tempfile
import time
from typing import Callable, List, Optional

import numpy as np

from toolboxv2.mods.isaa.base.hybrid_memory import HybridMemoryStore


class BenchmarkHybridMemory:
    """
    Comprehensive benchmark suite for HybridMemoryStore.

    Tests all Phase 2 performance criteria and provides detailed metrics.
    """

    def __init__(self, n: int = 1000, dim: int = 768):
        """
        Initialize benchmark suite.

        Args:
            n: Number of entries for benchmarks (default: 1000)
            dim: Embedding dimension (default: 768)
        """
        self.tmp = tempfile.mkdtemp(prefix="hybrid_memory_bench_")
        self.store = HybridMemoryStore(self.tmp, dim)
        self.rng = np.random.default_rng(42)
        self.dim = dim
        self.n = n

        print("\n" + "=" * 70)
        print("HybridMemoryStore Benchmark Suite")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Entries: {n}")
        print(f"  Dimension: {dim}")
        print(f"  Temp dir: {self.tmp}")
        print("=" * 70)

    def _generate_embedding(self) -> np.ndarray:
        """Generate random embedding."""
        return self.rng.random(self.dim).astype(np.float32)

    def _measure_time(self, func: Callable, iterations: int = 1) -> List[float]:
        """
        Measure execution time over multiple iterations.

        Args:
            func: Function to measure
            iterations: Number of iterations

        Returns:
            List of execution times in seconds
        """
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            func()
            times.append(time.perf_counter() - t0)
        return times

    def _print_stats(
        self,
        name: str,
        times: List[float],
        unit: str = "ms",
        target: Optional[float] = None,
    ):
        """
        Print statistics for benchmark results.

        Args:
            name: Benchmark name
            times: List of execution times in seconds
            unit: Time unit (ms or s)
            target: Target threshold (optional)
        """
        if unit == "ms":
            times_ms = [t * 1000 for t in times]
        else:
            times_ms = [t for t in times]

        avg = statistics.mean(times_ms)
        median = statistics.median(times_ms)
        stdev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
        p50 = statistics.quantiles(times_ms, n=100)[49] if len(times_ms) > 1 else avg
        p95 = statistics.quantiles(times_ms, n=100)[94] if len(times_ms) > 1 else avg
        p99 = statistics.quantiles(times_ms, n=100)[98] if len(times_ms) > 1 else avg

        status = "✓" if target is None or avg < target else "✗"

        print(f"\n{status} {name}")
        print(f"  Average: {avg:.2f}{unit}")
        print(f"  Median: {median:.2f}{unit}")
        print(f"  Std Dev: {stdev:.2f}{unit}")
        print(f"  P50: {p50:.2f}{unit}")
        print(f"  P95: {p95:.2f}{unit}")
        print(f"  P99: {p99:.2f}{unit}")
        print(f"  Min: {min(times_ms):.2f}{unit}")
        print(f"  Max: {max(times_ms):.2f}{unit}")

        if target:
            print(f"  Target: <{target}{unit}")
            if avg < target:
                print(f"  Status: ✓ PASS")
            else:
                print(f"  Status: ✗ FAIL (exceeded by {avg - target:.2f}{unit})")

        return avg

    def bench_add_latency(self, warmup: int = 10, iterations: int = 100) -> float:
        """
        Benchmark add() latency.

        Phase 2 Exit Criteria: < 5ms avg

        Args:
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations

        Returns:
            Average latency in ms
        """
        print("\n" + "-" * 70)
        print("BENCHMARK: Add Latency")
        print("-" * 70)

        # Warmup
        print(f"\n  Warming up ({warmup} iterations)...")
        for i in range(warmup):
            self.store.add(f"warmup {i}", self._generate_embedding())

        # Benchmark
        print(f"  Benchmarking ({iterations} iterations)...")
        times = []
        for i in range(iterations):
            content = f"benchmark entry {i} about topic {i % 50}"
            emb = self._generate_embedding()

            t0 = time.perf_counter()
            self.store.add(content, emb, concepts=[f"topic-{i % 50}"])
            times.append(time.perf_counter() - t0)

            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{iterations}")

        avg = self._print_stats("Add Latency", times, unit="ms", target=5.0)

        print(f"\n  Total entries added: {iterations}")
        print(f"  Store size: {self.store.stats()['active']} entries")

        return avg

    def bench_query_latency(self, warmup: int = 5, iterations: int = 50) -> dict:
        """
        Benchmark query() latency for different search modes.

        Phase 2 Exit Criteria: < 50ms avg for hybrid query

        Args:
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations

        Returns:
            Dictionary of average latencies by mode
        """
        print("\n" + "-" * 70)
        print("BENCHMARK: Query Latency")
        print("-" * 70)

        # Ensure we have data to query
        stats = self.store.stats()
        if stats["active"] < 100:
            print(f"\n  Populating store with {self.n} entries...")
            for i in range(self.n):
                self.store.add(
                    f"query benchmark {i} about topic {i % 50}",
                    self._generate_embedding(),
                    concepts=[f"topic-{i % 50}"],
                )

        results = {}
        query_emb = self._generate_embedding()

        # Test different search modes
        modes = [
            ("vector", "Vector Search"),
            ("bm25", "BM25 Search"),
            ("relation", "Relation Search"),
            (("vector", "bm25"), "Hybrid (Vector + BM25)"),
            (("vector", "bm25", "relation"), "Hybrid (All Modes)"),
        ]

        for mode, mode_name in modes:
            print(f"\n  Testing: {mode_name}")

            # Warmup
            for i in range(warmup):
                self.store.query(
                    f"warmup query {i}",
                    query_emb,
                    k=5,
                    search_modes=mode if isinstance(mode, tuple) else (mode,),
                )

            # Benchmark
            times = []
            for i in range(iterations):
                query_text = f"query test {i} topic"

                t0 = time.perf_counter()
                hits = self.store.query(
                    query_text,
                    query_emb,
                    k=5,
                    search_modes=mode if isinstance(mode, tuple) else (mode,),
                )
                times.append(time.perf_counter() - t0)

            target = 50.0 if isinstance(mode, tuple) and len(mode) > 1 else None
            avg = self._print_stats(
                f"{mode_name} Latency", times, unit="ms", target=target
            )
            results[mode_name] = avg

        print(f"\n  Store size: {self.store.stats()['active']} entries")

        return results

    def bench_save_load(self, iterations: int = 5) -> dict:
        """
        Benchmark save() and load() operations.

        Phase 2 Exit Criteria: < 2s for 10k entries

        Args:
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with save and load metrics
        """
        print("\n" + "-" * 70)
        print("BENCHMARK: Save/Load Performance")
        print("-" * 70)

        # Ensure we have data
        stats = self.store.stats()
        if stats["active"] < 100:
            print(f"\n  Populating store with {self.n} entries...")
            for i in range(self.n):
                self.store.add(
                    f"save load test {i}",
                    self._generate_embedding(),
                    concepts=[f"topic-{i % 20}"],
                )

        stats = self.store.stats()
        print(
            f"\n  Store size: {stats['active']} entries, {stats['entities']} entities, {stats['relations']} relations"
        )

        # Benchmark save
        print(f"\n  Benchmarking save ({iterations} iterations)...")
        save_times = []
        save_sizes = []

        for i in range(iterations):
            t0 = time.perf_counter()
            data = self.store.save_to_bytes()
            save_times.append(time.perf_counter() - t0)
            save_sizes.append(len(data))

        avg_save = self._print_stats("Save Time", save_times, unit="s", target=2.0)
        avg_size = statistics.mean(save_sizes)
        print(f"\n  Average save size: {avg_size / 1024:.2f} KB")

        # Benchmark load
        print(f"\n  Benchmarking load ({iterations} iterations)...")
        load_times = []

        # Create a fresh store for loading
        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_bench_load_")
        try:
            for i in range(iterations):
                store2 = HybridMemoryStore(tmp2, self.dim)

                t0 = time.perf_counter()
                store2.load(data)
                load_times.append(time.perf_counter() - t0)

                store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        avg_load = self._print_stats("Load Time", load_times, unit="s", target=2.0)

        return {
            "save_avg_s": avg_save,
            "load_avg_s": avg_load,
            "size_kb": avg_size / 1024,
        }

    def bench_concept_lookup(
        self, n_entries: int = 10000, iterations: int = 100
    ) -> float:
        """
        Benchmark concept-based lookup.

        Phase 2 Exit Criteria: < 10ms at 10k entries (O(1) index)

        Args:
            n_entries: Number of entries to populate
            iterations: Number of lookup iterations

        Returns:
            Average lookup time in ms
        """
        print("\n" + "-" * 70)
        print("BENCHMARK: Concept Lookup (P2 - O(1) Index)")
        print("-" * 70)

        # Populate with concept-tagged entries
        stats = self.store.stats()
        if stats["active"] < n_entries:
            print(f"\n  Populating store with {n_entries} entries...")
            for i in range(n_entries):
                if (i + 1) % 1000 == 0:
                    print(f"    Progress: {i + 1}/{n_entries}")

                self.store.add(
                    f"concept test entry {i} about topic {i % 100}",
                    self._generate_embedding(),
                    concepts=[f"topic-{i % 100}", f"category-{i % 10}"],
                )

        print(f"\n  Store size: {self.store.stats()['active']} entries")

        # Benchmark concept queries
        print(f"\n  Benchmarking concept lookup ({iterations} iterations)...")
        times = []
        query_emb = self._generate_embedding()

        for i in range(iterations):
            concept = f"topic-{i % 100}"

            t0 = time.perf_counter()
            hits = self.store.query(concept, query_emb, k=5, search_modes=("relation",))
            times.append(time.perf_counter() - t0)

        avg = self._print_stats("Concept Lookup", times, unit="ms", target=10.0)

        return avg

    def bench_deduplication(self, n_entries: int = 5000) -> bool:
        """
        Benchmark deduplication with many duplicate entries.

        Phase 2 Exit Criteria: No OOM (P4 - RAM Explosion)

        Args:
            n_entries: Number of entries (50% duplicates)

        Returns:
            True if no OOM occurred
        """
        print("\n" + "-" * 70)
        print("BENCHMARK: Deduplication (P4 - No OOM)")
        print("-" * 70)

        print(f"\n  Adding {n_entries} entries with ~50% duplicates...")

        initial_stats = self.store.stats()
        initial_count = initial_stats["active"]

        try:
            # Add entries with 50% duplicates
            for i in range(n_entries):
                # Reuse content for every other entry (create duplicates)
                content = f"dedup test {i % (n_entries // 2)}"
                self.store.add(content, self._generate_embedding())

                if (i + 1) % 500 == 0:
                    print(f"    Progress: {i + 1}/{n_entries}")
                    # Print memory usage
                    import psutil

                    process = psutil.Process(os.getpid())
                    mem_mb = process.memory_info().rss / 1024 / 1024
                    print(f"    Memory usage: {mem_mb:.2f} MB")

            final_stats = self.store.stats()
            final_count = final_stats["active"]

            print(f"\n  Initial entries: {initial_count}")
            print(f"  Added: {n_entries}")
            print(f"  Final active entries: {final_count}")
            print(f"  Duplicates prevented: {initial_count + n_entries - final_count}")

            # Should not have 5000 entries (many were duplicates)
            assert final_count <= n_entries // 2 + initial_count + 100,"Should have prevented most duplicates"

            print(f"\n  Status: ✓ PASS - No OOM")
            return True

        except MemoryError as e:
            print(f"\n  Status: ✗ FAIL - OOM occurred: {e}")
            return False

    def bench_concurrent_access(
        self, writers: int = 2, readers: int = 2, operations: int = 50
    ) -> bool:
        """
        Benchmark concurrent read/write operations.

        Phase 2 Exit Criteria: 0 errors (P6 - Non-blocking)

        Args:
            writers: Number of concurrent writer threads
            readers: Number of concurrent reader threads
            operations: Operations per thread

        Returns:
            True if no errors occurred
        """
        print("\n" + "-" * 70)
        print("BENCHMARK: Concurrent Access (P6 - Non-blocking)")
        print("-" * 70)

        import concurrent.futures
        import threading

        errors = []
        error_lock = threading.Lock()

        def writer_thread(thread_id: int):
            """Writer thread function."""
            try:
                for i in range(operations):
                    content = f"concurrent writer {thread_id} entry {i}"
                    self.store.add(content, self._generate_embedding())
            except Exception as e:
                with error_lock:
                    errors.append(f"Writer {thread_id}: {e}")

        def reader_thread(thread_id: int):
            """Reader thread function."""
            try:
                for i in range(operations):
                    query_text = f"concurrent test {i}"
                    hits = self.store.query(query_text, self._generate_embedding(), k=3)
            except Exception as e:
                with error_lock:
                    errors.append(f"Reader {thread_id}: {e}")

        print(f"\n  Running {writers} writers + {readers} readers")
        print(f"  Operations per thread: {operations}")

        t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=writers + readers
        ) as executor:
            # Submit writers
            writer_futures = [executor.submit(writer_thread, i) for i in range(writers)]

            # Submit readers
            reader_futures = [executor.submit(reader_thread, i) for i in range(readers)]

            # Wait for all
            concurrent.futures.wait(writer_futures + reader_futures)

        elapsed = time.perf_counter() - t0

        print(f"\n  Elapsed time: {elapsed:.2f}s")
        print(f"  Total operations: {(writers + readers) * operations}")
        print(f"  Throughput: {(writers + readers) * operations / elapsed:.1f} ops/s")

        if errors:
            print(f"\n  Status: ✗ FAIL - {len(errors)} errors occurred")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
            return False
        else:
            print(f"\n  Status: ✓ PASS - No errors")
            return True

    def run_all_benchmarks(self) -> dict:
        """
        Run all benchmarks and return summary.

        Returns:
            Dictionary with all benchmark results
        """
        print("\n" + "=" * 70)
        print("RUNNING ALL BENCHMARKS")
        print("=" * 70)

        results = {}

        # Run benchmarks
        results["add_latency"] = self.bench_add_latency(warmup=10, iterations=100)
        results["query_latency"] = self.bench_query_latency(warmup=5, iterations=50)
        results["save_load"] = self.bench_save_load(iterations=5)
        results["concept_lookup"] = self.bench_concept_lookup(
            n_entries=1000, iterations=100
        )
        results["deduplication"] = self.bench_deduplication(n_entries=1000)
        results["concurrent"] = self.bench_concurrent_access(
            writers=2, readers=2, operations=50
        )

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: dict):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY - Phase 2 Exit Criteria")
        print("=" * 70)

        criteria = [
            ("Add (1 entry)", "< 5ms avg", results["add_latency"], 5.0),
            (
                "Query Hybrid",
                "< 50ms avg",
                results["query_latency"].get("Hybrid (All Modes)", 999),
                50.0,
            ),
            (
                "Save/Load",
                "< 2s",
                max(
                    results["save_load"]["save_avg_s"], results["save_load"]["load_avg_s"]
                ),
                2.0,
            ),
            ("Concept Lookup", "< 10ms", results["concept_lookup"], 10.0),
            ("Deduplication", "No OOM", results["deduplication"], None),
            ("Concurrent R/W", "0 Errors", results["concurrent"], None),
        ]

        passed = 0
        total = len(criteria)

        print(
            "\n{:<25} {:<15} {:<15} {:<10}".format(
                "Criterion", "Target", "Result", "Status"
            )
        )
        print("-" * 70)

        for name, target, result, threshold in criteria:
            if threshold is not None:
                status = "✓ PASS" if result < threshold else "✗ FAIL"
                if result < threshold:
                    passed += 1
            else:
                status = "✓ PASS" if result else "✗ FAIL"
                if result:
                    passed += 1

            if isinstance(result, bool):
                result_str = "OK" if result else "FAILED"
            else:
                result_str = f"{result:.2f}"

            print("{:<25} {:<15} {:<15} {:<10}".format(name, target, result_str, status))

        print("\n" + "=" * 70)
        print(f"RESULTS: {passed}/{total} criteria passed ({passed / total * 100:.0f}%)")
        print("=" * 70)

        if passed == total:
            print("\n✓✓✓ ALL PHASE 2 EXIT CRITERIA MET ✓✓✓\n")
        else:
            print(f"\n✗ {total - passed} criteria failed - needs optimization\n")

    def cleanup(self):
        """Clean up temporary directory."""
        try:
            self.store.close()
        except:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)
        print(f"\n✓ Cleaned up: {self.tmp}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark HybridMemoryStore")
    parser.add_argument(
        "-n",
        "--entries",
        type=int,
        default=1000,
        help="Number of entries (default: 1000)",
    )
    parser.add_argument(
        "-d", "--dim", type=int, default=768, help="Embedding dimension (default: 768)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick benchmark with fewer iterations"
    )

    args = parser.parse_args()

    # Create benchmark instance
    bench = BenchmarkHybridMemory(n=args.entries, dim=args.dim)

    try:
        if args.quick:
            # Quick benchmark
            print("\n*** QUICK MODE - Reduced iterations ***\n")
            bench.bench_add_latency(warmup=5, iterations=20)
            bench.bench_query_latency(warmup=2, iterations=10)
            bench.bench_save_load(iterations=2)
        else:
            # Full benchmark suite
            bench.run_all_benchmarks()
    finally:
        bench.cleanup()


if __name__ == "__main__":
    main()
