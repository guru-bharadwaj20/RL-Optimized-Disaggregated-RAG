"""
Benchmark utility for the RL-Optimized Disaggregated RAG system.

Compares cache configurations (exact OFF, exact ON, semantic ON) and
worker counts (1, 2, 4) across a fixed question set.

Output:
    benchmarks/results/report.md       — human-readable markdown table
    benchmarks/results/results.csv     — raw per-request CSV
    benchmarks/results/summary.json    — structured summary for CI/dashboards

Usage:
    python benchmarks/run_benchmark.py [--url http://localhost:8000]
                                       [--workers 1,2,4]
                                       [--requests 20]
                                       [--concurrency 4]
                                       [--timeout 60]

Requirements (install locally, not inside Docker):
    pip install requests numpy
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import requests as _requests

    def http_post(url: str, payload: dict, timeout: int) -> Tuple[int, dict, float]:
        t0 = time.perf_counter()
        r = _requests.post(url, json=payload, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000
        return r.status_code, r.json() if r.status_code == 200 else {}, latency

    def http_patch(url: str, payload: dict, timeout: int = 10) -> bool:
        try:
            r = _requests.patch(url, json=payload, timeout=timeout)
            return r.status_code < 300
        except Exception:
            return False

except ImportError:
    print("ERROR: 'requests' package required. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ── Constants ─────────────────────────────────────────────────────────────────

QUESTION_BANK = [
    "What is retrieval-augmented generation?",
    "How does vector search work?",
    "Explain Redis message queues",
    "What is load balancing?",
    "How does caching reduce latency?",
    "What is the CAP theorem?",
    "What are word embeddings?",
    "Explain microservices architecture",
    "What is Prometheus monitoring?",
    "How does semantic search differ from keyword search?",
    "What is Qdrant used for?",
    "Explain sentence transformers",
    "What is Docker and why is it used?",
    "How does an API gateway work?",
    "What is asynchronous message processing?",
    "Explain the role of Kubernetes in deployment",
    "What are cosine similarity scores?",
    "What is a vector database?",
    "How does RAG improve LLM accuracy?",
    "What is distributed system fault tolerance?",
]

RESULTS_DIR = Path(__file__).parent / "results"


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    scenario: str
    worker_count: int
    question: str
    status_code: int
    latency_ms: float
    cache_status: str
    llm_backend: str
    replica: str
    success: bool


@dataclass
class ScenarioStats:
    scenario: str
    worker_count: int
    n_requests: int
    n_success: int
    latencies: List[float] = field(default_factory=list)
    cache_hits: int = 0
    semantic_hits: int = 0
    miss: int = 0

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_requests if self.n_requests else 0

    @property
    def cache_hit_rate(self) -> float:
        if self.n_requests == 0:
            return 0
        return (self.cache_hits + self.semantic_hits) / self.n_requests

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0
        if _HAS_NUMPY:
            return float(np.percentile(self.latencies, p))
        s = sorted(self.latencies)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0

    @property
    def throughput(self) -> float:
        if not self.latencies or self.mean == 0:
            return 0
        total_wall_ms = sum(self.latencies)
        return (self.n_success / total_wall_ms) * 1000 if total_wall_ms else 0


# ── Admin helpers ─────────────────────────────────────────────────────────────

def wait_for_gateway(url: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


def set_cache_config(url: str, exact: bool, semantic: bool, timeout: int = 10) -> bool:
    """Toggle cache flags via admin endpoint (if present) or skip gracefully."""
    try:
        r = _requests.patch(
            f"{url}/admin/cache",
            json={"exact_cache_enabled": exact, "semantic_cache_enabled": semantic},
            timeout=timeout,
        )
        return r.status_code < 300
    except Exception:
        return False


def flush_caches(url: str, timeout: int = 10) -> bool:
    """Flush Redis exact cache + semantic cache (if endpoint exists)."""
    try:
        r = _requests.post(f"{url}/admin/flush_cache", timeout=timeout)
        return r.status_code < 300
    except Exception:
        return False


# ── Core benchmark runner ─────────────────────────────────────────────────────

def run_single_request(
    url: str, question: str, scenario: str, worker_count: int, timeout: int
) -> RequestResult:
    try:
        status, data, latency = http_post(
            f"{url}/query", {"query": question}, timeout
        )
        success = status == 200
        return RequestResult(
            scenario=scenario,
            worker_count=worker_count,
            question=question,
            status_code=status,
            latency_ms=latency,
            cache_status=data.get("cache_status", "miss") if success else "error",
            llm_backend=data.get("llm_backend", "unknown") if success else "unknown",
            replica=data.get("selected_replica", "unknown") if success else "unknown",
            success=success,
        )
    except Exception as e:
        return RequestResult(
            scenario=scenario,
            worker_count=worker_count,
            question=question,
            status_code=0,
            latency_ms=0,
            cache_status="error",
            llm_backend="unknown",
            replica="unknown",
            success=False,
        )


def run_scenario(
    url: str,
    scenario: str,
    worker_count: int,
    questions: List[str],
    concurrency: int,
    timeout: int,
    verbose: bool = True,
) -> List[RequestResult]:
    results = []
    print(f"\n  Running {len(questions)} requests with concurrency={concurrency}…")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(run_single_request, url, q, scenario, worker_count, timeout): q
            for q in questions
        }
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            if verbose:
                status = "✓" if r.success else "✗"
                cache = r.cache_status[:3].upper()
                print(
                    f"  [{i:03d}/{len(questions)}] {status} {cache} "
                    f"{r.latency_ms:6.0f}ms  {r.question[:55]}"
                )

    return results


def aggregate(results: List[RequestResult], scenario: str, worker_count: int) -> ScenarioStats:
    relevant = [r for r in results if r.scenario == scenario and r.worker_count == worker_count]
    stats = ScenarioStats(
        scenario=scenario,
        worker_count=worker_count,
        n_requests=len(relevant),
        n_success=sum(1 for r in relevant if r.success),
        latencies=[r.latency_ms for r in relevant if r.success],
        cache_hits=sum(1 for r in relevant if r.cache_status == "hit"),
        semantic_hits=sum(1 for r in relevant if r.cache_status == "semantic_hit"),
        miss=sum(1 for r in relevant if r.cache_status == "miss"),
    )
    return stats


# ── Output writers ────────────────────────────────────────────────────────────

SCENARIO_LABELS = {
    "exact_off": "Exact Cache OFF",
    "exact_on": "Exact Cache ON",
    "semantic_on": "Semantic Cache ON",
}


def write_csv(all_results: List[RequestResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scenario", "worker_count", "question", "status_code",
                "latency_ms", "cache_status", "llm_backend", "replica", "success",
            ],
        )
        w.writeheader()
        for r in all_results:
            w.writerow(
                dict(
                    scenario=r.scenario,
                    worker_count=r.worker_count,
                    question=r.question,
                    status_code=r.status_code,
                    latency_ms=f"{r.latency_ms:.2f}",
                    cache_status=r.cache_status,
                    llm_backend=r.llm_backend,
                    replica=r.replica,
                    success=r.success,
                )
            )
    print(f"\nCSV  → {path}")


def write_json(stats_list: List[ScenarioStats], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for s in stats_list:
        data.append(
            {
                "scenario": s.scenario,
                "scenario_label": SCENARIO_LABELS.get(s.scenario, s.scenario),
                "worker_count": s.worker_count,
                "n_requests": s.n_requests,
                "n_success": s.n_success,
                "success_rate": round(s.success_rate, 4),
                "mean_latency_ms": round(s.mean, 2),
                "p50_latency_ms": round(s.p50, 2),
                "p95_latency_ms": round(s.p95, 2),
                "p99_latency_ms": round(s.p99, 2),
                "throughput_rps": round(s.throughput, 4),
                "cache_hit_rate": round(s.cache_hit_rate, 4),
                "exact_cache_hits": s.cache_hits,
                "semantic_cache_hits": s.semantic_hits,
                "cache_misses": s.miss,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": data}, f, indent=2)
    print(f"JSON → {path}")


def write_markdown(stats_list: List[ScenarioStats], path: Path, n_questions: int, concurrency: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# RAG Pipeline Benchmark Report\n")
    lines.append(f"- **Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    lines.append(f"- **Requests per scenario:** {n_questions}")
    lines.append(f"- **Concurrency:** {concurrency}")
    lines.append("")

    # Group by worker count
    worker_counts = sorted(set(s.worker_count for s in stats_list))
    for wc in worker_counts:
        subset = [s for s in stats_list if s.worker_count == wc]
        lines.append(f"\n## Workers: {wc}\n")
        lines.append(
            "| Scenario | Requests | Success% | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | RPS | Cache Hit% |"
        )
        lines.append(
            "|----------|----------|----------|-----------|----------|----------|----------|-----|------------|"
        )
        for s in subset:
            label = SCENARIO_LABELS.get(s.scenario, s.scenario)
            lines.append(
                f"| {label} "
                f"| {s.n_requests} "
                f"| {s.success_rate:.0%} "
                f"| {s.mean:.0f} "
                f"| {s.p50:.0f} "
                f"| {s.p95:.0f} "
                f"| {s.p99:.0f} "
                f"| {s.throughput:.2f} "
                f"| {s.cache_hit_rate:.0%} |"
            )

    lines.append("\n## Cache Breakdown\n")
    lines.append("| Scenario | Workers | Exact Hits | Semantic Hits | Misses |")
    lines.append("|----------|---------|-----------|---------------|--------|")
    for s in stats_list:
        label = SCENARIO_LABELS.get(s.scenario, s.scenario)
        lines.append(
            f"| {label} | {s.worker_count} | {s.cache_hits} | {s.semantic_hits} | {s.miss} |"
        )

    lines.append("\n---")
    lines.append("*Generated by `benchmarks/run_benchmark.py`*\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"MD   → {path}")


# ── Scenario definitions ──────────────────────────────────────────────────────

SCENARIOS = [
    # (key, exact_cache_enabled, semantic_cache_enabled)
    ("exact_off", False, False),
    ("exact_on", True, False),
    ("semantic_on", True, True),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="API Gateway URL")
    parser.add_argument("--workers", default="1,2,4", help="Comma-separated worker counts to simulate")
    parser.add_argument("--requests", type=int, default=20, help="Questions per scenario")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent threads per scenario")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout seconds")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR), help="Output directory")
    parser.add_argument("--skip-cache-toggle", action="store_true",
                        help="Skip /admin/cache calls (if endpoint not present)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    url = args.url.rstrip("/")
    n_req = args.requests
    worker_counts_raw = [w.strip() for w in args.workers.split(",")]
    worker_counts = []
    for w in worker_counts_raw:
        try:
            worker_counts.append(int(w))
        except ValueError:
            pass

    # Build question list (cycle through bank if needed)
    questions = []
    for i in range(n_req):
        questions.append(QUESTION_BANK[i % len(QUESTION_BANK)])

    print("=" * 70)
    print("RL-Optimized RAG — Benchmark Suite")
    print("=" * 70)
    print(f"  API URL       : {url}")
    print(f"  Questions     : {n_req}")
    print(f"  Concurrency   : {args.concurrency}")
    print(f"  Timeout       : {args.timeout}s")
    print(f"  Worker counts : {worker_counts}")
    print(f"  Scenarios     : {[s[0] for s in SCENARIOS]}")

    print("\nWaiting for API Gateway to be ready…")
    if not wait_for_gateway(url, timeout=30):
        print(f"ERROR: {url}/health not reachable after 30s. Is docker-compose up?", file=sys.stderr)
        sys.exit(1)
    print("  Gateway OK.")

    all_results: List[RequestResult] = []
    all_stats: List[ScenarioStats] = []

    for wc in worker_counts:
        print(f"\n{'='*70}")
        print(f"  WORKER COUNT: {wc}")
        print(f"{'='*70}")

        for scenario_key, exact_en, semantic_en in SCENARIOS:
            label = SCENARIO_LABELS[scenario_key]
            print(f"\n[{label}]")

            if not args.skip_cache_toggle:
                ok = set_cache_config(url, exact_en, semantic_en)
                if ok:
                    print(f"  Cache: exact={exact_en} semantic={semantic_en} → configured")
                else:
                    print(
                        f"  NOTE: /admin/cache not found — run with --skip-cache-toggle or add endpoint. "
                        f"Continuing with current gateway config."
                    )

            flush_caches(url)

            # Warm-up: one request before timing (not counted)
            print("  Warm-up request…")
            run_single_request(url, questions[0], "warmup", wc, args.timeout)

            scenario_results = run_scenario(
                url=url,
                scenario=scenario_key,
                worker_count=wc,
                questions=questions,
                concurrency=args.concurrency,
                timeout=args.timeout,
            )
            all_results.extend(scenario_results)
            stats = aggregate(all_results, scenario_key, wc)
            all_stats.append(stats)

            print(
                f"\n  → mean={stats.mean:.0f}ms  p50={stats.p50:.0f}ms  "
                f"p95={stats.p95:.0f}ms  p99={stats.p99:.0f}ms  "
                f"rps={stats.throughput:.2f}  hit%={stats.cache_hit_rate:.0%}"
            )

    # Write outputs
    print(f"\n{'='*70}")
    print("Writing results…")
    write_csv(all_results, out_dir / "results.csv")
    write_json(all_stats, out_dir / "summary.json")
    write_markdown(all_stats, out_dir / "report.md", n_req, args.concurrency)

    # Print final summary table to stdout
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Scenario':<22} {'Workers':>7} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'RPS':>7} {'Hit%':>6}"
    print(header)
    print("-" * len(header))
    for s in all_stats:
        label = SCENARIO_LABELS.get(s.scenario, s.scenario)
        print(
            f"{label:<22} {s.worker_count:>7} {s.mean:>7.0f}ms "
            f"{s.p50:>6.0f}ms {s.p95:>6.0f}ms {s.p99:>6.0f}ms "
            f"{s.throughput:>6.2f} {s.cache_hit_rate:>5.0%}"
        )
    print(f"\nDone. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
