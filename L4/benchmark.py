"""
Benchmark ONNX Runtime inference across providers.
Usage:
    python benchmark.py --model model.onnx --n 100
"""

import argparse
import time
import numpy as np


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def run_benchmark(model_path: str, provider: str, n: int = 100) -> dict:
    import onnxruntime as ort

    from app.inference import PROVIDER_MAP, _build_session

    session = _build_session(model_path, provider)
    actual_provider = session.get_providers()[0]
    input_name = session.get_inputs()[0].name

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warm-up
    for _ in range(5):
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    total_s = latencies.sum() / 1000
    return {
        "provider":       actual_provider,
        "n":              n,
        "mean_ms":        round(float(latencies.mean()), 2),
        "median_ms":      round(float(np.median(latencies)), 2),
        "p95_ms":         round(float(np.percentile(latencies, 95)), 2),
        "p99_ms":         round(float(np.percentile(latencies, 99)), 2),
        "min_ms":         round(float(latencies.min()), 2),
        "max_ms":         round(float(latencies.max()), 2),
        "throughput_rps": round(n / total_s, 1),
    }


def print_table(results: list[dict]):
    headers = ["Provider", "Mean ms", "Median ms", "P95 ms", "P99 ms", "Min ms", "Max ms", "RPS"]
    rows = [
        [
            r["provider"].replace("ExecutionProvider", ""),
            r["mean_ms"], r["median_ms"], r["p95_ms"],
            r["p99_ms"], r["min_ms"], r["max_ms"], r["throughput_rps"],
        ]
        for r in results
    ]
    col_w = [max(len(str(rows[i][j])) for i in range(len(rows))) for j in range(len(headers))]
    col_w = [max(col_w[j], len(headers[j])) for j in range(len(headers))]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.onnx")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--providers", nargs="+", default=["cpu", "cuda", "tensorrt"])
    args = parser.parse_args()

    results = []
    for p in args.providers:
        print(f"\nBenchmarking provider: {p} ...")
        try:
            r = run_benchmark(args.model, p, args.n)
            results.append(r)
            print(f"  mean={r['mean_ms']} ms  rps={r['throughput_rps']}")
        except Exception as e:
            print(f"  SKIPPED ({e})")

    if results:
        print("\n=== Results ===")
        print_table(results)
