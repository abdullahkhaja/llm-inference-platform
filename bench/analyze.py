#!/usr/bin/env python3
"""Aggregate benchmark CSVs and emit simple performance plots.

All Phase 1 outputs are labeled as SIMULATED.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from bench.load_test import percentile


def load_request_csvs(results_dir: Path, stamp: Optional[str] = None) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("simulated_requests_*.csv")):
        if stamp and stamp not in path.name:
            continue
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(
                    {
                        "concurrency": int(row["concurrency"]),
                        "ok": row["ok"].lower() == "true",
                        "latency_s": float(row["latency_s"]),
                        "queue_wait_ms": float(row["queue_wait_ms"]),
                        "inference_ms": float(row["inference_ms"]),
                        "output_tokens": int(row["output_tokens"]),
                    }
                )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    by_c: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["ok"]:
            by_c[row["concurrency"]].append(row)

    summaries = []
    for c in sorted(by_c):
        group = by_c[c]
        lat = [r["latency_s"] for r in group]
        qw = [r["queue_wait_ms"] for r in group]
        summaries.append(
            {
                "simulated": True,
                "concurrency": c,
                "n": len(group),
                "mean_latency_s": sum(lat) / len(lat),
                "p50_latency_s": percentile(lat, 50),
                "p95_latency_s": percentile(lat, 95),
                "p99_latency_s": percentile(lat, 99),
                "mean_queue_wait_ms": sum(qw) / len(qw),
            }
        )
    return summaries


def plot_summaries(summaries: list[dict], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plots. Install requirements.txt"
        ) from exc

    if not summaries:
        print("No data to plot")
        return

    xs = [s["concurrency"] for s in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Phase 1 SIMULATED mock benchmarks (not real model performance)")

    axes[0].plot(xs, [s["p50_latency_s"] for s in summaries], marker="o", label="p50")
    axes[0].plot(xs, [s["p95_latency_s"] for s in summaries], marker="o", label="p95")
    axes[0].plot(xs, [s["p99_latency_s"] for s in summaries], marker="o", label="p99")
    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel("Latency (s) — simulated")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, [s["mean_queue_wait_ms"] for s in summaries], marker="o", color="tab:orange")
    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("Mean queue wait (ms) — simulated")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote plot {out_path}")


def analyze_dir(results_dir: Path, stamp: Optional[str] = None) -> list[dict]:
    rows = load_request_csvs(results_dir, stamp=stamp)
    summaries = aggregate(rows)
    out_json = results_dir / (
        f"simulated_analysis_{stamp}.json" if stamp else "simulated_analysis_latest.json"
    )
    out_json.write_text(
        json.dumps(
            {
                "simulated": True,
                "note": "Phase 1 mock results only",
                "p99_caveat": "p99 from small samples is unreliable",
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_path = results_dir / (
        f"simulated_latency_vs_concurrency_{stamp}.png"
        if stamp
        else "simulated_latency_vs_concurrency_latest.png"
    )
    plot_summaries(summaries, plot_path)
    print(f"Wrote {out_json}")
    return summaries


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Phase 1 simulated benchmark CSVs")
    p.add_argument("--results-dir", type=Path, default=Path("bench/results"))
    p.add_argument("--stamp", default=None, help="Optional timestamp filter matching CSV names")
    args = p.parse_args()
    analyze_dir(args.results_dir, stamp=args.stamp)


if __name__ == "__main__":
    main()
