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


def load_request_csvs(
    results_dir: Path,
    stamp: Optional[str] = None,
    prefix: str = "*",
) -> list[dict]:
    rows: list[dict] = []
    patterns = (
        [f"{prefix}_requests_*.csv"]
        if prefix != "*"
        else ["simulated_requests_*.csv", "real_requests_*.csv"]
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(results_dir.glob(pattern)))
    for path in paths:
        if stamp and stamp not in path.name:
            continue
        simulated = path.name.startswith("simulated_")
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
                        "simulated": (
                            row.get("simulated", str(simulated)).lower() == "true"
                        ),
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
        simulated = all(r.get("simulated", True) for r in group)
        summaries.append(
            {
                "simulated": simulated,
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


def plot_summaries(summaries: list[dict], out_path: Path, *, simulated: bool) -> None:
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
    title = (
        "SIMULATED mock benchmarks (not real model performance)"
        if simulated
        else "REAL remote vLLM benchmarks"
    )
    fig.suptitle(title)
    label = "simulated" if simulated else "real"

    axes[0].plot(xs, [s["p50_latency_s"] for s in summaries], marker="o", label="p50")
    axes[0].plot(xs, [s["p95_latency_s"] for s in summaries], marker="o", label="p95")
    axes[0].plot(xs, [s["p99_latency_s"] for s in summaries], marker="o", label="p99")
    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel(f"Latency (s) — {label}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, [s["mean_queue_wait_ms"] for s in summaries], marker="o", color="tab:orange")
    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel(f"Mean queue wait (ms) — {label}")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote plot {out_path}")


def analyze_dir(
    results_dir: Path,
    stamp: Optional[str] = None,
    prefix: str = "simulated",
) -> list[dict]:
    rows = load_request_csvs(results_dir, stamp=stamp, prefix=prefix)
    summaries = aggregate(rows)
    simulated = prefix != "real" and all(s.get("simulated", True) for s in summaries)
    tag = "simulated" if simulated else "real"
    out_json = results_dir / (
        f"{tag}_analysis_{stamp}.json" if stamp else f"{tag}_analysis_latest.json"
    )
    out_json.write_text(
        json.dumps(
            {
                "simulated": simulated,
                "note": (
                    "Mock results only"
                    if simulated
                    else "Real remote vLLM results — not fabricated"
                ),
                "p99_caveat": "p99 from small samples is unreliable",
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_path = results_dir / (
        f"{tag}_latency_vs_concurrency_{stamp}.png"
        if stamp
        else f"{tag}_latency_vs_concurrency_latest.png"
    )
    plot_summaries(summaries, plot_path, simulated=simulated)
    print(f"Wrote {out_json}")
    return summaries


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze benchmark CSVs (simulated or real)")
    p.add_argument("--results-dir", type=Path, default=Path("bench/results"))
    p.add_argument("--stamp", default=None, help="Optional timestamp filter matching CSV names")
    p.add_argument(
        "--prefix",
        default="simulated",
        choices=("simulated", "real", "*"),
        help="Which request CSV prefix to load",
    )
    args = p.parse_args()
    analyze_dir(args.results_dir, stamp=args.stamp, prefix=args.prefix)


if __name__ == "__main__":
    main()
