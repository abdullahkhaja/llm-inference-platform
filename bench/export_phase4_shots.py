"""Render a small set of Grafana PNGs from completed Phase 4 windows.

Runs against Grafana on localhost. Does not start a model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from bench.capture_grafana import capture_one, windows_from_results

DASHBOARD = "phase3-serving"
# Panel ids match ops/grafana/dashboards/phase3_serving.json.
SHOTS = [
    ("grafana-overview.png", "ALL", None),
    ("prefix-cache-off.png", "A-repeated-off", None),
    ("prefix-cache-on.png", "A-repeated-on", None),
    ("prefix-cache-hit-ratio.png", "A-repeated-on", 9),
    ("admission-overload.png", "B-above-on", None),
    ("admission-rejections.png", "B-above-on", 6),
    ("concurrency-throughput.png", "C-sweep", 2),
    ("concurrency-latency.png", "C-sweep", 3),
    ("gpu-kv-cache.png", "C-sweep", 8),
]


def main() -> None:
    root = Path(os.environ.get("PHASE4_RESULTS", "/work/phase4"))
    dest_dir = Path(os.environ.get("PHASE4_SHOTS", "docs/images/phase4"))
    windows = {row["experiment_id"]: row for row in windows_from_results(root)}
    concurrency = [row for key, row in windows.items() if key.startswith("C-")]
    if concurrency:
        windows["C-sweep"] = {
            "experiment_id": "C-sweep",
            "start": min(row["start"] for row in concurrency),
            "end": max(row["end"] for row in concurrency),
            "prefix_caching": concurrency[-1].get("prefix_caching"),
            "admission": concurrency[-1].get("admission"),
            "concurrency": "1-16",
            "workload": concurrency[-1].get("workload"),
        }
    if windows:
        measured = [row for key, row in windows.items() if key not in {"ALL", "C-sweep"}]
        windows["ALL"] = {
            "experiment_id": "ALL",
            "start": min(row["start"] for row in measured),
            "end": max(row["end"] for row in measured),
            "prefix_caching": None,
            "admission": None,
            "concurrency": None,
            "workload": "all-completed",
        }
    manifest = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename, experiment_id, panel_id in SHOTS:
        window = windows.get(experiment_id)
        dest = dest_dir / filename
        if window is None:
            manifest.append({
                "file": filename,
                "experiment_id": experiment_id,
                "ok": False,
                "error": "missing experiment window",
            })
            continue
        start = datetime.fromisoformat(window["start"])
        end = datetime.fromisoformat(window["end"])
        try:
            row = capture_one(
                base_url=os.environ.get("GRAFANA_URL", "http://127.0.0.1:3000"),
                token=os.environ.get("GRAFANA_TOKEN", ""),
                dashboard_uid=DASHBOARD,
                panel_id=panel_id,
                start=start,
                end=end,
                dest=dest,
            )
            row["experiment_id"] = experiment_id
            row["config"] = {
                "prefix_caching": window.get("prefix_caching"),
                "admission": window.get("admission"),
                "concurrency": window.get("concurrency"),
                "workload": window.get("workload"),
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "file": str(dest),
                "experiment_id": experiment_id,
                "panel_id": panel_id,
                "from": window["start"],
                "to": window["end"],
                "ok": False,
                "error": str(exc)[:300],
            }
        manifest.append(row)
        print(json.dumps({"file": filename, "ok": row.get("ok"), "experiment_id": experiment_id}))
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not any(row.get("ok") for row in manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
