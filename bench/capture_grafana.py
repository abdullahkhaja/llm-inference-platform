"""Export Grafana panels for a recorded benchmark window.

Credentials come from the environment (GRAFANA_URL, GRAFANA_TOKEN).
This module does not start a GPU workload.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from bench.phase4 import grafana_range_ms, is_png, redact_secrets


def render_path(dashboard_uid: str, panel_id: int | None, start_ms: int, end_ms: int) -> str:
    query = urlencode({
        "from": start_ms,
        "to": end_ms,
        "width": 1600,
        "height": 900,
        "tz": "UTC",
        "orgId": 1,
    })
    if panel_id is None:
        return f"/render/d/{dashboard_uid}?{query}"
    return f"/render/d-solo/{dashboard_uid}?{query}&panelId={panel_id}"


def save_png(path: Path, blob: bytes) -> None:
    if not is_png(blob):
        raise ValueError(f"not a PNG: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


def capture_one(
    *,
    base_url: str,
    token: str,
    dashboard_uid: str,
    panel_id: int | None,
    start: datetime,
    end: datetime,
    dest: Path,
) -> dict:
    import httpx

    start_ms, end_ms = grafana_range_ms(start, end)
    path = render_path(dashboard_uid, panel_id, start_ms, end_ms)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    auth = None
    user = os.environ.get("GRAFANA_USER", "")
    password = os.environ.get("GRAFANA_PASSWORD", "")
    if user and password and not token:
        auth = (user, password)
    response = httpx.get(
        base_url.rstrip("/") + path,
        headers=headers,
        auth=auth,
        timeout=120.0,
    )
    response.raise_for_status()
    save_png(dest, response.content)
    return {
        "file": str(dest),
        "dashboard_uid": dashboard_uid,
        "panel_id": panel_id,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "bytes": len(response.content),
        "ok": True,
    }


def windows_from_results(root: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            continue
        start = summary.get("measured_start") or summary.get("started_at")
        end = summary.get("ended_at")
        if not start or not end:
            continue
        rows.append({
            "experiment_id": summary.get("experiment_id") or summary_path.parent.name,
            "start": start,
            "end": end,
            "prefix_caching": summary.get("prefix_caching"),
            "admission": summary.get("admission"),
            "concurrency": summary.get("concurrency"),
            "workload": summary.get("workload"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one Grafana panel to PNG")
    parser.add_argument("--dashboard-uid", default="phase3-serving")
    parser.add_argument("--panel-id", type=int)
    parser.add_argument("--start", required=True, help="ISO-8601 UTC")
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    try:
        row = capture_one(
            base_url=os.environ.get("GRAFANA_URL", "http://127.0.0.1:13000"),
            token=os.environ.get("GRAFANA_TOKEN", ""),
            dashboard_uid=args.dashboard_uid,
            panel_id=args.panel_id,
            start=start,
            end=end,
            dest=args.out,
        )
    except Exception as exc:  # noqa: BLE001
        row = {"ok": False, "error": redact_secrets(str(exc)), "file": str(args.out)}
    print(json.dumps(row, indent=2))
    if not row.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
