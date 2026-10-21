#!/usr/bin/env python3
"""Async load tester for POST /generate.

Phase 1 results are SIMULATED mock timings — not real model performance.

Default is a lightweight smoke test. Use --requests 1000+ for more meaningful
tail-latency estimates. p99 from small samples is unreliable.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

DEFAULT_QUESTION = "List all employees earning more than 50000."
DEFAULT_SCHEMA = "CREATE TABLE employees (id INTEGER, name TEXT, salary INTEGER);"


@dataclass
class RequestRecord:
    concurrency: int
    index: int
    ok: bool
    http_status: int
    latency_s: float
    queue_wait_ms: Optional[float]
    inference_ms: Optional[float]
    output_tokens: int
    input_tokens: int
    error: str
    simulated: bool = True
    ttft_ms: Optional[float] = None


def arrival_times(rate: float, duration_s: float, start: float = 0.0) -> list[float]:
    """Monotonic open-loop schedule. Offered count does not depend on latency."""
    if rate <= 0 or duration_s <= 0:
        return []
    step = 1.0 / rate
    times: list[float] = []
    t = start
    end = start + duration_s
    while t < end - 1e-12:
        times.append(t)
        t += step
    return times


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def summarize(
    records: list[RequestRecord],
    wall_s: float,
    *,
    arrival_window_s: Optional[float] = None,
) -> dict:
    ok = [r for r in records if r.ok]
    latencies = [r.latency_s for r in ok]
    queue_waits = [r.queue_wait_ms for r in ok if r.queue_wait_ms is not None]
    failures = len(records) - len(ok)
    timeouts = sum(1 for r in records if "timeout" in r.error.lower())
    rejected = [r for r in records if r.http_status == 503]
    total_out_tokens = sum(r.output_tokens for r in ok)
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    offer_window = arrival_window_s if arrival_window_s and arrival_window_s > 0 else wall_s
    simulated = all(r.simulated for r in ok) if ok else True
    tok_key = (
        "simulated_output_tokens_per_sec" if simulated else "output_tokens_per_sec"
    )
    note = (
        "Mock backend — not real model performance"
        if simulated
        else "Real remote vLLM backend — engine token counts"
    )
    return {
        "simulated": simulated,
        "note": note,
        "requests": len(records),
        "successes": len(ok),
        "failures": failures,
        "timeouts": timeouts,
        "rejections": len(rejected),
        "offered": len(records),
        "failure_rate": failures / len(records) if records else 0.0,
        "timeout_rate": timeouts / len(records) if records else 0.0,
        "wall_seconds": wall_s,
        "arrival_window_s": arrival_window_s,
        "offered_rps": len(records) / offer_window if offer_window > 0 else 0.0,
        "rps": len(ok) / wall_s if wall_s > 0 else 0.0,
        tok_key: total_out_tokens / wall_s if wall_s > 0 else 0.0,
        "mean_latency_s": statistics.fmean(latencies) if latencies else float("nan"),
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "p99_latency_s": percentile(latencies, 99),
        "mean_queue_wait_ms": statistics.fmean(queue_waits) if queue_waits else float("nan"),
        "p50_queue_wait_ms": percentile(queue_waits, 50) if queue_waits else float("nan"),
        "p95_queue_wait_ms": percentile(queue_waits, 95) if queue_waits else float("nan"),
        "p99_caveat": (
            "p99 estimates from small samples are unreliable; "
            "use --requests 1000+ for stronger tail-latency signal"
            if len(ok) < 1000
            else "empirical percentile of successful requests only"
        ),
        "p50_rejection_latency_s": percentile([r.latency_s for r in rejected], 50)
        if rejected
        else None,
        "mean_ttft_ms": statistics.fmean(ttfts) if ttfts else None,
        "p50_ttft_ms": percentile(ttfts, 50) if ttfts else None,
        "p95_ttft_ms": percentile(ttfts, 95) if ttfts else None,
    }


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    *,
    concurrency: int,
    index: int,
    max_new_tokens: int,
    timeout_s: float,
    question: str = DEFAULT_QUESTION,
    schema: str = DEFAULT_SCHEMA,
) -> RequestRecord:
    payload = {
        "question": question,
        "schema": schema,
        "max_new_tokens": max_new_tokens,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=timeout_s)
        latency = time.perf_counter() - t0
        if resp.status_code != 200:
            return RequestRecord(
                concurrency=concurrency,
                index=index,
                ok=False,
                http_status=resp.status_code,
                latency_s=latency,
                queue_wait_ms=None,
                inference_ms=None,
                output_tokens=0,
                input_tokens=0,
                error=resp.text[:200],
            )
        data = resp.json()
        qw = data.get("queue_wait_ms")
        inf = data.get("inference_ms")
        return RequestRecord(
            concurrency=concurrency,
            index=index,
            ok=True,
            http_status=200,
            latency_s=latency,
            queue_wait_ms=float(qw) if qw is not None else None,
            inference_ms=float(inf) if inf is not None else None,
            output_tokens=int(data.get("output_tokens", 0)),
            input_tokens=int(data.get("input_tokens", 0)),
            error="",
            simulated=bool(data.get("simulated", True)),
        )
    except httpx.TimeoutException as exc:
        return RequestRecord(
            concurrency=concurrency,
            index=index,
            ok=False,
            http_status=0,
            latency_s=time.perf_counter() - t0,
            queue_wait_ms=None,
            inference_ms=None,
            output_tokens=0,
            input_tokens=0,
            error=f"timeout: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 — record any client failure
        return RequestRecord(
            concurrency=concurrency,
            index=index,
            ok=False,
            http_status=0,
            latency_s=time.perf_counter() - t0,
            queue_wait_ms=None,
            inference_ms=None,
            output_tokens=0,
            input_tokens=0,
            error=str(exc),
        )


def first_content_token_ttft_s(blocks: list[tuple[float, str]]) -> Optional[float]:
    """Elapsed seconds to the first SSE token event with nonempty text.

    ``blocks`` are ``(elapsed_s, sse_chunk)``. A ``meta`` event is not content.
    """
    for elapsed, chunk in blocks:
        event = "message"
        data_lines: list[str] = []
        for line in chunk.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if event != "token" or not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        text = payload.get("text") if isinstance(payload, dict) else None
        if isinstance(text, str) and text.strip():
            return elapsed
    return None


async def one_stream_request(
    client: httpx.AsyncClient,
    url: str,
    *,
    concurrency: int,
    index: int,
    max_new_tokens: int,
    timeout_s: float,
    question: str = DEFAULT_QUESTION,
    schema: str = DEFAULT_SCHEMA,
) -> RequestRecord:
    """POST /generate/stream and time the first nonempty content token."""
    payload = {
        "question": question,
        "schema": schema,
        "max_new_tokens": max_new_tokens,
    }
    t0 = time.perf_counter()
    blocks: list[tuple[float, str]] = []
    buffer = ""
    usage: dict = {}
    simulated = True
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", errors="replace")[:200]
                return RequestRecord(
                    concurrency=concurrency,
                    index=index,
                    ok=False,
                    http_status=resp.status_code,
                    latency_s=time.perf_counter() - t0,
                    queue_wait_ms=None,
                    inference_ms=None,
                    output_tokens=0,
                    input_tokens=0,
                    error=body,
                )
            async for piece in resp.aiter_text():
                buffer += piece
                while "\n\n" in buffer:
                    raw, buffer = buffer.split("\n\n", 1)
                    elapsed = time.perf_counter() - t0
                    blocks.append((elapsed, raw))
                    if "event: usage" in raw:
                        data_lines = [
                            line.split(":", 1)[1].strip()
                            for line in raw.splitlines()
                            if line.startswith("data:")
                        ]
                        if data_lines:
                            try:
                                usage = json.loads("\n".join(data_lines))
                            except json.JSONDecodeError:
                                usage = {}
        latency = time.perf_counter() - t0
        ttft = first_content_token_ttft_s(blocks)
        simulated = bool(usage.get("simulated", True)) if usage else True
        return RequestRecord(
            concurrency=concurrency,
            index=index,
            ok=True,
            http_status=200,
            latency_s=latency,
            queue_wait_ms=None,
            inference_ms=(
                float(usage["inference_s"]) * 1000.0
                if usage.get("inference_s") is not None
                else None
            ),
            output_tokens=int(usage.get("output_tokens") or 0),
            input_tokens=int(usage.get("input_tokens") or 0),
            error="",
            simulated=simulated,
            ttft_ms=None if ttft is None else round(ttft * 1000.0, 3),
        )
    except httpx.TimeoutException as exc:
        return RequestRecord(
            concurrency, index, False, 0, time.perf_counter() - t0,
            None, None, 0, 0, f"timeout: {exc}", True, None,
        )
    except Exception as exc:  # noqa: BLE001
        return RequestRecord(
            concurrency, index, False, 0, time.perf_counter() - t0,
            None, None, 0, 0, str(exc), True, None,
        )


async def run_concurrency(
    *,
    base_url: str,
    concurrency: int,
    requests: int,
    max_new_tokens: int,
    timeout_s: float,
    workload: str = "short",
    seed: int = 42,
) -> tuple[list[RequestRecord], dict]:
    from bench.workloads import item

    url = base_url.rstrip("/") + "/generate"
    sem = asyncio.Semaphore(concurrency)
    records: list[RequestRecord] = []

    async with httpx.AsyncClient() as client:
        async def worker(i: int) -> None:
            async with sem:
                work = item(workload, i + seed)
                rec = await one_request(
                    client,
                    url,
                    concurrency=concurrency,
                    index=i,
                    max_new_tokens=max_new_tokens,
                    timeout_s=timeout_s,
                    question=work.question,
                    schema=work.schema,
                )
                records.append(rec)

        wall0 = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(requests)))
        wall_s = time.perf_counter() - wall0

    records.sort(key=lambda r: r.index)
    summary = summarize(records, wall_s)
    summary["concurrency"] = concurrency
    summary["max_new_tokens"] = max_new_tokens
    summary["workload"] = workload
    summary["pattern"] = "closed-loop"
    summary["synthetic_workload"] = True
    return records, summary


async def run_open_loop(
    *,
    base_url: str,
    rate: float,
    duration_s: float,
    max_new_tokens: int,
    timeout_s: float,
    workload: str = "short",
    seed: int = 42,
) -> tuple[list[RequestRecord], dict]:
    """Schedule arrivals on a monotonic clock. Do not wait for completions."""
    from bench.workloads import item

    url = base_url.rstrip("/") + "/generate"
    times = arrival_times(rate, duration_s)
    records: list[RequestRecord] = []

    async with httpx.AsyncClient() as client:
        async def launch(i: int, when: float, origin: float) -> RequestRecord:
            delay = origin + when - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            work = item(workload, i + seed)
            return await one_request(
                client,
                url,
                concurrency=0,
                index=i,
                max_new_tokens=max_new_tokens,
                timeout_s=timeout_s,
                question=work.question,
                schema=work.schema,
            )

        origin = time.perf_counter()
        wall0 = origin
        records = list(
            await asyncio.gather(
                *(launch(i, when, origin) for i, when in enumerate(times))
            )
        )
        wall_s = time.perf_counter() - wall0

    summary = summarize(records, wall_s, arrival_window_s=duration_s)
    summary["concurrency"] = 0
    summary["arrival_rate"] = rate
    summary["offered_rate"] = rate
    summary["pattern"] = "open-loop"
    summary["workload"] = workload
    summary["synthetic_workload"] = True
    summary["max_new_tokens"] = max_new_tokens
    return records, summary


def write_csv(path: Path, records: list[RequestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else [
        "concurrency",
        "index",
        "ok",
        "http_status",
        "latency_s",
        "queue_wait_ms",
        "inference_ms",
        "output_tokens",
        "input_tokens",
        "error",
        "simulated",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load test for /generate (mock or real vLLM)")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument(
        "--concurrency",
        default="1,2,4,8,16",
        help="Comma-separated concurrency levels",
    )
    p.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Requests per concurrency level (default: smoke=20; use 1000+ for p99)",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=42, help="Reserved for future payload RNG")
    p.add_argument(
        "--label",
        default="auto",
        help="Filename prefix: auto|simulated|real (auto from first successful response)",
    )
    p.add_argument(
        "--backend-info",
        default="",
        help="Optional free-text hardware / backend note stored in rollup JSON",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bench/results"),
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Run analyze.py after writing CSVs",
    )
    p.add_argument(
        "--workload",
        default="short",
        help="short|medium|long|repeated-schema|unrelated-schema (synthetic)",
    )
    p.add_argument(
        "--pattern",
        default="closed",
        choices=("closed", "open"),
        help="closed: fixed concurrency. open: arrival rate independent of latency",
    )
    p.add_argument("--arrival-rate", type=float, default=1.0, help="Open-loop requests/sec")
    p.add_argument("--duration", type=float, default=5.0, help="Open-loop seconds")
    p.add_argument("--experiment-id", default="")
    p.add_argument("--gpu", default="")
    p.add_argument("--prefix-caching", default="", help="true|false if known for this run")
    p.add_argument("--admission", default="", help="true|false if known for this run")
    p.add_argument(
        "--latency-objective",
        type=float,
        default=None,
        help="Seconds. goodput counts successes at or under this latency",
    )
    return p.parse_args()


async def amain() -> None:
    args = parse_args()
    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Load test — labels simulated/real from API responses")
    print(f"base_url={args.base_url} requests/level={args.requests} tokens={args.max_new_tokens}")
    if args.requests < 200:
        print(
            "NOTE: small sample size — p95/p99 estimates are unreliable. "
            "Re-run with --requests 1000+ for stronger tail metrics."
        )
    print("=" * 60)

    all_summaries: list[dict] = []
    label = args.label
    if args.pattern == "open":
        levels = [0]
    for c in levels:
        print(f"\n--- pattern={args.pattern} concurrency={c} ---")
        if args.pattern == "open":
            records, summary = await run_open_loop(
                base_url=args.base_url,
                rate=args.arrival_rate,
                duration_s=args.duration,
                max_new_tokens=args.max_new_tokens,
                timeout_s=args.timeout,
                workload=args.workload,
                seed=args.seed,
            )
        else:
            records, summary = await run_concurrency(
            base_url=args.base_url,
            concurrency=c,
            requests=args.requests,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.timeout,
            workload=args.workload,
            seed=args.seed,
        )
        summary["experiment_id"] = args.experiment_id or None
        summary["gpu"] = args.gpu or None
        summary["prefix_caching"] = args.prefix_caching or None
        summary["admission"] = args.admission or None
        summary["vllm_version"] = "v0.8.5" if label == "real" or args.label == "real" else None
        if args.latency_objective is not None:
            good = sum(
                1
                for rec in records
                if rec.ok and rec.latency_s <= args.latency_objective
            )
            summary["latency_objective_s"] = args.latency_objective
            wall = float(summary["wall_seconds"])
            summary["goodput_rps"] = good / wall if wall > 0 else 0.0
        if label == "auto":
            label = "simulated" if summary.get("simulated", True) else "real"
        if args.backend_info:
            summary["backend_info"] = args.backend_info
        csv_path = out_dir / f"{label}_requests_c{c}_{stamp}.csv"
        write_csv(csv_path, records)
        summary_path = out_dir / f"{label}_summary_c{c}_{stamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        all_summaries.append(summary)
        keys = [
            "concurrency",
            "rps",
            "simulated",
            "mean_latency_s",
            "p50_latency_s",
            "p95_latency_s",
            "p99_latency_s",
            "mean_queue_wait_ms",
            "failure_rate",
        ]
        for extra in ("simulated_output_tokens_per_sec", "output_tokens_per_sec"):
            if extra in summary:
                keys.append(extra)
        print(json.dumps({k: summary[k] for k in keys if k in summary}, indent=2))
        print(f"wrote {csv_path}")

    rollup = out_dir / f"{label}_summary_all_{stamp}.json"
    rollup.write_text(
        json.dumps(
            {
                "simulated": label == "simulated",
                "label": label,
                "backend_info": args.backend_info or None,
                "levels": all_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nRollup: {rollup}")

    if args.analyze:
        from bench.analyze import analyze_dir

        analyze_dir(out_dir, stamp=stamp)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
