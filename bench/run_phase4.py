"""Run one Phase 4 condition against an already-running API.

Dry-run prints the plan and does not open a socket.
This module does not create a RunPod pod and does not touch a local GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from bench.load_test import run_concurrency, run_open_loop, write_csv
from bench.phase4 import json_safe, utc_now


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one Phase 4 benchmark condition")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--out", type=Path, default=Path("bench/results/phase4"))
    p.add_argument("--mode", choices=("dry-run", "smoke", "full"), default="dry-run")
    p.add_argument("--pattern", choices=("closed", "open"), default="closed")
    p.add_argument("--workload", default="repeated-schema")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--arrival-rate", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--prefix-caching", default="")
    p.add_argument("--admission", default="")
    p.add_argument("--gpu", default="NVIDIA L4")
    p.add_argument("--hourly-usd", type=float, default=0.59)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _scaled(args: argparse.Namespace) -> tuple[int, int, float]:
    if args.mode == "smoke":
        return min(args.requests, 4), min(args.warmup, 1), min(args.duration, 4.0)
    if args.mode == "dry-run":
        return args.requests, args.warmup, args.duration
    return args.requests, args.warmup, args.duration


async def _run(args: argparse.Namespace) -> dict:
    requests, warmup, duration = _scaled(args)
    dest = args.out / args.experiment_id
    summary_path = dest / "summary.json"
    if summary_path.exists() and not args.force and args.mode != "dry-run":
        return json.loads(summary_path.read_text(encoding="utf-8"))

    plan = {
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "pattern": args.pattern,
        "workload": args.workload,
        "concurrency": args.concurrency,
        "requests": requests,
        "warmup": warmup,
        "arrival_rate": args.arrival_rate,
        "duration_s": duration,
        "stream": args.stream,
        "prefix_caching": args.prefix_caching or None,
        "admission": args.admission or None,
        "gpu": args.gpu,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.mode == "dry-run":
        plan["status"] = "dry-run"
        print(json.dumps(plan, indent=2))
        return plan

    from bench.load_test import one_request, one_stream_request
    import httpx
    from bench.workloads import item

    started = utc_now()
    endpoint = "/generate/stream" if args.stream else "/generate"
    url = args.base_url.rstrip("/") + endpoint

    async def warmup_calls() -> None:
        if warmup <= 0:
            return
        async with httpx.AsyncClient() as client:
            for i in range(warmup):
                work = item(args.workload, i + args.seed)
                call = one_stream_request if args.stream else one_request
                await call(
                    client,
                    url,
                    concurrency=1,
                    index=i,
                    max_new_tokens=args.max_new_tokens,
                    timeout_s=args.timeout,
                    question=work.question,
                    schema=work.schema,
                )

    await warmup_calls()
    measured_start = utc_now()
    if args.stream and args.pattern == "closed":
        import time
        from bench.load_test import RequestRecord, summarize

        async with httpx.AsyncClient() as client:
            sem = asyncio.Semaphore(args.concurrency)

            async def worker(i: int) -> RequestRecord:
                async with sem:
                    work = item(args.workload, i + args.seed)
                    return await one_stream_request(
                        client,
                        url,
                        concurrency=args.concurrency,
                        index=i,
                        max_new_tokens=args.max_new_tokens,
                        timeout_s=args.timeout,
                        question=work.question,
                        schema=work.schema,
                    )

            t0 = time.perf_counter()
            records = list(await asyncio.gather(*(worker(i) for i in range(requests))))
            wall = time.perf_counter() - t0
        records.sort(key=lambda r: r.index)
        summary = summarize(records, wall)
        summary["pattern"] = "closed-loop"
        summary["workload"] = args.workload
        summary["concurrency"] = args.concurrency
        summary["stream"] = True
    elif args.pattern == "open":
        records, summary = await run_open_loop(
            base_url=args.base_url,
            rate=args.arrival_rate,
            duration_s=duration,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.timeout,
            workload=args.workload,
            seed=args.seed,
        )
    else:
        records, summary = await run_concurrency(
            base_url=args.base_url,
            concurrency=args.concurrency,
            requests=requests,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.timeout,
            workload=args.workload,
            seed=args.seed,
        )
    ended = utc_now()

    summary.update(plan)
    summary["status"] = "complete"
    summary["started_at"] = started.isoformat()
    summary["measured_start"] = measured_start.isoformat()
    summary["ended_at"] = ended.isoformat()
    summary["endpoint"] = "/generate" if args.pattern == "open" else endpoint
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "requests.csv", records)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps(json_safe({
        "experiment_id": args.experiment_id,
        "successes": summary.get("successes"),
        "rejections": summary.get("rejections"),
        "simulated": summary.get("simulated"),
        "p99_latency_s": summary.get("p99_latency_s"),
        "output_tokens_per_sec": summary.get("output_tokens_per_sec"),
        "mean_ttft_ms": summary.get("mean_ttft_ms"),
    }), indent=2))
    return summary


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
