# GPU Cost Tracking & Inference Economics

This file records rented GPU time so inference cost can be separated from model-quality results. Per-session figures are estimates (`duration × quoted $/hr`). The running total is the amount actually spent. After Phase 4, that total covers every phase.

There is no fixed spending cap. Sessions should stay short: terminate idle pods, keep images pinned, and avoid leaving a GPU up after the script finishes. A few dollars of additional credit is enough to expect Phases 3 and 4 to fit, if runs stay bounded. That is a planning note, not a limit. Add credit if an experiment needs it.

## GPU session history

Quoted rate for these sessions: **$0.59/hour** (NVIDIA L4, Secure Cloud). Estimated dollars are `duration_hours × 0.59`, rounded to the cent. The Phase 2 rows add up to the **$0.46** spent through Phase 2. The Phase 3 row is not part of that confirmed total.

| Date (UTC) | Phase | Purpose | GPU | $/hr (quoted) | Duration (min) | Est. $ | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-15 | 2 | Balance probe | L4 Secure | 0.59 | ~2 | 0.02 | Create succeeded only as a balance check; terminated |
| 2026-10-15 | 2 | Failed startup | L4 Secure | 0.59 | ~37 | 0.36 | Boot script called `python`; image only has `python3`. Proxy never became ready. Terminated |
| 2026-10-15 | 2 | Crash-loop retry | L4 Secure | 0.59 | ~1 | 0.01 | Same `python` entrypoint. Terminated immediately |
| 2026-10-15 | 2 | Successful smoke test | L4 Secure | 0.59 | ~7 | 0.07 | EU-RO-1. `spider-sft` served real SQL (`simulated: false`). Terminated after the smoke |
| 2026-10-17 | 3 | Phase 3 validation | L4 Secure | 0.59 | ~27 | 0.27 | EU-RO-1. Terminated after the validation |

## Cost summary

Figures below are estimates, not an invoice.

| Bucket | Amount (USD) |
| --- | --- |
| Phase 2 session estimates | 0.46 |
| Phase 3 session estimate | 0.27 |
| Estimated sessions so far | **0.73** |
| All phases, after Phase 4 | — |

Failed startups are development cost. They are not part of inference cost-per-request.

The Phase 3 row is `duration × $0.59`, so $0.27 is an estimate.

## Cost per successful request

```text
cost per successful request = attributable GPU cost / successful requests
```

Which cost goes in the numerator depends on the question:

| Quantity | What it includes | Use |
| --- | --- | --- |
| Benchmark-window cost | GPU time while the measured requests were in flight | Inference efficiency of that run |
| Full session cost | Startup, weight download, load, benchmark, and idle time until terminate | What that rental actually cost |
| Development spend | Probes, failed boots, smokes, and other non-benchmark sessions | Project spend, not a serving SLO |

Hourly price times wall-clock hours sets the session cost. Higher throughput or higher concurrency lowers cost per request only when the GPU is doing useful work during that window. Cost per request also moves with model size, sequence length, batching, and how long the machine sits idle.

For cost per request, use the **final successful benchmark run** only. Do not fold the Phase 2 sessions into that ratio. The all-phases total in the summary above is a separate number: update it after Phase 4 to Phase 2 plus every later session.

Phase 2 did not record a request count intended for this ratio. The smoke proved the path; it is not an efficiency result.

## Phase 4 measurements

When a final benchmark exists, record at least:

- GPU type and quoted hourly rate
- Run duration and whether the cost is the benchmark window or the full session
- Requests issued and requests succeeded
- Tokens generated
- Average output tokens/sec and requests/sec
- p95 and p99 latency
- Cost per successful request
- Cost per million generated tokens (`session_or_window_cost / generated_tokens × 1e6`)

No Phase 4 row exists yet, so those values are not filled in.
