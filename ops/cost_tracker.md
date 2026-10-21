# GPU Cost Tracking & Inference Economics

This file records rented GPU time so inference cost stays separate from model-quality results. Per-session figures are estimates (`duration × quoted $/hr`) unless a row says an invoice posted that amount. None of the rows below are invoice-confirmed.

There is no spending cap in this repository. Terminate the pod when the artifacts are downloaded. Stopping vLLM, Prometheus, or Grafana does not stop RunPod billing.

## GPU session history

Quoted rate for these sessions: **$0.59/hour** (NVIDIA L4, Secure Cloud). Estimated dollars are `duration_hours × 0.59`, rounded to the cent. The Phase 2 rows add up to **$0.46**. Later rows are the same kind of estimate. They are not part of that Phase 2 total.

| Date (UTC) | Phase | Purpose | GPU | $/hr (quoted) | Duration (min) | Est. $ | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-15 | 2 | Balance probe | L4 Secure | 0.59 | ~2 | 0.02 | Create succeeded only as a balance check; terminated |
| 2026-10-15 | 2 | Failed startup | L4 Secure | 0.59 | ~37 | 0.36 | Boot script called `python`; image only has `python3`. Proxy never became ready. Terminated |
| 2026-10-15 | 2 | Crash-loop retry | L4 Secure | 0.59 | ~1 | 0.01 | Same `python` entrypoint. Terminated immediately |
| 2026-10-15 | 2 | Successful smoke test | L4 Secure | 0.59 | ~7 | 0.07 | EU-RO-1. `spider-sft` served real SQL (`simulated: false`). Terminated after the smoke |
| 2026-10-17 | 3 | Phase 3 validation | L4 Secure | 0.59 | ~27 | 0.27 | EU-RO-1. Terminated after the validation |
| 2026-10-20 | 4 | Phase 4 benchmark | L4 Secure | 0.59 | ~81 | 0.80 | EU-RO-1. Caching, admission, and concurrency, then Grafana export. Terminated after the PNGs and request records were downloaded |

The Phase 4 clock is 20:25:39Z to 21:46:32Z. The end is the SSH close after the pod delete returned 204, not a billing-API timestamp.

## Cost summary

| Bucket | Amount (USD) | Basis |
| --- | --- | --- |
| Phase 2 session estimates | 0.46 | Four rows above. Treated as the spent Phase 2 total |
| Phase 3 session estimate | 0.27 | ~27 min × $0.59 |
| Phase 4 session estimate | 0.80 | ~81 min × $0.59 |
| Estimated sessions so far | **1.53** | Sum of the three lines. Not an invoice |

Failed startups are development cost. They are not part of inference cost-per-request. Do not add the $0.46 or the $0.27 to the Phase 4 efficiency ratios.

## Phase 4 inference economics

Two different clocks:

| Clock | What it includes | Hours | Est. $ |
| --- | --- | --- | --- |
| Full session | Create through terminate, including model load and Grafana | 1.348 | 0.80 |
| Measured windows | Sum of condition wall clocks from the request records | 0.9445 | 0.557 |

The efficiency ratios use the measured-window estimate only.

| Input | Value |
| --- | --- |
| Successful requests | 10,388 |
| Generated output tokens | 136,554 |
| Window cost | $0.557 |
| Cost per successful request | $0.000054 |
| Cost per million output tokens | $4.08 |

```text
cost_per_successful_request = 0.557 / 10388 = 0.0000536
cost_per_1M_output_tokens   = 0.557 / 136554 × 1_000_000 = 4.08
```

The window excludes weight download, the vLLM restarts between conditions, and image rendering. It still mixes several workloads, so it is a campaign cost, not the cost of the concurrency-16 row alone. Using the $0.80 session instead of $0.557 would charge load and Grafana time to those requests.

Phase 2 did not record a request count for this ratio. The smoke proved the path. It is not an efficiency result.
