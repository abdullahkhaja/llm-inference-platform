#!/usr/bin/env bash
# Stop GPU containers on the pod. Does NOT terminate the RunPod instance.
# Stopping containers ≠ ending billing: you must TERMINATE (or stop) the pod
# in the RunPod console / CLI, and delete unused network volumes if any.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "Stopping docker compose GPU stack..."
docker compose -f docker-compose.gpu.yml down || true

echo "Container status:"
docker ps -a --filter name=llm- || true

cat <<'EOF'

IMPORTANT billing notes:
  1. This script only stops Docker containers on the VM.
  2. An active RunPod POD continues to bill GPU time until you STOP or TERMINATE it.
  3. STOP may retain disk and still incur storage charges depending on product.
  4. TERMINATE releases the GPU; copy smoke_results/ off-pod first.
  5. Network volumes (if you attached any) bill separately until deleted.

Verify in console that no GPU pod remains running, then log the session in
ops/cost_tracker.md (dev/test vs final benchmark columns).
EOF
