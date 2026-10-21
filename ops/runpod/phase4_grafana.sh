#!/usr/bin/env bash
# Start Grafana and the standalone image renderer on the pod CPU.
# Call this only after Phase 4 measurements have finished.
# Does not start vLLM and does not use the GPU.
set -u
ROOT="${ROOT:-/work/src}"
OUT="${OUT:-/work/phase4}"
GRAFANA_HOME=/opt/grafana
RENDERER=/opt/grafana-image-renderer
mkdir -p "$OUT" /work/grafana
cd "$ROOT"

log() { echo "$*" | tee -a "$OUT/grafana.log"; }
fail() { echo "FAIL: $*" | tee -a "$OUT/grafana.log"; exit 1; }

log "cpu=$(nproc)"
free -m | tee -a "$OUT/grafana.log"
df -h / /work /opt 2>/dev/null | tee -a "$OUT/grafana.log" || true
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$OUT/gpu_mem_before_grafana.csv" || true

avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
log "mem_available_mb=$avail_mb"
if (( avail_mb < 4096 )); then
  log "renderer-skipped low-memory avail_mb=$avail_mb"
  echo "low-memory" > "$OUT/renderer_mode.txt"
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
if ! command -v google-chrome >/dev/null 2>&1 && [[ ! -x /usr/bin/google-chrome ]]; then
  curl -fsSL -o /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  apt-get update
  apt-get install -y /tmp/chrome.deb || apt-get install -y -f
fi
CHROME=$(command -v google-chrome || true)
[[ -n "$CHROME" ]] || fail "chrome missing"

if [[ ! -x "$RENDERER" ]]; then
  curl -fsSL -o "$RENDERER" \
    https://github.com/grafana/grafana-image-renderer/releases/download/v5.12.7/grafana-image-renderer-linux-amd64
  chmod 755 "$RENDERER"
fi
"$RENDERER" server --help > "$OUT/renderer_help.txt" 2>&1 || true
"$RENDERER" --version > "$OUT/renderer_version.txt" 2>&1 || true

if [[ ! -x "$GRAFANA_HOME/bin/grafana" ]]; then
  tmp=/tmp/grafana.tgz
  if ! curl -fsSL -o "$tmp" https://dl.grafana.com/oss/release/grafana-12.1.0.linux-amd64.tar.gz; then
    curl -fsSL -o "$tmp" https://dl.grafana.com/oss/release/grafana-11.6.3.linux-amd64.tar.gz
  fi
  mkdir -p /opt
  tar -xzf "$tmp" -C /opt
  extracted=$(find /opt -maxdepth 1 -type d -name 'grafana-*' | head -1)
  rm -rf "$GRAFANA_HOME"
  mv "$extracted" "$GRAFANA_HOME"
fi
"$GRAFANA_HOME/bin/grafana" --version > "$OUT/grafana_version.txt" 2>&1 || true

if [[ ! -f /work/grafana/renderer.token ]]; then
  umask 077
  python3 -c 'import secrets; print(secrets.token_hex(24))' > /work/grafana/renderer.token
fi
if [[ ! -f /work/grafana/admin_password ]]; then
  umask 077
  python3 -c 'import secrets; print(secrets.token_urlsafe(18))' > /work/grafana/admin_password
fi
RENDER_TOKEN=$(cat /work/grafana/renderer.token)
ADMIN_PASSWORD=$(cat /work/grafana/admin_password)

pkill -f grafana-image-renderer || true
pkill -f grafana-server || true
sleep 1

# Software rendering only. Do not initialize a GPU.
export CHROMIUM_FLAGS="--disable-gpu --use-gl=swiftshader --disable-dev-shm-usage"
nohup "$RENDERER" server \
  --server.addr=127.0.0.1:8081 \
  --server.auth-token="$RENDER_TOKEN" \
  --browser.path="$CHROME" \
  > "$OUT/renderer.log" 2>&1 &
echo $! > "$OUT/renderer.pid"

prov="$GRAFANA_HOME/conf/provisioning"
mkdir -p "$prov/datasources" "$prov/dashboards/json"
cat > "$prov/datasources/prometheus.yml" <<'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:9090
    isDefault: true
    editable: false
EOF
cp -f "$ROOT/ops/grafana/dashboards/phase3_serving.json" "$prov/dashboards/json/phase3.json"
cat > "$prov/dashboards/dash.yml" <<'EOF'
apiVersion: 1
providers:
  - name: phase4
    type: file
    options:
      path: /opt/grafana/conf/provisioning/dashboards/json
EOF

export GF_SERVER_HTTP_ADDR=127.0.0.1
export GF_SERVER_HTTP_PORT=3000
export GF_SECURITY_ADMIN_USER=admin
export GF_SECURITY_ADMIN_PASSWORD="$ADMIN_PASSWORD"
export GF_AUTH_ANONYMOUS_ENABLED=false
export GF_RENDERING_SERVER_URL=http://127.0.0.1:8081/render
export GF_RENDERING_CALLBACK_URL=http://127.0.0.1:3000/
export GF_RENDERING_RENDERER_TOKEN="$RENDER_TOKEN"
export GF_PATHS_PROVISIONING="$prov"
nohup "$GRAFANA_HOME/bin/grafana" server --homepath "$GRAFANA_HOME" > "$OUT/grafana-server.log" 2>&1 &
echo $! > "$OUT/grafana.pid"

for i in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:3000/api/health -o "$OUT/grafana_health.json"; then
    log "grafana-ready"
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:3000/api/health >/dev/null || fail "grafana health"
if curl -fsS http://127.0.0.1:8081/healthz -o "$OUT/renderer_health.json" \
  || curl -fsS -H "Authorization: Bearer $RENDER_TOKEN" http://127.0.0.1:8081/render -o /dev/null; then
  log "renderer-port-open"
fi
ss -ltnp 2>/dev/null | awk 'NR==1 || /:3000|:8081|:9090|:8000|:8001/' | tee -a "$OUT/grafana.log" || true
log "grafana-cpu-stack-started"
echo "renderer" > "$OUT/renderer_mode.txt"
