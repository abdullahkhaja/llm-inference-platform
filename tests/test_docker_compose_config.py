"""Static checks for Docker Compose / ops config (Docker may be unavailable)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_excludes_training_via_dockerignore() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "training/" in dockerignore
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY training" not in dockerfile
    assert "COPY app ./app" in dockerfile


def test_compose_exposes_expected_ports() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"8000:8000"' in compose
    assert '"9090:9090"' in compose
    assert '"3000:3000"' in compose
    assert "prometheus" in compose
    assert "grafana" in compose


def test_grafana_dashboard_and_datasource_provisioned() -> None:
    ds = ROOT / "ops/grafana/provisioning/datasources/prometheus.yml"
    dash_prov = ROOT / "ops/grafana/provisioning/dashboards/dashboards.yml"
    dash = ROOT / "ops/grafana/dashboards/phase1_mock_serving.json"
    assert ds.is_file()
    assert "uid: prometheus" in ds.read_text(encoding="utf-8")
    assert dash_prov.is_file()
    assert dash.is_file()
    content = dash.read_text(encoding="utf-8")
    assert "SIMULATED" in content
    assert "llm_request_latency_seconds" in content


def test_prometheus_scrapes_api() -> None:
    prom = (ROOT / "ops/prometheus.yml").read_text(encoding="utf-8")
    assert "api:8000" in prom
    assert "/metrics" in prom
