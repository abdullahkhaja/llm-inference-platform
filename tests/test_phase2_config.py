"""Phase 2 config / docker / ops static checks (CPU only)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_compose_and_dockerfile_exist() -> None:
    assert (ROOT / "Dockerfile.gpu").is_file()
    assert (ROOT / "docker-compose.gpu.yml").is_file()
    gpu = (ROOT / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    assert "vllm" in gpu
    assert "nvidia" in gpu
    # Local default compose must remain mock/CPU
    local = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "BACKEND=mock" in local
    assert "nvidia" not in local


def test_runpod_scripts_present() -> None:
    for name in ("setup.sh", "start.sh", "smoke_test.sh", "cleanup.sh", "README.md"):
        assert (ROOT / "ops" / "runpod" / name).is_file()
    assert (ROOT / "ops" / "runpod" / "resolve_adapter.py").is_file()
    assert (ROOT / "ops" / "cost_tracker.md").is_file()


def test_phase2_grafana_dashboard() -> None:
    dash = ROOT / "ops/grafana/dashboards/phase2_real_serving.json"
    assert dash.is_file()
    content = dash.read_text(encoding="utf-8")
    assert "real" in content and "simulated" in content
    assert "llm_ttft_seconds" in content
    assert "llm_remote_errors_total" in content


def test_dockerfile_gpu_pins_official_vllm_image() -> None:
    text = (ROOT / "Dockerfile.gpu").read_text(encoding="utf-8")
    assert "vllm/vllm-openai:" in text
    assert "resolve_adapter" in text
