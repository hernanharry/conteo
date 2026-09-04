"""Tests F7 — Observabilidad: /metrics, alertas, logging, compose.

Cubre:
  - /metrics devuelve texto/plain con métricas de Prometheus
  - /health ya testeado en test_health_f7.py
  - AlertMonitor no rompe con worker None o cámaras caídas
  - docker-compose.observability.yml existe y tiene los servicios
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from testutil import FakeWorker, login_client  # noqa: E402


@pytest.fixture(scope="module")
def obs_server_env(tmp_path_factory):
    import threading, db
    db_path = str(tmp_path_factory.mktemp("f7obs") / "app.db")
    db.DB_PATH = db_path
    db.init_db()
    import camera_manager
    camera_manager.reset_for_tests()
    camera_manager.set_worker_factory(FakeWorker)
    import server
    try:
        yield server
    finally:
        server.camera_manager.stop_all()
        server.camera_manager.reset_for_tests()
        server.notifications.reset_for_tests()
        db.reset_gallery_retention_for_tests()
        for t in threading.enumerate():
            if getattr(t, "name", None) in ("watchdog", "alert-monitor") and t.is_alive():
                t.join(timeout=2.0)


def test_metrics_endpoint(obs_server_env):
    client = obs_server_env.app.test_client()
    login_client(client)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"object_tracker" in resp.data


def test_metrics_content_type(obs_server_env):
    client = obs_server_env.app.test_client()
    login_client(client)
    resp = client.get("/metrics")
    assert "text/plain" in resp.content_type


def test_metrics_publico_sin_sesion(obs_server_env):
    client = obs_server_env.app.test_client()
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_alert_monitor_no_rompe_con_worker_none():
    import alerts
    from unittest.mock import MagicMock
    cm = MagicMock()
    cm.get_worker.return_value = None
    mon = alerts.AlertMonitor(cm, notif_worker_fn=None)
    mon._check_cameras()
    # no crash = success


def test_compose_observability_exists():
    path = REPO_ROOT / "docker-compose.observability.yml"
    assert path.exists(), "docker-compose.observability.yml no existe"


def test_compose_observability_has_services():
    path = REPO_ROOT / "docker-compose.observability.yml"
    content = path.read_text()
    assert "prometheus" in content
    assert "grafana" in content
    assert "observability" in content


def test_prometheus_yml_exists():
    path = REPO_ROOT / "prometheus.yml"
    assert path.exists(), "prometheus.yml no existe"


def test_prometheus_yml_targets():
    path = REPO_ROOT / "prometheus.yml"
    content = path.read_text()
    assert "object-tracker:8001" in content
    assert "/metrics" in content
