"""Tests F7 — Observabilidad: /health y /api/health.

El health check expone estado de la app para el orquestador (Docker HEALTHCHECK)
y diagnostico. Garantias:
  - /health y /api/health responden 200 JSON con campos estables.
  - Son PUBLICOS siempre (no requieren sesion), para que el HEALTHCHECK de
    Docker funcione sin login.
  - Ante una BD caida responden 503 sin romper el proceso.
"""

import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from testutil import FakeWorker  # noqa: E402


@pytest.fixture(scope="module")
def health_server_env(tmp_path_factory):
    """Importa server una vez con DB aislada y camera_manager limpio."""
    import db

    db_path = str(tmp_path_factory.mktemp("f7") / "app.db")
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
            if getattr(t, "name", None) == "watchdog" and t.is_alive():
                t.join(timeout=2.0)


def test_health_responde_200_con_estructura_estable(health_server_env):
    server = health_server_env
    client = server.app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert isinstance(data["uptime_s"], (int, float))
    assert "python" in data
    assert isinstance(data["threads"], int)
    assert "users" in data
    assert isinstance(data["users"], int)
    assert data["db"] == {"ok": True, "error": None}
    assert isinstance(data["cameras"], dict)


def test_health_alias_en_health_endpoint(health_server_env):
    """F7.2: /health devuelve el mismo payload que /api/health."""
    server = health_server_env
    client = server.app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "uptime_s" in data


def test_health_incluye_camaras_con_estado(health_server_env, camera_dict):
    server = health_server_env
    import db

    db.add_camera(camera_dict)
    server.camera_manager.reset_for_tests()
    server.camera_manager.set_worker_factory(FakeWorker)
    server.camera_manager.start_camera(camera_dict)

    client = server.app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "entrada-principal" in data["cameras"]
    assert "status" in data["cameras"]["entrada-principal"]
    server.camera_manager.reset_for_tests()
    db.delete_camera("entrada-principal")


def test_health_es_publico_sin_sesion(health_server_env):
    """F7.2: /health y /api/health nunca exigen sesion (para HEALTHCHECK Docker)."""
    server = health_server_env
    client = server.app.test_client()
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    resp_api = client.get("/api/health")
    assert resp_api.status_code == 200
    resp_index = client.get("/")
    assert resp_index.status_code == 302
    assert "/login" in resp_index.headers["Location"]
