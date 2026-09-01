"""Tests F7 — Observabilidad: /api/health.

El health check expone estado de la app para el orquestador (Docker HEALTHCHECK)
y diagnostico. Garantias:
  - /api/health responde 200 JSON con campos estables (ok, uptime_s, python,
    threads, auth_enabled, db, cameras) cuando la BD responde.
  - Es PUBLICO siempre (no requiere credenciales de auth F5), para que el
    HEALTHCHECK de Docker funcione sin guardar credenciales.
  - Ante una BD caida responde 503 (segun estado del campo ok) sin romper el
    proceso.
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
    """Importa `server` una vez con DB aislada y camera_manager limpio."""
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
    assert data["auth_enabled"] in (True, False)
    assert data["db"] == {"ok": True, "error": None}
    assert isinstance(data["cameras"], dict)


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


def test_health_es_publico_sin_auth(health_server_env):
    """F7: /api/health nunca exige credenciales (para el HEALTHCHECK de Docker).

    Se fuerza la auth activa en el modulo auth y se verifica que /api/health
    responde 200 mientras que / (index) queda 401."""
    server = health_server_env
    import auth

    prev_user, prev_pass = auth.WEB_USER, auth.WEB_PASSWORD
    prev_conf = auth._credentials_configured
    auth.WEB_USER = "admin"
    auth.WEB_PASSWORD = "secreto"
    auth._credentials_configured = True
    try:
        client = server.app.test_client()
        resp_health = client.get("/api/health")
        assert resp_health.status_code == 200
        resp_index = client.get("/")
        assert resp_index.status_code == 401
    finally:
        auth.WEB_USER = prev_user
        auth.WEB_PASSWORD = prev_pass
        auth._credentials_configured = prev_conf