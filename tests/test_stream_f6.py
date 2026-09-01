"""Tests F6 — Streaming MJPEG robusto (server.stream).

El endpoint /stream/<name> sirve el vivo como MJPEG. F6 agrega:
  - headers anti-cache (Cache-Control/Pragma) para que un navegador o proxy no
    bufferize un flujo por definicion efimero;
  - X-Accel-Buffering: no para proxies tipo nginx;
  - GeneratorExit limpio cuando el cliente corta;
  - STREAM_POLL_INTERVAL configurable.

Estos tests NO necesitan el stack ML completo: inyectan un FakeWorker (con
get_latest_jpeg propio) en camera_manager y verifican el contrato del endpoint
(404 para camaras inexistentes, headers, contenido multipart). Importar `server`
corre con camera_manager/notificaciones; se limpia todo en el teardown.
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
def stream_server_env(tmp_path_factory):
    """Importa `server` una vez con DB aislada y camera_manager limpio."""
    import db

    db_path = str(tmp_path_factory.mktemp("f6") / "app.db")
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


@pytest.fixture()
def fake_worker_registered(stream_server_env, camera_dict):
    """Registra una camara con un FakeWorker que produce un JPEG estatico."""
    server = stream_server_env
    server.camera_manager.reset_for_tests()
    server.camera_manager.set_worker_factory(FakeWorker)
    server.camera_manager.start_camera(camera_dict)
    worker = server.camera_manager.get_worker("entrada-principal")
    worker.set_latest_jpeg(b"\xff\xd8\xff\xe0fake-jpeg-1")
    yield server, worker
    server.camera_manager.reset_for_tests()


def test_stream_404_para_camara_inexistente(stream_server_env):
    server = stream_server_env
    client = server.app.test_client()
    resp = client.get("/stream/no-existe")
    assert resp.status_code == 404


def test_stream_responde_headers_mime_y_anticache(fake_worker_registered):
    server, worker = fake_worker_registered
    client = server.app.test_client()
    resp = client.get("/stream/entrada-principal")

    assert resp.status_code == 200
    assert resp.mimetype == "multipart/x-mixed-replace"
    assert "frame" in resp.headers.get("Content-Type", "")
    assert resp.headers.get("Cache-Control", "").startswith("no-store")
    assert resp.headers.get("Pragma") == "no-cache"
    assert resp.headers.get("X-Accel-Buffering") == "no"


def test_stream_emite_primer_frame_multipart(fake_worker_registered):
    server, worker = fake_worker_registered
    client = server.app.test_client()
    resp = client.get("/stream/entrada-principal")

    # El test client no consumio el body (we got headers before body); leemos
    # el primer chunp del generador y luego lo cerramos (GeneratorExit limpio).
    gen = iter(resp.response)
    try:
        first = next(gen)
    finally:
        gen.close()

    assert b"--frame" in first
    assert b"Content-Type: image/jpeg" in first
    assert b"fake-jpeg-1" in first


def test_poll_interval_default_y_parse(stream_server_env, monkeypatch):
    import os
    import importlib
    import server

    # default: 0.05
    monkeypatch.delenv("STREAM_POLL_INTERVAL", raising=False)
    importlib.reload(server)
    assert server.STREAM_POLL_INTERVAL == 0.05

    # valor custom parseable
    monkeypatch.setenv("STREAM_POLL_INTERVAL", "0.33")
    importlib.reload(server)
    assert server.STREAM_POLL_INTERVAL == 0.33
    # restaurar el server default para no contaminar otros tests
    monkeypatch.delenv("STREAM_POLL_INTERVAL", raising=False)
    importlib.reload(server)


def test_stream_generador_cierra_con_GeneratorExit(fake_worker_registered):
    """El generador sale prolijo cuando el cliente corta (GeneratorExit)."""
    server, worker = fake_worker_registered
    client = server.app.test_client()
    resp = client.get("/stream/entrada-principal")
    gen = iter(resp.response)
    # consumir un frame y luego cerrar explicitamente
    next(gen)
    gen.close()
    # cerrar dos veces no rompe
    gen.close()