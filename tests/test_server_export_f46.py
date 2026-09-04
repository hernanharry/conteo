"""Tests F4.6 — Exportacion ZIP sin pico de RAM (server.export_zip).

`export_zip` antes materializaba el ZIP completo en un io.BytesIO en RAM
(hasta 500 imagenes). Ahora escribe a un archivo temporal en disco y lo sirve
con un generador que lee el archivo y lo elimina en `finally` (despues de
cerrar el handle; funciona tambien en Windows donde un archivo abierto no
puede borrarse).

Estos tests son docker-marked: importar `server` arranca camera_manager
(watchdog) y necesita el stack completo (cv2/torch/flask), por eso NO corren en
el Python local de desarrollo.

Cubren:
  - el endpoint responde 200 y produce un ZIP valido con las entradas reales
    (paridad de contenido), con el tope de 500 imagenes;
  - el ZIP se construye via archivo temporal en disco (mkstemp), no en RAM
    (BytesIO) -- se espia que se usa mkstemp;
  - el archivo temporal no queda huerfano tras la respuesta (el finally del
    generador lo borra).
"""

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "tests"))
from testutil import login_client  # noqa: E402


@pytest.fixture(scope="module")
def server_env(tmp_path_factory):
    """Importa `server` una sola vez con DB aislada y devuelve el modulo + el
    DB_PATH usado. Importar `server` arranca camera_manager (watchdog); en el
    teardown limpiamos TODOS los globales compartidos (camera_manager,
    notificaciones, worker de retencion) y exhalamos hilos watchdog residuales
    para no contaminar a otros archivos de test."""
    import threading

    import db

    db_path = str(tmp_path_factory.mktemp("f46") / "app.db")
    db.DB_PATH = db_path
    db.init_db()

    import server
    try:
        yield server, db_path
    finally:
        server.camera_manager.stop_all()
        server.camera_manager.reset_for_tests()
        server.notifications.reset_for_tests()
        db.reset_gallery_retention_for_tests()
        for t in threading.enumerate():
            if getattr(t, "name", None) == "watchdog" and t.is_alive():
                t.join(timeout=2.0)


@pytest.mark.docker
def test_export_zip_temporal_paridad_y_sin_huerfanos(server_env, tmp_path_factory):
    """F4.6: el ZIP se escribe a archivo temporal (no BytesIO RAM), contiene
    las mismas entradas que los archivos reales, y el temporal no queda."""
    server, db_path = server_env
    import db

    gal_dir = str(tmp_path_factory.mktemp("f46-gal") / "gallery")
    db_path2 = str(tmp_path_factory.mktemp("f46-db") / "app.db")
    db.DB_PATH = db_path2
    db.init_db()

    # 3 imagenes reales de prueba en la galeria
    files = []
    for i in range(3):
        cls = "persona" if i % 2 == 0 else "auto"
        p = os.path.join(gal_dir, cls, f"cam-a_{i}.jpg")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"IMG-" * 100 + f"-{i}".encode())
        db.add_detection("cam-a", cls, i, f"2026-08-28T10:00:0{i}.000", p)
        files.append(p)

    # espiar mkstemp para probar que se usa archivo temporal (no RAM)
    original_mkstemp = tempfile.mkstemp
    created_tmp = {}

    def recording_mkstemp(**kwargs):
        fd, path = original_mkstemp(**kwargs)
        created_tmp["path"] = path
        return fd, path

    server.tempfile.mkstemp = recording_mkstemp
    client = server.app.test_client()
    login_client(client)
    try:
        resp = client.get("/gallery/export.zip?camera=cam-a")
    finally:
        server.tempfile.mkstemp = original_mkstemp

    assert resp.status_code == 200
    assert resp.mimetype == "application/zip"
    assert "path" in created_tmp, "export_zip deberia usar un archivo temporal en disco"

    # paridad de contenido: el ZIP contiene exactamente las 3 imagenes
    payload = io.BytesIO(resp.data)
    with zipfile.ZipFile(payload) as zf:
        names = sorted(zf.namelist())
    assert names == sorted(["persona/cam-a_0.jpg", "auto/cam-a_1.jpg", "persona/cam-a_2.jpg"])

    # el temporal no queda huerfano tras la respuesta (el generador lo borra
    # en su `finally`, tras cerrar el handle)
    assert not os.path.exists(created_tmp["path"]), "quedo un archivo temporal huerfano"


@pytest.mark.docker
def test_export_zip_respeta_tope_500(server_env, tmp_path_factory):
    """F4.6: se mantiene el tope de 500 imagenes por peticion."""
    server, _ = server_env
    import db

    db_path2 = str(tmp_path_factory.mktemp("f46-db2") / "app.db")
    db.DB_PATH = db_path2
    db.init_db()

    gal = str(tmp_path_factory.mktemp("f46-gal2") / "gallery")
    for i in range(520):
        cls = "persona"
        p = os.path.join(gal, cls, f"cam-a_{i}.jpg")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x" * 100)
        db.add_detection("cam-a", cls, i, f"2026-08-28T10:00:00.{i:06d}", p)

    client = server.app.test_client()
    login_client(client)
    resp = client.get("/gallery/export.zip?camera=cam-a")
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
        # 520 detecciones pero tope 500 => a lo sumo 500 entradas
        assert len(zf.namelist()) <= 500
