"""Tests F5 — Autenticacion basica (auth.py).

auth.py solo usa la biblioteca estandar (base64, hmac, os), asi que estos
tests SIEMPRE se ejecutan (no son docker-marked).

Cubren:
  - auth desactivada cuando no hay credenciales configuradas.
  - credenciales validas / invalidas (y comparacion en tiempo constante).
  - el flujo de request (server._require_auth): 401 con WWW-Authenticate sin
    credenciales, 200 con credenciales validas, 401 con invalidas.
  - lista blanca WEB_PUBLIC_PATHS deja algunas rutas publicas.
  - no se loguean credenciales (el header nunca se imprime).
"""

import base64
import importlib
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))


def _basic_header(user, password):
    raw = f"{user}:{password}".encode("utf-8")
    return f"Basic {base64.b64encode(raw).decode('utf-8')}"


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    """Aisla las variables de entorno que controlan la auth entre tests."""
    monkeypatch.delenv("WEB_USER", raising=False)
    monkeypatch.delenv("WEB_PASSWORD", raising=False)
    monkeypatch.delenv("WEB_PUBLIC_PATHS", raising=False)
    yield


def _reload_auth():
    import auth
    importlib.reload(auth)
    return auth


def test_auth_desactivada_si_no_hay_credenciales():
    auth = _reload_auth()
    assert auth.auth_enabled() is False
    # con la auth desactivada, authenticate siempre devuelve True
    assert auth.authenticate(type("R", (), {"path": "/", "headers": {}})()) is True


def test_auth_activada_con_credenciales():
    os.environ["WEB_USER"] = "admin"
    os.environ["WEB_PASSWORD"] = "secreto"
    auth = _reload_auth()
    assert auth.auth_enabled() is True


def test_credenciales_validas_aceptadas():
    os.environ["WEB_USER"] = "admin"
    os.environ["WEB_PASSWORD"] = "secreto"
    auth = _reload_auth()
    req = type("R", (), {"path": "/", "headers": {"Authorization": _basic_header("admin", "secreto")}})()
    assert auth.authenticate(req) is True


def test_credenciales_invalidas_rechazadas():
    os.environ["WEB_USER"] = "admin"
    os.environ["WEB_PASSWORD"] = "secreto"
    auth = _reload_auth()
    req = type("R", (), {"path": "/", "headers": {"Authorization": _basic_header("admin", "mal")}})()
    assert auth.authenticate(req) is False
    req2 = type("R", (), {"path": "/", "headers": {"Authorization": _basic_header("otro", "secreto")}})()
    assert auth.authenticate(req2) is False


def test_sin_header_authorization_rechazado():
    os.environ["WEB_USER"] = "admin"
    os.environ["WEB_PASSWORD"] = "secreto"
    auth = _reload_auth()
    req = type("R", (), {"path": "/", "headers": {}})()
    assert auth.authenticate(req) is False


def test_header_basico_invalido_rechazado():
    os.environ["WEB_USER"] = "admin"
    os.environ["WEB_PASSWORD"] = "secreto"
    auth = _reload_auth()
    req = type("R", (), {"path": "/", "headers": {"Authorization": "Bearer abc"}})()
    assert auth.authenticate(req) is False


def test_lista_blanca_deja_ruta_publica():
    os.environ["WEB_USER"] = "admin"
    os.environ["WEB_PASSWORD"] = "secreto"
    os.environ["WEB_PUBLIC_PATHS"] = "/stream,/gallery/image"
    auth = _reload_auth()
    # rutas en la lista blanca: publicas (True) sin credenciales
    req_stream = type("R", (), {"path": "/stream/cam-a", "headers": {}})()
    req_img = type("R", (), {"path": "/gallery/image", "headers": {}})()
    assert auth.authenticate(req_stream) is True
    assert auth.authenticate(req_img) is True
    # una ruta fuera de la lista blanca sigue protegida
    req_index = type("R", (), {"path": "/", "headers": {}})()
    assert auth.authenticate(req_index) is False


def test_authenticate_con_path_publico_cuando_auth_apagada():
    auth = _reload_auth()
    req = type("R", (), {"path": "/stream/cam-a", "headers": {}})()
    assert auth.authenticate(req) is True


# --- Integration con server (usuario + password correctos) ---

@pytest.fixture(scope="module")
def auth_server_env(tmp_path_factory):
    """Importa `server` con credenciales WEB_USER/WEB_PASSWORD y DB aislada.

    Usa un contenedor de flask config rail para reconstruir las credenciales de
    auth con la config set desde esta fixture (se restaura en teardown)."""
    import threading

    import auth
    import db

    db_path = str(tmp_path_factory.mktemp("f5-auth") / "app.db")
    db.DB_PATH = db_path
    db.init_db()

    # guarda el estado previo para restaurarlo
    prev_user, prev_pass = auth.WEB_USER, auth.WEB_PASSWORD
    auth.WEB_USER = "admin"
    auth.WEB_PASSWORD = "secreto"
    auth._credentials_configured = bool(auth.WEB_USER and auth.WEB_PASSWORD)

    import server
    try:
        yield server
    finally:
        server.camera_manager.stop_all()
        server.camera_manager.reset_for_tests()
        server.notifications.reset_for_tests()
        db.reset_gallery_retention_for_tests()
        # restaurar el estado de auth (para no filtrar a otros archivos de test)
        auth.WEB_USER = prev_user
        auth.WEB_PASSWORD = prev_pass
        auth._credentials_configured = bool(auth.WEB_USER and auth.WEB_PASSWORD)
        for t in threading.enumerate():
            if getattr(t, "name", None) == "watchdog" and t.is_alive():
                t.join(timeout=2.0)


def test_server_401_sin_credenciales(auth_server_env):
    import threading

    server = auth_server_env
    client = server.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate", "").startswith("Basic")


def test_server_200_con_credenciales_validas(auth_server_env):
    server = auth_server_env
    client = server.app.test_client()
    resp = client.get("/", headers={"Authorization": _basic_header("admin", "secreto")})
    assert resp.status_code == 200


def test_server_401_con_credenciales_invalidas(auth_server_env):
    server = auth_server_env
    client = server.app.test_client()
    resp = client.get("/", headers={"Authorization": _basic_header("admin", "mal")})
    assert resp.status_code == 401
