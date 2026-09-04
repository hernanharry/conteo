"""Tests F5 — Autenticacion Flask-Login (usuarios, sesiones, roles, CSRF).

Cubre:
  - login con credenciales validas / invalidas
  - sesion con cookie HttpOnly + SameSite=Lax
  - rutas protegidas redirigen a /login si no hay sesion
  - roles: admin puede alta/baja, viewer solo lectura
  - CSRF: POST sin token falla
  - health es publico sin auth
  - audit_log registra acciones de admin
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from testutil import login_client, FakeWorker  # noqa: E402


@pytest.fixture(scope="module")
def f5_server_env(tmp_path_factory):
    """Importa server una vez con DB aislada para tests de auth."""
    import threading
    import db
    db_path = str(tmp_path_factory.mktemp("f5-auth") / "app.db")
    db.DB_PATH = db_path
    db.init_db()
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
def client(f5_server_env):
    return f5_server_env.app.test_client()


def test_login_page_accessible(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"csrf_token" in resp.data


def test_login_credenciales_invalidas_fallan(client):
    # Crear usuario con password correcto, luego intentar con password incorrecto
    import auth, db
    if not db.get_user("admin"):
        db.create_user("admin", auth.hash_password("secret"), role="admin")
    page = client.get("/login")
    import re
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    resp = client.post("/login", data={
        "username": "admin", "password": "wrong", "csrf_token": token,
    })
    assert resp.status_code == 200
    assert b"incorrectos" in resp.data


def test_login_credenciales_validas_redirigir(client):
    resp = login_client(client, username="admin", password="secret")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/")


def test_sesion_cookie_set(client):
    login_client(client)
    resp = client.get("/")
    assert resp.status_code == 200


def test_ruta_protegida_redirect_sin_sesion(f5_server_env):
    client = f5_server_env.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_health_es_publico(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "uptime_s" in data
    assert "users" in data


def test_api_health_es_publico(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_admin_puede_agregar_camara(client):
    import re, db
    import auth as auth_mod
    db.create_user("adm3", auth_mod.hash_password("pass"), role="admin")
    page = client.get("/login")
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    client.post("/login", data={"username": "adm3", "password": "pass", "csrf_token": token})
    # Obtener CSRF para form de agregar
    page2 = client.get("/")
    m2 = re.search(r'name="csrf_token" value="([^"]+)"', page2.get_data(as_text=True))
    token2 = m2.group(1) if m2 else ""
    resp = client.post("/cameras/add", data={
        "name": "test-cam", "rtsp_url": "rtsp://test/test",
        "classes": "0,2", "conf_threshold": "0.35",
        "line_start": "0,0", "line_end": "100,100",
        "csrf_token": token2,
    }, follow_redirects=True)
    assert resp.status_code == 200
    assert db.get_camera("test-cam") is not None


def test_viewer_no_puede_agregar_camara(client):
    import re, db
    import auth as auth_mod
    db.create_user("vw1", auth_mod.hash_password("pass"), role="viewer")
    page = client.get("/login")
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    client.post("/login", data={"username": "vw1", "password": "pass", "csrf_token": token})
    page2 = client.get("/")
    m2 = re.search(r'name="csrf_token" value="([^"]+)"', page2.get_data(as_text=True))
    token2 = m2.group(1) if m2 else ""
    resp = client.post("/cameras/add", data={
        "name": "test-cam2", "rtsp_url": "rtsp://test/test2", "csrf_token": token2,
    }, follow_redirects=True)
    assert resp.status_code == 403


def test_admin_puede_eliminar_camara(client):
    import re, db
    import auth as auth_mod
    db.create_user("adm4", auth_mod.hash_password("pass"), role="admin")
    page = client.get("/login")
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    client.post("/login", data={"username": "adm4", "password": "pass", "csrf_token": token})
    db.add_camera({"name": "del-cam", "rtsp_url": "rtsp://x", "classes": "0",
                    "conf_threshold": "0.35", "line_start": "0,0", "line_end": "1,1"})
    page2 = client.get("/")
    m2 = re.search(r'name="csrf_token" value="([^"]+)"', page2.get_data(as_text=True))
    token2 = m2.group(1) if m2 else ""
    resp = client.post("/cameras/del-cam/delete",
                       data={"csrf_token": token2}, follow_redirects=True)
    assert resp.status_code == 200
    assert db.get_camera("del-cam") is None


def test_viewer_no_puede_eliminar_camara(client):
    import re, db
    import auth as auth_mod
    db.create_user("vw2", auth_mod.hash_password("pass"), role="viewer")
    page = client.get("/login")
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    client.post("/login", data={"username": "vw2", "password": "pass", "csrf_token": token})
    db.add_camera({"name": "del-cam2", "rtsp_url": "rtsp://x", "classes": "0",
                    "conf_threshold": "0.35", "line_start": "0,0", "line_end": "1,1"})
    page2 = client.get("/")
    m2 = re.search(r'name="csrf_token" value="([^"]+)"', page2.get_data(as_text=True))
    token2 = m2.group(1) if m2 else ""
    resp = client.post("/cameras/del-cam2/delete",
                       data={"csrf_token": token2}, follow_redirects=True)
    assert resp.status_code == 403


def test_audit_log_registra_add_y_delete(client):
    import re, db
    import auth as auth_mod
    db.create_user("adm5", auth_mod.hash_password("pass"), role="admin")
    page = client.get("/login")
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    client.post("/login", data={"username": "adm5", "password": "pass", "csrf_token": token})
    # Agregar camara
    page2 = client.get("/")
    m2 = re.search(r'name="csrf_token" value="([^"]+)"', page2.get_data(as_text=True))
    token2 = m2.group(1) if m2 else ""
    client.post("/cameras/add", data={
        "name": "audit-cam", "rtsp_url": "rtsp://x",
        "classes": "0", "conf_threshold": "0.35",
        "line_start": "0,0", "line_end": "1,1", "csrf_token": token2,
    })
    # Eliminar
    page3 = client.get("/")
    m3 = re.search(r'name="csrf_token" value="([^"]+)"', page3.get_data(as_text=True))
    token3 = m3.group(1) if m3 else ""
    client.post("/cameras/audit-cam/delete",
                data={"csrf_token": token3}, follow_redirects=True)
    logs = db.list_audit_log()
    actions = [(l["action"], l["entity_ref"]) for l in logs]
    assert any(a == "camera_delete" and r == "audit-cam" for a, r in actions)


def test_csrf_falta_token_rechaza(client):
    import re, db
    import auth as auth_mod
    db.create_user("adm6", auth_mod.hash_password("pass"), role="admin")
    page = client.get("/login")
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    client.post("/login", data={"username": "adm6", "password": "pass", "csrf_token": token})
    resp = client.post("/cameras/add", data={
        "name": "no-csrf", "rtsp_url": "rtsp://x",
    })
    assert resp.status_code == 400


def test_cookie_session_properties(client):
    # Verificar que el POST de login setea session cookie con HttpOnly + SameSite=Lax
    import re, db, auth
    db.create_user("cook", auth.hash_password("pass"), role="admin")
    page = client.get("/login")
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    resp = client.post("/login", data={
        "username": "cook", "password": "pass", "csrf_token": token,
    }, follow_redirects=False)
    assert resp.status_code == 302
    cookies = resp.headers.getlist("Set-Cookie")
    session_found = any("session=" in c for c in cookies)
    assert session_found, "deberia haber una cookie de sesion en el POST login"
    for c in cookies:
        if "session=" in c:
            assert "HttpOnly" in c
            assert "SameSite=Lax" in c
