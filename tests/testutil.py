"""Utilidades para tests: dobles deterministas del stack pesado (F1) y
helpers de login para los tests de servidor (F5)."""

import re
import time


def login_client(client, username="admin", password="secret", role="admin"):
    """Inicia sesion en un test_client de Flask (F5).

    Crea el usuario en la BD aislada si no existe y hace un POST real a /login
    (con CSRF y rate limit por defecto altos en tests). Devuelve la respuesta
    del POST. Despues de esto, el client queda autenticado en su sesion."""
    import auth as auth_mod
    import db

    if not db.get_user(username):
        db.create_user(username, auth_mod.hash_password(password), role=role)

    # extraer el token CSRF del formulario de login
    page = client.get("/login")
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    return client.post(
        "/login",
        data={"username": username, "password": password, "csrf_token": token},
    )


def wait_until(condition, timeout=5.0, interval=0.02):
    """Poll sencillo: devuelve True apenas la condicion se cumple (o al final
    del timeout). Incluye un chequeo final para el caso limite."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return bool(condition())


def login_client(client, username="admin", password="secret", role="admin"):
    """Loguea un usuario en un Flask test_client creando el user en la DB aislada."""
    import re
    import auth
    import db
    pw_hash = auth.hash_password(password)
    existing = db.get_user(username)
    if existing:
        # Actualizar password y rol siempre para evitar residuos entre tests
        import sqlite3
        conn = db.get_conn()
        conn.execute("UPDATE users SET password_hash=?, role=? WHERE username=?",
                     (pw_hash, role, username))
        conn.commit()
        conn.close()
    else:
        db.create_user(username, pw_hash, role=role)
    # Obtener token CSRF del formulario de login
    page = client.get("/login")
    m = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    token = m.group(1) if m else ""
    resp = client.post("/login", data={
        "username": username, "password": password, "csrf_token": token,
    }, follow_redirects=False)
    return resp


class FakeCap:
    """Doble de cv2.VideoCapture para _FrameGrabber.

    - frames: lista de objetos-frame que se recorren en ciclo (infinito si
      hay al menos uno).
    - fail_for: si > 0, las primeras N lecturas devuelven (False, None)
      (simula corte de red al arrancar).
    - always_fail: True -> nunca entrega frames.
    """

    def __init__(self, frames=None, fail_for=0, always_fail=False):
        self.frames = list(frames or [])
        self.fail_for = fail_for
        self.always_fail = always_fail
        self.idx = 0
        self.reads = 0
        self.released = False

    def set(self, *args, **kwargs):
        return True

    def read(self):
        self.reads += 1
        if self.always_fail:
            return (False, None)
        if self.reads <= self.fail_for:
            return (False, None)
        if not self.frames:
            return (False, None)
        frame = self.frames[self.idx % len(self.frames)]
        self.idx += 1
        return (True, frame)

    def release(self):
        self.released = True


class FakeWorker:
    """Doble de CameraWorker para el manager: controles manuales de vida y
    estado; stop() mata al worker salvo que el test lo impida (stuck)."""

    def __init__(self, camera, initial_in=0, initial_out=0, alive=True):
        self.camera = camera
        self.name = camera["name"]
        self.in_count = initial_in
        self.out_count = initial_out
        self.status = "en vivo"
        self.last_frame_time = time.time()
        self.last_detection_time = time.time()
        self._alive = alive
        self.stopped = False
        self.joined = False
        self._config_error = False
        self._stuck = False
        self._latest_jpeg = None

    def get_latest_jpeg(self):
        return self._latest_jpeg

    def set_latest_jpeg(self, jpeg_bytes):
        self._latest_jpeg = jpeg_bytes

    def start(self):
        pass

    def stop(self):
        self.stopped = True
        if not self._stuck:
            self.status = "detenida"
            self._alive = False

    def set_stuck(self, value):
        """Si True, stop() no mata al worker: simula un thread que no termina."""
        self._stuck = value

    def set_alive(self, value):
        self._alive = value

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        if self.stopped and not self._stuck:
            self._alive = False
        self.joined = True