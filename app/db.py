import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "/app/data/app.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
_lock = threading.Lock()


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock, get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                rtsp_url TEXT NOT NULL,
                classes TEXT NOT NULL DEFAULT '0,2,3,5,7',
                conf_threshold REAL NOT NULL DEFAULT 0.35,
                line_start TEXT NOT NULL DEFAULT '0,300',
                line_end TEXT NOT NULL DEFAULT '1280,300',
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                tracker_id INTEGER,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL
            )
            """
        )
        # migracion: agrega columnas de conteo acumulado si no existen (para
        # instalaciones que ya tenian la tabla cameras creada antes de esto)
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(cameras)")}
        if "cumulative_in" not in existing_cols:
            conn.execute("ALTER TABLE cameras ADD COLUMN cumulative_in INTEGER NOT NULL DEFAULT 0")
        if "cumulative_out" not in existing_cols:
            conn.execute("ALTER TABLE cameras ADD COLUMN cumulative_out INTEGER NOT NULL DEFAULT 0")

        # F4.3: indices minimos sobre detections para las consultas reales de
        # la app (list_detections filtra por camera_name/class_name/timestamp y
        # ordena por id DESC; distinct_classes por class_name). Son IF NOT
        # EXISTS: idempotentes sobre instalaciones existentes y no rompen nada.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_camera ON detections(camera_name, id DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name, id DESC)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(timestamp)")

        # F5.1: usuarios de la UI (sesiones con Flask-Login). password_hash se
        # genera con werkzeug (generate_password_hash). role es admin|viewer.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer'
                    CHECK (role IN ('admin', 'viewer')),
                created_at TEXT NOT NULL
            )
            """
        )
        # F5.9: auditoria de acciones sensibles (altas/bajas de camaras y
        # borrado de detecciones/galeria). Solo lectura para la UI.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                entity TEXT NOT NULL,
                entity_ref TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )


def list_cameras():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM cameras ORDER BY id")]


def get_camera(name):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM cameras WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None


def add_camera(data):
    with _lock, get_conn() as conn:
        conn.execute(
            """
            INSERT INTO cameras (name, rtsp_url, classes, conf_threshold, line_start, line_end)
            VALUES (:name, :rtsp_url, :classes, :conf_threshold, :line_start, :line_end)
            """,
            data,
        )


def delete_camera(name):
    """Elimina la camara y su registro asociado (F4.2):

    - la fila de `cameras`;
    - las filas de `detections` de esa camara (antes quedaban huerfanas, sin FK);
    - los archivos fisicos de los recortes de galeria (`image_path`) de esa
      camara.

    Todo dentro del mismo `_lock`. El borrado fisico de archivos es tolerante
    a errores (un archivo ya no existente no rompe el borrado)."""
    paths_to_delete = []
    with _lock, get_conn() as conn:
        rows = conn.execute(
            "SELECT image_path FROM detections WHERE camera_name=?", (name,)
        ).fetchall()
        paths_to_delete = [r["image_path"] for r in rows]
        conn.execute("DELETE FROM detections WHERE camera_name=?", (name,))
        conn.execute("DELETE FROM cameras WHERE name=?", (name,))
    for path in paths_to_delete:
        try:
            os.remove(path)
        except OSError:
            # archivo ya inexistente o sin permisos: no bloquea el borrado
            pass


def update_camera_line(name, line_start, line_end):
    with _lock, get_conn() as conn:
        conn.execute(
            "UPDATE cameras SET line_start=?, line_end=? WHERE name=?",
            (line_start, line_end, name),
        )


def update_camera_counts(name, in_count, out_count):
    with _lock, get_conn() as conn:
        conn.execute(
            "UPDATE cameras SET cumulative_in=?, cumulative_out=? WHERE name=?",
            (in_count, out_count, name),
        )


def add_detection(camera_name, class_name, tracker_id, timestamp, image_path):
    with _lock, get_conn() as conn:
        conn.execute(
            """
            INSERT INTO detections (camera_name, class_name, tracker_id, timestamp, image_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (camera_name, class_name, tracker_id, timestamp, image_path),
        )


def list_detections(class_name=None, camera_name=None, date_from=None, date_to=None, limit=300):
    query = "SELECT * FROM detections"
    conds, params = [], []
    if class_name:
        conds.append("class_name=?")
        params.append(class_name)
    if camera_name:
        conds.append("camera_name=?")
        params.append(camera_name)
    if date_from:
        conds.append("timestamp >= ?")
        params.append(date_from)
    if date_to:
        # date_to viene como YYYY-MM-DD; incluye todo ese dia HASTA el ultimo
        # microsegundo. Los timestamps se guardan con microsegundos
        # (datetime.now().isoformat()), asi que un tope de "T23:59:59" excluiria
        # TODO [23:59:59.000, 23:59:59.999] (BUG-DB-001).
        if len(date_to) <= 10:
            date_to = f"{date_to}T23:59:59.999999"
        conds.append("timestamp <= ?")
        params.append(date_to)
    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(query, params)]


def distinct_classes():
    with get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT class_name FROM detections ORDER BY class_name")
        return [r[0] for r in rows]


# F5 ----------------------------------------------------------------
# Usuarios + auditoria (F5.1, F5.9). Los hashes se generan/verifican con
# werkzeug (F5.2) desde auth.py/server.py; esta capa solo persiste y consulta.

def list_users():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY id"
        )]


def get_user(username):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_user(username, password_hash, role="viewer"):
    with _lock, get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, role, datetime.now().isoformat()),
        )


def update_user_role(username, role):
    with _lock, get_conn() as conn:
        conn.execute("UPDATE users SET role=? WHERE username=?", (role, username))


def user_exists(username):
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
        return row is not None


def log_audit(username, action, entity, entity_ref=None):
    """F5.9: registra una accion sensible en audit_log (altas/bajas de camaras,
    borrado de detecciones/galeria). Nunca hace fallar la operacion por un
    error de auditoria."""
    try:
        with _lock, get_conn() as conn:
            conn.execute(
                "INSERT INTO audit_log (username, action, entity, entity_ref, timestamp)"
                " VALUES (?, ?, ?, ?, ?)",
                (username, action, entity, entity_ref, datetime.now().isoformat()),
            )
    except Exception as exc:  # noqa: BLE001 - la auditoria no debe romper la app
        logger.warning("no se pudo auditar %s/%s: %s", action, entity, exc)


def list_audit_log(limit=200):
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        )]


# F4.5 ------------------------------------------------
GALLERY_MAX_FILES = int(os.getenv("GALLERY_MAX_FILES", "0"))
GALLERY_MAX_AGE_DAYS = int(os.getenv("GALLERY_MAX_AGE_DAYS", "0"))


def _delete_detections_with_files(conn, ids):
    """Borra las filas de detections indicadas y sus archivos fisicos.

    Devuelve la lista de rutas ya borradas de la BD para eliminarlas de disco
    fuera de la transaccion (de forma tolerante)."""
    paths = []
    if ids:
        placeholders = ",".join("?" for _ in ids)
        paths = [
            r["image_path"]
            for r in conn.execute(
                f"SELECT image_path FROM detections WHERE id IN ({placeholders})", ids
            )
        ]
        conn.execute(f"DELETE FROM detections WHERE id IN ({placeholders})", ids)
    return paths


def _remove_files(paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            # archivo ya inexistente o sin permisos: no bloquea la limpieza
            pass


def enforce_gallery_retention(max_files=None, max_age_days=None):
    """F4.5: retencion automatica opt-in de la galeria.

    Valores 0/None significan "sin retencion" (comportamiento actual): no
    elimina nada. Si `max_files` > 0, deja como mucho los `max_files`
    registros mas recientes (borra los mas antiguos, menor id). Si
    `max_age_days` > 0, borra los registros mas antiguos que esa cantidad de
    dias (comparando el timestamp isoformat). Borra fila + archivo fisico.

    Devuelve el numero de registros eliminados."""
    if max_files is None:
        max_files = GALLERY_MAX_FILES
    if max_age_days is None:
        max_age_days = GALLERY_MAX_AGE_DAYS
    if max_files <= 0 and max_age_days <= 0:
        return 0

    paths_to_delete = []
    deleted = 0
    with _lock, get_conn() as conn:
        if max_age_days > 0:
            cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            ids = [r["id"] for r in conn.execute(
                "SELECT id FROM detections WHERE timestamp < ?", (cutoff,)
            )]
            paths_to_delete.extend(_delete_detections_with_files(conn, ids))
            deleted += len(ids)
        if max_files > 0:
            conn.execute("BEGIN")
            total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            if total > max_files:
                overflow = total - max_files
                ids = [r["id"] for r in conn.execute(
                    "SELECT id FROM detections ORDER BY id ASC LIMIT ?", (overflow,)
                )]
                paths_to_delete.extend(_delete_detections_with_files(conn, ids))
                deleted += len(ids)
    _remove_files(paths_to_delete)
    return deleted


# ---------------------------------------------------------------------------
# GalleryRetentionWorker (F4.5)
# ---------------------------------------------------------------------------
# La limpieza de la galeria NO debe bloquear el hilo caliente de deteccion.
# Por eso la retencion se ejecuta en un UNICO hilo daemon periodico propiedad
# de esta capa (que ya posee la logica de retencion), y no en
# camera_worker._save_gallery_crop. Es el mismo idioma que ya usa HourClock en
# notifications.py (hilo ligero + stop/join acotado), pero sin cola y sin
# acoplarse a notificaciones.
#
# Garantias:
#  - La limpieza se ejecuta SOLO dentro del worker.
#  - ensure_gallery_retention_worker() es idempotente y NO BLOQUEANTE: el hilo
#    de deteccion solo lo invoca (al guardar el primer recorte) y sigue.
#  - Con GALLERY_MAX_FILES=0 y GALLERY_MAX_AGE_DAYS=0 el worker no hace
#    limpieza y se duerme sin hacer trabajo innecesario.
GALLERY_CLEANUP_INTERVAL_MIN = float(os.getenv("GALLERY_CLEANUP_INTERVAL_MIN", "5"))


class GalleryRetentionWorker:
    def __init__(self, interval_min=None, enforce_fn=None):
        self.interval_min = interval_min if interval_min is not None else GALLERY_CLEANUP_INTERVAL_MIN
        self._enforce = enforce_fn or enforce_gallery_retention
        self._stop_event = threading.Event()
        self._thread = None

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_alive:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="gallery-retention", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self):
        # pasada inicial en background: prunnea cualquier galeria preexistente
        self._run_cleanup()
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval_min * 60):
                break
            self._run_cleanup()

    def _run_cleanup(self):
        try:
            self._enforce()
        except Exception as exc:  # noqa: BLE001 - el worker nunca muere por un error
            logger.warning("error en limpieza de galeria: %s", exc)


# Singleton global del worker de retencion.
_retention_worker = None
_retention_worker_lock = threading.Lock()


def ensure_gallery_retention_worker():
    """Arranca (idempotente) el worker de retencion. NO BLOQUEANTE: no espera
    ninguna limpieza. Si la retencion no esta configurada (0/0), aun asi puede
    arrancar, pero el loop solo ejecuta una pasada que no elimina nada y luego
    duerme sin trabajo.

    Este metodo es el unico punto de contacto del hilo de deteccion: retorna
    inmediatamente sin ejecutar SELECT/DELETE/commit/os.remove."""
    global _retention_worker
    with _retention_worker_lock:
        if _retention_worker is None:
            _retention_worker = GalleryRetentionWorker()
            _retention_worker.start()
    return _retention_worker


def stop_gallery_retention_worker(timeout=10.0):
    """Shutdown global del worker de retencion (idempotente): lo detiene y hace
    join acotado para no dejar hilos huerfanos."""
    global _retention_worker
    with _retention_worker_lock:
        worker = _retention_worker
        _retention_worker = None
    if worker is not None:
        worker.stop()
        worker.join(timeout=timeout)


def reset_gallery_retention_for_tests():
    """Vuelve al estado vacio determinista (aislamiento entre tests)."""
    stop_gallery_retention_worker(timeout=5.0)
