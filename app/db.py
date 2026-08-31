import os
import sqlite3
import threading

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
