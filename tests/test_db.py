"""Tests de baseline sobre la capa SQLite (db.py).

Solo dependencias stdlib: se ejecutan SIEMPRE, incluso en el Python local de
desarrollo (walk 1). Cubren el comportamiento EXISTENTE, sin modificar la
aplicacion. Donde el test documenta un problema ya conocido, va acompanado de
una nota `GAP ` (para resolver en F1+).
"""

import sqlite3

import pytest

import db


def test_init_db_crea_tablas_y_columnas_de_migracion(db):
    """init_db crea cameras/detections y agrega columnas cumulativas."""
    with db.get_conn() as conn:
        tables = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"cameras", "detections"} <= tables
    with db.get_conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(cameras)")}
    assert {"cumulative_in", "cumulative_out"} <= cols


def test_respetar_init_idempotente(db):
    """init_db puede llamarse varias veces sin romperse."""
    db.init_db()
    db.init_db()
    with db.get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
    assert count == 0


def test_init_crea_indices_de_detections(db):
    """F4.3: init_db crea los indices minimos sobre detections que cubren las
    consultas reales (filtro por camera/class/timestamp + ORDER BY id DESC)."""
    with db.get_conn() as conn:
        idxs = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    assert {
        "idx_detections_camera",
        "idx_detections_class",
        "idx_detections_ts",
    } <= idxs


def test_indices_idempotentes(db):
    """F4.3: volver a llamar init_db no duplica ni rompe los indices."""
    db.init_db()
    db.init_db()
    with db.get_conn() as conn:
        idxs = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    assert len(idxs & {"idx_detections_camera", "idx_detections_class", "idx_detections_ts"}) == 3


def test_add_y_get_camera_roundtrip(db, camera_dict):
    db.add_camera(camera_dict)
    cam = db.get_camera(camera_dict["name"])
    assert cam is not None
    assert cam["name"] == "entrada-principal"
    assert cam["rtsp_url"] == camera_dict["rtsp_url"]
    assert cam["classes"] == "0,2,3,5,7"
    assert cam["active"] == 1


def test_get_camera_inexistente_devuelve_none(db):
    assert db.get_camera("no-existe") is None


def test_add_camera_duplicada_lanza_integrity_error(db, camera_dict):
    """GAP-DB: hoy una camara duplicada muestra HTTP 500 sin feedback (F1/F5)."""
    db.add_camera(camera_dict)
    with pytest.raises(sqlite3.IntegrityError):
        db.add_camera(camera_dict)


def test_list_cameras_orden_por_id(db, camera_dict):
    for i in range(3):
        cam = dict(camera_dict)
        cam["name"] = f"cam-{i}"
        db.add_camera(cam)
    names = [c["name"] for c in db.list_cameras()]
    assert names == ["cam-0", "cam-1", "cam-2"]


def test_delete_camera(db, camera_dict):
    db.add_camera(camera_dict)
    db.delete_camera(camera_dict["name"])
    assert db.get_camera(camera_dict["name"]) is None


def test_update_camera_line(db, camera_dict):
    db.add_camera(camera_dict)
    db.update_camera_line("entrada-principal", "100,400", "900,400")
    cam = db.get_camera("entrada-principal")
    assert cam["line_start"] == "100,400"
    assert cam["line_end"] == "900,400"


def test_update_camera_counts_persiste(db, camera_dict):
    """Los conteos (hoy cache en memoria) persisten via SQLite."""
    db.add_camera(camera_dict)
    db.update_camera_counts("entrada-principal", 12, 7)
    cam = db.get_camera("entrada-principal")
    assert cam["cumulative_in"] == 12
    assert cam["cumulative_out"] == 7


def test_add_detection_y_listar_todas(db):
    db.add_detection("cam-a", "persona", 101, "2026-08-28T09:00:01.000", "/tmp/a.jpg")
    db.add_detection("cam-a", "auto", 202, "2026-08-28T09:05:00.000", "/tmp/b.jpg")
    db.add_detection("cam-b", "persona", 303, "2026-08-28T10:00:00.000", "/tmp/c.jpg")
    rows = db.list_detections()
    assert len(rows) == 3


def test_list_detections_filtro_clase_y_camara(db):
    for i in range(4):
        db.add_detection(
            f"cam-{i % 2}", "persona" if i % 2 else "auto", i, "2026-08-28T10:00:00.000", "/tmp/x.jpg"
        )
    assert {d["class_name"] for d in db.list_detections(class_name="auto")} == {"auto"}
    assert {d["camera_name"] for d in db.list_detections(camera_name="cam-0")} == {"cam-0"}


def test_list_detections_fecha_hasta_incluye_dia_completo(db):
    """F4.1 (BUG-DB-001): date_to='YYYY-MM-DD' debe incluir el ultimo segundo
    del dia completo, incluyendo los microsegundos (los timestamps se guardan
    con isoformat()). Antes el tope "T23:59:59" excluia
    [23:59:59.000, 23:59:59.999]."""
    db.add_detection("cam-a", "persona", 1, "2026-08-27T23:59:58.999", "/tmp/1.jpg")
    db.add_detection("cam-a", "persona", 2, "2026-08-28T23:59:59.900", "/tmp/2.jpg")
    db.add_detection("cam-a", "persona", 3, "2026-08-29T00:00:00.100", "/tmp/3.jpg")
    rows = db.list_detections(date_from="2026-08-28", date_to="2026-08-28")
    assert len(rows) == 1  # solo la del dia 28


def test_list_detections_fecha_hasta_limite_microsegundos(db):
    """F4.1 (BUG-DB-001): los valores de borde del ultimo segundo entran y el
    primer microsegundo del dia siguiente sale."""
    db.add_detection("cam-a", "p", 1, "2026-08-28T23:59:59.999998", "/tmp/1.jpg")
    db.add_detection("cam-a", "p", 2, "2026-08-28T23:59:59.999999", "/tmp/2.jpg")
    db.add_detection("cam-a", "p", 3, "2026-08-29T00:00:00.000000", "/tmp/3.jpg")
    rows = db.list_detections(date_from="2026-08-28", date_to="2026-08-28")
    assert {int(r["tracker_id"]) for r in rows} == {1, 2}


def test_list_detections_fecha_hasta_con_hora_no_se_rompe(db):
    """F4.1: si date_to ya incluye hora (uso no actual de la app), no se
    sobreescribe el formato."""
    db.add_detection("cam-a", "p", 1, "2026-08-28T12:00:00.000", "/tmp/1.jpg")
    db.add_detection("cam-a", "p", 2, "2026-08-28T13:00:00.000", "/tmp/2.jpg")
    rows = db.list_detections(date_to="2026-08-28T12:59:59.999999")
    assert {int(r["tracker_id"]) for r in rows} == {1}


def test_list_detections_order_y_limit(db):
    for i in range(10):
        db.add_detection("cam-a", "persona", i, "2026-08-28T10:00:00.000", "/tmp/x.jpg")
    rows = db.list_detections(limit=3)
    ids = [int(r["tracker_id"]) for r in rows]
    assert ids == [9, 8, 7]  # ORDER BY id DESC LIMIT 3


def test_distinct_classes(db):
    db.add_detection("cam-a", "persona", 1, "2026-08-28T10:00:00.000", "/tmp/1.jpg")
    db.add_detection("cam-a", "auto", 2, "2026-08-28T10:00:00.000", "/tmp/2.jpg")
    db.add_detection("cam-a", "persona", 3, "2026-08-28T10:00:00.000", "/tmp/3.jpg")
    assert db.distinct_classes() == ["auto", "persona"]


def test_delete_camera_limpia_detecciones_y_archivos(db, camera_dict, tmp_path):
    """F4.2: borrar una camara elimina sus detecciones y sus recortes en
    disco, y no toca los de otras camaras."""
    import os

    db.add_camera(camera_dict)
    gal = str(tmp_path / "gallery")
    p1 = os.path.join(gal, "persona", "entrada-principal_20260828_000001_1.jpg")
    p2 = os.path.join(gal, "auto", "entrada-principal_20260828_000002_2.jpg")
    p3 = os.path.join(gal, "persona", "otra-cam_20260828_000003_3.jpg")
    for p in (p1, p2, p3):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("x")
    db.add_detection("entrada-principal", "persona", 1, "2026-08-28T10:00:00.000", p1)
    db.add_detection("entrada-principal", "auto", 2, "2026-08-28T10:00:00.000", p2)
    db.add_detection("otra-cam", "persona", 3, "2026-08-28T10:00:00.000", p3)

    db.delete_camera("entrada-principal")

    # filas y archivos de la camara borrada desaparecen...
    assert db.list_detections(camera_name="entrada-principal") == []
    assert not os.path.exists(p1)
    assert not os.path.exists(p2)
    # ...y se preservan los de la otra camara
    assert len(db.list_detections(camera_name="otra-cam")) == 1
    assert os.path.exists(p3)
    assert db.get_camera("entrada-principal") is None


def test_delete_camera_tolera_archivo_faltante(db, camera_dict, tmp_path):
    """F4.2: si un recorte ya no existe en disco, el borrado no falla."""
    import os

    db.add_camera(camera_dict)
    gal = str(tmp_path / "gallery")
    p1 = os.path.join(gal, "persona", "entrada-principal_20260828_000001_1.jpg")  # no se crea
    db.add_detection("entrada-principal", "persona", 1, "2026-08-28T10:00:00.000", p1)
    db.delete_camera("entrada-principal")
    assert db.list_detections(camera_name="entrada-principal") == []
    assert db.get_camera("entrada-principal") is None


def test_delete_camera_sin_detecciones(db, camera_dict):
    """F4.2: borrar una camara sin detecciones no rompe nada."""
    db.add_camera(camera_dict)
    db.delete_camera(camera_dict["name"])
    assert db.get_camera(camera_dict["name"]) is None


def test_delete_camera_idempotente(db):
    """F4.2: borrar una camara inexistente es inofensivo."""
    db.delete_camera("no-existe")
    assert db.get_camera("no-existe") is None