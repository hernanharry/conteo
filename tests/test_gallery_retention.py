"""Tests F4.5 — GalleryRetentionWorker (app/db.py).

Cubre los requisitos de la subfase:
  1. worker arranca correctamente;
  2. ensure es idempotente y no crea multiples workers;
  3. shutdown limpio;
  4. stop hace join acotado (no bloquea por una limpieza lenta);
  5. 0/0 no elimina nada;
  6. limites por cantidad funcionan;
  7. limites por antiguedad funcionan;
  8. limpieza inicial ocurre en background;
  9. ensure_gallery_retention_worker() es NO BLOQUEANTE (contrato: no espera
     una limpieza artificialmente lenta);
 10. _save_gallery_crop no ejecuta limpieza sincronica (docker);
 11. no quedan hilos huerfanos entre tests.

Los tests construyen instancias de GalleryRetentionWorker con una `enforce_fn`
inyectada y/o `interval_min` alto para mantenerlos deterministas y aislados
del singleton global (que se resetea vía el fixture autouse de conftest).
"""

import os
import threading
import time

import pytest

import db


# ---------- 1) arranque ----------


def test_worker_arranca_correctamente():
    w = db.GalleryRetentionWorker(interval_min=1000, enforce_fn=lambda: None)
    w.start()
    try:
        assert w.is_alive
    finally:
        w.stop()
        w.join(timeout=2.0)
    assert not w.is_alive


# ---------- 2) ensure idempotente (no crea multiples workers) ----------


def test_ensure_idempotente_no_crea_multiples_workers():
    try:
        w1 = db.ensure_gallery_retention_worker()
        w2 = db.ensure_gallery_retention_worker()
        w3 = db.ensure_gallery_retention_worker()
        assert w1 is w2 is w3  # mismo singleton / mismo hilo
        assert w1.is_alive
    finally:
        db.reset_gallery_retention_for_tests()


# ---------- 3) shutdown limpio ----------


def test_shutdown_limpio():
    w = db.ensure_gallery_retention_worker()
    assert w.is_alive
    db.stop_gallery_retention_worker(timeout=3.0)
    assert db._retention_worker is None  # singleton vacio, worker detenido


# ---------- 4) stop hace join acotado (no bloquea por limpieza lenta) ----------


def test_stop_join_acotado_no_bloquea_por_limpieza_lenta():
    release = threading.Event()

    def slow_enforce():
        release.wait(60)  # limpieza "lenta" hasta que la liberemos

    w = db.GalleryRetentionWorker(interval_min=0.00001, enforce_fn=slow_enforce)
    w.start()  # la pasada inicial queda dormida en background
    w.stop()
    start = time.perf_counter()
    w.join(timeout=0.5)  # join acotado: NO espera los 60s de la limpieza
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    release.set()  # liberamos la limpieza lenta
    w.join(timeout=2.0)  # y re-apamos el hilo del todo (sin huerfanos)
    assert not w.is_alive


# ---------- 5) 0/0 no elimina nada ----------


def test_limites_cero_no_eliminan_nada(db):
    for i in range(5):
        db.add_detection("cam-a", "persona", i, "2026-08-28T10:00:00.000", f"/tmp/{i}.jpg")
    n = db.enforce_gallery_retention(0, 0)
    assert n == 0
    assert len(db.list_detections()) == 5  # nada borrado


# ---------- 6) limites por cantidad ----------


def test_limite_por_cantidad_a_traves_del_worker(db, tmp_path):
    gal = str(tmp_path / "gallery")
    paths = [os.path.join(gal, "persona", f"c_{i}.jpg") for i in range(6)]
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("x")
    for i, p in enumerate(paths):
        db.add_detection("cam-a", "persona", i, f"2026-08-28T10:00:0{i}.000", p)

    w = db.GalleryRetentionWorker(
        interval_min=1000,
        enforce_fn=lambda: db.enforce_gallery_retention(max_files=3, max_age_days=0),
    )
    w._run_cleanup()  # lo que el worker ejecuta en su hilo
    assert {int(d["tracker_id"]) for d in db.list_detections()} == {3, 4, 5}
    assert not os.path.exists(paths[0])
    w.stop()
    w.join(timeout=1.0)


# ---------- 7) limites por antiguedad ----------


def test_limite_por_antiguedad_a_traves_del_worker(db, tmp_path):
    gal = str(tmp_path / "gallery")
    old = os.path.join(gal, "persona", "old.jpg")
    new = os.path.join(gal, "persona", "new.jpg")
    for p in (old, new):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("x")
    db.add_detection("cam-a", "persona", 1, "2020-01-01T10:00:00.000", old)
    db.add_detection("cam-a", "persona", 2, "2050-12-31T10:00:00.000", new)

    w = db.GalleryRetentionWorker(
        interval_min=1000,
        enforce_fn=lambda: db.enforce_gallery_retention(max_files=0, max_age_days=1),
    )
    w._run_cleanup()
    assert {int(d["tracker_id"]) for d in db.list_detections()} == {2}
    assert not os.path.exists(old)
    w.stop()
    w.join(timeout=1.0)


# ---------- 8) limpieza inicial ocurre en background ----------


def test_limpieza_inicial_ocurre_en_background():
    ran_in_worker = {}

    def slow_enforce():
        ran_in_worker["thread"] = threading.current_thread().name
        time.sleep(0.3)  # limpieza inicial "lenta"

    w = db.GalleryRetentionWorker(interval_min=1000, enforce_fn=slow_enforce)
    start = time.perf_counter()
    w.start()  # el arranque NO ejecuta la limpieza en este hilo
    elapsed_start = time.perf_counter() - start
    assert elapsed_start < 0.1  # start devuelve sin esperar la limpieza
    try:
        deadline = time.time() + 3.0
        while "thread" not in ran_in_worker and time.time() < deadline:
            time.sleep(0.02)
        assert "thread" in ran_in_worker, "la limpieza inicial no se ejecuto"
        assert ran_in_worker["thread"] != threading.current_thread().name, \
            "la limpieza inicial se ejecuto en el hilo llamante (NO background)"
    finally:
        w.stop()
        w.join(timeout=2.0)


# ---------- 9) ensure es NO BLOQUEANTE (contrato) ----------


def test_ensure_no_bloquea_por_limpieza_lenta():
    """IMPORTANTE: ensure_gallery_retention_worker() debe retornar sin esperar
    a que termine una limpieza artificialmente lenta que corre en background."""
    release = threading.Event()

    def slow_enforce():
        release.wait(60)  # limpieza lenta, en el hilo del worker

    w = db.GalleryRetentionWorker(interval_min=0.001, enforce_fn=slow_enforce)
    w.start()  # la pasada inicial (lenta) queda en background
    db._retention_worker = w
    try:
        start = time.perf_counter()
        got = db.ensure_gallery_retention_worker()  # ya esta corriendo
        elapsed = time.perf_counter() - start
        assert got is w
        assert elapsed < 2.0, "ensure bloqueo esperando una limpieza lenta"
    finally:
        db._retention_worker = None
        w.stop()
        release.set()
        w.join(timeout=2.0)


# ---------- 10) _save_gallery_crop no ejecuta limpieza sincronica (docker) ----------


@pytest.mark.docker
def test_save_gallery_crop_no_llama_a_enforce(db, tmp_path, monkeypatch):
    """Requerimiento 9 del plan: _save_gallery_crop SOLO puede llamar a ensure
    (arranque no bloqueante) y NUNCA a enforce (limpieza sincronica). Aislamos
    monkeypatching AMBAS: si _save_gallery_crop llamara a enforce en-linea lo
    detectariamos; y parcheamos ensure como no-op para que no arranque un
    worker real cuyo pasada inicial (background) incremente el contador.
    Requiere stack completo (cv2/numpy), por eso es docker-marked."""
    import importlib

    import numpy as np

    import camera_worker as cw

    importlib.reload(cw)
    cw.GALLERY_DIR = str(tmp_path / "gallery")

    enforce_calls = {"n": 0}
    ensure_calls = {"n": 0}

    def recording_enforce(*a, **k):
        enforce_calls["n"] += 1

    def recording_ensure():
        ensure_calls["n"] += 1

    monkeypatch.setattr(db, "enforce_gallery_retention", recording_enforce)
    monkeypatch.setattr(cw, "ensure_gallery_retention_worker", recording_ensure)

    obj = cw.CameraWorker.__new__(cw.CameraWorker)
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    obj._save_gallery_crop(crop, "persona", "cam-a", 1)

    assert enforce_calls["n"] == 0, "_save_gallery_crop ejecuto limpieza sincronica"
    assert ensure_calls["n"] == 1, "_save_gallery_crop deberia llamar a ensure (no bloqueante)"


# ---------- 11) no quedan hilos huerfanos entre tests ----------


def test_no_quedan_hilos_huerfanos():
    db.ensure_gallery_retention_worker()
    db.reset_gallery_retention_for_tests()
    leaked = [t for t in threading.enumerate() if t.name == "gallery-retention"]
    assert leaked == []
