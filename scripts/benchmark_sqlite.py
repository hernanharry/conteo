"""Benchmark F4.4 — concurrencia / volumen de la capa SQLite (db.py).

NO es una suite de tests: es una herramienta de medicion (gate) para decidir
si F4 necesita cambios de concurrencia. Mide, sobre el codigo actual:

1) Throughput de add_detection bajo N hilos concurrentes (simula N camaras
   escribiendo a la vez), que hoy pasa por el lock global `_lock` + una
   conexion nueva por operacion + commit por INSERT.
2) Tiempo de list_detections (filtro por camera/class/ts) con y SIN indices
   (F4.3) sobre un volumen realista.
3) Referencia: insersion en bloque (muchas filas, UN commit) para cuantificar
   el costo del patron "conexion+commit por fila" que usa db.py hoy.

No afirma ningun valor: reporta numeros para decidir con datos, no por teoria.

Uso (desde la raiz del repo):
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_sqlite.py
"""

import os
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "app"))

import db

DB_DISK = os.getenv("F4_BENCH_CONC_ROWS", "2000")   # filas en la seccion de concurrencia
DB_BULK = int(os.getenv("F4_BENCH_LIST_ROWS", "20000"))  # filas en la seccion de indices


def _reset_db(path):
    db.DB_PATH = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    db.init_db()


def bench_add_detection(path, rows, threads):
    """Mide throughput de add_detection con N hilos concurrentes."""
    _reset_db(path)
    barrier = threading.Barrier(threads)
    per_thread = rows // threads
    errors = []

    def worker(idx):
        worker_rows = [
            (f"cam-{idx % 4}", "persona", idx * 1000 + i, f"2026-08-28T10:00:00.{idx}",
             f"/tmp/{idx}_{i}.jpg")
            for i in range(per_thread)
        ]
        barrier.wait()
        for r in worker_rows:
            try:
                db.add_detection(*r)
            except Exception as e:  # pragma: no cover
                errors.append(e)

    start = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(i,), name=f"w{i}") for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    elapsed = time.perf_counter() - start
    total = per_thread * threads
    rate = total / elapsed
    print(f"  add_detection: {threads} hilo(s), {total} filas en {elapsed:.2f}s "
          f"-> {rate:,.0f} ins/s" + (f" (ERRORES: {len(errors)})" if errors else ""), flush=True)
    return rate


def bench_batch_reference(path, rows):
    """Referencia: mismo volumen pero en un solo executemany + commit, para
    aislar el costo del patron conexion+commit por fila de db.py."""
    _reset_db(path)
    bulk = [
        (f"cam-{i % 4}", "persona", i, f"2026-08-28T10:00:00.{i}00000", f"/tmp/{i}.jpg")
        for i in range(rows)
    ]
    conn = sqlite3.connect(path)
    start = time.perf_counter()
    conn.executemany(
        "INSERT INTO detections (camera_name, class_name, tracker_id, timestamp, image_path) VALUES (?,?,?,?,?)",
        bulk,
    )
    conn.commit()
    conn.close()
    elapsed = time.perf_counter() - start
    print(f"  batch executemany+commit (referencia): {rows} filas en {elapsed:.2f}s "
          f"-> {rows / elapsed:,.0f} ins/s", flush=True)


def bench_list_detections(path, rows, with_indices):
    """Mide tiempo de list_detections con filtros reales, con o sin indices F4.3."""
    _reset_db(path)
    conn = sqlite3.connect(path)
    if not with_indices:
        for idx in ("idx_detections_camera", "idx_detections_class", "idx_detections_ts"):
            conn.execute(f"DROP INDEX IF EXISTS {idx}")
    conn.commit()
    conn.close()

    bulk = [
        (f"cam-{i % 4}", ("persona", "auto", "bici")[i % 3], i,
         f"2026-08-{20 + (i % 10):02d}T{(i * 7) % 24:02d}:{(i * 3) % 60:02d}:00.{i:06d}", f"/tmp/{i}.jpg")
        for i in range(rows)
    ]
    conn = db.get_conn()
    conn.executemany(
        "INSERT INTO detections (camera_name, class_name, tracker_id, timestamp, image_path) VALUES (?,?,?,?,?)",
        bulk,
    )
    conn.commit()

    samples = []
    for _ in range(5):
        start = time.perf_counter()
        db.list_detections(camera_name="cam-0", class_name="persona")
        db.list_detections(camera_name="cam-2")
        db.list_detections(date_from="2026-08-20", date_to="2026-08-25")
        db.distinct_classes()
        samples.append(time.perf_counter() - start)
    samples.sort()
    med = samples[len(samples) // 2]
    label = "CON indices" if with_indices else "SIN indices"
    print(f"  list_detections({rows} filas) {label}: mediana {med * 1000:.2f} ms "
          f"(media {sum(samples) / len(samples) * 1000:.2f} ms)", flush=True)


def main():
    print("=== Benchmark F4.4 (gate de concurrencia SQLite) ===")
    print("db.DB_PATH se redirige a DB temporal por medicion.\n", flush=True)

    tmp = tempfile.mkdtemp(prefix="f4-bench-")
    conc_rows = int(DB_DISK)

    print(f"-- [1] add_detection multi-hilo (lock global + conexion/op + commit); {conc_rows} filas --")
    bench_add_detection(os.path.join(tmp, "c1.db"), conc_rows, 1)
    bench_add_detection(os.path.join(tmp, "c2.db"), conc_rows, 4)
    bench_add_detection(os.path.join(tmp, "c3.db"), conc_rows, 8)

    print(f"\n-- [1b] Referencia batch (mismo volumen, un commit): {conc_rows} filas --")
    bench_batch_reference(os.path.join(tmp, "bf.db"), conc_rows)

    print(f"\n-- [2] list_detections: aporte de indices F4.3 ({DB_BULK} filas) --")
    bench_list_detections(os.path.join(tmp, "li.db"), DB_BULK, with_indices=False)
    bench_list_detections(os.path.join(tmp, "li2.db"), DB_BULK, with_indices=True)

    print("\nListo. Interpretacion y decision en F4.4 (gate).", flush=True)


if __name__ == "__main__":
    main()
