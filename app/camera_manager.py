import os
import threading
import time

from camera_worker import CameraWorker
from db import get_camera, list_cameras, update_camera_counts

_workers = {}
_lock = threading.Lock()
_watchdog_started = False

# Cada cuanto revisa el watchdog el estado de las camaras, y cuanto tiempo
# sin frames nuevos considera "colgada" (mas que unas cuantas reconexiones
# de RECONNECT_DELAY_SECONDS, para no reiniciar de mas por un corte normal).
WATCHDOG_INTERVAL_SECONDS = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "15"))
WATCHDOG_STALE_SECONDS = int(os.getenv("WATCHDOG_STALE_SECONDS", "60"))


def start_all():
    for cam in list_cameras():
        if cam["active"]:
            start_camera(cam)
    _start_watchdog()


def start_camera(cam: dict):
    with _lock:
        if cam["name"] in _workers:
            return
        initial_in = cam.get("cumulative_in") or 0
        initial_out = cam.get("cumulative_out") or 0
        worker = CameraWorker(cam, initial_in=initial_in, initial_out=initial_out)
        worker.start()
        _workers[cam["name"]] = worker


def stop_camera(name: str):
    with _lock:
        worker = _workers.pop(name, None)
    if worker:
        worker.stop()


def get_worker(name: str):
    return _workers.get(name)


def restart_camera(cam: dict):
    """Para el worker actual (si existe), conserva su conteo in/out
    acumulado hasta el momento, y arranca uno nuevo con ese conteo como
    punto de partida -- asi un reinicio (manual o del watchdog) no pierde
    la cuenta de entradas/salidas del dia."""
    name = cam["name"]
    with _lock:
        old_worker = _workers.pop(name, None)

    carry_in, carry_out = 0, 0
    if old_worker:
        carry_in = old_worker.in_count
        carry_out = old_worker.out_count
        old_worker.stop()

    if carry_in or carry_out:
        update_camera_counts(name, carry_in, carry_out)

    cam = dict(cam)
    cam["cumulative_in"] = carry_in
    cam["cumulative_out"] = carry_out
    start_camera(cam)


def _watchdog_loop():
    while True:
        time.sleep(WATCHDOG_INTERVAL_SECONDS)
        now = time.time()
        with _lock:
            names = list(_workers.keys())
        for name in names:
            worker = _workers.get(name)
            if worker is None:
                continue

            last_frame = getattr(worker, "last_frame_time", None)
            last_detection = getattr(worker, "last_detection_time", None)
            stale_video = (now - last_frame) if last_frame else 0
            stale_detection = (now - last_detection) if last_detection else 0
            stale_for = max(stale_video, stale_detection)

            if stale_for > WATCHDOG_STALE_SECONDS:
                motivo = "video" if stale_video >= stale_detection else "deteccion"
                print(
                    f"[watchdog] '{name}' sin progreso ({motivo}) hace {int(stale_for)}s "
                    f"(limite {WATCHDOG_STALE_SECONDS}s) -- reiniciando worker"
                )
                cam = get_camera(name)
                if cam:
                    restart_camera(cam)


def _start_watchdog():
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True
    threading.Thread(target=_watchdog_loop, daemon=True).start()
