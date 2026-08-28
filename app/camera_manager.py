"""Ciclo de vida de los CameraWorker + watchdog.

Fase 1 (lifecycle/robustez):
- INVARIANTE central: como mucho UN worker activo por camara. Antes de crear
  un worker nuevo, restart_camera/start_camera verifican que el anterior haya
  terminado (stop + join con WORKER_STOP_TIMEOUT_SECONDS). Si el viejo no
  termina (p.ej. cap.read() bloqueado para siempre), el slot queda en
  _quarantined y NO se crea un worker duplicado; el watchdog retoma el slot
  cuando el worker finalmente muere.
- El watchdog usa un evaluador puro (_evaluate_worker) que distingue:
  muerto / fallo de config (no auto-reinicia) / reconectando (no reinicia:
  es un corte normal, no un cuelgue) / deteniendo (espera) / stale (reinicia).
- Anti loop: WATCHDOG_MAX_RESTARTS_PER_WINDOW reinicios automaticos en
  WATCHDOG_RESTART_WINDOW_SECONDS, WATCHDOG_MIN_RESTART_INTERVAL_SECONDS entre
  reintentos y jitter en el intervalo para no sincronizar en T.
- _worker_factory es inyectable (tests con FakeWorker sin stack pesado).
"""

import logging
import os
import random
import threading
import time

from camera_worker import (
    STATE_RUNNING,
    STATE_STOPPING,
    STATE_FAILED,
    CameraWorker,
)
from db import get_camera, list_cameras, update_camera_counts

logger = logging.getLogger(__name__)

_workers = {}
_quarantined = {}
_restart_stats = {}
_lock = threading.Lock()
_watchdog_started = False
_watchdog_stop = threading.Event()

# Fabrica de workers: inyectable para tests (set_worker_factory).
_worker_factory = CameraWorker

# Cada cuanto revisa el watchdog el estado de las camaras, y cuanto tiempo
# sin frames nuevos considera "colgada" (mas que unas cuantas reconexiones
# de RECONNECT_DELAY_SECONDS, para no reiniciar de mas por un corte normal).
WATCHDOG_INTERVAL_SECONDS = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "15"))
WATCHDOG_STALE_SECONDS = int(os.getenv("WATCHDOG_STALE_SECONDS", "60"))
# Jitter (s) que se suma al intervalo para evitar que todos los workers se
# revisen en sincronia.
WATCHDOG_JITTER_SECONDS = float(os.getenv("WATCHDOG_JITTER_SECONDS", "2"))
# Minimo tiempo entre reinicios automaticos de la misma camara.
WATCHDOG_MIN_RESTART_INTERVAL_SECONDS = float(
    os.getenv("WATCHDOG_MIN_RESTART_INTERVAL_SECONDS", "30")
)
# Anti loop: maximo de reinicios automaticos por ventana; superado, la camara
# queda parada (el operador decide) en vez de martillar reinicios.
WATCHDOG_MAX_RESTARTS_PER_WINDOW = int(os.getenv("WATCHDOG_MAX_RESTARTS_PER_WINDOW", "3"))
WATCHDOG_RESTART_WINDOW_SECONDS = int(os.getenv("WATCHDOG_RESTART_WINDOW_SECONDS", "300"))
# Cuanto se espera a que un worker muera tras stop() antes de pasar el slot a
# cuarentena (y no duplicar workers).
WORKER_STOP_TIMEOUT_SECONDS = float(os.getenv("WORKER_STOP_TIMEOUT_SECONDS", "5"))


# ---------- Fabrica de workers (inyeccion para tests) ----------

def set_worker_factory(factory):
    global _worker_factory
    with _lock:
        _worker_factory = factory


def _make_worker(cam, initial_in=0, initial_out=0):
    return _worker_factory(cam, initial_in=initial_in, initial_out=initial_out)


# ---------- Arranque / detencion ----------

def start_all():
    for cam in list_cameras():
        if cam["active"]:
            start_camera(cam)
    _start_watchdog()


def start_camera(cam: dict):
    name = cam["name"]
    with _lock:
        # INVARIANTE: si el slot esta ocupado (worker vivo o en cuarentena) no
        # se arranca otro -- siempre hay como mucho uno por camara.
        if name in _workers or name in _quarantined:
            return
        initial_in = cam.get("cumulative_in") or 0
        initial_out = cam.get("cumulative_out") or 0
        worker = _make_worker(cam, initial_in=initial_in, initial_out=initial_out)
        try:
            worker.start()
        except Exception as exc:
            logger.exception("[manager] no se pudo arrancar worker '%s': %s", name, exc)
            return
        _workers[name] = worker


def stop_camera(name: str):
    with _lock:
        worker = _workers.pop(name, None)
    if worker is None:
        return
    try:
        worker.stop()
    except Exception as exc:
        logger.warning("[manager] error pidiendo stop a '%s': %s", name, exc)
        return
    if not _join_worker(worker, name):
        # el worker no termino a tiempo: pasa a cuarentena para no duplicar
        # si alguien intenta arrancar la misma camara de nuevo.
        with _lock:
            if _quarantined.get(name) is None:
                _quarantined[name] = {
                    "worker": worker,
                    "cam": None,
                    "carry_in": getattr(worker, "in_count", 0),
                    "carry_out": getattr(worker, "out_count", 0),
                }
        logger.warning(
            "[manager] el worker de '%s' no termino en %ss -- slot en cuarentena "
            "(evita workers duplicados)",
            name,
            WORKER_STOP_TIMEOUT_SECONDS,
        )


def get_worker(name: str):
    return _workers.get(name)


def _join_worker(worker, name):
    """Espera a que el worker muera (bounded). Devuelve True si termino."""
    try:
        if worker.is_alive():
            worker.join(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        return not worker.is_alive()
    except Exception as exc:
        logger.warning("[manager] join de '%s' fallo: %s", name, exc)
        return False


def restart_camera(cam: dict):
    """Para el worker actual (si existe), conserva su conteo in/out acumulado
    hasta el momento, y arranca uno nuevo con ese conteo como punto de partida
    -- asi un reinicio (manual o del watchdog) no pierde la cuenta del dia.

    INVARIANTE: solo arranca el worker nuevo cuando el anterior confirmo
    is_alive()==False. Si no muere en WORKER_STOP_TIMEOUT_SECONDS, el slot va
    a _quarantined (sin worker devuelto) y el watchdog lo retoma al confirmar
    la muerte -- nunca hay dos workers procesando la misma camara."""
    name = cam["name"]
    with _lock:
        old_worker = _workers.pop(name, None)
        quarantined = _quarantined.pop(name, None)

    candidates = [old_worker]
    if quarantined is not None:
        candidates.append(quarantined["worker"])

    carry_in, carry_out = 0, 0
    stuck = None
    for w in candidates:
        if w is None:
            continue
        c_in = getattr(w, "in_count", 0)
        c_out = getattr(w, "out_count", 0)
        if c_in > carry_in:
            carry_in = c_in
        if c_out > carry_out:
            carry_out = c_out
        try:
            w.stop()
        except Exception as exc:
            logger.warning("[manager] error pidiendo stop a '%s': %s", name, exc)
        if not _join_worker(w, name):
            stuck = (w, c_in, c_out)
            break

    if stuck is not None:
        w, c_in, c_out = stuck
        with _lock:
            _quarantined[name] = {
                "worker": w,
                "cam": dict(cam),
                "carry_in": c_in,
                "carry_out": c_out,
            }
        logger.warning(
            "[manager] el worker de '%s' no termino en %ss -- slot en cuarentena "
            "hasta que muera (sin worker duplicado)",
            name,
            WORKER_STOP_TIMEOUT_SECONDS,
        )
        return

    if carry_in or carry_out:
        try:
            update_camera_counts(name, carry_in, carry_out)
        except Exception as exc:
            logger.warning("[manager] no se pudieron persistir conteos de '%s': %s", name, exc)

    cam = dict(cam)
    cam["cumulative_in"] = carry_in
    cam["cumulative_out"] = carry_out
    start_camera(cam)


def stop_all():
    """Detiene todo (shutdown de la app / tests): watchdog + workers + join."""
    _stop_watchdog()
    with _lock:
        workers = list(_workers.values())
        quarantined = list(_quarantined.values())
        _workers.clear()
        _quarantined.clear()
    todos = workers + [q["worker"] for q in quarantined]
    for w in todos:
        try:
            w.stop()
        except Exception:
            pass
    for w in todos:
        try:
            if w.is_alive():
                w.join(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        except Exception:
            pass


def reset_for_tests():
    """Vuelve a estado vacio determinista (para aislamiento entre tests)."""
    global _worker_factory
    _stop_watchdog()
    with _lock:
        _workers.clear()
        _quarantined.clear()
        _restart_stats.clear()
    _worker_factory = CameraWorker


# ---------- Watchdog ----------

def _start_watchdog():
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True
    _watchdog_stop.clear()
    threading.Thread(target=_watchdog_loop, name="watchdog", daemon=True).start()


def _stop_watchdog():
    global _watchdog_started
    _watchdog_stop.set()
    _watchdog_started = False


def _watchdog_loop():
    while not _watchdog_stop.is_set():
        delay = WATCHDOG_INTERVAL_SECONDS + random.random() * WATCHDOG_JITTER_SECONDS
        if _watchdog_stop.wait(delay):
            break
        try:
            _watchdog_pass()
        except Exception as exc:
            logger.exception("[watchdog] error en la pasada: %s", exc)


def _watchdog_pass():
    now = time.time()

    # 1) slots en cuarentena: si el worker finalmente murio, recuperar el slot
    for name in list(_quarantined.keys()):
        q = _quarantined.get(name)
        if q is None or name in _workers:
            continue
        if not q["worker"].is_alive():
            with _lock:
                _quarantined.pop(name, None)
            cam = q.get("cam")
            if cam is None:
                # camara borrada: no hay nada que relanzar, solo liberar slot
                continue
            if _restart_allowed(name, now):
                cam = dict(cam)
                cam["cumulative_in"] = q.get("carry_in", 0)
                cam["cumulative_out"] = q.get("carry_out", 0)
                if q.get("carry_in") or q.get("carry_out"):
                    try:
                        update_camera_counts(name, cam["cumulative_in"], cam["cumulative_out"])
                    except Exception as exc:
                        logger.warning("[watchdog] no se pudieron persistir conteos de '%s': %s", name, exc)
                _record_restart(name, now)
                start_camera(cam)
            else:
                logger.warning(
                    "[watchdog] '%s' recuperada de cuarentena pero en crash-loop: "
                    "queda detenida hasta decision manual",
                    name,
                )

    # 2) workers registrados
    with _lock:
        names = list(_workers.keys())
    for name in names:
        worker = _workers.get(name)
        if worker is None:
            continue
        decision = _evaluate_worker(worker, time.time())
        if decision == "restart":
            cam = get_camera(name)
            if cam is not None:
                _maybe_restart_from_watchdog(name, cam, time.time())
        elif decision == "dead":
            logger.warning("[watchdog] el worker de '%s' termino solo -- se libera el slot", name)
            with _lock:
                _workers.pop(name, None)
        elif decision == "failed":
            logger.warning(
                "[watchdog] '%s' con fallo de configuracion: no se auto-reinicia "
                "(corregir desde la UI)",
                name,
            )


def _evaluate_worker(worker, now):
    """Evaluador PURO (sin efectos) de un worker.

    Devuelve:
      - "dead":    el thread ya termino (nadie lo detuvo desde el manager).
      - "failed":  fallo de CONFIGURACION -> no auto-reiniciar (el fallo en
                   runtime "excepcion inesperada" SI es restartable).
      - "wait":    deteniendo / reconectando (corte normal de la camara, no un
                   cuelgue) / iniciando -- se espera, no se reinicia.
      - "restart": en vivo pero sin progreso de video o deteccion.
      - "ok":      healthy.
    """
    if not worker.is_alive():
        return "dead"
    state = getattr(worker, "status", None)
    if state == STATE_FAILED:
        return "failed" if getattr(worker, "_config_error", False) else "restart"
    if state in (STATE_STOPPING,):
        return "wait"
    if state != STATE_RUNNING:  # iniciando / conectando / reconectando
        return "wait"

    last_frame = getattr(worker, "last_frame_time", None)
    last_detection = getattr(worker, "last_detection_time", None)
    stale_video = (now - last_frame) if last_frame else 0
    stale_detection = (now - last_detection) if last_detection else 0
    stale_for = max(stale_video, stale_detection)
    if stale_for > WATCHDOG_STALE_SECONDS:
        return "restart"
    return "ok"


def _maybe_restart_from_watchdog(name, cam, now):
    """Reinicio automatico con guardas anti loop (no bloqueante)."""
    if not _restart_allowed(name, now):
        logger.warning("[watchdog] '%s' en crash-loop: se omite el reinicio automatico", name)
        return
    last = _restart_stats.get(name)
    if last and last["times"] and (now - last["times"][-1]) < WATCHDOG_MIN_RESTART_INTERVAL_SECONDS:
        return
    logger.warning("[watchdog] '%s' sin progreso o fallo -- reiniciando worker", name)
    _record_restart(name, time.time())
    restart_camera(cam)


def _restart_allowed(name, now):
    """Anti loop: ¿puede reiniciarse 'name' ahora? Poda la ventana y marca
    bloqueo si se supero el maximo de reinicios por ventana."""
    entry = _restart_stats.get(name)
    cutoff = now - WATCHDOG_RESTART_WINDOW_SECONDS
    if entry is None:
        return True
    entry["times"] = [t for t in entry["times"] if t > cutoff]
    if entry.get("blocked"):
        if not entry["times"]:
            entry["blocked"] = False
            return True
        return False
    if len(entry["times"]) >= WATCHDOG_MAX_RESTARTS_PER_WINDOW:
        entry["blocked"] = True
        return False
    return True


def _record_restart(name, now):
    entry = _restart_stats.setdefault(name, {"times": [], "blocked": False})
    entry["times"] = [
        t for t in entry["times"] if t > (now - WATCHDOG_RESTART_WINDOW_SECONDS)
    ]
    entry["times"].append(now)
    entry["blocked"] = False