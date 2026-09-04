"""Alertas proactivas (F7.3): monitorea estado de cámaras y envía alertas
vía Telegram cuando se detectan problemas persistentes.

Tipos de alerta:
  - Cámara caída (>N minutos sin frames nuevos)
  - Worker reiniciado repetidamente por el watchdog
  - Errores de detección persistentes

Se integra con el NotificationWorker existente de F3 (cola acotada, sin
bloquear al productor). Corre en un hilo daemon independiente.
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

ALERT_CHECK_INTERVAL = float(os.getenv("ALERT_CHECK_INTERVAL", "60"))
ALERT_CAMERA_DOWN_MINUTES = float(os.getenv("ALERT_CAMERA_DOWN_MINUTES", "5"))
ALERT_MIN_ALERT_INTERVAL = float(os.getenv("ALERT_MIN_ALERT_INTERVAL", "300"))


class AlertMonitor:
    def __init__(self, camera_manager_module, notif_worker_fn=None):
        self._cm = camera_manager_module
        self._notif_fn = notif_worker_fn
        self._stop = threading.Event()
        self._thread = None
        self._last_alert = {}  # camera -> timestamp

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="alert-monitor", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        while not self._stop.is_set():
            self._check_cameras()
            self._stop.wait(ALERT_CHECK_INTERVAL)

    def _check_cameras(self):
        now = time.time()
        try:
            from db import list_cameras
            for cam in list_cameras():
                name = cam["name"]
                w = self._cm.get_worker(name)
                if w is None:
                    self._maybe_alert(name, "caida", now,
                                      f"⚠️ Cámara '{name}' sin worker activo")
                    continue
                status = getattr(w, "status", "detenida")
                if status != "en vivo":
                    self._maybe_alert(name, "caida", now,
                                      f"⚠️ Cámara '{name}' en estado '{status}'")
                else:
                    self._last_alert.pop(f"{name}:caida", None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("alert monitor error: %s", exc)

    def _maybe_alert(self, camera, kind, now, text):
        key = f"{camera}:{kind}"
        last = self._last_alert.get(key, 0)
        if now - last < ALERT_MIN_ALERT_INTERVAL:
            return
        self._last_alert[key] = now
        self._send_alert(text)

    def _send_alert(self, text):
        if self._notif_fn is None:
            return
        try:
            self._notif_fn("", text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("no se pudo enviar alerta: %s", exc)


_monitor = None
_monitor_lock = threading.Lock()


def start_alert_monitor(camera_manager_module, notif_worker_fn=None):
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = AlertMonitor(camera_manager_module, notif_worker_fn)
            _monitor.start()
    return _monitor


def stop_alert_monitor(timeout=5.0):
    global _monitor
    with _monitor_lock:
        m = _monitor
        _monitor = None
    if m:
        m.stop()
        m.join(timeout=timeout)


def reset_for_tests():
    stop_alert_monitor()
