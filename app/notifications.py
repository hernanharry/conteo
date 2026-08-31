"""Notificaciones desacopladas (F3): n8n / Telegram fuera del hilo de deteccion.

F3 resuelve: el pipeline de camara (RTSP -> YOLO -> ByteTrack -> LineZone ->
conteo) NO debe bloquearse ni retrasarse por operaciones de red de los
reportes (webhook n8n, Telegram, envio de fotos). Un timeout o fallo de una
request NO debe detener la deteccion ni el conteo.

Principio:
- El hilo de deteccion SOLO enquelea (no hace requests).
- Uno o pocos hilos worker consumen la cola y ejecutan las requests.
- Cola acotada + politicas claras: coalescencia y descarte, NUNCA bloquear al
  productor indefinidamente.
- Shutdown limpio (no deja threads huerfanos).
- Tokens de Telegram jamas se loguean completos (sanitizacion).

Solo usa la biblioteca estandar: importable y testeable sin el stack pesado
(cv2/torch/ultralytics/requests se cargan lazy en el worker).
"""

import logging
import os
import queue
import re
import threading
import time

logger = logging.getLogger(__name__)

# --- Configuracion global (migrada desde camera_worker) ---
WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
WEBHOOK_TIMEOUT = float(os.getenv("N8N_WEBHOOK_TIMEOUT_SECONDS", "5"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_SEND_PHOTO = os.getenv("TELEGRAM_SEND_PHOTO", "true").lower() == "true"
TELEGRAM_TIMEOUT = float(os.getenv("TELEGRAM_TIMEOUT_SECONDS", "10"))

# Cola: tamano maximo. Si se llena, se aplica la politica de descarte
# (coalescencia por camara + descarte del trabajo mas nuevo, con log).
NOTIF_QUEUE_SIZE = int(os.getenv("NOTIF_QUEUE_SIZE", "256"))
# Cuantos hilos consumen la cola de notificaciones.
NOTIF_WORKERS = int(os.getenv("NOTIF_WORKERS", "1"))
# Cada cuanto el HorologIO revisa si cambio la hora para emitir reportes
# horarios aunque una camara este temporalmente sin frames.
HOUR_TICK_SECONDS = float(os.getenv("HOUR_TICK_SECONDS", "1.0"))


# ---------------------------------------------------------------------------
# Sanitizacion de tokens
# ---------------------------------------------------------------------------

# Patron para redactar el token de Telegram en cadenas como
# "https://api.telegram.org/bot123456:ABC.../sendMessage". Redacta desde
# "bot" hasta el primer delimitador (/ , espacio, comilla, fin de cadena).
_BOT_TOKEN_RE = re.compile(r"(bot)([A-Za-z0-9_:\-]{6,})(?=[/&\"'\s]|$)", re.IGNORECASE)


def sanitize_token(text):
    """Redacta tokens de Telegram en una cadena para que jamas se logueen
    completos. Reemplaza el segmento del token por '<REDACTADO>'."""
    if not isinstance(text, str):
        text = str(text)
    return _BOT_TOKEN_RE.sub(r"\1<REDACTADO>", text)


# ---------------------------------------------------------------------------
# Envios (requests) - ejecutados SOLO en el worker de notificaciones
# ---------------------------------------------------------------------------

def send_webhook(url, payload, timeout):
    """POST del payload JSON al webhook. Devuelve True si 2xx."""
    import requests

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return True


def send_telegram(token, chat_id, text, photo_bytes=None, send_photo=True, timeout=10):
    """Manda un reporte a Telegram (mensaje y/o foto). Devuelve True si ok.
    Las excepciones se propagan al worker (que las loguea sanitizadas)."""
    import requests

    if not (token and chat_id):
        return False
    base = f"https://api.telegram.org/bot{token}"
    if photo_bytes and send_photo:
        requests.post(
            f"{base}/sendPhoto",
            data={"chat_id": chat_id, "caption": text},
            files={"photo": ("frame.jpg", photo_bytes, "image/jpeg")},
            timeout=timeout,
        )
    else:
        requests.post(
            f"{base}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=timeout,
        )
    return True


# ---------------------------------------------------------------------------
# Worker de notificaciones: cola acotada + hilos consumidores
# ---------------------------------------------------------------------------
#
# Politica de la cola (definida explicitamente):
#  - Tamano maximo: NOTIF_QUEUE_SIZE (acotado, no consume memoria infinita).
#  - Coalescencia: se mantiene a lo sumo UNA notificacion pendiente por
#    camara. Enviar otra notificacion para la misma camara reemplaza a la
#    pendiente (evita duplicados y acota la cola). Ver _pending_by_camera.
#  - Cuando la cola real esta llena y llega una camara nueva: se loguea el
#    evento y se DESCARTA el trabajo mas nuevo (policy: piso la notificacion
#    pendiente mas antigua de otra camara? No: descarto la nueva). Jamas
#    bloquea al productor (enqueue usa put_nowait).
#  - Prioridades: no hay; el trabajo pendiente mas antiguo se procesa primero
#    (FIFO de la cola real, con coalescencia previa).


class NotificationWorker:
    def __init__(
        self,
        queue_size=None,
        num_workers=None,
        send_webhook_fn=None,
        send_telegram_fn=None,
        webhook_url=None,
        telegram_token=None,
        telegram_chat_id=None,
    ):
        self.queue_size = queue_size or NOTIF_QUEUE_SIZE
        self.num_workers = num_workers or NOTIF_WORKERS
        self.webhook_url = webhook_url if webhook_url is not None else WEBHOOK_URL
        self.telegram_token = (
            telegram_token if telegram_token is not None else TELEGRAM_BOT_TOKEN
        )
        self.telegram_chat_id = (
            telegram_chat_id if telegram_chat_id is not None else TELEGRAM_CHAT_ID
        )
        # dependencias inyectables (tests) - defaults reales
        self._send_webhook = send_webhook_fn or send_webhook
        self._send_telegram = send_telegram_fn or send_telegram

        self._queue = queue.Queue(maxsize=self.queue_size)
        self._pending_by_camera = {}  # camera -> job; coalescencia por camara
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads = []
        # contadores expuestos (diagnostico / tests)
        self.processed_count = 0
        self.dropped_count = 0
        self.error_count = 0

    # ---- ciclo de vida ----

    def start(self):
        if self._threads:
            return
        self._stop_event.clear()
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._consume_loop, name=f"notif-{i}", daemon=True
            )
            t.start()
            self._threads.append(t)

    def stop(self):
        """Senal de stop: los consumidores drenan lo pendiente o terminan. No
        crea threads huerfanos."""
        self._stop_event.set()

    def join(self, timeout=None):
        """Espera a que terminen los hilos consumidores (bounded)."""
        threads = self._threads
        for t in threads:
            t.join(timeout=timeout)
        self._threads = []

    @property
    def is_alive(self):
        return any(t.is_alive() for t in self._threads)

    def reset_counters(self):
        self.processed_count = 0
        self.dropped_count = 0
        self.error_count = 0

    # ---- API para el productor (hilo de deteccion) ----

    def submit_hourly_report(self, camera_name, hour_label, in_count, out_count, jpeg=None):
        """Enquelea un reporte horario de forma NO BLOQUEANTE. Devuelve True
        si quedo en cola, False si se descarto (cola llena / coalescido)."""
        job = {
            "kind": "hourly",
            "camera": camera_name,
            "hour": hour_label,
            "in": in_count,
            "out": out_count,
            "jpeg": jpeg,
        }
        return self._enqueue(camera_name, job)

    def submit_message(self, camera_name, text):
        """Enquelea un mensaje libre (Telegram) de forma NO BLOQUEANTE."""
        job = {"kind": "text", "camera": camera_name, "text": text, "jpeg": None}
        return self._enqueue(camera_name, job)

    def _enqueue(self, camera_name, job):
        """Politica de enqueue: coalescencia por camara + cola acotada + sin
        bloqueo. Devuelve True si el trabajo quedo efectivamente en la cola."""
        if self._stop_event.is_set():
            logger.debug("cola de notificaciones detenida; se descarta trabajo de '%s'", camera_name)
            return False

        with self._pending_lock:
            existing = self._pending_by_camera.get(camera_name)
            if existing is not None:
                # coalescer: reemplazo el pendiente de la misma camara
                self._pending_by_camera[camera_name] = job
                logger.debug(
                    "notificacion de '%s' coalescida (reemplaza a pendiente anterior)", camera_name
                )
                return True
            try:
                # marcamos el pendiente ANTES de put_nowait para no perder la
                # referencia si el put llena la cola
                self._pending_by_camera[camera_name] = job
                self._queue.put_nowait(job)
                return True
            except queue.Full:
                # Deshacemos el marcado y descartamos el trabajo nuevo (nunca
                # bloqueamos al productor).
                self._pending_by_camera.pop(camera_name, None)
                self.dropped_count += 1
                logger.warning(
                    "cola de notificaciones llena (%d): se descarta reporte de '%s'",
                    self.queue_size,
                    camera_name,
                )
                return False

    # ---- consumidor ----

    def _consume_loop(self):
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_job(job)
            except Exception as exc:  # noqa: BLE001 - el worker nunca muere
                self.error_count += 1
                logger.warning(
                    "[%s] error procesando notificacion (%s): %s; sigue en marcha",
                    job.get("camera", "?"),
                    job.get("kind"),
                    sanitize_token(exc),
                )
            finally:
                with self._pending_lock:
                    # si lo pendiente que drenamos sigue siendo el mismo, lo quitamos
                    if (
                        self._pending_by_camera.get(job.get("camera"))
                        is job
                    ):
                        self._pending_by_camera.pop(job.get("camera"), None)
                self._queue.task_done()
                self.processed_count += 1

    def _process_job(self, job):
        kind = job.get("kind")
        camera = job.get("camera", "?")
        if kind == "hourly":
            self._dispatch_hourly(camera, job)
        elif kind == "text":
            if self._send_telegram is not None:
                self._send_telegram(
                    self.telegram_token,
                    self.telegram_chat_id,
                    job["text"],
                    photo_bytes=None,
                )
        else:
            logger.warning("[%s] notificacion desconocida: %s", camera, kind)

    def _dispatch_hourly(self, camera, job):
        payload = {
            "camera": camera,
            "hour": job["hour"],
            "in_count": job["in"],
            "out_count": job["out"],
        }
        if self.webhook_url and self._send_webhook is not None:
            # una excepcion del webhook se propaga al consumidor, que la
            # loguea y sigue -- no mata el worker ni la cola
            self._send_webhook(self.webhook_url, payload, WEBHOOK_TIMEOUT)

        text = (
            f"Camara {camera} - reporte {job['hour']}\n"
            f"Entradas: {job['in']} | Salidas: {job['out']}"
        )
        if self._send_telegram is not None:
            self._send_telegram(
                self.telegram_token,
                self.telegram_chat_id,
                text,
                photo_bytes=job.get("jpeg"),
                send_photo=TELEGRAM_SEND_PHOTO,
                timeout=TELEGRAM_TIMEOUT,
            )


# ---------------------------------------------------------------------------
# Registro de hora por camara (fallback para camara caida)
# ---------------------------------------------------------------------------
#
# El reporte horario normal se dispara en el hilo de deteccion al cambiar la
# hora (ver _DetectorContext._maybe_hourly_report). Si la camara esta caida no
# hay frame nuevo y ese hilo no corre. Para no perder el reporte del cambio de
# hora, un HorologIO (hilo ligero, independiente de frames) invoca el
# on_hour_tick de cada camara registrada. El hook es idempotente por hora
# (guardado por last_report_hour), asi no duplica si el propio hilo de
# deteccion ya lo disparo.


class HourClock:
    def __init__(self, tick_seconds=None):
        self.tick_seconds = tick_seconds or HOUR_TICK_SECONDS
        self._callbacks = {}  # camera -> callable
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_hour = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._last_hour = None
        self._thread = threading.Thread(
            target=self._loop, name="hour-clock", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def register(self, camera_name, on_hour_tick):
        with self._lock:
            self._callbacks[camera_name] = on_hour_tick

    def unregister(self, camera_name):
        with self._lock:
            self._callbacks.pop(camera_name, None)

    def _loop(self):
        while not self._stop_event.is_set():
            now = time.time()
            hour = time.strftime("%Y-%m-%d %H:00", time.localtime(now))
            if hour != self._last_hour:
                self._last_hour = hour
                with self._lock:
                    callbacks = list(self._callbacks.values())
                for cb in callbacks:
                    try:
                        cb()
                    except Exception as exc:  # noqa: BLE001 - un hook no debe tirar el clock
                        logger.warning("error en on_hour_tick de camara: %s", sanitize_token(exc))
            self._stop_event.wait(self.tick_seconds)


# ---------------------------------------------------------------------------
# Singleton de la aplicacion
# ---------------------------------------------------------------------------

_worker = None
_worker_lock = threading.Lock()
_clock = None


def get_notification_worker():
    """Worker global de notificaciones (singleton). Se arranca lazily la primera
    vez que un handler de reporte lo necesita (el worker de camara lo usa al
    enqueue)."""
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = NotificationWorker()
            _worker.start()
        return _worker


def get_hour_clock():
    global _clock
    with _worker_lock:
        if _clock is None:
            _clock = HourClock()
            _clock.start()
        return _clock


def enqueue_hourly_report(camera_name, hour_label, in_count, out_count, jpeg=None):
    """API usada por el hilo de deteccion: construye el trabajo y lo enquelea
    sin hacer requests. No bloquea."""
    return get_notification_worker().submit_hourly_report(
        camera_name, hour_label, in_count, out_count, jpeg
    )


def stop_all(timeout=15.0):
    """Shutdown global limpio: detiene clock + worker, espera join acotado.
    Idempotente. No deja threads huerfanos."""
    global _worker, _clock
    if _clock is not None:
        _clock.stop()
        _clock.join(timeout=min(timeout, 5.0))
        _clock = None
    if _worker is not None:
        _worker.stop()
        _worker.join(timeout=min(timeout, 10.0))
        _worker = None


def reset_for_tests():
    """Vuelve a estado vacio determinista (aislamiento entre tests)."""
    global _worker, _clock
    if _worker is not None:
        _worker.stop()
        _worker.join(timeout=5.0)
        _worker = None
    if _clock is not None:
        _clock.stop()
        _clock.join(timeout=5.0)
        _clock = None
