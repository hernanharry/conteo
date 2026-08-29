"""Hilo por camara: RTSP -> YOLO -> ByteTrack -> LineZone -> conteos -> galeria.

Fase 1 (lifecycle/robustez):
- Las dependencias pesadas (cv2 / supervisioin / torch / ultralytics / requests)
  se importan de forma LAZY dentro de las funciones que las usan. Este modulo
  queda importable con solo la biblioteca estandar, lo que permite testear el
  lifecycle completo (estados, shutdown, errores) sin cargar el stack de ML.
- El parseo de configuracion vive ahora en camera_config.py (testeable).
- El worker corre todo su ciclo dentro de run() con try/except/finally: ante
  cualquier excepcion pasa a estado "error" (sin morir en silencio) y el bloque
  finally garantiza el shutdown (detener grabber/hilo de render, vaciar galeria,
  persistir conteos). Nunca se usa except Exception: pass sin loguear.
- Nombre de estados compatibles con la UI: detenida / iniciando / conectando /
  en vivo / reconectando / deteniendo / error.
"""

import logging
import os
import threading
import time
from datetime import datetime

from camera_config import parse_camera_config
from db import add_detection, update_camera_counts

logger = logging.getLogger(__name__)

# Timeout de socket para RTSP (en segundos). Sin esto, cv2.VideoCapture puede
# quedarse colgado para siempre esperando datos si la camara/red tiene un
# hipo, sin devolver error ni activar la reconexion.
RTSP_TIMEOUT_SECONDS = int(os.getenv("RTSP_TIMEOUT_SECONDS", "15"))
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    f"rtsp_transport;tcp|stimeout;{RTSP_TIMEOUT_SECONDS * 1_000_000}",
)

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_SEND_PHOTO = os.getenv("TELEGRAM_SEND_PHOTO", "true").lower() == "true"
GALLERY_DIR = os.getenv("GALLERY_DIR", "/app/data/gallery")
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY_SECONDS", "5"))

# Cuantos threads de CPU puede usar PyTorch por inferencia (se aplica de forma
# Lazy al cargar el modelo, no al importar este modulo).
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "4"))

# La deteccion (YOLO) corre cada N frames nuevos como maximo -- el vivo NO
# depende de esto, se muestra siempre a su propio ritmo en un hilo aparte.
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))

IMGSZ = int(os.getenv("IMGSZ", "640"))
CROP_PADDING_PERCENT = int(os.getenv("CROP_PADDING_PERCENT", "20"))

# --- Ancho de banda del vivo ---
# Calidad JPEG del stream (1-100). Bajarlo reduce mucho el peso de cada
# frame con perdida de nitidez apenas perceptible para monitoreo.
LIVE_JPEG_QUALITY = int(os.getenv("LIVE_JPEG_QUALITY", "70"))
# Ancho maximo (px) al que se re-escala el frame ANTES de mandarlo al vivo.
# La deteccion sigue usando el frame original a full resolucion -- esto
# solo afecta lo que se ve en el navegador.
LIVE_MAX_WIDTH = int(os.getenv("LIVE_MAX_WIDTH", "960"))
# Limite de FPS de salida del stream hacia el navegador.
LIVE_STREAM_FPS = float(os.getenv("LIVE_STREAM_FPS", "12"))

# --- Estados del worker (strings compatibles con la UI) ---
STATE_STARTED = "iniciando"
STATE_CONNECTING = "conectando"
STATE_RUNNING = "en vivo"
STATE_RECONNECTING = "reconectando"
STATE_STOPPING = "deteniendo"
STATE_STOPPED = "detenida"
STATE_FAILED = "error"

# IMPORTANTE: hay un segundo nombre para el estado inicial ("iniciando"): la UI
# y CameraManager usan estas constantes; ninguna otra capa hardcodea los strings.

_model_lock = threading.Lock()
_shared_model = None
_torch_threads_configured = False

# F2: todas las camaras comparten un unico modelo en memoria. Este lock
# serializa la INFERENCIA (a diferencia de _model_lock, que solo protege la
# carga lazy): garantiza UN forward YOLO a la vez en todo el proceso. Sin el,
# N camaras x TORCH_THREADS threads sobresuscriben la CPU y el throughput se
# degrada de forma no determinista. ByteTrack/LineZone son por camara y NO se
# serializan aca -- solo el forward comparte estado.
_INFERENCE_LOCK = threading.Lock()


def _run_inference(model, frame, *, classes, conf, imgsz):
    """Ejecuta el forward YOLO sobre el modelo compartido de forma single-flight.

    Devuelve el objeto Result sin indexar (el caller decide que alumbrar). El
    lock se libera con `with` aunque el modelo lance una excepcion, para que
    ningun worker quede trabado esperando un forward que nunca termina."""
    with _INFERENCE_LOCK:
        return model(
            frame,
            verbose=False,
            classes=classes,
            conf=conf,
            imgsz=imgsz,
        )


def get_model():
    """Un solo modelo YOLO cargado en memoria, compartido por todas las camaras.

    Se importa torch/ultralytics de forma lazy recien aca: el primer worker que
    necesita el modelo lo carga (y aplica set_num_threads). Si la carga falla la
    excepcion se propaga y el worker queda en estado "error" (el watchdog decide
    que hacer)."""
    global _shared_model, _torch_threads_configured
    with _model_lock:
        if _shared_model is None:
            import torch
            from ultralytics import YOLO

            if not _torch_threads_configured:
                try:
                    torch.set_num_threads(TORCH_THREADS)
                finally:
                    _torch_threads_configured = True
            _shared_model = YOLO(MODEL_PATH)
        return _shared_model


def notify_telegram(text, photo_bytes=None):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    try:
        import requests

        if photo_bytes and TELEGRAM_SEND_PHOTO:
            requests.post(
                f"{base}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": text},
                files={"photo": ("frame.jpg", photo_bytes, "image/jpeg")},
                timeout=10,
            )
        else:
            requests.post(
                f"{base}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                timeout=10,
            )
    except Exception as exc:
        logger.warning("Error enviando a Telegram: %s", exc)


def _default_cap(url):
    """Captura cv2 (FFMPEG) por defecto; crea la conexion RTSP con TCP."""
    import cv2

    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


class _FrameGrabber(threading.Thread):
    """Lee el RTSP en su propio hilo, sin parar nunca, y guarda solo el
    ultimo frame recibido.

    Con RTSP_TIMEOUT_SECONDS, un corte real hace que cap.read() falle a los N
    segundos (en vez de bloquear el hilo para siempre) y esta clase reconecta
    sola. `connected` solo es True tras un read() valido -- asi el watchdog no
    da por viva a una camara cuya conexion no esta entregando frames.

    `cap_factory` y `reconnect_delay` son inyectables para tests (un FakeCap
    sin cv2) y para ajustar la espera entre reintentos."""

    def __init__(self, rtsp_url: str, cap_factory=None, reconnect_delay=None):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self._cap_factory = cap_factory or _default_cap
        self._reconnect_delay = (
            reconnect_delay if reconnect_delay is not None else RECONNECT_DELAY
        )
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._connected = False
        self.frame_count = 0
        self.reconnect_count = 0
        self.last_error = None
        self.last_frame_time = None

    def stop(self):
        self._stop_event.set()

    @property
    def connected(self):
        with self._lock:
            return self._connected

    def _set_connected(self, value):
        with self._lock:
            self._connected = value

    def get_latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame

    @staticmethod
    def _apply_capture_options(cap):
        """Config de buffer best-effort (FakeCap de tests no tiene cv2)."""
        if not hasattr(cap, "set"):
            return
        try:
            import cv2

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as exc:
            logger.debug("No se pudo configurar BUFFERSIZE: %s", exc)

    def run(self):
        cap = None
        try:
            while not self._stop_event.is_set():
                if cap is None:
                    try:
                        cap = self._cap_factory(self.rtsp_url)
                        self._apply_capture_options(cap)
                    except Exception as exc:
                        # la apertura en si fallo (red caida): reintento
                        self.last_error = f"no se pudo abrir captura: {exc}"
                        self._set_connected(False)
                        if self._stop_event.wait(self._reconnect_delay):
                            break
                        continue

                try:
                    ok, frame = cap.read()
                except Exception as exc:
                    ok, frame = False, None
                    self.last_error = f"error en read(): {exc}"

                if not ok:
                    self._set_connected(False)
                    self.last_error = "read() sin frame (corte/red)"
                    self.reconnect_count += 1
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    if self._stop_event.wait(self._reconnect_delay):
                        break
                    continue

                self._set_connected(True)
                self.frame_count += 1
                self.last_frame_time = time.time()
                with self._lock:
                    self._frame = frame
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass


def _crop_with_padding(frame, xyxy):
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    box_w, box_h = x2 - x1, y2 - y1
    pad_x = box_w * (CROP_PADDING_PERCENT / 100.0)
    pad_y = box_h * (CROP_PADDING_PERCENT / 100.0)
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))
    return frame[y1:y2, x1:x2].copy()


class _DetectorContext:
    """Porte EXACTO del bucle de deteccion/conteo original (F0 -> F1, sin
    cambios de comportamiento): YOLO -> ByteTrack -> LineZone -> in/out ->
    recorte de tracks a galeria -> reporte horario.

    Se separa de CameraWorker.run() solo para poder inyectar un doble
    deterministico (_InjectedContext) en los tests de lifecycle, sin cargar
    supervision / yolov8. Los efectos secundarios (galeria, webhook/telegram,
    persistencia) se inyectan como callbacks para mantener el contexto puro."""

    def __init__(self, cfg, model_fn=get_model, infer_fn=None):
        self.cfg = cfg
        self.model_fn = model_fn
        self._infer = infer_fn or _run_inference
        self.model = None
        self.tracker = None
        self.line_zone = None
        self.active_tracks = {}
        self.last_report_hour = None
        self.base_in = 0
        self.base_out = 0
        self.in_count = 0
        self.out_count = 0
        self.detections = None
        self.labels = []
        self.ready = False
        # F2: contadores minimos de inferencia (ms por forward completado,
        # incluye la espera por el lock si otras camaras estan infiriendo).
        self.inference_count = 0
        self.inference_total_ms = 0.0
        self.last_inference_ms = 0.0
        # callbacks inyectados por el worker (None = efecto deshabilitado)
        self.save_crop = None
        self.send_report = None
        self.persist_counts = None

    def setup(self, worker=None, initial_in=0, initial_out=0):
        import supervision as sv

        self.model = self.model_fn()
        self.tracker = sv.ByteTrack()
        self.line_zone = sv.LineZone(
            start=sv.Point(*self.cfg.line_start), end=sv.Point(*self.cfg.line_end)
        )
        if worker is not None:
            worker._line_zone = self.line_zone
        self.base_in = initial_in
        self.base_out = initial_out
        self.in_count = initial_in
        self.out_count = initial_out
        self.ready = True
        try:
            self.detections = sv.Detections.empty()
        except Exception:
            # sin supervision no hay "detections vacio"; el render loop salta
            self.detections = None

    def process(self, frame):
        import supervision as sv

        t0 = time.perf_counter()
        results = self._infer(
            self.model,
            frame,
            classes=self.cfg.classes or None,
            conf=self.cfg.conf_threshold,
            imgsz=IMGSZ,
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.inference_count += 1
        self.inference_total_ms += elapsed_ms
        self.last_inference_ms = elapsed_ms

        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        self.line_zone.trigger(detections)
        self.in_count = self.base_in + self.line_zone.in_count
        self.out_count = self.base_out + self.line_zone.out_count

        labels = []
        current_tracker_ids = set()
        for i in range(len(detections)):
            class_id = int(detections.class_id[i])
            tracker_id = (
                int(detections.tracker_id[i])
                if detections.tracker_id is not None
                else None
            )
            conf = float(detections.confidence[i])
            class_name = self.model.names[class_id]
            labels.append(f"#{tracker_id} {class_name} {conf:.2f}")

            if tracker_id is None:
                continue
            current_tracker_ids.add(tracker_id)

            x1, y1, x2, y2 = detections.xyxy[i]
            area = (x2 - x1) * (y2 - y1)
            existing = self.active_tracks.get(tracker_id)
            if existing is None or area > existing["area"]:
                self.active_tracks[tracker_id] = {
                    "area": area,
                    "crop": _crop_with_padding(frame, (x1, y1, x2, y2)),
                    "class_name": class_name,
                }

        lost_ids = set(self.active_tracks.keys()) - current_tracker_ids
        for lost_id in lost_ids:
            track = self.active_tracks.pop(lost_id)
            if self.save_crop is not None:
                self.save_crop(track["crop"], track["class_name"], self.cfg.name, lost_id)

        self.detections = detections
        self.labels = labels

        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        if current_hour != self.last_report_hour:
            if self.last_report_hour is not None:
                in_c = self.line_zone.in_count
                out_c = self.line_zone.out_count
                if self.send_report is not None:
                    self.send_report(self.cfg.name, self.last_report_hour, in_c, out_c)
                self.base_in += in_c
                self.base_out += out_c
                self.line_zone.in_count = 0
                self.line_zone.out_count = 0
                if self.persist_counts is not None:
                    self.persist_counts(self.cfg.name, self.base_in, self.base_out)
            self.last_report_hour = current_hour

        return detections

    def flush(self):
        """Vacia tracks activos a galeria y persiste el conteo final. Solo si
        el setup completo existosamente (sino no hay nada que persistir)."""
        if not self.ready:
            return
        try:
            for tracker_id, track in list(self.active_tracks.items()):
                self.active_tracks.pop(tracker_id)
                if self.save_crop is not None:
                    self.save_crop(
                        track["crop"], track["class_name"], self.cfg.name, tracker_id
                    )
        except Exception as exc:
            logger.warning("[%s] error vaciando galeria: %s", self.cfg.name, exc)
        try:
            if self.persist_counts is not None:
                self.persist_counts(self.cfg.name, self.in_count, self.out_count)
        except Exception as exc:
            logger.warning("[%s] error persistiendo conteos: %s", self.cfg.name, exc)


class _InjectedContext:
    """Doble deterministico de _DetectorContext para tests de lifecycle: no
    toca YOLO / ByteTrack / LineZone / cv2. Solo cuenta frames procesados y
    mantiene los conteos que recibe como base."""

    def __init__(self):
        self.detections = None
        self.labels = []
        self.in_count = 0
        self.out_count = 0
        self.processed_frames = 0

    def setup(self, worker=None, initial_in=0, initial_out=0):
        self.in_count = initial_in
        self.out_count = initial_out

    def process(self, frame):
        self.processed_frames += 1
        return self.detections

    def flush(self):
        pass


class CameraWorker(threading.Thread):
    """Hilo principal por camara. Arranca un _FrameGrabber (lee RTSP) y un
    hilo de render (dibuja y sirve el vivo, siempre a su propio ritmo);
    este hilo (run) queda dedicado exclusivamente a la deteccion YOLO, que
    puede correr mas lento sin frenar el video.

    Lifecycle F1: run() envuelve todo en try/except -> "error" + finally ->
    _shutdown(). stop() es idempotente y reactivo. `processor_factory` y
    `cap_factory` son inyectables para tests."""

    def __init__(
        self,
        camera: dict,
        initial_in: int = 0,
        initial_out: int = 0,
        enable_render: bool = True,
        processor_factory=None,
        cap_factory=None,
        reconnect_delay=None,
    ):
        name = camera.get("name", "?")
        super().__init__(name=f"worker-{name}", daemon=True)
        self.camera = camera
        self._stop_event = threading.Event()
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg = None
        self._detections_lock = threading.Lock()
        self._last_detections = None
        self._last_labels = []
        self._line_zone = None
        self._grabber = None
        self._render_thread = None
        self._processor_factory = processor_factory
        self._cap_factory = cap_factory
        self._reconnect_delay = reconnect_delay
        self._enable_render = enable_render
        self._state_lock = threading.Lock()
        self.state_log = []

        self._base_in = initial_in
        self._base_out = initial_out
        self.in_count = initial_in
        self.out_count = initial_out
        self._set_state(STATE_STARTED)

        self._failed_reason = None
        self._config_error = False

        # heartbeats separados: el watchdog reinicia si CUALQUIERA de los
        # dos deja de progresar (video congelado, o deteccion trabada aunque
        # el video se siga viendo)
        self.last_frame_time = time.time()
        self.last_detection_time = time.time()

        # F2: contadores de rendimiento (frames nuevos observados vs frames
        # procesados; frames_skipped = frames_available - frames_processed).
        # El tiempo de inferencia se lee del procesador real (0 si es un
        # doble de tests que no toca YOLO).
        self.frames_available = 0
        self.frames_processed = 0
        self.inference_count = 0
        self.inference_total_ms = 0.0
        self.last_inference_ms = 0.0

    # ---------- API publica (UI / CameraManager) ----------

    def stop(self):
        self._stop_event.set()
        if self._grabber is not None:
            try:
                self._grabber.stop()
            except Exception as exc:
                logger.warning("[%s] error pidiendo stop al grabber: %s", self.camera.get("name", "?"), exc)
        current = self.get_state()
        if current not in (STATE_STOPPED, STATE_FAILED):
            self._set_state(STATE_STOPPING)

    def get_state(self):
        with self._state_lock:
            return self.status

    def _set_state(self, state):
        with self._state_lock:
            self.status = state
            self.state_log.append((state, time.time()))
            if len(self.state_log) > 100:
                self.state_log = self.state_log[-50:]

    def _fail(self, reason):
        self._failed_reason = reason
        self._set_state(STATE_FAILED)

    def get_latest_jpeg(self):
        with self._jpeg_lock:
            return self._latest_jpeg

    def _set_latest_jpeg(self, jpeg_bytes):
        with self._jpeg_lock:
            self._latest_jpeg = jpeg_bytes

    def _set_last_detections(self, detections, labels):
        with self._detections_lock:
            self._last_detections = detections
            self._last_labels = labels

    def _get_last_detections(self):
        with self._detections_lock:
            return self._last_detections, self._last_labels

    # ---------- Construccion interna (inyectable en tests) ----------

    def _default_processor(self, cfg):
        ctx = _DetectorContext(cfg, model_fn=get_model)
        ctx.save_crop = self._save_gallery_crop
        ctx.send_report = self._send_hourly_report
        ctx.persist_counts = update_camera_counts
        return ctx

    def _make_processor(self, cfg):
        if self._processor_factory is not None:
            return self._processor_factory(cfg)
        return self._default_processor(cfg)

    def _start_grabber(self, url):
        grabber = _FrameGrabber(
            url,
            cap_factory=self._cap_factory,
            reconnect_delay=self._reconnect_delay,
        )
        self._grabber = grabber
        grabber.start()
        return grabber

    # ---------- Ciclo de vida ----------

    def run(self):
        name = self.camera.get("name", "?")
        try:
            cfg = parse_camera_config(self.camera)
        except ValueError as exc:
            self._config_error = True
            self._fail(f"config invalida: {exc}")
            logger.error("[%s] config invalida: %s", name, exc)
            return

        processor = None
        try:
            processor = self._make_processor(cfg)
            self._run_loop(name, cfg, processor)
        except Exception as exc:
            self._fail(f"excepcion inesperada: {exc.__class__.__name__}: {exc}")
            logger.exception("[%s] el worker fallo: %s", name, exc)
        finally:
            self._shutdown(name, processor)

    def _run_loop(self, name, cfg, processor):
        # setup (modelo/tracker/linea) puede fallar (ej. no cargo el modelo):
        # la excepcion sube a run() y el worker pasa a "error".
        processor.setup(self, initial_in=self._base_in, initial_out=self._base_out)

        self._set_state(STATE_CONNECTING)
        grabber = self._start_grabber(cfg.rtsp_url)

        if self._enable_render:
            self._render_thread = threading.Thread(
                target=self._render_loop, name=f"render-{name}", daemon=True
            )
            self._render_thread.start()

        frame_counter = 0
        last_processed_frame = None

        while not self._stop_event.is_set():
            frame = grabber.get_latest_frame()
            if frame is None or frame is last_processed_frame:
                self._set_state(STATE_RUNNING if grabber.connected else STATE_RECONNECTING)
                self._stop_event.wait(0.01)
                continue

            last_processed_frame = frame
            self._set_state(STATE_RUNNING)
            self.last_detection_time = time.time()
            frame_counter += 1
            self.frames_available += 1

            if FRAME_SKIP > 1 and frame_counter % FRAME_SKIP != 0:
                continue

            self.frames_processed += 1
            processor.process(frame)
            self.in_count = processor.in_count
            self.out_count = processor.out_count
            # heartbeat de deteccion al COMPLETAR la inferencia (no al
            # encolarla): si un worker queda esperando el lock de inferencia,
            # el watchdog sigue viendo que no avanza y puede reiniciarlo.
            self.last_detection_time = time.time()
            self.inference_count = getattr(processor, "inference_count", 0)
            self.inference_total_ms = getattr(processor, "inference_total_ms", 0.0)
            self.last_inference_ms = getattr(processor, "last_inference_ms", 0.0)
            self._set_last_detections(processor.detections, processor.labels)

    def _shutdown(self, name, processor):
        """Cierre garantizado (finally): detener grabber/render, vaciar la
        galeria y persistir conteos. Nunca sobrescribe el estado "error"."""
        if self.get_state() != STATE_FAILED:
            self._set_state(STATE_STOPPING)
        self._stop_event.set()
        if self._grabber is not None:
            self._grabber.stop()

        if processor is not None:
            try:
                processor.flush()
            except Exception as exc:
                logger.warning("[%s] error en flush del procesador: %s", name, exc)

        if self._render_thread is not None and self._render_thread.is_alive():
            self._render_thread.join(timeout=2.0)
        if self._grabber is not None and self._grabber.is_alive():
            self._grabber.join(timeout=2.0)

        if self.get_state() != STATE_FAILED:
            self._set_state(STATE_STOPPED)

    # ---------- Render del vivo (hilo aparte, siempre a su ritmo) ----------

    def _render_loop(self):
        """Corre en paralelo a la deteccion: siempre toma el frame mas
        reciente del grabber y lo dibuja con las ultimas detecciones
        conocidas, a su propio ritmo -- nunca espera a YOLO.

        F1: el loop entero esta protegido; ante un error puntual avisa y
        reintenta (0.5s) en vez de morir en silencio."""
        name = self.camera.get("name", "?")
        try:
            import cv2
            import supervision as sv

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            line_annotator = sv.LineZoneAnnotator()
        except Exception as exc:
            logger.warning("[%s] no se pudo inicializar el render vivo: %s", name, exc)
            return

        last_rendered_frame = None
        while not self._stop_event.is_set():
            try:
                if self._grabber is None:
                    self._stop_event.wait(0.05)
                    continue

                frame = self._grabber.get_latest_frame()
                if frame is None or frame is last_rendered_frame:
                    self._stop_event.wait(0.01)
                    continue

                last_rendered_frame = frame
                self.last_frame_time = time.time()

                detections, labels = self._get_last_detections()
                annotated = frame.copy()
                if detections is not None:
                    annotated = box_annotator.annotate(annotated, detections)
                    annotated = label_annotator.annotate(annotated, detections, labels=labels)
                if self._line_zone is not None:
                    annotated = line_annotator.annotate(annotated, self._line_zone)

                h, w = annotated.shape[:2]
                if LIVE_MAX_WIDTH and w > LIVE_MAX_WIDTH:
                    scale = LIVE_MAX_WIDTH / w
                    annotated = cv2.resize(annotated, (LIVE_MAX_WIDTH, int(h * scale)))

                ok_jpeg, buf = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_JPEG_QUALITY]
                )
                if ok_jpeg:
                    self._set_latest_jpeg(buf.tobytes())

                if LIVE_STREAM_FPS > 0:
                    self._stop_event.wait(1.0 / LIVE_STREAM_FPS)
            except Exception as exc:
                logger.warning("[%s] error en render loop: %s", name, exc)
                self._stop_event.wait(0.5)

    # ---------- Efectos de la galeria / reporte (F3 conserva comportamiento) ----------

    def _save_gallery_crop(self, crop, class_name, camera_name, tracker_id):
        try:
            if crop is None or getattr(crop, "size", 0) == 0:
                return
            import cv2

            class_dir = os.path.join(GALLERY_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_name}_{ts}_{tracker_id}.jpg"
            path = os.path.join(class_dir, filename)
            cv2.imwrite(path, crop)
            add_detection(camera_name, class_name, tracker_id, datetime.now().isoformat(), path)
        except Exception as exc:
            logger.warning("Error guardando recorte de galeria: %s", exc)

    def _send_hourly_report(self, name, hour_label, in_count, out_count):
        payload = {
            "camera": name,
            "hour": hour_label,
            "in_count": in_count,
            "out_count": out_count,
        }
        if WEBHOOK_URL:
            try:
                import requests

                requests.post(WEBHOOK_URL, json=payload, timeout=5)
            except Exception as exc:
                logger.warning("[%s] Error enviando webhook a n8n: %s", name, exc)

        text = (
            f"Camara {name} - reporte {hour_label}\n"
            f"Entradas: {in_count} | Salidas: {out_count}"
        )
        notify_telegram(text, self.get_latest_jpeg())