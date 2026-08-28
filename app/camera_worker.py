import os
import threading
import time
from datetime import datetime

import cv2
import requests
import supervision as sv
import torch
from ultralytics import YOLO

from db import add_detection, update_camera_counts

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

# Cuantos threads de CPU puede usar PyTorch por inferencia.
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "4"))
torch.set_num_threads(TORCH_THREADS)

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

_model_lock = threading.Lock()
_shared_model = None


def get_model():
    """Un solo modelo YOLO cargado en memoria, compartido por todas las camaras."""
    global _shared_model
    with _model_lock:
        if _shared_model is None:
            _shared_model = YOLO(MODEL_PATH)
        return _shared_model


def notify_telegram(text, photo_bytes=None):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    try:
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
        print(f"Error enviando a Telegram: {exc}")


class _FrameGrabber(threading.Thread):
    """Lee el RTSP en su propio hilo, sin parar nunca, y guarda solo el
    ultimo frame recibido. Con RTSP_TIMEOUT_SECONDS, un corte real hace que
    cap.read() falle a los N segundos (en vez de bloquear el hilo para
    siempre) y esta clase reconecta sola."""

    def __init__(self, rtsp_url: str):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self.connected = False

    def stop(self):
        self._stop_event.set()

    def get_latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connected = True
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                self.connected = False
                cap.release()
                time.sleep(RECONNECT_DELAY)
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.connected = True
                continue
            with self._lock:
                self._frame = frame
        cap.release()


class CameraWorker(threading.Thread):
    """Hilo principal por camara. Arranca un _FrameGrabber (lee RTSP) y un
    hilo de render (dibuja y sirve el vivo, siempre a su propio ritmo);
    este hilo (run) queda dedicado exclusivamente a la deteccion YOLO, que
    puede correr mas lento sin frenar el video."""

    def __init__(self, camera: dict, initial_in: int = 0, initial_out: int = 0):
        super().__init__(daemon=True)
        self.camera = camera
        self._stop_event = threading.Event()
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg = None
        self._detections_lock = threading.Lock()
        self._last_detections = sv.Detections.empty()
        self._last_labels = []
        self._line_zone = None
        self._grabber = None

        self._base_in = initial_in
        self._base_out = initial_out
        self.in_count = initial_in
        self.out_count = initial_out
        self.status = "iniciando"

        # heartbeats separados: el watchdog reinicia si CUALQUIERA de los
        # dos deja de progresar (video congelado, o deteccion trabada aunque
        # el video se siga viendo)
        self.last_frame_time = time.time()
        self.last_detection_time = time.time()

    def stop(self):
        self._stop_event.set()

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

    def run(self):
        name = self.camera["name"]
        rtsp_url = self.camera["rtsp_url"]
        classes = [int(c) for c in self.camera["classes"].split(",") if c.strip() != ""]
        conf_threshold = float(self.camera["conf_threshold"])
        line_start = tuple(int(v) for v in self.camera["line_start"].split(","))
        line_end = tuple(int(v) for v in self.camera["line_end"].split(","))

        model = get_model()
        tracker = sv.ByteTrack()
        self._line_zone = sv.LineZone(start=sv.Point(*line_start), end=sv.Point(*line_end))

        self.status = "conectando"
        self._grabber = _FrameGrabber(rtsp_url)
        self._grabber.start()

        render_thread = threading.Thread(target=self._render_loop, daemon=True)
        render_thread.start()

        active_tracks = {}
        last_report_hour = None
        frame_counter = 0
        last_processed_frame = None

        while not self._stop_event.is_set():
            frame = self._grabber.get_latest_frame()
            if frame is None or frame is last_processed_frame:
                self.status = "en vivo" if self._grabber.connected else "reconectando"
                time.sleep(0.01)
                continue

            last_processed_frame = frame
            self.status = "en vivo"
            self.last_detection_time = time.time()
            frame_counter += 1

            if FRAME_SKIP > 1 and frame_counter % FRAME_SKIP != 0:
                continue

            results = model(frame, verbose=False, classes=classes or None, conf=conf_threshold, imgsz=IMGSZ)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            self._line_zone.trigger(detections)
            self.in_count = self._base_in + self._line_zone.in_count
            self.out_count = self._base_out + self._line_zone.out_count

            labels = []
            current_tracker_ids = set()
            for i in range(len(detections)):
                class_id = int(detections.class_id[i])
                tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None
                conf = float(detections.confidence[i])
                class_name = model.names[class_id]
                labels.append(f"#{tracker_id} {class_name} {conf:.2f}")

                if tracker_id is None:
                    continue
                current_tracker_ids.add(tracker_id)

                x1, y1, x2, y2 = detections.xyxy[i]
                area = (x2 - x1) * (y2 - y1)
                existing = active_tracks.get(tracker_id)
                if existing is None or area > existing["area"]:
                    active_tracks[tracker_id] = {
                        "area": area,
                        "crop": self._crop_with_padding(frame, (x1, y1, x2, y2)),
                        "class_name": class_name,
                    }

            lost_ids = set(active_tracks.keys()) - current_tracker_ids
            for lost_id in lost_ids:
                track = active_tracks.pop(lost_id)
                self._save_gallery_crop(track["crop"], track["class_name"], name, lost_id)

            self._set_last_detections(detections, labels)

            current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
            if current_hour != last_report_hour:
                if last_report_hour is not None:
                    self._send_hourly_report(name, last_report_hour)
                    self._base_in += self._line_zone.in_count
                    self._base_out += self._line_zone.out_count
                    self._line_zone.in_count = 0
                    self._line_zone.out_count = 0
                    update_camera_counts(name, self._base_in, self._base_out)
                last_report_hour = current_hour

        for tracker_id, track in active_tracks.items():
            self._save_gallery_crop(track["crop"], track["class_name"], name, tracker_id)

        update_camera_counts(name, self.in_count, self.out_count)
        self._grabber.stop()
        self.status = "detenida"

    def _render_loop(self):
        """Corre en paralelo a la deteccion: siempre toma el frame mas
        reciente del grabber y lo dibuja con las ultimas detecciones
        conocidas, a su propio ritmo -- nunca espera a YOLO."""
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        line_annotator = sv.LineZoneAnnotator()
        last_rendered_frame = None

        while not self._stop_event.is_set():
            if self._grabber is None:
                time.sleep(0.05)
                continue

            frame = self._grabber.get_latest_frame()
            if frame is None or frame is last_rendered_frame:
                time.sleep(0.01)
                continue

            last_rendered_frame = frame
            self.last_frame_time = time.time()

            detections, labels = self._get_last_detections()
            annotated = box_annotator.annotate(frame.copy(), detections)
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
                time.sleep(1.0 / LIVE_STREAM_FPS)

    def _crop_with_padding(self, frame, xyxy):
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

    def _save_gallery_crop(self, crop, class_name, camera_name, tracker_id):
        try:
            if crop is None or crop.size == 0:
                return
            class_dir = os.path.join(GALLERY_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_name}_{ts}_{tracker_id}.jpg"
            path = os.path.join(class_dir, filename)
            cv2.imwrite(path, crop)
            add_detection(camera_name, class_name, tracker_id, datetime.now().isoformat(), path)
        except Exception as exc:
            print(f"Error guardando recorte de galeria: {exc}")

    def _send_hourly_report(self, name, hour_label):
        payload = {
            "camera": name,
            "hour": hour_label,
            "in_count": self._line_zone.in_count,
            "out_count": self._line_zone.out_count,
        }
        if WEBHOOK_URL:
            try:
                requests.post(WEBHOOK_URL, json=payload, timeout=5)
            except Exception as exc:
                print(f"[{name}] Error enviando webhook a n8n: {exc}")

        text = (
            f"Camara {name} - reporte {hour_label}\n"
            f"Entradas: {self._line_zone.in_count} | Salidas: {self._line_zone.out_count}"
        )
        notify_telegram(text, self.get_latest_jpeg())
