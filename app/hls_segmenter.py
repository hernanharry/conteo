"""Segmentador HLS (F6) — toma frames JPEG del CameraWorker y produce
segmentos .m3u8 + .ts mediante ffmpeg.

Cada cámara tiene su propio proceso ffmpeg que lee JPEGs por stdin (pipe)
y escribe segmentos en un directorio bajo ./data/hls/<nombre>/.

Configuracion (variables de entorno):
  HLS_SEGMENT_TIME  — duracion de cada segmento en segundos (default: 2)
  HLS_SEGMENTS_LIST — cuantos segmentos conservar en la playlist (default: 5)
  HLS_DIR           — directorio raiz para los segments (default: /app/data/hls)

Garantias:
  - Un hilo por cámara; si ffmpeg muere se reintenta tras HLS_RECONNECT_DELAY.
  - Limpieza automática de segmentos viejos (patrón GalleryRetentionWorker).
  - Si HLS no está habilitado (ffmpeg no encontrado o HLS_DIR no existe),
    el endpoint /hls/<name>/stream.m3u8 devuelve 404 y no se arranca nada.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time

logger = logging.getLogger(__name__)

HLS_SEGMENT_TIME = float(os.getenv("HLS_SEGMENT_TIME", "2"))
HLS_SEGMENTS_LIST = int(os.getenv("HLS_SEGMENTS_LIST", "5"))
HLS_DIR = os.getenv("HLS_DIR", "/app/data/hls")
HLS_RECONNECT_DELAY = float(os.getenv("HLS_RECONNECT_DELAY", "5"))
HLS_CLEANUP_INTERVAL_MIN = float(os.getenv("HLS_CLEANUP_INTERVAL_MIN", "2"))
HLS_MAX_AGE_MIN = float(os.getenv("HLS_MAX_AGE_MIN", "10"))

_ffmpeg_path = shutil.which("ffmpeg")
_enabled = _ffmpeg_path is not None


def hls_enabled():
    return _enabled


def get_hls_dir(camera_name):
    return os.path.join(HLS_DIR, camera_name)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class HLSSegmenter:
    """Un segmentador HLS por cámara: recibe JPEGs via put_frame() y los
    pipea a ffmpeg que produce .m3u8 + .ts en disco."""

    def __init__(self, camera_name):
        self.camera_name = camera_name
        self.out_dir = get_hls_dir(camera_name)
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._current_frame = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_alive:
            return
        _ensure_dir(self.out_dir)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"hls-{self.camera_name}", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._kill_proc()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def put_frame(self, jpeg_bytes):
        with self._frame_lock:
            self._current_frame = jpeg_bytes
            self._frame_event.set()

    def _kill_proc(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    def _loop(self):
        while not self._stop.is_set():
            self._start_ffmpeg()
            while not self._stop.is_set() and self._proc and self._proc.poll() is None:
                self._frame_event.wait(timeout=1)
                self._frame_event.clear()
                with self._frame_lock:
                    frame = self._current_frame
                if frame and self._proc and self._proc.poll() is None:
                    try:
                        self._proc.stdin.write(frame)
                        self._proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        break
            self._kill_proc()
            if not self._stop.is_set():
                logger.warning("hls %s: ffmpeg murió, reconectando en %ss",
                               self.camera_name, HLS_RECONNECT_DELAY)
                self._stop.wait(HLS_RECONNECT_DELAY)

    def _start_ffmpeg(self):
        playlist = os.path.join(self.out_dir, "stream.m3u8")
        pattern = os.path.join(self.out_dir, "seg_%05d.ts")
        cmd = [
            _ffmpeg_path,
            "-y",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-r", "5",
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", str(int(HLS_SEGMENT_TIME * 5)),
            "-sc_threshold", "0",
            "-f", "hls",
            "-hls_time", str(HLS_SEGMENT_TIME),
            "-hls_list_size", str(HLS_SEGMENTS_LIST),
            "-hls_flags", "delete_segments+append_list",
            "-hls_delete_threshold", str(HLS_SEGMENTS_LIST + 2),
            playlist,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            logger.info("hls %s: ffmpeg arrancado", self.camera_name)
        except Exception as exc:
            logger.error("hls %s: no se pudo arrancar ffmpeg: %s", self.camera_name, exc)
            self._proc = None


# --- Gestión global de segmentadores (una instancia por cámara) ---

_segmenters = {}
_lock = threading.Lock()


def get_segmenter(camera_name):
    with _lock:
        return _segmenters.get(camera_name)


def start_segmenter(camera_name):
    if not _enabled:
        return None
    with _lock:
        if camera_name in _segmenters:
            seg = _segmenters[camera_name]
            if seg.is_alive:
                return seg
        seg = HLSSegmenter(camera_name)
        _segmenters[camera_name] = seg
        seg.start()
        return seg


def stop_segmenter(camera_name):
    with _lock:
        seg = _segmenters.pop(camera_name, None)
    if seg:
        seg.stop()
        seg.join(timeout=5)


def stop_all_segmenters():
    with _lock:
        names = list(_segmenters.keys())
    for name in names:
        stop_segmenter(name)


def push_frame(camera_name, jpeg_bytes):
    seg = get_segmenter(camera_name)
    if seg and seg.is_alive:
        seg.put_frame(jpeg_bytes)


def cleanup_old_segments():
    """Limpia segmentos .ts viejos de todas las cámaras."""
    if not _enabled or not os.path.isdir(HLS_DIR):
        return
    cutoff = time.time() - (HLS_MAX_AGE_MIN * 60)
    for cam_dir in os.listdir(HLS_DIR):
        seg_dir = os.path.join(HLS_DIR, cam_dir)
        if not os.path.isdir(seg_dir):
            continue
        for fname in os.listdir(seg_dir):
            if fname.endswith(".ts"):
                fpath = os.path.join(seg_dir, fname)
                try:
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                except OSError:
                    pass
        # También limpiar m3u8 si el directorio quedó casi vacío
        m3u8 = os.path.join(seg_dir, "stream.m3u8")
        ts_files = [f for f in os.listdir(seg_dir) if f.endswith(".ts")]
        if not ts_files and os.path.exists(m3u8):
            try:
                os.remove(m3u8)
            except OSError:
                pass


class HLSCleanupWorker:
    """Worker daemon que limpia segmentos viejos periódicamente."""

    def __init__(self, interval_min=None):
        self.interval_min = interval_min or HLS_CLEANUP_INTERVAL_MIN
        self._stop = threading.Event()
        self._thread = None

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="hls-cleanup", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            if self._stop.wait(self.interval_min * 60):
                break
            try:
                cleanup_old_segments()
            except Exception as exc:
                logger.warning("hls cleanup error: %s", exc)


_cleanup_worker = None
_cleanup_lock = threading.Lock()


def ensure_cleanup_worker():
    global _cleanup_worker
    with _cleanup_lock:
        if _cleanup_worker is None and _enabled:
            _cleanup_worker = HLSCleanupWorker()
            _cleanup_worker.start()
    return _cleanup_worker


def stop_cleanup_worker(timeout=5.0):
    global _cleanup_worker
    with _cleanup_lock:
        w = _cleanup_worker
        _cleanup_worker = None
    if w:
        w.stop()
        w.join(timeout=timeout)


def reset_for_tests():
    stop_all_segmenters()
    stop_cleanup_worker()
