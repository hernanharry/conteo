"""Tests de lifecycle de CameraWorker (F1): estados, shutdown, errores.

Deterministas: se inyectan _InjectedContext (sin YOLO) y FakeCap (sin cv2),
con render deshabilitado. Corren SIEMPRE en local."""

from camera_worker import CameraWorker, _InjectedContext
from testutil import FakeCap, wait_until

STATE_RUNNING = "en vivo"
STATE_STOPPED = "detenida"
STATE_STOPPING = "deteniendo"
STATE_FAILED = "error"


def _frames():
    return [object() for _ in range(3)]


def _build_worker(camera_dict, processor=None, initial_in=0, initial_out=0, **kwargs):
    cap = FakeCap(frames=_frames())
    worker = CameraWorker(
        dict(camera_dict),
        initial_in=initial_in,
        initial_out=initial_out,
        enable_render=False,
        processor_factory=(lambda cfg: processor) if processor is not None else None,
        cap_factory=lambda url: cap,
        reconnect_delay=0.02,
        **kwargs,
    )
    return worker, cap


def _start_join(worker, timeout=5.0):
    worker.start()
    try:
        worker.stop()
    finally:
        worker.join(timeout=timeout)


def test_worker_procesa_frames_y_termina_en_detenida(camera_dict):
    proc = _InjectedContext()
    worker, _ = _build_worker(camera_dict, processor=proc, initial_in=5, initial_out=3)
    worker.start()
    try:
        assert wait_until(lambda: proc.processed_frames >= 1)
        assert worker.status == STATE_RUNNING
        # los conteos base se preservan durante el run loop
        assert worker.in_count == 5
        assert worker.out_count == 3
    finally:
        worker.stop()
        worker.join(timeout=5.0)
    assert worker.status == STATE_STOPPED
    assert proc.processed_frames >= 1


class CountingProcessor(_InjectedContext):
    """Suma +1 por frame procesado para verificar que el worker refleja los
    conteos del procesador."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        self.in_count += 1
        return self.detections


def test_worker_refleja_conteos_del_procesador(camera_dict):
    proc = CountingProcessor()
    worker, _ = _build_worker(camera_dict, processor=proc, initial_in=5, initial_out=3)
    worker.start()
    try:
        assert wait_until(lambda: proc.calls >= 2)
        assert worker.in_count == proc.in_count == 5 + proc.calls
    finally:
        worker.stop()
        worker.join(timeout=5.0)
    assert worker.in_count == 5 + proc.calls


class RaisingProcessor:
    def __init__(self):
        self.detections = None
        self.labels = []
        self.in_count = 0
        self.out_count = 0
        self.flushed = False

    def setup(self, worker=None, initial_in=0, initial_out=0):
        self.in_count = initial_in
        self.out_count = initial_out

    def process(self, frame):
        raise RuntimeError("boom en deteccion")

    def flush(self):
        self.flushed = True


def test_excepcion_en_procesamiento_pasa_a_error_sin_morir_en_silencio(camera_dict):
    proc = RaisingProcessor()
    worker, _ = _build_worker(camera_dict, processor=proc)
    worker.start()
    try:
        assert wait_until(lambda: worker.status == STATE_FAILED)
        assert "excepcion inesperada" in (worker._failed_reason or "")
        assert "boom" in worker._failed_reason
        assert worker._config_error is False
    finally:
        # stop() posterior no debe sobrescribir el estado "error"
        worker.stop()
        worker.join(timeout=5.0)
    assert worker.status == STATE_FAILED
    assert proc.flushed is True  # el finally del shutdown corrio igual


class SetupFailingProcessor:
    """Simula que cargar el modelo explota al inicializar."""

    def __init__(self):
        self.detections = None
        self.labels = []
        self.in_count = 0
        self.out_count = 0

    def setup(self, worker=None, initial_in=0, initial_out=0):
        raise RuntimeError("no cargo el modelo")

    def process(self, frame):
        raise AssertionError("no deberia procesarse")

    def flush(self):
        pass


def test_error_en_setup_pasa_a_error(camera_dict):
    worker, _ = _build_worker(camera_dict, processor=SetupFailingProcessor())
    worker.start()
    try:
        assert wait_until(lambda: worker.status == STATE_FAILED)
        assert "excepcion inesperada" in (worker._failed_reason or "")
    finally:
        worker.stop()
        worker.join(timeout=5.0)
    assert worker.status == STATE_FAILED


def test_config_invalida_pasa_a_error_y_marca_config_error(camera_dict):
    bad = dict(camera_dict, name="  ")
    worker = CameraWorker(
        bad,
        enable_render=False,
        cap_factory=lambda url: FakeCap(frames=_frames()),
        reconnect_delay=0.02,
    )
    worker.start()
    try:
        assert wait_until(lambda: worker.status == STATE_FAILED)
        assert worker._config_error is True
        assert "config invalida" in (worker._failed_reason or "")
    finally:
        worker.stop()
        worker.join(timeout=5.0)


def test_stop_inmediato_termina_en_detenida(camera_dict):
    proc = _InjectedContext()
    worker, _ = _build_worker(camera_dict, processor=proc)
    worker.start()
    worker.stop()
    worker.stop()  # idempotente
    worker.join(timeout=5.0)
    assert worker.status == STATE_STOPPED


def test_stop_antes_de_empezar_es_seguro(camera_dict):
    worker, _ = _build_worker(camera_dict, processor=_InjectedContext())
    # nadie arranco el thread: stop() no debe explotar
    worker.stop()
    assert worker.status == STATE_STOPPING
    # despues de todo se arranca y termina limpio
    worker.start()
    worker.join(timeout=5.0)