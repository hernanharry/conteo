"""F2: serializacion de la inferencia YOLO sobre el modelo compartido.

El problema central resuelto: N camaras comparten un unico modelo YOLO y cada
worker llamaria `model(...)` en su propio hilo, generando N forwards
concurrentes (N x TORCH_THREADS threads sobrescribiendo la CPU). `_run_inference`
garantiza UN forward a la vez (single-flight) via `_INFERENCE_LOCK`.

Estos tests no necesitan el stack pesado (cv2/supervision/torch/ultralytics):
`_run_inference` recibe el modelo como callable y camera_worker.py importa
solo la biblioteca estandar. Los contadores del worker se prueban con
_InjectedContext + FakeCap (analogo a los tests de lifecycle F1)."""

import threading
import time

import pytest

import camera_worker
from camera_worker import (
    CameraWorker,
    _DetectorContext,
    _InjectedContext,
    _run_inference,
)
from testutil import FakeCap, wait_until


def _frames():
    return [object() for _ in range(3)]


# ---------- Single-flight sobre el lock global ----------

def test_run_inference_single_flight():
    """Cualquier cantidad de hilos llamando al modelo compartido a la vez:
    jamas hay mas de un forward en ejecucion."""
    in_flight = 0
    peak = 0
    state_lock = threading.Lock()

    def slow_model(*args, **kwargs):
        nonlocal in_flight, peak
        with state_lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.02)
        with state_lock:
            in_flight -= 1
        return "detecciones"

    barrier = threading.Barrier(8)
    errors = []

    def work():
        try:
            barrier.wait(timeout=5)
            for _ in range(15):
                assert camera_worker._run_inference(
                    slow_model, "frame", classes=None, conf=0.5, imgsz=320
                ) == "detecciones"
        except Exception as exc:  # noqa: BLE001 - propagado al hilo principal
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert peak == 1


def test_run_inference_libera_lock_si_el_modelo_falla():
    """Un forward que explota no deja el lock tomado: el siguiente worker
    procede sin quedar colgado (el watchdog F1 depende de esto)."""

    def exploding_model(*args, **kwargs):
        raise RuntimeError("boom en inferencia")

    with pytest.raises(RuntimeError, match="boom"):
        camera_worker._run_inference(exploding_model, "x", classes=None, conf=0.5, imgsz=320)

    ran = {"ok": False}

    def ok_model(*args, **kwargs):
        ran["ok"] = True
        return "fine"

    camera_worker._run_inference(ok_model, "x", classes=None, conf=0.5, imgsz=320)
    assert ran["ok"] is True


def test_detector_context_usa_la_inferencia_serializada():
    """El contexto de deteccion default enruta SIEMPRE por _run_inference
    (el lock del modulo): no hay camino alternativo no serializado."""
    ctx = _DetectorContext(object())
    assert ctx._infer is camera_worker._run_inference


# ---------- Contadores de rendimiento del worker ----------

def _build_worker(camera_dict, processor=None, **kwargs):
    cap = FakeCap(frames=_frames())
    return CameraWorker(
        dict(camera_dict),
        enable_render=False,
        processor_factory=(lambda cfg: processor) if processor is not None else None,
        cap_factory=lambda url: cap,
        reconnect_delay=0.02,
        **kwargs,
    )


def test_worker_contadores_frames_con_procesador_inyectado(camera_dict):
    """frames_available >= frames_processed (nunca mas procesados que vistos);
    frames descartados = available - processed (FRAME_SKIP o espera por lock);
    el worker no inventa inferencias cuando no hay YOLO real."""
    proc = _InjectedContext()
    worker = _build_worker(camera_dict, processor=proc)
    worker.start()
    try:
        assert wait_until(lambda: worker.frames_processed >= 3)
        assert worker.frames_available >= worker.frames_processed >= 1
        assert worker.frames_available - worker.frames_processed >= 0
        # _InjectedContext no toca YOLO: los contadores de inferencia quedan 0
        assert worker.inference_count == 0
        assert worker.last_inference_ms == 0.0
    finally:
        worker.stop()
        worker.join(timeout=5.0)