"""Tests de _FrameGrabber (F1): ciclo de frames, reconexion y stop.

Usan FakeCap (sin cv2): corren siempre en local.
"""

import threading
import time

from camera_worker import _FrameGrabber
from testutil import FakeCap, wait_until


def _make_grabber(cap_factory, reconnect_delay=0.05, url="rtsp://fake"):
    def factory(_url):
        return cap_factory

    return _FrameGrabber(url, cap_factory=factory, reconnect_delay=reconnect_delay)


def test_lee_frames_y_marca_connected():
    cap = FakeCap(frames=[object(), object(), object()])
    g = _make_grabber(cap)

    assert g.connected is False
    g.start()
    try:
        assert wait_until(lambda: g.frame_count > 0)
        assert g.connected is True
        assert g.get_latest_frame() is not None
        assert g.last_frame_time is not None
    finally:
        g.stop()
        g.join(timeout=2.0)
    assert cap.released is True


def test_sin_frames_nunca_marca_connected():
    cap = FakeCap(always_fail=True)
    g = _make_grabber(cap)
    g.start()
    try:
        # deja correr unos ciclos de reconnect(rapida)
        time.sleep(0.3)
        assert g.connected is False
        assert g.reconnect_count > 0
        assert g.last_error is not None
    finally:
        g.stop()
        g.join(timeout=2.0)


def test_reconecta_luego_de_fallos_iniciales():
    # falla en las primeras 2 lecturas y despues entrega frames
    cap = FakeCap(frames=[object()], fail_for=2)
    g = _make_grabber(cap)
    g.start()
    try:
        assert wait_until(lambda: g.connected is True)
        assert wait_until(lambda: g.frame_count > 0)
        assert g.reconnect_count >= 1
    finally:
        g.stop()
        g.join(timeout=2.0)


def test_stop_detiene_el_hilo_y_libera_la_captura():
    cap = FakeCap(frames=[object()])
    g = _make_grabber(cap)
    g.start()
    assert wait_until(lambda: g.frame_count > 0)
    g.stop()
    g.join(timeout=2.0)
    assert not g.is_alive()
    assert cap.released is True


def test_la_apertura_que_falla_reintenta():
    # cap_factory que lanza (no pudo abrir la captura): debe reintentar
    attempts = {"n": 0}

    def factory(_url):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("no se pudo abrir")
        return FakeCap(frames=[object()])

    g = _FrameGrabber("rtsp://fake", cap_factory=factory, reconnect_delay=0.05)
    g.start()
    try:
        assert wait_until(lambda: g.connected is True)
        assert attempts["n"] >= 2
    finally:
        g.stop()
        g.join(timeout=2.0)


def test_grabber_max_fps_espacia_las_lecturas(monkeypatch):
    # GRABBER_MAX_FPS>0 ralentiza la decodificacion para liberar CPU en
    # maquinas debiles: entre lecturas exitosas debe pasar ~1/GRABBER_MAX_FPS.
    # El test solo verifica el espacio minimo (holgado a 0.025s) para no
    # destinar en CI: lo que nunca debe pasar es que NO haya espacio.
    monkeypatch.setattr("camera_worker.GRABBER_MAX_FPS", 8.0)  # gap esperado ~0.125s
    cap = FakeCap(frames=[object()] * 20)
    g = _make_grabber(cap)
    stamps = []

    original_set = g._set_connected

    def patched_set_connected(value):
        if value and g.frame_count > 0 and len(stamps) < 4:
            stamps.append(time.time())
        original_set(value)

    g._set_connected = patched_set_connected  # type: ignore[assignment]
    g.start()
    try:
        assert wait_until(lambda: len(stamps) >= 4, timeout=3.0)
    finally:
        g.stop()
        g.join(timeout=2.0)

    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert gaps, "deberia haber al menos 2 lecturas espaciadas"
    assert all(g >= 0.02 for g in gaps), f"gaps demasiado cortos: {gaps}"
    assert g.frame_count <= len(stamps) + 1