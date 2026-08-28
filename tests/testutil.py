"""Utilidades para tests de lifecycle (F1): dobles deterministas del
stack pesado (sin cv2 / supervision / torch / ultralytics / requests)."""

import time


def wait_until(condition, timeout=5.0, interval=0.02):
    """Poll sencillo: devuelve True apenas la condicion se cumple (o al final
    del timeout). Incluye un chequeo final para el caso limite."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return bool(condition())


class FakeCap:
    """Doble de cv2.VideoCapture para _FrameGrabber.

    - frames: lista de objetos-frame que se recorren en ciclo (infinito si
      hay al menos uno).
    - fail_for: si > 0, las primeras N lecturas devuelven (False, None)
      (simula corte de red al arrancar).
    - always_fail: True -> nunca entrega frames.
    """

    def __init__(self, frames=None, fail_for=0, always_fail=False):
        self.frames = list(frames or [])
        self.fail_for = fail_for
        self.always_fail = always_fail
        self.idx = 0
        self.reads = 0
        self.released = False

    def set(self, *args, **kwargs):
        return True

    def read(self):
        self.reads += 1
        if self.always_fail:
            return (False, None)
        if self.reads <= self.fail_for:
            return (False, None)
        if not self.frames:
            return (False, None)
        frame = self.frames[self.idx % len(self.frames)]
        self.idx += 1
        return (True, frame)

    def release(self):
        self.released = True


class FakeWorker:
    """Doble de CameraWorker para el manager: controles manuales de vida y
    estado; stop() mata al worker salvo que el test lo impida (stuck)."""

    def __init__(self, camera, initial_in=0, initial_out=0, alive=True):
        self.camera = camera
        self.name = camera["name"]
        self.in_count = initial_in
        self.out_count = initial_out
        self.status = "en vivo"
        self.last_frame_time = time.time()
        self.last_detection_time = time.time()
        self._alive = alive
        self.stopped = False
        self.joined = False
        self._config_error = False
        self._stuck = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True
        if not self._stuck:
            self.status = "detenida"
            self._alive = False

    def set_stuck(self, value):
        """Si True, stop() no mata al worker: simula un thread que no termina."""
        self._stuck = value

    def set_alive(self, value):
        self._alive = value

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        if self.stopped and not self._stuck:
            self._alive = False
        self.joined = True