"""Tests de baseline sobre configuracion de camaras.

OJO GAP-F0 (documentado, no refactorizado): el parseo de config vive INLINE
dentro de CameraWorker.run() (app/camera_worker.py:181-184), sin funciones
independientes que poder testear. Por F0 no se refactoriza. Estos tests fijan
el CONTRATO que ese parseo deberia cumplir, usando las mismas expresiones;
cuando la capa pesada este disponible se ejecutan ademas los tests del modulo
real (test_*_worker/manager). El destino: extraer este parseo a una funcion
testeable en F1.
"""

import pytest


def _parse_worker_config(cam):
    """Espejo exacto del parseo actual de CameraWorker.run (sin imports pesados)."""
    classes = [int(c) for c in cam["classes"].split(",") if c.strip() != ""]
    conf_threshold = float(cam["conf_threshold"])
    line_start = tuple(int(v) for v in cam["line_start"].split(","))
    line_end = tuple(int(v) for v in cam["line_end"].split(","))
    return classes, conf_threshold, line_start, line_end


class TestParseoContrato:
    def test_config_valida_por_defecto(self, camera_dict):
        classes, conf, start, end = _parse_worker_config(camera_dict)
        assert classes == [0, 2, 3, 5, 7]
        assert conf == pytest.approx(0.35)
        assert start == (0, 300)
        assert end == (1280, 300)

    def test_classes_spacios_y_vacios_ignorados(self, camera_dict):
        cam = dict(camera_dict, classes="0, 2, , 7, ")
        classes, *_ = _parse_worker_config(cam)
        assert classes == [0, 2, 7]

    def test_classes_vacias_devuelve_lista_vacia(self, camera_dict):
        cam = dict(camera_dict, classes=", ,")
        classes, *_ = _parse_worker_config(cam)
        assert classes == []

    def test_classes_invalidas_lanzan_value_error(self, camera_dict):
        """GAP-F1: hoy un 'persona' en classes mata CameraWorker.run (sin
        try/except) y el watchdog lo reinicia en bucle."""
        cam = dict(camera_dict, classes="persona")
        with pytest.raises(ValueError):
            _parse_worker_config(cam)

    def test_conf_invalida_lanza_value_error(self, camera_dict):
        cam = dict(camera_dict, conf_threshold="alta")
        with pytest.raises(ValueError):
            _parse_worker_config(cam)

    def test_linea_invalida_lanza_value_error(self, camera_dict):
        cam = dict(camera_dict, line_start="abc,def")
        with pytest.raises(ValueError):
            _parse_worker_config(cam)

    def test_linea_escalar_no_lanza_en_parseo_pero_es_latente(self, camera_dict):
        """GAP-F1: '300' parsea a (300,) sin error, pero LineZone espera 2
        coordenadas; recien fallaria en sv.Point(*line_start) dentro de run()
        (TypeError). El parseo no protege este caso."""
        cam = dict(camera_dict, line_start="300")
        classes, conf, start, end = _parse_worker_config(cam)
        assert start == (300,)


class TestParseoRealWorker:
    """Estos importan camera_worker (cv2/supervision/torch/ultralytics). Se
    ejecutan solo donde el stack completo esta instalado (Docker / venv).
    GAP-F0: no se puede correr en el Python local de desarrollo."""

    @pytest.mark.docker
    def test_camera_worker_importa_con_config_valida(self, camera_dict):
        pytest.importorskip("cv2")
        pytest.importorskip("torch")
        import camera_worker  # noqa: F401

        assert camera_worker.FRAME_SKIP >= 1
        assert camera_worker.IMGSZ > 0