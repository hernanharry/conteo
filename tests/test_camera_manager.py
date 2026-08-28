"""Tests de baseline sobre CameraManager (y CameraWorker como minimo).

GAP-F0 (documentado, no refactorizado): camera_manager.py importa
camera_worker.py en el topo del modulo, que a su vez importa cv2 /
supervision / torch / ultralytics. Ademas CameraWorker es una clase concreta
instanciada directamente (sin inyeccion de dependencias): para testearla desde
cero se necesitaria refactorizar -- prohibido en F0. Por eso:

- Todos los tests de este modulo estan marcados `docker` (se saltan por
  defecto; se activan con --run-docker en un entorno con el stack completo).
- El comportamiento basico que aqui se fija debe verificarse en el entorno
  objetivo (contenedor o venv con requirements completos).
- Los contratos de conftest/camera_dict valen para siembra de datos reales.
"""

import pytest


@pytest.mark.docker
def test_start_all_con_db_vacia_no_crea_workers(camera_dict):
    """Con la DB vaciac (sin cameras), start_all no debe levantar workers."""
    pytest.importorskip("cv2")
    import camera_manager

    # Aislamiento: resetear estado global del modulo (GAP-F1: es global mutable)
    camera_manager._workers.clear()

    camera_manager.start_all()
    assert len(camera_manager._workers) == 0


@pytest.mark.docker
def test_worker_arranca_con_conteos_acumulados_iniciales(camera_dict):
    """Un worker nuevo toma cumulative_in/out como base (no pierde conteos
    persistidos al reiniciar). GAP-F1: estado global de _workers a limpiar."""
    pytest.importorskip("cv2")
    import camera_manager

    camera_manager._workers.clear()
    cam = dict(camera_dict, cumulative_in=5, cumulative_out=3)
    from camera_manager import start_camera, stop_camera

    start_camera(cam)
    try:
        worker = camera_manager.get_worker("entrada-principal")
        assert worker is not None
        assert worker.in_count == 5
        assert worker.out_count == 3
    finally:
        stop_camera("entrada-principal")


@pytest.mark.docker
def test_restart_camera_conserva_conteo_acumulado(camera_dict):
    """restart_camera no debe perder el conteo acumulado del worker anterior."""
    pytest.importorskip("cv2")
    import camera_manager

    camera_manager._workers.clear()
    cam = dict(camera_dict, cumulative_in=2, cumulative_out=1)
    camera_manager.start_camera(cam)
    try:
        camera_manager.restart_camera(cam)
        worker = camera_manager.get_worker("entrada-principal")
        assert worker is not None
        assert worker.in_count >= 2
        assert worker.out_count >= 1
    finally:
        camera_manager.stop_camera("entrada-principal")


@pytest.mark.docker
def test_stop_remove_worker_del_diccionario(camera_dict):
    pytest.importorskip("cv2")
    import camera_manager

    camera_manager._workers.clear()
    camera_manager.start_camera(dict(camera_dict, cumulative_in=0, cumulative_out=0))
    assert "entrada-principal" in camera_manager._workers
    camera_manager.stop_camera("entrada-principal")
    assert camera_manager.get_worker("entrada-principal") is None