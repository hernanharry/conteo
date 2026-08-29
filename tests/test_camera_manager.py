"""Tests de baseline sobre CameraManager, reconciliados con el contrato F1.

F0 los dejo marcados `docker` (importan camera_worker -> cv2/torch/ultralytics)
para verificarse en el entorno objetivo.

F1 cambio el ciclo de vida de CameraManager (invariante de UN worker por camara,
cuarentena de workers que no terminan en WORKER_STOP_TIMEOUT_SECONDS y watchdog
evaluador). El contrato nuevo se cubre de forma determinista en
test_camera_manager_f1.py con FakeWorker.

Estos tests conservan el nombre e intencion de la F0 pero validan el CONTRATO F1:
aislan el estado global con reset_for_tests() (en F0 el estado mutable global
hacia que un test que dejaba cuarentena ensuciara a los siguientes) y usan
FakeWorker donde el resultado depende de que el worker muera a tiempo.
"""

import pytest

from testutil import FakeWorker


def _load_camera_manager():
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    import camera_manager

    camera_manager.reset_for_tests()
    return camera_manager


@pytest.fixture()
def fake_factory():
    """Factory inyectable de FakeWorker: restart/stop son deterministas
    (el worker muere en stop(), sin depender de un RTSP real)."""
    camera_manager = _load_camera_manager()
    created = []

    def factory(cam, initial_in=0, initial_out=0):
        w = FakeWorker(cam, initial_in=initial_in, initial_out=initial_out)
        created.append(w)
        return w

    camera_manager.set_worker_factory(factory)
    return created


@pytest.mark.docker
def test_start_all_con_db_vacia_no_crea_workers(db):
    """Con la DB vacia (sin cameras), start_all no debe levantar workers."""
    camera_manager = _load_camera_manager()

    camera_manager.start_all()
    assert len(camera_manager._workers) == 0


@pytest.mark.docker
def test_worker_arranca_con_conteos_acumulados_iniciales(camera_dict, fake_factory):
    """Un worker nuevo toma cumulative_in/out como base (no pierde conteos
    persistidos al reiniciar)."""
    camera_manager = _load_camera_manager()
    cam = dict(camera_dict, cumulative_in=5, cumulative_out=3)

    camera_manager.start_camera(cam)
    try:
        worker = camera_manager.get_worker("entrada-principal")
        assert worker is not None
        assert worker.in_count == 5
        assert worker.out_count == 3
    finally:
        camera_manager.stop_camera("entrada-principal")


@pytest.mark.docker
def test_restart_camera_conserva_conteo_acumulado(camera_dict, fake_factory):
    """restart_camera no pierde el conteo acumulado del worker anterior.

    F1: el worker nuevo solo se arranca cuando el anterior confirmo que termino;
    con FakeWorker (determinista) restart completa y conserva la cuenta."""
    camera_manager = _load_camera_manager()
    cam = dict(camera_dict, cumulative_in=2, cumulative_out=1)

    camera_manager.start_camera(cam)
    try:
        camera_manager.restart_camera(cam)
        worker = camera_manager.get_worker("entrada-principal")
        assert worker is not None
        assert worker.in_count == 2
        assert worker.out_count == 1
        assert "entrada-principal" not in camera_manager._quarantined
    finally:
        camera_manager.stop_camera("entrada-principal")


@pytest.mark.docker
def test_stop_remove_worker_del_diccionario(camera_dict, fake_factory):
    camera_manager = _load_camera_manager()

    camera_manager.start_camera(dict(camera_dict, cumulative_in=0, cumulative_out=0))
    assert "entrada-principal" in camera_manager._workers
    camera_manager.stop_camera("entrada-principal")
    assert camera_manager.get_worker("entrada-principal") is None