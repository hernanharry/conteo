"""Tests del CameraManager (F1): invariante de UN worker por camara,
cuarentena de workers atascados, watchdog evaluador y anti crash-loop.

Usan FakeWorker (inyectado via set_worker_factory): corren SIEMPRE en local,
sin stack pesado."""

import time

import pytest

import camera_manager
from testutil import FakeWorker, wait_until


@pytest.fixture(autouse=True)
def _manager_limpio():
    camera_manager.reset_for_tests()
    yield
    camera_manager.reset_for_tests()


@pytest.fixture()
def fake_factory():
    """Factory que registra cada worker creado (para poder distinguirlos)."""
    created = []

    def factory(cam, initial_in=0, initial_out=0):
        w = FakeWorker(cam, initial_in=initial_in, initial_out=initial_out)
        created.append(w)
        return w

    camera_manager.set_worker_factory(factory)
    return created


def _envejecer(worker):
    worker.last_frame_time = time.time() - 600
    worker.last_detection_time = time.time() - 600


# ---------- start / stop / restart ----------

def test_start_y_stop_basico(camera_dict, fake_factory):
    camera_manager.start_camera(dict(camera_dict, cumulative_in=4, cumulative_out=2))
    worker = camera_manager.get_worker("entrada-principal")
    assert worker is not None
    assert worker.in_count == 4
    assert worker.out_count == 2

    camera_manager.stop_camera("entrada-principal")
    assert camera_manager.get_worker("entrada-principal") is None
    assert worker.is_alive() is False


def test_start_camera_no_duplica_worker(camera_dict, fake_factory):
    camera_manager.start_camera(camera_dict)
    primero = camera_manager.get_worker("entrada-principal")
    camera_manager.start_camera(camera_dict)
    assert camera_manager.get_worker("entrada-principal") is primero
    assert len(fake_factory) == 1


def test_restart_camera_conserva_conteos_y_no_duplica(camera_dict, fake_factory):
    cam = dict(camera_dict, cumulative_in=5, cumulative_out=3)
    camera_manager.start_camera(cam)
    viejo = camera_manager.get_worker("entrada-principal")

    camera_manager.restart_camera(cam)

    nuevo = camera_manager.get_worker("entrada-principal")
    assert nuevo is not None
    assert nuevo is not viejo
    assert nuevo.in_count == 5
    assert nuevo.out_count == 3
    assert viejo.is_alive() is False
    assert len(fake_factory) == 2
    assert "entrada-principal" not in camera_manager._quarantined


# ---------- Cuarentena: worker que no termina ----------

def test_restart_con_worker_atascado_pone_slot_en_cuarentena_sin_duplicar(camera_dict, fake_factory):
    camera_manager.start_camera(camera_dict)
    viejo = camera_manager.get_worker("entrada-principal")
    viejo.set_stuck(True)

    camera_manager.restart_camera(camera_dict)

    # NO se creo otro worker (invarianza) y el slot quedo en cuarentena.
    assert camera_manager.get_worker("entrada-principal") is None
    assert "entrada-principal" in camera_manager._quarantined
    assert len(fake_factory) == 1
    assert viejo.is_alive() is True


def test_start_camera_no_salta_por_encima_de_cuarentena(camera_dict, fake_factory):
    travieso = FakeWorker(camera_dict)
    travieso.set_stuck(True)
    with camera_manager._lock:
        camera_manager._quarantined["entrada-principal"] = {
            "worker": travieso, "cam": None, "carry_in": 0, "carry_out": 0,
        }

    camera_manager.start_camera(camera_dict)

    assert camera_manager.get_worker("entrada-principal") is None
    assert len(fake_factory) == 0


def test_watchdog_recupera_el_slot_cuando_el_worker_cuarentenado_muere(camera_dict, fake_factory):
    camera_manager.start_camera(camera_dict)
    viejo = camera_manager.get_worker("entrada-principal")
    viejo.set_stuck(True)
    camera_manager.restart_camera(camera_dict)
    assert "entrada-principal" in camera_manager._quarantined

    # "muere" la espera (simulando que cap.read devolvio)
    viejo.set_alive(False)

    camera_manager._watchdog_pass()

    nuevo = camera_manager.get_worker("entrada-principal")
    assert nuevo is not None
    assert nuevo is not viejo
    assert "entrada-principal" not in camera_manager._quarantined
    assert len(fake_factory) == 2


# ---------- Watchdog evaluador (puro) ----------

def test_evaluate_muerto():
    w = FakeWorker({"name": "x"}, alive=False)
    assert camera_manager._evaluate_worker(w, time.time()) == "dead"


def test_evaluate_failed_config_no_auto():
    w = FakeWorker({"name": "x"}, alive=True)
    w.status = "error"
    w._config_error = True
    assert camera_manager._evaluate_worker(w, time.time()) == "failed"


def test_evaluate_failed_runtime_si_es_restartable():
    w = FakeWorker({"name": "x"}, alive=True)
    w.status = "error"
    w._config_error = False
    assert camera_manager._evaluate_worker(w, time.time()) == "restart"


def test_evaluate_reconectando_no_reinicia():
    w = FakeWorker({"name": "x"}, alive=True)
    w.status = "reconectando"
    _envejecer(w)
    assert camera_manager._evaluate_worker(w, time.time()) == "wait"


def test_evaluate_deteniendo_espera():
    w = FakeWorker({"name": "x"}, alive=True)
    w.status = "deteniendo"
    _envejecer(w)
    assert camera_manager._evaluate_worker(w, time.time()) == "wait"


def test_evaluate_iniciando_espera():
    w = FakeWorker({"name": "x"}, alive=True)
    w.status = "iniciando"
    _envejecer(w)
    assert camera_manager._evaluate_worker(w, time.time()) == "wait"


def test_evaluate_stale_reinicia():
    w = FakeWorker({"name": "x"}, alive=True)
    w.status = "en vivo"
    _envejecer(w)
    assert camera_manager._evaluate_worker(w, time.time()) == "restart"


def test_evaluate_ok():
    w = FakeWorker({"name": "x"}, alive=True)
    w.status = "en vivo"
    assert camera_manager._evaluate_worker(w, time.time()) == "ok"


# ---------- Watchdog en accion ----------

def test_watchdog_reinicia_worker_stale(db, camera_dict, fake_factory):
    db.add_camera(camera_dict)
    camera_manager.start_camera(camera_dict)
    viejo = camera_manager.get_worker("entrada-principal")
    _envejecer(viejo)

    camera_manager._watchdog_pass()

    nuevo = camera_manager.get_worker("entrada-principal")
    assert nuevo is not None
    assert nuevo is not viejo
    assert viejo.is_alive() is False
    assert camera_manager._restart_stats["entrada-principal"]["times"]


def test_watchdog_no_reinicia_worker_reconectando(db, camera_dict, fake_factory):
    db.add_camera(camera_dict)
    camera_manager.start_camera(camera_dict)
    w = camera_manager.get_worker("entrada-principal")
    w.status = "reconectando"
    _envejecer(w)

    camera_manager._watchdog_pass()

    assert camera_manager.get_worker("entrada-principal") is w
    assert "entrada-principal" not in camera_manager._restart_stats


def test_watchdog_crash_loop_bloquea_reinicios(monkeypatch, db, camera_dict, fake_factory):
    monkeypatch.setattr(camera_manager, "WATCHDOG_MAX_RESTARTS_PER_WINDOW", 2)
    monkeypatch.setattr(camera_manager, "WATCHDOG_RESTART_WINDOW_SECONDS", 60)
    monkeypatch.setattr(camera_manager, "WATCHDOG_MIN_RESTART_INTERVAL_SECONDS", 0)

    db.add_camera(camera_dict)
    camera_manager.start_camera(camera_dict)

    for creado in fake_factory:
        _envejecer(creado)
    camera_manager._watchdog_pass()  # reinicio #1 (crea worker #2)
    for creado in fake_factory:
        _envejecer(creado)
    camera_manager._watchdog_pass()  # reinicio #2 (crea worker #3)
    for creado in fake_factory:
        _envejecer(creado)
    camera_manager._watchdog_pass()  # bloqueado: NO reinicia

    assert len(fake_factory) == 3
    assert camera_manager._restart_stats["entrada-principal"]["blocked"] is True


def test_stop_all_detiene_workers_y_cuarentenas(camera_dict, fake_factory):
    camera_manager.start_camera(camera_dict)
    normal = camera_manager.get_worker("entrada-principal")

    stuck = FakeWorker(dict(camera_dict, name="otra"))
    stuck.set_stuck(True)
    with camera_manager._lock:
        camera_manager._quarantined["otra"] = {"worker": stuck, "cam": None, "carry_in": 0, "carry_out": 0}

    camera_manager.stop_all()

    assert camera_manager._workers == {}
    assert camera_manager._quarantined == {}
    assert normal.is_alive() is False
    assert stuck.stopped is True