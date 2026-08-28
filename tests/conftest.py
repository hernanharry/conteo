"""Configuracion global de los tests de baseline (Fase 0).

Reglas de F0:
- No refactorizar la aplicacion para hacerla testeable.
- No cambiar comportamiento funcional.
- Requisito: las dependencias pesadas (cv2/supervision/torch/ultralytics/flask)
  NO estan presentes en el Python local de desarrollo (Windows, Python 3.14).
  Por eso los tests que tocan camera_worker.py/camera_manager.py importan
  esas dependencias y se ejecutan solo cuando el stack completo esta
  disponible (contenedor Docker o venv con requirements completos).
  Los tests de la capa db.py usan solo stdlib y SIEMPRE se ejecutan.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Los modulos de la aplicacion viven en app/ (import db, camera_worker, ...).
sys.path.insert(0, str(REPO_ROOT / "app"))

# La BD y la galeria jamac tocan ./data de la aplicacion: siempre a env temporales.
TEST_DB_DIR = tempfile.mkdtemp(prefix="f0-db-")
os.environ["DB_PATH"] = os.path.join(TEST_DB_DIR, "baseline.db")
os.environ["GALLERY_DIR"] = tempfile.mkdtemp(prefix="f0-gallery-")


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker",
        action="store_true",
        default=False,
        help="ejecuta tests que requieren stack completo / Docker (workers reales, RTSP sintetico)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(
            reason="requiere stack completo/Docker; usar --run-docker en el entorno objetivo"
        )
        for item in items:
            if item.get_closest_marker("docker"):
                item.add_marker(skip_docker)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """BD SQLite aislada por test: cada test usa su propia base temporal."""
    import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    return db


@pytest.fixture()
def camera_dict():
    """Config de camara tal como llega de la UI/DB a CameraWorker."""
    return {
        "name": "entrada-principal",
        "rtsp_url": "rtsp://usuario:password@192.168.1.64:554/Streaming/Channels/101",
        "classes": "0,2,3,5,7",
        "conf_threshold": "0.35",
        "line_start": "0,300",
        "line_end": "1280,300",
        "active": 1,
        "cumulative_in": 0,
        "cumulative_out": 0,
    }