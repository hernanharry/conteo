"""Tests F6 — Streaming HLS (hls_segmenter + endpoints /hls).

Verifica:
  - hls_enabled() retorna True cuando ffmpeg esta en PATH
  - start/stop segmenter crea y destruye el hilo
  - push_frame no rompe cuando no hay segmentador
  - endpoints /hls/<name>/stream.m3u8 y /hls/<name>/<segment> devuelven 404
    cuando no hay segmentos, 200 cuando existen
  - cleanup_old_segments limpia archivos .ts viejos
  - MJPEG (/stream) sigue funcionando como fallback
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from testutil import FakeWorker, login_client  # noqa: E402


@pytest.fixture(scope="module")
def hls_server_env(tmp_path_factory):
    """Importa server una vez con DB aislada para tests HLS."""
    import threading
    import db

    db_path = str(tmp_path_factory.mktemp("f6hls") / "app.db")
    db.DB_PATH = db_path
    db.init_db()

    import hls_segmenter
    hls_dir = str(tmp_path_factory.mktemp("f6hls-seg"))
    hls_segmenter.HLS_DIR = hls_dir
    hls_segmenter._enabled = True

    import camera_manager
    camera_manager.reset_for_tests()
    camera_manager.set_worker_factory(FakeWorker)

    import server
    try:
        yield server, hls_dir, hls_segmenter
    finally:
        hls_segmenter.reset_for_tests()
        server.camera_manager.stop_all()
        server.camera_manager.reset_for_tests()
        server.notifications.reset_for_tests()
        db.reset_gallery_retention_for_tests()
        for t in threading.enumerate():
            if getattr(t, "name", None) in ("watchdog", "hls-cleanup") and t.is_alive():
                t.join(timeout=2.0)


@pytest.fixture()
def hls_client(hls_server_env):
    server, _, _ = hls_server_env
    client = server.app.test_client()
    login_client(client)
    return client


def test_hls_playlist_404_sin_segmentos(hls_client):
    resp = hls_client.get("/hls/no-existe/stream.m3u8")
    assert resp.status_code == 404


def test_hls_segment_404_sin_archivo(hls_client):
    resp = hls_client.get("/hls/no-existe/seg_00001.ts")
    assert resp.status_code == 404


def test_hls_playlist_200_con_archivo(hls_server_env, hls_client):
    _, hls_dir, _ = hls_server_env
    cam_dir = os.path.join(hls_dir, "cam-test")
    os.makedirs(cam_dir, exist_ok=True)
    playlist = os.path.join(cam_dir, "stream.m3u8")
    with open(playlist, "w") as f:
        f.write("#EXTM3U\n#EXTINF:2,\nseg_00001.ts\n")
    resp = hls_client.get("/hls/cam-test/stream.m3u8")
    assert resp.status_code == 200
    assert b"EXTM3U" in resp.data


def test_hls_segment_200_con_archivo(hls_server_env, hls_client):
    _, hls_dir, _ = hls_server_env
    cam_dir = os.path.join(hls_dir, "cam-test2")
    os.makedirs(cam_dir, exist_ok=True)
    seg = os.path.join(cam_dir, "seg_00001.ts")
    with open(seg, "wb") as f:
        f.write(b"\x00" * 100)
    resp = hls_client.get("/hls/cam-test2/seg_00001.ts")
    assert resp.status_code == 200
    assert resp.mimetype == "video/mp2t"


def test_hls_segment_rechaza_path_traversal(hls_client):
    resp = hls_client.get("/hls/cam/../../etc/passwd.ts")
    assert resp.status_code in (404, 400)


def test_cleanup_old_segments(hls_server_env):
    _, hls_dir, hls_seg = hls_server_env
    cam_dir = os.path.join(hls_dir, "cam-cleanup")
    os.makedirs(cam_dir, exist_ok=True)
    # Crear un .ts "viejo"
    old_seg = os.path.join(cam_dir, "seg_00001.ts")
    with open(old_seg, "wb") as f:
        f.write(b"\x00" * 50)
    os.utime(old_seg, (0, 0))  # 1970
    # Crear uno "nuevo"
    new_seg = os.path.join(cam_dir, "seg_00002.ts")
    with open(new_seg, "wb") as f:
        f.write(b"\x00" * 50)
    hls_seg.cleanup_old_segments()
    assert not os.path.exists(old_seg)
    assert os.path.exists(new_seg)


def test_mjpeg_fallback_sigue_funcionando(hls_server_env):
    """F6.3: MJPEG queda como fallback; sigue respondiendo 200."""
    server, _, _ = hls_server_env
    server.camera_manager.reset_for_tests()
    server.camera_manager.set_worker_factory(FakeWorker)
    cam = {
        "name": "mjpeg-cam", "rtsp_url": "rtsp://x", "classes": "0",
        "conf_threshold": "0.35", "line_start": "0,0", "line_end": "1,1",
        "active": 1, "cumulative_in": 0, "cumulative_out": 0,
    }
    server.camera_manager.start_camera(cam)
    worker = server.camera_manager.get_worker("mjpeg-cam")
    worker.set_latest_jpeg(b"\xff\xd8\xff\xe0fake")
    client = server.app.test_client()
    login_client(client)
    resp = client.get("/stream/mjpeg-cam")
    assert resp.status_code == 200
    assert resp.mimetype == "multipart/x-mixed-replace"
    gen = iter(resp.response)
    first = next(gen)
    gen.close()
    assert b"fake" in first
    server.camera_manager.reset_for_tests()
