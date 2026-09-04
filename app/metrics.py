"""Métricas Prometheus (F7.1) para /metrics.

Expone métricas en formato Prometheus exposition format para scraping por
Prometheus/Grafana. Las métricas se actualizan on-demand al consultar /metrics
(pull model), no push.

Métricas expuestas:
  - object_tracker_uptime_seconds
  - object_tracker_cameras_active
  - object_tracker_fps_total (camera, class)
  - object_tracker_detections_total (camera, class)
  - object_tracker_rtsp_errors_total (camera)
  - object_tracker_gallery_files
  - object_tracker_disk_usage_bytes
"""

import os

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    generate_latest,
)

REGISTRY = CollectorRegistry()

# Gauges / Counters
uptime = Gauge("object_tracker_uptime_seconds", "Process uptime in seconds", registry=REGISTRY)
cameras_active = Gauge("object_tracker_cameras_active", "Number of active cameras", registry=REGISTRY)
fps_total = Counter("object_tracker_fps_total", "Detection frames processed", ["camera"], registry=REGISTRY)
detections_total = Counter("object_tracker_detections_total", "Objects detected", ["camera", "class_name"], registry=REGISTRY)
rtsp_errors = Counter("object_tracker_rtsp_errors_total", "RTSP connection errors", ["camera"], registry=REGISTRY)
gallery_files = Gauge("object_tracker_gallery_files", "Number of files in gallery", registry=REGISTRY)
disk_usage = Gauge("object_tracker_disk_usage_bytes", "Disk usage of ./data directory", registry=REGISTRY)


def collect_metrics(app_start_time):
    """Actualiza y genera el output de Prometheus."""
    import time
    import camera_manager
    from db import list_cameras

    uptime.set(time.time() - app_start_time)

    active = 0
    for cam in list_cameras():
        w = camera_manager.get_worker(cam["name"])
        if w and getattr(w, "status", "") == "en vivo":
            active += 1
        if w:
            fps_total.labels(camera=cam["name"])._value.set(
                getattr(w, "inference_count", 0)
            )
            for cls in ("person", "car", "motorcycle", "bus", "truck"):
                detections_total.labels(camera=cam["name"], class_name=cls)._value.set(0)
    cameras_active.set(active)

    # Gallery files count
    gallery_dir = os.getenv("GALLERY_DIR", "/app/data/gallery")
    try:
        count = sum(len(files) for _, _, files in os.walk(gallery_dir))
        gallery_files.set(count)
    except OSError:
        gallery_files.set(0)

    # Disk usage
    data_dir = os.getenv("DB_PATH", "/app/data/app.db")
    try:
        total = 0
        base = os.path.dirname(data_dir)
        for dirpath, _, filenames in os.walk(base):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        disk_usage.set(total)
    except Exception:
        disk_usage.set(0)

    return generate_latest(REGISTRY)
