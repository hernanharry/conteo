"""Parsing de configuracion de camaras (F1).

Extrae el parseo que vivia inline dentro de CameraWorker.run() a una funcion
independiente, testeable y con errores controlados. Mantiene exactamente la
misma semantica del parseo original (enteros por coma, float, puntos x,y) y
agrega validacion estricta de exactamente 2 coordenadas por punto (antes eso
reventaba recien dentro de sv.Point()).

Solo usa la biblioteca estandar: se puede importar y testear sin cargar
torch / ultralytics / supervision / opencv.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class CameraConfig:
    name: str
    rtsp_url: str
    classes: List[int]
    conf_threshold: float
    line_start: Tuple[int, int]
    line_end: Tuple[int, int]


def parse_camera_config(camera) -> CameraConfig:
    """Convierte el dict de camara (viene de DB/UI) en una CameraConfig.

    Lanza ValueError con mensajes claros ante configuracion invalida:
    - nombre o rtsp_url vacios
    - classes no separables en enteros COCO
    - conf_threshold no numerico
    - line_start / line_end que no sean 'x,y' con exactamente 2 enteros.
    """
    name = str(camera.get("name") or "").strip()
    if not name:
        raise ValueError("nombre de camara vacio")
    rtsp_url = str(camera.get("rtsp_url") or "").strip()
    if not rtsp_url:
        raise ValueError("rtsp_url vacio")

    classes = _parse_classes(camera.get("classes"))
    conf_threshold = _parse_number(camera.get("conf_threshold") or "0.35", "conf_threshold")
    line_start = _parse_point(camera.get("line_start") or "0,300", "line_start")
    line_end = _parse_point(camera.get("line_end") or "1280,300", "line_end")

    return CameraConfig(name=name, rtsp_url=rtsp_url, classes=classes,
                        conf_threshold=conf_threshold, line_start=line_start, line_end=line_end)


def _parse_classes(raw) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        parts = raw
    else:
        parts = [p.strip() for p in str(raw).split(",") if p.strip() != ""]
    try:
        return [int(p) for p in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"classes invalidas: {raw!r} — deben ser IDs COCO enteros separados por coma"
        ) from exc


def _parse_number(raw, label: str) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} invalido: {raw!r} — debe ser numerico") from exc


def _parse_point(raw, label: str) -> Tuple[int, int]:
    try:
        parts = [int(p.strip()) for p in str(raw).split(",")]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} invalido: {raw!r} — debe ser 'x,y' con enteros") from exc
    if len(parts) != 2:
        raise ValueError(f"{label} invalido: {raw!r} — debe tener exactamente 2 coordenadas (x,y)")
    return (parts[0], parts[1])