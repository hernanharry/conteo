"""Estabilizador de IDs de tracking (F10).

ByteTrack asigna un `tracker_id` nuevo cuando el objeto se mueve demasiado
entre muestras (autos rapidos con cadencia de deteccion baja). LineZone de
supervision solo cuenta un cruce si el MISMO tracker_id se ve de un lado de
la linea y del otro: ante un ID switch el cruce se pierde ("cruza pero no
cuenta").

Este modulo convierte los IDs raw de ByteTrack en `entity_id` estables a lo
largo de ID switches: si una deteccion no continua un tracker_id conocido,
se intenta asociarla geometricamente (misma clase + proximidad espacial) con
una deteccion reciente de otro tracker_id. El resultado se anota como
`tracker_id` en el stream que llega a LineZone, que asi ve la transicion de
lado y cuenta.

Reglas:
- El match por tracker_id tiene prioridad (continuidad de ByteTrack).
- El match geometrico exige: misma clase, no estar expirado, desplazamiento
  del centro acotado a una fraccion del tamano del bbox anterior, y que ni el
  candidato ni la entidad hayan sido ya reclamados en este frame.
- Una entidad expira tras ID_ASSOC_MAX_AGE_FRAMES frames procesados sin ser
  vista (evita fusionar un objeto nuevo con uno viejo lejano).
- Solo usa la biblioteca estandar: testeable sin numpy/supervision.
"""

import os
from dataclasses import dataclass

MAX_AGE_FRAMES = int(os.getenv("ID_ASSOC_MAX_AGE_FRAMES", "15"))
# Debe ser > 1.0: para que LineZone cuente, la caja pasa de estar COMPLETA de
# un lado a COMPLETA del otro, y eso implica un desplazamiento de centro de al
# menos el alto/width del bbox (ratio >= 1.0). 2.0 deja margen para esa
# transicion sin abrir la puerta a fusionar objetos lejanos.
MAX_DISPLACEMENT = float(os.getenv("ID_ASSOC_MAX_DISPLACEMENT", "2.0"))
MIN_OVERLAP = float(os.getenv("ID_ASSOC_MIN_OVERLAP", "0.02"))


@dataclass
class _Entity:
    entity_id: int
    class_id: int
    last_tracker_id: int
    last_box: tuple  # (x1, y1, x2, y2)
    age_frames: int = 0


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _displacement_ratio(prev_box, curr_box):
    """Desplazamiento del centro en unidades del tamano del bbox anterior."""
    x1, y1, x2, y2 = prev_box
    pw, ph = max(0.0, x2 - x1), max(0.0, y2 - y1)
    size = max(pw, ph)
    if size <= 0:
        return float("inf")
    cx0, cy0 = _box_center(prev_box)
    cx1, cy1 = _box_center(curr_box)
    return ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5 / size


class StableIDAssigner:
    """Asigna un entity_id estable a cada deteccion del frame."""

    def __init__(self, max_age_frames=None, max_displacement=None, min_overlap=None):
        self.max_age_frames = max_age_frames if max_age_frames is not None else MAX_AGE_FRAMES
        self.max_displacement = (
            max_displacement if max_displacement is not None else MAX_DISPLACEMENT
        )
        self.min_overlap = min_overlap if min_overlap is not None else MIN_OVERLAP
        self._entities = {}  # entity_id -> _Entity
        self._next_id = 1
        self.frame_index = 0

    def update(self, detections):
        """detections: lista de tuplas (tracker_id, class_id, (x1,y1,x2,y2)) o
        dicts con claves tracker_id/class_id/xyxy. Devuelve lista de entity_id
        alineada con la entrada (None donde no hubo bbox valido)."""
        frame_dets = []
        for det in detections:
            if isinstance(det, dict):
                tid = det.get("tracker_id")
                cid = det.get("class_id")
                box = tuple(det["xyxy"])
            else:
                tid, cid, box = det
            if box is None or len(box) != 4:
                frame_dets.append((tid, cid, None))
            else:
                frame_dets.append((tid, cid, (float(box[0]), float(box[1]), float(box[2]), float(box[3]))))

        self.frame_index += 1
        self._expire()

        # entidades reclamadas en este frame (match exacto o geometrico)
        claimed_entities = set()

        def _match_by_tracker(tid, cid):
            if tid is None:
                return None
            for ent_id, ent in self._entities.items():
                if ent_id in claimed_entities:
                    continue
                if ent.last_tracker_id == tid:
                    return ent_id
            return None

        def _match_geometric(cid, box):
            best_id, best_score = None, float("inf")
            cx, cy = _box_center(box)
            for ent_id, ent in self._entities.items():
                if ent_id in claimed_entities:
                    continue
                if ent.class_id != cid:
                    continue
                if _box_iou(ent.last_box, box) >= self.min_overlap:
                    score = -_box_iou(ent.last_box, box)
                else:
                    disp = _displacement_ratio(ent.last_box, box)
                    if disp > self.max_displacement:
                        continue
                    score = disp
                if score < best_score:
                    best_id, best_score = ent_id, score
            return best_id

        entity_ids = []
        for tid, cid, box in frame_dets:
            if box is None:
                entity_ids.append(None)
                continue
            ent_id = _match_by_tracker(tid, cid)
            if ent_id is None:
                ent_id = _match_geometric(cid, box)
            if ent_id is None:
                ent_id = self._next_id
                self._next_id += 1
                self._entities[ent_id] = _Entity(
                    entity_id=ent_id, class_id=cid, last_tracker_id=tid, last_box=box
                )
                claimed_entities.add(ent_id)
            else:
                self._entities[ent_id].last_tracker_id = tid
                self._entities[ent_id].last_box = box
                self._entities[ent_id].age_frames = 0
                claimed_entities.add(ent_id)
            entity_ids.append(ent_id)
        return entity_ids

    def _expire(self):
        stale = [
            eid for eid, ent in self._entities.items()
            if ent.age_frames >= self.max_age_frames
        ]
        for eid in stale:
            self._entities.pop(eid, None)
        for ent in self._entities.values():
            ent.age_frames += 1

    def entity_count(self):
        return len(self._entities)