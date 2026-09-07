"""F10: estabilizador de IDs ante ID switches de ByteTrack (autos rapidos).

Estos tests usan SOLO la biblioteca estandar (el modulo id_stabilizer no
depende de numpy/supervision), asi que corren en el venv local sin el stack
pesado. Reproducen el sintoma documentado: LineZone no cuenta si ByteTrack
re-ida al objeto entre frames (ver test_line_zone_f3.py::test_line_zone_no
_cuenta_si_tracker_id_cambia).
"""

from id_stabilizer import StableIDAssigner

# La linea de conteo en los tests del repo esta en y=300. ABAJO/ARRIBA quedan
# COMPLETOS de su lado (LineZone exige eso para contar el cruce) y con un
# desplazamiento de centro <= 2.0x el alto del bbox (vease id_stabilizer.py).
ABAJO = (500, 320, 700, 520)
ARRIBA = (500, 80, 700, 280)
LINEA_Y = 300


def _det(tid, class_id, box):
    return (tid, class_id, box)


def test_mismo_tracker_id_mismo_entity():
    a = StableIDAssigner()
    e1 = a.update([_det(5, 2, ABAJO)])
    e2 = a.update([_det(5, 2, ARRIBA)])
    assert e1[0] == e2[0]  # ByteTrack mantuvo el id: entidad continua


def test_id_switch_mismo_objeto_mismo_entity():
    """El auto rapido: ByteTrack re-ida en cada frame pero la geometria lo
    asocia -> la entidad NO cambia (LineZone si puede contar el cruce)."""
    a = StableIDAssigner()
    e1 = a.update([_det(5, 2, ABAJO)])
    e2 = a.update([_det(7, 2, ARRIBA)])  # mismo objeto, otro tracker_id
    assert e1[0] == e2[0]


def test_id_switch_continua_varias_veces():
    a = StableIDAssigner()
    ids = a.update([_det(5, 2, ABAJO)])
    ids = a.update([_det(6, 2, (560, 200, 660, 320))])
    ids = a.update([_det(9, 2, ARRIBA)])
    assert len(set(ids)) == 1


def test_no_fusiona_objetos_distintos_cercanos():
    """Dos autos de la misma clase que se mueven juntos NO se fusionan."""
    a = StableIDAssigner()
    box1, box2 = (100, 300, 200, 400), (900, 300, 1000, 400)
    e1 = a.update([_det(1, 2, box1), _det(2, 2, box2)])
    # avanzan un poco manteniendo separacion
    e2 = a.update([_det(1, 2, (110, 305, 210, 405)), _det(2, 2, (910, 305, 1010, 405))])
    assert e1[0] != e1[1]
    assert e2[0] == e1[0] and e2[1] == e1[1]


def test_no_fusiona_con_cambio_de_clase():
    a = StableIDAssigner()
    e1 = a.update([_det(5, 2, ABAJO)])
    # mismo lugar geometrico pero clase distinta (nunca es el mismo objeto)
    e2 = a.update([_det(6, 0, (510, 405, 610, 505))])
    assert e1[0] != e2[0]


def test_expira_y_reap_entity_nuevo():
    a = StableIDAssigner(max_age_frames=3)
    e1 = a.update([_det(5, 2, ABAJO)])
    # el objeto desaparece mas de max_age_frames frames procesados
    for _ in range(4):
        a.update([])
    e2 = a.update([_det(5, 2, ABAJO)])
    assert e1[0] != e2[0]


def test_nueva_entidad_es_unica_en_el_mismo_frame():
    """Dos detecciones simultaneas no pueden reclamar la misma entidad."""
    a = StableIDAssigner()
    ids = a.update([_det(1, 2, (100, 300, 200, 400)), _det(2, 2, (700, 300, 800, 400))])
    assert len(set(ids)) == 2


def test_bbox_invalido_devuelve_none():
    a = StableIDAssigner()
    ids = a.update([(1, 2, None), (2, 2, ABAJO)])
    assert ids[0] is None
    assert ids[1] is not None


# --- Mini replica de la semantica de LineZone (solo stdlib) ---
# LineZone cuenta una transicion de lado solo si el MISMO id la ve de un lado
# y luego del otro. Con los ids estabilizados, el ID switch del auto rapido ya
# no rompe ese requisito.


def _centro(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _lado(box, linea_y=LINEA_Y):
    """Firma del lado respecto de una linea horizontal y=LINEA_Y."""
    _, cy = _centro(box)
    return "abajo" if cy > linea_y else "arriba"


def _contar_cruces_con_ids_estables(assigner, frames):
    """Corre el assigner frame a frame y cuenta transiciones de lado usando
    SOLO la continuidad que el assigner mantiene (mimica a LineZone con un
    estado por id estable). Devuelve el numero de cruces."""
    estado = {}
    cruces = 0
    for frame in frames:
        entity_ids = assigner.update(frame)
        for box, eid in zip(frame, entity_ids):
            lado = _lado(box[2])
            if lado != estado.get(eid):
                if eid in estado:
                    cruces += 1
                estado[eid] = lado
    return cruces


def test_auto_rapido_con_id_switch_si_es_contado():
    """Con stabilizer, el auto que ByteTrack re-ida entre frames cruza la linea
    y la transicion de lado se cuenta (con raw IDs nunca)."""
    frames = [[_det(5, 2, ABAJO)], [_det(9, 2, ARRIBA)]]
    a = StableIDAssigner()
    assert _contar_cruces_con_ids_estables(a, frames) == 1


def test_auto_rapido_sin_id_switch_contado_igual():
    a = StableIDAssigner()
    frames = [[_det(3, 2, ABAJO)], [_det(3, 2, ARRIBA)]]
    assert _contar_cruces_con_ids_estables(a, frames) == 1