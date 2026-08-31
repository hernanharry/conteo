"""F3: primitiva de conteo de supervision (LineZone) tal como la usa la app.

Estos tests REQUIEREN el stack pesado (numpy + supervision + cv2), por eso
llevan la marca `docker`: corren en el entorno objetivo con --run-docker (o en
un venv con requirements completos) y validan el comportamiento EXACTO de la
version instalada de supervision. Documentan la causa mas tipica de "cruce
visible pero 0 conteo": LineZone exige el MISMO tracker_id viendo el bbox de
un lado de la linea y luego del otro; si ByteTrack cambia de id (o la linea
no esta en las coordenadas del frame nativo), el contador no suma.

Regla: nunca tocan YOLO/torch — solo sv.Detections + LineZone.
"""

import pytest


@pytest.mark.docker
def test_line_zone_cuenta_cruce_con_mismo_tracker_id():
    """Lo esperado: un bbox que pasa de abajo a arriba de la linea, con id
    estable, incrementa in_count en exactamente 1."""
    import numpy as np
    import supervision as sv

    zone = sv.LineZone(sv.Point(0, 300), sv.Point(1280, 300))
    zone.trigger(sv.Detections(xyxy=np.array([[500, 400, 600, 500]]), tracker_id=np.array([7])))
    zone.trigger(sv.Detections(xyxy=np.array([[500, 100, 600, 200]]), tracker_id=np.array([7])))

    assert zone.in_count == 1
    assert zone.out_count == 0


@pytest.mark.docker
def test_line_zone_cuenta_el_mismo_tracker_una_sola_vez():
    """Oscilar alrededor de la linea con el mismo id no infla el contador:
    cuenta una vez por TRANSICION de lado, no por frame.

    Semantica real de supervision 0.24.0 (supervision/detection/line_zone.py,
    metodo trigger): LineZone cuenta cambios de lado del tracker, no cruces por
    ciclo.

    - La PRIMERA deteccion de un tracker solo inicializa tracker_state (NO suma).
    - Cada transicion posterior de lado suma in O out (una sola direccion por
      cambio de lado; jamas ambos en un mismo ciclo).
    - Por eso, con el primer estado sembrado en "abajo", N ciclos abajo->arriba
      producen in=N y out=N-1: la primera deteccion "abajo" del ciclo 1 consume
      el contador de inicializacion en vez de sumar un "out".

    Este test valida exactamente eso (in=3, out=2), no una simetria in==out."""
    import numpy as np
    import supervision as sv

    zone = sv.LineZone(sv.Point(0, 300), sv.Point(1280, 300))
    abajo = sv.Detections(xyxy=np.array([[500, 400, 600, 500]]), tracker_id=np.array([7]))
    arriba = sv.Detections(xyxy=np.array([[500, 100, 600, 200]]), tracker_id=np.array([7]))

    for _ in range(3):
        zone.trigger(abajo)
        zone.trigger(arriba)

    assert zone.in_count == 3
    assert zone.out_count == 2


@pytest.mark.docker
def test_line_zone_no_cuenta_si_tracker_id_cambia():
    """ByteTrack que re-ida al objeto entre frames: LineZone inicializa el
    estado nuevo y jamas ve la transicion -> 0 (sintoma "cruce pero no cuenta")."""
    import numpy as np
    import supervision as sv

    zone = sv.LineZone(sv.Point(0, 300), sv.Point(1280, 300))
    zone.trigger(sv.Detections(xyxy=np.array([[500, 400, 600, 500]]), tracker_id=np.array([1])))
    # el frame siguiente "el mismo objeto" aparece con otro id
    zone.trigger(sv.Detections(xyxy=np.array([[500, 100, 600, 200]]), tracker_id=np.array([2])))

    assert zone.in_count == 0


@pytest.mark.docker
def test_line_zone_no_cuenta_detecciones_sin_tracker_id():
    """Sin tracker_id, LineZone no puede atribuir el cruce: avisa y no suma."""
    import numpy as np
    import supervision as sv

    zone = sv.LineZone(sv.Point(0, 300), sv.Point(1280, 300))
    zone.trigger(sv.Detections(xyxy=np.array([[500, 400, 600, 500]])))
    zone.trigger(sv.Detections(xyxy=np.array([[500, 100, 600, 200]])))

    assert zone.in_count == 0
    assert zone.out_count == 0


@pytest.mark.docker
def test_line_zone_no_cuenta_si_la_linea_no_esta_donde_cruza():
    """Un bbox que se mueve en paralelo a la linea (misma y, cruzando en x)
    jamas cruza la linea real: el contador queda en 0 aunque "visualmente"
    pasara por delante. Encaja con una linea movida fuera de escala."""
    import numpy as np
    import supervision as sv

    zone = sv.LineZone(sv.Point(0, 300), sv.Point(1280, 300))
    derecha = sv.Detections(xyxy=np.array([[900, 350, 1000, 450]]), tracker_id=np.array([7]))
    izquierda = sv.Detections(xyxy=np.array([[100, 350, 200, 450]]), tracker_id=np.array([7]))
    zone.trigger(derecha)
    zone.trigger(izquierda)

    assert zone.in_count == 0
    assert zone.out_count == 0