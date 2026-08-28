"""Tests de parseo/validacion de configuracion de camaras (F1).

camera_config.py es stdlib-puro: estos tests corren SIEMPRE, sin stack pesado.
"""

import pytest

from camera_config import CameraConfig, parse_camera_config


def _camera(**over):
    base = {
        "name": "entrada-principal",
        "rtsp_url": "rtsp://usuario:password@192.168.1.64:554/Streaming/Channels/101",
        "classes": "0,2,3,5,7",
        "conf_threshold": "0.35",
        "line_start": "0,300",
        "line_end": "1280,300",
    }
    base.update(over)
    return base


def test_parse_valida_para_camara_completa(camera_dict):
    cfg = parse_camera_config(camera_dict)
    assert isinstance(cfg, CameraConfig)
    assert cfg.name == "entrada-principal"
    assert cfg.rtsp_url == camera_dict["rtsp_url"]
    assert cfg.classes == [0, 2, 3, 5, 7]
    assert cfg.conf_threshold == 0.35
    assert cfg.line_start == (0, 300)
    assert cfg.line_end == (1280, 300)


def test_parse_detalle_camara(camera_dict):
    parse_camera_config(dict(camera_dict, line_start="1,2", line_end="3,4"))


def test_parse_con_spacios_y_comas_sobrantes(camera_dict):
    cfg = parse_camera_config(dict(camera_dict, classes="0, 2, 3, ,, 5"))
    assert cfg.classes == [0, 2, 3, 5]


def test_parse_acepta_lista_de_enteros_desde_la_ui(camera_dict):
    # La UI puede mandar la lista ya convertida (datos que llegaron como JSON)
    cfg = parse_camera_config(dict(camera_dict, classes=[0, 2]))
    assert cfg.classes == [0, 2]


def test_parse_nombre_vacio_es_invalido(camera_dict):
    with pytest.raises(ValueError, match="nombre"):
        parse_camera_config(dict(camera_dict, name="  "))


def test_parse_rtsp_vacio_es_invalido(camera_dict):
    with pytest.raises(ValueError, match="rtsp_url"):
        parse_camera_config(dict(camera_dict, rtsp_url=""))


def test_parse_classes_no_enteras_es_invalido(camera_dict):
    with pytest.raises(ValueError, match="classes"):
        parse_camera_config(dict(camera_dict, classes="0,persona,5"))


def test_parse_conf_no_numerica_es_invalido(camera_dict):
    with pytest.raises(ValueError, match="conf_threshold"):
        parse_camera_config(dict(camera_dict, conf_threshold="alto"))


def test_parse_punto_de_una_sola_coordenada_es_invalido(camera_dict):
    with pytest.raises(ValueError, match="line_start"):
        parse_camera_config(dict(camera_dict, line_start="300"))


def test_parse_punto_de_tres_coordenadas_es_invalido(camera_dict):
    with pytest.raises(ValueError, match="line_end"):
        parse_camera_config(dict(camera_dict, line_end="1,2,3"))


def test_parse_punto_no_numerico_es_invalido(camera_dict):
    with pytest.raises(ValueError, match="line_start"):
        parse_camera_config(dict(camera_dict, line_start="a,b"))