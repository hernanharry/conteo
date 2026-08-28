# Fuente RTSP sintética para tests (Fase 0)

Mecanismo reproducible para probar la app **sin cámara IP real**:
un servidor RTSP (`mediaMTX`) + `ffmpeg` publicando un patrón de test con
objetos en movimiento. No modifica el `Dockerfile` ni el `docker-compose.yml`
de la aplicación.

## Prerrequisitos (entorno objetivo)

- Docker (para `compose-test-rtsp.yml`) **o** el binario `mediamtx` nativo.
- `ffmpeg` en el PATH (la imagen de la app ya lo incluye; para el host hay que
  instalarlo: `winget install ffmpeg` / `apt install ffmpeg`).
- La aplicación corriendo (contenedor o `python server.py` con el stack
  completo instalado).

## Pasos

### 1) Levantar el servidor RTSP

Con Docker:

```bash
docker compose -f scripts/rtsp-test-source/compose-test-rtsp.yml up -d
# RTSP disponible en rtsp://127.0.0.1:8554/  (puerto default de mediaMTX)
```

Sin Docker (binario nativo): descargar `mediamtx` (GitHub releases), ejecutar
`./mediamtx` (usa el puerto 8554 por defecto).

### 2) Publicar un stream sintético con objetos en movimiento

```powershell
# Windows
powershell -File scripts/rtsp-test-source/ffmpeg_publish.ps1 -Target rtsp://127.0.0.1:8554/feed1
```

```bash
# Linux (equivalente)
ffmpeg -re -stream_loop -1 -f lavfi -i "testsrc=size=1280x720:rate=25" \
  -vf "drawbox=x=100+t*20:y=300:w=60:h=60:color=white@0.9:t=fill" \
  -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/feed1
```

El `drawbox` mueve un objeto a lo largo del eje `x`, de modo que cruce una
línea de conteo vertical/`x` para verificar cruces reales.

### 3) Configurar la cámara en la app

Desde la UI (http://localhost:8001, pestaña Cámaras):

- Nombre: `test-sintetica`
- URL RTSP: `rtsp://127.0.0.1:8554/feed1`
- Clases: por defecto (`0,2,3,5,7`)
- Línea: por defecto

### 4) Verificar con el smoke check

```powershell
powershell -File scripts/smoke_check.ps1 -BaseUrl http://localhost:8001 -Camera test-sintetica
```

El script comprueba (En orden):

1. `/` responde 200 (app arrancada).
2. `/gallery` responde 200 (galería inicializada).
3. `/live/test-sintetica` responde 200 (pagina de vivo).
4. `/stream/test-sintetica` devuelve MJPEG (multipart) — la cámara está conectada.
5. Estado de la cámara en `/` refleja `en vivo` o `reconectando`.

## Restricciones conocidas de F0

- En la máquina de desarrollo actual (Windows, sin Docker, sin stack pesado)
  este mecanismo **no pudo ejecutarse**: queda preparado para el entorno
  objetivo. Resultados reales deben registrarse en `docs/baseline.md` cuando
  se ejecute (NO inventar números).
- El aumento de cruces registrado por la app debe compararse contra los cruces
  reales del patrón sintético (ground truth conocida) cuando el mecanismo corra
  junto con el stack completo.