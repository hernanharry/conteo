# AGENTS.md

## Qué hace este repositorio

Aplicación web autocontenida en Docker para **detección, seguimiento y conteo de objetos** en streams RTSP de cámaras Hikvision (u otras compatibles con RTSP).

Usa **YOLOv8** (Ultralytics) para detectar objetos y **ByteTrack** (vía `supervision`) para asignarles un ID de tracking persistente. Cuenta cuántos objetos cruzan una **línea virtual** configurada por cámara (entradas y salidas), y expone todo desde un navegador:

- **Cámaras**: alta, baja y configuración de cámaras (nombre, URL RTSP, clases COCO, umbral de confianza, línea de conteo) desde la UI, sin editar archivos.
- **Vivo** (`/live/<nombre>`): stream MJPEG anotado en tiempo real (cajas, IDs, línea de conteo).
- **Galería**: recortes de cada objeto detectado, agrupados por clase y cámara, con filtros y exportación CSV/ZIP.
- **Notificaciones opcionales**: reporte horario vía webhook de n8n y/o mensaje (con foto) a Telegram.

Los datos persisten en `./data/` (SQLite + imágenes de galería) montado como volumen Docker.

## Arquitectura

```
RTSP (cámara)
    │
    ▼
_FrameGrabber (hilo por cámara) ──► último frame en memoria
    │
    ├──► _render_loop (hilo) ──► anota frame + detecciones ──► JPEG ──► /stream/<nombre> (MJPEG)
    │
    └──► CameraWorker.run (hilo principal) ──► YOLO + ByteTrack + LineZone
              │
              ├──► contadores in/out (persistidos en SQLite)
              ├──► recortes a galería (al perder un track)
              └──► reporte horario (n8n / Telegram)

Flask (server.py) ──► UI web + API REST mínima
CameraManager ──► arranca/detiene/reinicia workers + watchdog
SQLite (db.py) ──► cámaras registradas + índice de detecciones
```

**Puntos clave del diseño:**

- Un **único modelo YOLO** compartido en memoria entre todas las cámaras (`get_model()` en `camera_worker.py`).
- Cada cámara corre en su propio `CameraWorker` con tres hilos: grabber RTSP, render del vivo, y detección YOLO (esta última puede ir más lenta sin congelar el video).
- Un **watchdog** (`camera_manager.py`) reinicia workers colgados conservando el conteo acumulado.
- La detección corre cada `FRAME_SKIP` frames; el vivo se renderiza a su propio ritmo (`LIVE_STREAM_FPS`).

## Estructura de Carpetas

```
conteo v6/
├── app/
│   ├── server.py           # Flask: rutas de cámaras, vivo, galería, exportaciones
│   ├── camera_manager.py   # Ciclo de vida de workers + watchdog
│   ├── camera_worker.py    # Loop RTSP → YOLO → tracking → conteo → galería
│   ├── db.py               # SQLite: tablas cameras y detections
│   ├── templates/          # index.html, live.html, gallery.html, base.html
│   └── static/style.css
├── data/                   # (generado en runtime) app.db + gallery/<clase>/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
└── AGENTS.md
```

## Comandos para Ejecutar

```bash
# Configuración inicial
cp .env.example .env
# Editar .env con modelo YOLO, webhook n8n, Telegram, etc.

# Levantar con Docker (forma recomendada)
docker compose up -d --build
docker compose logs -f object-tracker

# Acceder a la UI
# http://<IP-SERVIDOR>:8001

# Desarrollo local (sin Docker)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd app && python server.py
```

La red externa `vision-net` en `docker-compose.yml` debe apuntar a la red Docker donde corre n8n (por defecto `n8n_default`).

## Dependencias

| Paquete | Propósito |
|---------|-----------|
| `ultralytics` | Modelo YOLOv8 para detección de objetos |
| `supervision` | ByteTrack, LineZone, anotadores visuales |
| `opencv-python-headless` | Captura RTSP, procesamiento de frames, JPEG |
| `torch` / `torchvision` | Backend de inferencia (instalado en Dockerfile) |
| `flask` | Servidor web y templates Jinja2 |
| `requests` | Webhook n8n y API de Telegram |

Variables de entorno relevantes: ver `.env.example` (`MODEL_PATH`, `FRAME_SKIP`, `IMGSZ`, `TORCH_THREADS`, `GRABBER_MAX_FPS`, `N8N_WEBHOOK_URL`, `TELEGRAM_*`, `WATCHDOG_*`, `LIVE_*`, `DB_PATH`, `GALLERY_DIR`, `GALLERY_MAX_FILES`, `GALLERY_MAX_AGE_DAYS`, `GALLERY_CLEANUP_INTERVAL_MIN`). Además (F5/F6): `WEB_USER`/`WEB_PASSWORD`/`WEB_PUBLIC_PATHS` (auth opcional), `STREAM_POLL_INTERVAL` (polling del MJPEG).

## Fases implementadas

- **F0** baseline: tests, smoke RTSP y `docs/baseline.md` (sin tocar `app/`).
- **F1** robustez: lifecycle de workers, DI inyectable, watchdog con anti-lockup.
- **F2** concurrencia: inferencia YOLO serializada sobre el modelo compartido + contadores `/api/perf`.
- **F3** notificaciones: n8n/Telegram en worker desacoplado (cola acotada), HourClock de respaldo.
- **F4** SQLite/galería/eventos: F4.1 BUG-DB-001, F4.2 integridad borrado, F4.3 índices, F4.4 benchmark SQLite, F4.5 retención de galería (worker daemon), F4.6 export ZIP por archivo temporal, F4.7 docs.
- **F5** seguridad: HTTP Basic Auth opcional (`web.properties`: `WEB_USER`/`WEB_PASSWORD`). `auth.py` solo stdlib; comparación en tiempo constante. `/api/health` queda público siempre (orquestador). Ver `tests/test_auth_f5.py`.
- **F6** streaming: MJPEG con headers anti-cache + `X-Accel-Buffering: no`, `STREAM_POLL_INTERVAL`, cierre limpio por GeneratorExit. Ver `tests/test_stream_f6.py`.
- **F7** observabilidad: `/api/health` (uptime, python, threads, estado por cámara, 503 si la BD cae). Ver `tests/test_health_f7.py`.
- **F8** Docker/producción: torch/torchvision pinneados (GAP-PROD-02), `HEALTHCHECK` contra `/api/health`, `.dockerignore` (no copia `.env`/`data/`), `stop_grace_period: 30s` para el shutdown limpio, red compose standalone (`external: false` con `name: n8n_default`).
- **F9** tuning de detección (hardware débil): benchmark real en docs/detection-tuning.md. Recomendado para CPUs de 2 núcleos: `IMGSZ=960` + `conf_threshold=0.25` + `TORCH_THREADS=2` + `GRABBER_MAX_FPS=8` (nuevo tope de decodificación) + `LIVE_STREAM_FPS`/`LIVE_MAX_WIDTH` bajos. OpenVINO/ONNX se midieron MÁS lentos que torch nativo en esta máquina (NO agregar al contenedor sin re-benchmark).

## Convenciones Importantes

- **Nomenclatura:** `snake_case` para funciones, variables y archivos Python; nombres de cámara en kebab-case o descriptivos (`entrada-principal`).
- **Idioma del código:** comentarios y mensajes de log en español; nombres de código en inglés.
- **Configuración por cámara** (RTSP, clases, línea, confianza) va en la base de datos, no en `.env`. El `.env` solo tiene parámetros globales.
- **Clases COCO por defecto:** `0,2,3,5,7` = persona, auto, moto, bus, camión.
- **Persistencia:** `./data` en el host montado a `/app/data` en el contenedor; no borrar sin backup.
- **Flujo de trabajo:** cambios en la línea de conteo desde `/live/<nombre>` disparan `restart_camera` para aplicar sin perder contadores.

## Detalles Adicionales

- Al agregar una cámara desde la UI, `camera_manager.start_camera()` arranca el worker automáticamente.
- Los recortes de galería se guardan cuando un `tracker_id` desaparece del frame (objeto salió de escena).
- El reporte horario resetea los contadores de la hora pero acumula en `cumulative_in` / `cumulative_out` en SQLite.
- Para mejor rendimiento: usar subflujo RTSP (`Channels/102`), subir `FRAME_SKIP`, bajar `IMGSZ` o usar `yolov8n.pt`. Para más precisión: `yolov8m.pt` (requiere más CPU o GPU con `nvidia-container-toolkit`).
- La exportación ZIP de galería tiene tope de 500 imágenes por petición; CSV hasta 100.000 registros.
- **F4 (persistencia/gallery hardening):**
  - `list_detections` con `date_to='YYYY-MM-DD'` incluye el día completo hasta `23:59:59.999999` (BUG-DB-001).
  - `delete_camera` elimina también las detecciones y los `.jpg` de esa cámara (integridad referencial).
  - `detections` tiene índices mínimos (`camera_name,id`, `class_name,id`, `timestamp`).
  - La retención de galería (`GALLERY_MAX_FILES` / `GALLERY_MAX_AGE_DAYS`, 0 = inactiva) corre en `GalleryRetentionWorker` (daemon en background), nunca en el hilo de detección. `_save_gallery_crop` solo hace `ensure_gallery_retention_worker()` (idempotente/no bloqueante).
  - `export_zip` genera el ZIP en un archivo temporal en disco y lo sirve por generador (sin pico de RAM), eliminando el temporal al terminar.
  - **Concurrencia SQLite:** `db.py` conserva el lock global `_lock` solo en escrituras + una conexión por operación. F4.4 midió (ver `scripts/benchmark_sqlite.py`) que el cuello real es la fsync/commit por `add_detection` en el hilo de detección (fuera del alcance F4), NO el lock. **No** se introducen WAL/batching/connection-pooling sin un benchmark que lo justifique.
- **F5 (seguridad):** HTTP Basic Auth opcional con Flask-Login + sessions. `auth.py` solo stdlib; comparación en tiempo constante. `/api/health` y `/metrics` quedan públicos siempre. `SECRET_KEY` requerido en `.env` para sesiones. CSRF (Flask-WTF) protege todos los POST; AJAX usa `X-CSRFToken` header del meta tag. Rate limit configurable en login.
- **F6 (streaming):** MJPEG con headers anti-cache + `X-Accel-Buffering: no`, `STREAM_POLL_INTERVAL`, cierre limpio por GeneratorExit. HLS segmentado (ffmpeg, `.m3u8` + `.ts`) como alternativa con hls.js. Ver `tests/test_stream_f6.py`.
- **F7 (observabilidad):** `/api/health` (uptime, python, threads, estado por cámara, 503 si la BD cae). `/metrics` endpoint Prometheus (uptime, camaras, detecciones, errores RTSP, disco). AlertMonitor daemon envía alertas Telegram proactivas (cámara caída). Docker-compose.observability.yml con Prometheus + Grafana (profile opt-in). Ver `tests/test_observability_f7.py`.
- **F8 (Docker/producción):** torch/torchvision pinneados (GAP-PROD-02), `HEALTHCHECK` contra `/api/health`, `.dockerignore` (no copia `.env`/`data/`), `stop_grace_period: 30s` para el shutdown limpio, usuario no-root `appuser`, limits de recursos (6G RAM / 3.5 CPU), labels OCI para build reproducible, Traefik reverse proxy opcional (profile `with-traefik`), backup automatizado de `./data` con rotación.
