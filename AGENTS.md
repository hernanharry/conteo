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

Variables de entorno relevantes: ver `.env.example` (`MODEL_PATH`, `FRAME_SKIP`, `IMGSZ`, `N8N_WEBHOOK_URL`, `TELEGRAM_*`, `WATCHDOG_*`, `LIVE_*`, `DB_PATH`, `GALLERY_DIR`).

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
