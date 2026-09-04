# Detección y conteo de objetos sobre cámaras Hikvision (YOLO + ByteTrack)

App web autocontenida en Docker: detecta y sigue objetos (personas, autos,
motos, buses, camiones) en streams RTSP de cámaras Hikvision con YOLOv8
(Ultralytics) + ByteTrack (vía `supervision`), cuenta cruces sobre una línea
virtual, y muestra todo desde el navegador:

- **Cámaras**: agregás cámaras (nombre + URL RTSP + parámetros) desde un
  formulario, sin tocar archivos de configuración.
- **Vivo**: vas a `/live/<nombre-camara>` y ves el stream anotado en tiempo
  real (cajas, ID de tracking, línea de conteo).
- **Galería**: pestaña con los recortes de cada objeto detectado, agrupados
  y filtrables por clase (persona, auto, moto...) y por cámara.
- Reporte horario opcional a un webhook de n8n y/o a un chat de Telegram.

## 1. Preparar el servidor Debian

```bash
sudo apt update && sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
```

Cerrá sesión y volvé a entrar (o `newgrp docker`) para que el grupo tome efecto.

### GPU (opcional)

Con CPU, `yolov8n.pt` anda bien para 1-3 cámaras a 5-10 FPS, de sobra para
conteo. Si sumás muchas cámaras o querés más precisión (`yolov8m.pt`),
instalá `nvidia-container-toolkit` y agregá `runtime: nvidia` al servicio
en el `docker-compose.yml`. Ojo: el modelo se comparte entre todas las
cámaras (un solo proceso YOLO en memoria), así que agregar cámaras nuevas
no implica cargar el modelo de nuevo.

## 2. Configuración inicial (global)

```bash
cd hikvision-object-tracker
cp .env.example .env
nano .env
```

Acá van los datos que aplican a todas las cámaras: webhook de n8n, token y
chat de Telegram, qué modelo YOLO usar. Los datos específicos de cada
cámara (URL RTSP, línea de conteo, clases a detectar) se cargan después
desde la web, no acá.

### Autenticación (opcional)

Si además definís `WEB_USER` y `WEB_PASSWORD` en `.env`, la UI, el vivo y las
exportaciones quedan protegidos por **HTTP Basic Auth** (F5). Sin esas
variables la app corre abierta, como siempre. Para el navegador es
transparente (te pide usuario/contraseña la primera vez). Se recomienda
habilitarla en producción:

```bash
WEB_USER=admin
WEB_PASSWORD=cambia-este-password
# Opcional: rutas que quedan públicas aun con auth (separadas por coma)
# WEB_PUBLIC_PATHS=/stream,/gallery/image
```

El health check `/api/health` queda público siempre, para que el `HEALTHCHECK`
de Docker funcione sin credenciales.

Si tu n8n corre en Docker y querés que el webhook le llegue por nombre de
servicio, poné el nombre real de esa red Docker en
`docker-compose.yml` → `networks.vision-net.name` (`docker network ls | grep n8n`
para verlo).

## 3. Levantar el servicio

```bash
docker compose up -d --build
docker compose logs -f object-tracker
```

Abrí `http://IP-DEL-SERVIDOR:8001` en el navegador.

## 4. Agregar una cámara desde la web

En la pestaña **Cámaras** completás:

- **Nombre**: identificador único (ej: `entrada-principal`).
- **URL RTSP**: formato Hikvision estándar:
  `rtsp://usuario:password@IP:554/Streaming/Channels/101` (canal principal)
  o `.../102` (subflujo, más liviano, alcanza para detección).
- **Clases a detectar**: IDs COCO separados por coma. Por defecto
  `0,2,3,5,7` = persona, auto, moto, bus, camión.
- **Línea de conteo**: coordenadas en píxeles del frame. Si no sabés bien
  dónde va, dejá los valores por defecto, guardá la cámara, entrá a su
  vivo, y ajustá después — se ve reflejado apenas la volvés a crear.

Al guardar, la cámara arranca a detectar automáticamente y aparece
disponible en la tabla, con su estado (`en vivo`, `reconectando`, etc.) y
los contadores de entradas/salidas.

## 5. Vivo

Click en el nombre de la cámara desde la tabla, o directo a
`/live/<nombre>`. Es un stream MJPEG (cajas + ID de tracking + línea de
conteo dibujadas en tiempo real) — liviano, sin plugins.

En la misma página se muestran los contadores de entradas/salidas
actualizados en vivo y un botón para **editar la línea de conteo**: hacé
click y arrastrá sobre el video para trazarla, y guardá. La línea se
interpreta en las coordenadas del frame original de la cámara (aunque el
vivo se vea re-escalado), así que lo que ves en pantalla es exactamente la
línea por la que cuenta.

### Diagnóstico cuando "cruza pero no cuenta"

Si un objeto cruza la línea visible pero el contador queda en 0, activá el
log de diagnóstico y repetí la prueba:

```bash
COUNTING_DEBUG=1 docker compose up -d --build
docker compose logs -f object-tracker
```

Cada frame logueado incluye `dets`, `ids` (tracker_ids de ByteTrack),
`lados` (signo del centro del bbox respecto de la línea) y `cruces`.
- Si `ids` cambia de frame en frame → ByteTrack re-identifica al objeto y
  LineZone no puede atribuirle el cruce (el contador exige el **mismo** id
  de un lado y del otro de la línea).
- Si `lados` nunca cambia → el bbox jamás cruzó la línea real: revisá que la
  línea esté donde cruza el objeto.
- Si `cruces` avanza pero la página no → el conteo ocurre (revisá la tabla
  de cámaras / `/api/perf`).

Los tests `tests/test_line_zone_f3.py` (marca `docker`) reproducen estos
casos sobre la primitiva de conteo real.

## 6. Galería

Pestaña **Galería**: cada vez que aparece un objeto nuevo (un ID de
tracking que no se había visto antes), se guarda un recorte de esa
detección clasificado por tipo de objeto. Filtrable por clase y por
cámara, pensado para revisar rápido "qué autos/personas pasaron hoy" sin
tener que mirar el video entero.

Las imágenes quedan en `./data/gallery/<clase>/` en el host (fuera del
contenedor), y el índice en `./data/app.db` (SQLite) — ambos persisten
entre reinicios gracias al volumen `./data:/app/data`.

### Retención automática (opt-in)

Por defecto la galería crece sin límite. Si querés acotar el disco, `.env`
acepta dos opciones (0 = sin retención):

- `GALLERY_MAX_FILES`: deja como mucho N registros/imágenes (borra los más
  antiguos).
- `GALLERY_MAX_AGE_DAYS`: borra los recortes más antiguos que N días.
- `GALLERY_CLEANUP_INTERVAL_MIN`: cada cuántos minutos el worker revisa la
  galería.

La limpieza corre en un **worker daemon en background** (nunca en el hilo de
detección) y borra tanto la fila del índice como el archivo físico.

### Borrar una cámara

Al borrar una cámara desde `index.html` se eliminan también su fila de la
tabla de cámaras, las detecciones asociadas **y** los recortes `.jpg` de esa
cámara en la galería.

### Exportación ZIP

`/gallery/export.zip` mantiene el tope de 500 imágenes por descarga; el ZIP
se genera en un archivo temporal en disco (no en memoria) para no consumir
RAM de golpe, y se elimina tras enviarse.

## 7. Notificaciones (n8n / Telegram)

Con `N8N_WEBHOOK_URL` completo en `.env`, cada cámara manda un POST por
hora:

```json
{
  "camera": "entrada-principal",
  "hour": "2026-08-16 09:00",
  "in_count": 42,
  "out_count": 37
}
```

Con `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` completos, se manda además
un mensaje con la foto anotada al mismo horario. Son independientes entre
sí — podés usar uno, otro, o ambos. Para crear el bot: hablale a
**@BotFather**, `/newbot`, copiá el token; agregalo a tu chat/grupo,
mandale un mensaje, y consultá
`https://api.telegram.org/bot<TOKEN>/getUpdates` para obtener el `chat_id`.

## 8. Ajustar rendimiento / precisión

- `MODEL_PATH` en `.env`: `yolov8n.pt` (rápido) → `yolov8s.pt` → `yolov8m.pt`
  (más preciso, más lento). Aplica a todas las cámaras (modelo compartido).
- `conf_threshold` por cámara: subilo si detecta falsos positivos, bajalo
  si pierde objetos reales.
- Usar el subflujo (`Channels/102`) en vez del canal principal reduce
  mucho el uso de CPU si no necesitás alta resolución para detectar.

## 9. Diagnóstico y salud

- **`/api/perf`**: contadores de rendimiento por cámara (frames procesados,
  frames saltados, tiempo de inferencia p50/p95, contadores in/out). Solo
  lectura, pensado para benchmarks.
- **`/api/health`**: estado de la app para el orquestador (F7). Responde 200
  con JSON (uptime, versión de Python, hilos vivos, estado y contadores por
  cámara) cuando la BD responde; 503 si no. Es público (sin auth) para que
  el `HEALTHCHECK` del contenedor lo use.
- **`/metrics`**: endpoint Prometheus (F7.1) con métricas de uptime, cámaras
  activas, detecciones por clase, errores RTSP, archivos de galería y uso
  de disco. Público para scraping sin auth.
- **`/stream/<nombre>`**: headers anti-cache + `X-Accel-Buffering: no` (F6)
  para que ningún proxy bufferice el MJPEG.

## 10. Observabilidad (F7)

Stack de observabilidad opcional con Prometheus + Grafana:

```bash
# Activar solo si necesitas métricas históricas / dashboards
docker compose -f docker-compose.yml -f docker-compose.observability.yml \
  --profile observability up -d

# Prometheus: http://IP:9090  |  Grafana: http://IP:3000 (admin/admin)
```

La app expone `/metrics` de forma nativa (sin el stack de observabilidad).

### Alertas proactivas (Telegram)

Si configurás `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`, el monitor de
alertas envía notificaciones cuando:
- Una cámara lleva más de 5 minutos caída (configurable con
  `ALERT_CAMERA_DOWN_MINUTES`).
- El intervalo mínimo entre alertas de la misma cámara es de 5 minutos
  (configurable con `ALERT_MIN_ALERT_INTERVAL`).

## 11. Seguridad (F5)

- **HTTP Basic Auth** opcional (`WEB_USER` / `WEB_PASSWORD`). Si no se
  definen, la app corre abierta.
- **CSRF** protection en todos los formularios POST (Flask-WTF).
- **Rate limit** en el login (5/min por IP en producción, configurable con
  `LOGIN_RATE_LIMIT`).
- **Cookies de sesión** HttpOnly + SameSite=Lax.
- **Audit log** de todas las acciones de escritura (quién, qué, cuándo).
- **Usuarios** con roles: `admin` (acceso total) y `viewer` (solo lectura).

## 12. Docker / Producción (F8)

### Healthcheck

El contenedor tiene un `HEALTHCHECK` nativo que consulta `/api/health` cada
30 segundos. Docker Desktop mostra el estado (`healthy`/`unhealthy`).

### Hardening

- Usuario `appuser:appuser` (no-root) dentro del contenedor.
- Límites de recursos: 6GB RAM / 3.5 CPU (ajustar en `docker-compose.yml`).
- `stop_grace_period: 30s` para shutdown limpio (persiste conteos).

### Traefik (opcional)

Si usás Traefik como reverse proxy:

```bash
docker compose --profile with-traefik up -d
# Dashboard Traefik: http://IP:8080
# App vía Traefik: http://IP/tracker
```

Sin Traefik, la app sigue accesible directamente en `http://IP:8001`.

### Backup automatizado

```bash
# Backup manual
./scripts/backup.sh

# Backup con cron (diario a las 3am)
0 3 * * * cd /ruta/al/proyecto && ./scripts/backup.sh >> ./data/backups/backup.log 2>&1
```

Los backups se guardan en `./data/backups/` (máximo 7 por defecto, configurable con `MAX_BACKUPS`).

### Build reproducible

```bash
# Con tag de versión
BUILD_VERSION=1.0.0 docker compose build

# Verificar labels
docker inspect object-tracker | grep -A5 "org.opencontainers"
```

## Estructura del proyecto

```
conteo-v6/
├── app/
│   ├── server.py           # rutas Flask: cámaras, vivo, galería, health, metrics
│   ├── auth.py             # HTTP Basic Auth + Flask-Login (F5)
│   ├── metrics.py          # Prometheus exposition (F7.1)
│   ├── alerts.py           # Monitor de alertas proactivas Telegram (F7.3)
│   ├── hls_segmenter.py    # HLS streaming (F6)
│   ├── camera_manager.py   # arranca/detiene el worker de cada cámara
│   ├── camera_worker.py    # loop de detección+tracking+galería por cámara
│   ├── db.py                # SQLite: cámaras, detecciones, users, audit_log
│   ├── notifications.py    # cola desacoplada de notificaciones (F3)
│   ├── gallery_retention.py # limpieza automática de galería (F4.5)
│   ├── templates/           # index, live, gallery, login, error, base
│   └── static/style.css
├── scripts/
│   ├── backup.sh            # backup automatizado de ./data (F8.4)
│   ├── create_admin.py      # bootstrap de usuario admin (F5.8)
│   └── benchmark_sqlite.py  # benchmark SQLite (F4.4)
├── tests/                    # suite completa (>140 tests)
├── Dockerfile               # torch pinneado + HEALTHCHECK + no-root (F8)
├── docker-compose.yml       # servicio principal + Traefik opcional
├── docker-compose.observability.yml  # Prometheus + Grafana (F7.5)
├── prometheus.yml           # config de scraping (F7.5)
├── requirements.txt
├── .env.example
└── README.md
```
