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

## Estructura del proyecto

```
hikvision-object-tracker/
├── app/
│   ├── server.py           # rutas Flask: cámaras, vivo, galería
│   ├── camera_manager.py   # arranca/detiene el worker de cada cámara
│   ├── camera_worker.py    # loop de detección+tracking+galería por cámara
│   ├── db.py                # SQLite: cámaras registradas y detecciones
│   ├── templates/           # index.html, live.html, gallery.html
│   └── static/style.css
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
