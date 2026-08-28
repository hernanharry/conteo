# Baseline de Fase 0 — estado inicial medido

Fecha de medición: 2026-08-28 · Autor: Fase 0 (plan de mejora YOLO+ByteTrack)

> Regla F0: **no se inventan números**. Todo lo que no pudo medirse en la
> máquina de desarrollo figura como `PENDIENTE` con el procedimiento exacto
> para medirlo en el entorno objetivo (máquina de despliegue con Docker,
> stack completo y cámaras o stream RTSP sintético).

## 1. Hardware utilizado (medido)

| Magnitud | Valor |
|---|---|
| CPU | Intel(R) Core(TM) i3-4330 CPU @ 3.50 GHz |
| Núcleos físicos | 2 |
| Hilos lógicos | 4 |
| RAM total | 12 687 511 552 bytes (~12.1 GB) |
| SO | Microsoft Windows 11 Pro (build 10.0.26100), 64 bits |
| Shell | Windows PowerShell 5.1 |

> Importante: es la máquina de DESARROLLO. El baseline de rendimiento de la
> aplicación debe medirse en el hardware de despliegue (Linux + Docker).

## 2. Entorno de desarrollo

| Componente | Versión (medida) |
|---|---|
| Python | 3.14.3 (64 bits) |
| pip | (del venv) |
| git | 2.53.0.windows.2 |
| pytest | 8.3.4 (venv `.venv`) |
| pytest-timeout | 2.3.1 (venv `.venv`) |
| Docker | **NO INSTALADO en esta máquina** (`docker` no encontrado) |
| ffmpeg | no verificado en PATH del host (no usado en F0) |

## 3. Versiones de la aplicación (declaradas, NO instaladas localmente)

No se instalaron las dependencias de runtime en el host (regla F0). Los pines
declarados en `requirements.txt` (deben verificarse como *medidos* recién en el
entorno objetivo):

| Paquete | Versión declarada |
|---|---|
| ultralytics | 8.3.40 |
| supervision | 0.24.0 |
| opencv-python-headless | 4.10.0.84 |
| requests | 2.32.3 |
| flask | 3.0.3 |
| torch/torchvision | sin pin en `Dockerfile` (imagen CPU) — GAP-PROD-02 |

## 4. Configuración utilizada (defaults de la app, sin cambios)

- `FRAME_SKIP=2`, `IMGSZ=640`, `TORCH_THREADS=4` (defaults en el código).
- Clases por defecto `0,2,3,5,7`; `conf_threshold=0.35`; línea `0,300→1280,300`.
- `LIVE_JPEG_QUALITY=70`, `LIVE_MAX_WIDTH=960`, `LIVE_STREAM_FPS=12`.
- `WATCHDOG_INTERVAL_SECONDS=15`, `WATCHDOG_STALE_SECONDS=60`.
- `DB_PATH=/app/data/app.db` (dentro del contenedor).
- **Cámaras configuradas: 0.** `./data/` vacío (sin SQLite ni imágenes).

## 5. Snapshot Git

- `v0-baseline-prev` = estado exacto previo a F0 (commit `d0abf81`).
- `HEAD` de F0 = estado con tests/scripts/docs (ningún archivo de `app/` modificado).
- Worktree: `git status` sin cambios funcionales.

## 6. Tests baseline ejecutados (real)

Comando: `& .venv\Scripts\python.exe -m pytest tests -v --timeout=60`

Resultado: **21 passed · 5 skipped · 1 xfailed · 0 errores** (27 tests).

| Grupo | Test | Resultado |
|---|---|---|
| config | parseo válido por defecto | PASS |
| config | espacios/comas vacías en classes | PASS |
| config | classes vacías → lista vacía | PASS |
| config | classes inválidas → ValueError (GAP-F1: mata run()) | PASS |
| config | conf inválida → ValueError | PASS |
| config | línea inválida "abc,def" → ValueError | PASS |
| config | línea escalar (300) parsea sin error = GAP latente (1-tupla→LineZone) | PASS |
| configuración del worker real (importa cv2/supervision/torch) | — | SKIP (`docker`) |
| camera_manager: start_all con DB vacía | — | SKIP (`docker`) |
| camera_manager: conteos acumulados iniciales | — | SKIP (`docker`) |
| camera_manager: restart conserva acumulado | — | SKIP (`docker`) |
| camera_manager: stop elimina del dict | — | SKIP (`docker`) |
| db: init crea tablas+migración | PASS |
| db: init idempotente | PASS |
| db: alta/lectura cámara | PASS |
| db: cámara inexistente → None | PASS |
| db: cámara duplicada → IntegrityError (GAP-F5) | PASS |
| db: listado ordenado | PASS |
| db: borrado cámara | PASS |
| db: update línea | PASS |
| db: persistencia cumulative_in/out | PASS |
| db: alta/listado detecciones | PASS |
| db: filtros clase/cámara | PASS |
| db: fecha_hasta incluye día completo | **XFAIL (bug real, F4)** |
| db: límite y orden | PASS |
| db: distinct classes | PASS |
| db: borrado cámara deja huérfanos (GAP integridad) | PASS |

### Hallazgo real descubierto por los tests (nuevo para el plan)

**BUG-DB-001 (confirmado por F0):** `db.list_detections` filtra el extremo
`date_to` con `timestamp <= f"{date_to}T23:59:59"` (`app/db.py:122`). Como los
timestamps se guardan con `datetime.isoformat()` (microsegundos), la
comparación lexicográfica **excluye todas las detecciones de
[23:59:59.000001, 23:59:59.999999]** del día elegido. El test
`test_list_detections_fecha_hasta_incluye_dia_completo` queda en `xfail`
(comportamiento deseado, se corrige en F4).

## 7. Smoke test

Mecanismo preparado (reproducible), **no ejecutado en esta máquina** porque
requiere Docker (fuente RTSP sintética mediaMTX) o cámara real + stack completo
(no disponibles aquí).

- Scripts: `scripts/rtsp-test-source/README.md`,
  `scripts/rtsp-test-source/compose-test-rtsp.yml`,
  `scripts/rtsp-test-source/ffmpeg_publish.ps1`, `scripts/smoke_check.ps1`.
- Procedimiento: levantar mediaMTX → publicar patrón sintético con objeto en
  movimiento → agregar cámara en la UI → `smoke_check.ps1`.
- Checklist verificado cuando se ejecute: 1) app arranca, 2) SQLite inicializa,
  3) CameraManager arranca, 4) cámara RTSP conecta, 5) stream MJPEG fluye,
  6) detección, 7) tracking, 8) LineZone, 9) conteo, 10) galería.

## 8. Baseline de rendimiento

**PENDIENTE — requiere entorno objetivo (Docker/cámara/sintética).**
Procedimiento para medir el estado ACTUAL (sin modificar parámetros):

1. Levantar la app con `docker compose up -d --build`.
2. Publicar stream sintético (script de §7) o usar cámara real.
3. Configurar `N` cámaras (1, 2, 4 y hasta 8 si el hardware lo permite).
4. Registrar **por nivel** con `docker stats` + logs:

| Métrica | Método de medición |
|---|---|
| FPS de stream (vivo) | frecuencia de actualización del MJPEG (intervalo entre JPEGs del cliente o log) |
| FPS de detección | conteo de frames procesados / tiempo (hoy sin log: medible por el flujo del vivo) |
| Inferencia p50/p95 (ms) | `time.perf_counter()` alrededor de `model(...)` — requiere instrumentar (F7) o perf externo |
| CPU % | `docker stats --no-stream --format "{{.CPUPerc}}"` |
| RAM | `docker stats` (MEM USAGE/LIMIT) + RSS del proceso |
| Threads | `threading.enumerate()` — requiere instrumentar (F7) |
| Cámaras simultáneas | número de workers `CameraWorker` corriendo con estado `en vivo` |
| Resolución de entrada | del stream (sub/principal, ej. 1280x720 / 1920x1080) |
| IMGSZ / FRAME_SKIP / TORCH_THREADS | fijos hoy (640/2/4) salvo justificación documentada |
| Latencia aproximada RTSP | edad del frame (hora de captura vs. hora de `read()`) |
| Tiempo de reconexión | cortar red (`docker network disconnect`) y cronometrar hasta primer frame nuevo |
| Detecciones | contador de filas `detections` incrementales |
| Cruces conocidos | ground truth del patrón sintético vs. contadores in/out de la app |

Regla: **no cambiar FRAME_SKIP/IMGSZ/TORCH_THREADS entre niveles** salvo notas
explícitas. Conteo de "pérdida aparente de frames": comparar FPS de entrada del
patcher de ffmpeg vs FPS de detección.

## 9. Soak baseline (≥ 2 h, ideal)

**PENDIENTE — requiere entorno objetivo.** Plantilla de registro:

| Métrica | Inicio | Fin | Observación |
|---|---|---|---|
| RAM (RSS) | — | — | — |
| Threads vivos (`threading.enumerate()`) | — | — | — |
| Errores en logs | — | — | — |
| Reconexiones | — | — | — |
| SQLite (bytes) | — | — | — |
| Gallery (bytes) | — | — | — |
| FPS de detección | — | — | — |

(El soak de 24-48 h queda para fases posteriores.)

## 10. Problemas encontrados en F0 (reales, no inventados)

1. **BUG-DB-001**: filtro `date_to` pierde el último segundo del día (confirmado
   por test, `xfail`; corrección en F4).
2. **GAP parseo config inline**: las validaciones de `classes/conf/línea` viven
   dentro de `CameraWorker.run()` sin función independiente → no testeable sin
   refactor (prohibido en F0). Documentado para F1.
3. **GAP sin DI en CameraManager/CameraWorker**: `CameraWorker` se instancia
   directamente; los tests de manager necesitan stack completo y quedan
   `SKIP` (marcados `docker`). Se ejecutarán en el entorno objetivo con
   `--run-docker`.
4. **GAP integridad**: borrar cámara deja detecciones huérfanas (test PASS
   documentando el comportamiento actual; F4/F5).
5. **GAP de secretos y credenciales**: `.env.example` documenta tokens pero no
   hay `.gitignore` previo (resuelto en F0: `.gitignore` creado; `.env`, `data/`
   y artefactos quedarán fuera del repo).
6. **GAP-ProD-02**: `torch`/`torchvision` sin pin en `Dockerfile`.

## 11. Limitaciones de la prueba

- Sin Docker ni stack completo en la máquina de desarrollo: la capa que
  importa cv2/supervision/torch/ultralytics (worker, manager, servidor) no pudo
  ejecutarse localmente; quedó documentada y encadenada a `--run-docker`.
- Sin cámaras reales ni fuente RTSP: no hubo medición de fps/CPU/RAM/latencia/
  reconexión (procedimiento definido en §8).
- No se tocó ningún archivo de `app/` ni de Docker: `git diff --stat` debe
  mostrar solo adiciones de F0.