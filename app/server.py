import atexit
import csv
import io
import logging
import os
import platform
import tempfile
import threading
import time
import zipfile

from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import current_user
from flask_wtf.csrf import CSRFProtect, CSRFError

import auth
import camera_manager
import notifications
from db import (
    add_camera,
    delete_camera,
    distinct_classes,
    get_camera,
    init_db,
    list_cameras,
    list_detections,
    log_audit,
    stop_gallery_retention_worker,
    update_camera_line,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

app = Flask(__name__)

# F5: secret key para firmar la sesion. SECRET_KEY es OBLIGATORIO ahora
# (F5.2/F5.6); se busca en el entorno. Sin el, la firma de sesiones no es posible.
app.secret_key = os.getenv("SECRET_KEY", "")

# F5.6: cookie de sesion siempre HttpOnly + SameSite=Lax. Secure se controla por
# entorno (false en LAN/HTTP, true tras ngrok/HTTPS).
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = auth.SESSION_COOKIE_SECURE
# CSRF (F5.4): se genera una key propia si no viene del entorno.
app.config["WTF_CSRF_TIME_LIMIT"] = int(os.getenv("WTF_CSRF_TIME_LIMIT", "3600"))

auth.login_manager.init_app(app)
_app_csrf = CSRFProtect(app)

# F5.5: rate limiting del /login (5 intentos/min por IP). Definido a nivel app
# para poder anotar la ruta login; el storage por defecto es en memoria.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# F6: intervalo (s) de polling del generador MJPEG cuando no hay frame nuevo.
# Regula el ancho de banda de "keep-alive" minimo del flujo.
STREAM_POLL_INTERVAL = float(os.getenv("STREAM_POLL_INTERVAL", "0.05"))

# F7: instante de arranque del proceso (para uptime del health check).
_APP_START_TIME = time.time()

_app_log = logging.getLogger("server")


def _get_csrf_token():
    from flask_wtf.csrf import generate_csrf

    return generate_csrf()


def _is_public_path(path: str) -> bool:
    """True si la ruta queda fuera del login (login, health, static, /metrics y
    las listas blancas configuradas). Todo lo demas exige sesion (F5.3)."""
    if not auth.is_public(path):
        return False
    # el health del orquestador se sirve en /health y /api/health (alias).
    return path.startswith(("/static/", "/login", "/health", "/api/health", "/metrics"))


@app.before_request
def _require_login():
    """F5.3: protege TODAS las rutas salvo las publicas. Las acciones de admin
    (alta/baja de camaras, borrado) se validan adicionalmente en cada ruta con
    require_admin()."""
    if _is_public_path(request.path):
        return None
    if not current_user.is_authenticated:
        # si no hay NINGUN usuario creado, redirigir al bootstrap no aplica
        # aqui: create_admin debe ser manual via CLI. Pedimos login.
        return redirect(url_for("login", next=request.full_path if request.full_path != "/" else None))
    return None


def require_admin():
    """F5.3: aborta 403 si el usuario actual no es admin."""
    if not (current_user.is_authenticated and current_user.is_admin()):
        abort(403)


@app.errorhandler(CSRFError)
def _csrf_error(e):
    return render_template(
        "error.html", code=400, message="Token CSRF invalido o expirado. Recargue el formulario."
    ), 400


@app.errorhandler(404)
def _not_found(e):
    return render_template("error.html", code=404, message="No se encontro el recurso."), 404


@app.errorhandler(403)
def _forbidden(e):
    return render_template("error.html", code=403, message="No tiene permisos para esta accion."), 403


# F6: HLS segmenter (si ffmpeg está disponible)
import hls_segmenter

atexit.register(hls_segmenter.stop_all_segmenters)
atexit.register(hls_segmenter.stop_cleanup_worker)

# F7.1: métricas Prometheus
import metrics as prom_metrics

# F7.3: alertas proactivas Telegram
import alerts as alert_monitor


def _start_alerts():
    try:
        import notifications
        alert_monitor.start_alert_monitor(
            camera_manager,
            notif_worker_fn=lambda cam, text: notifications.get_notification_worker().submit_message(cam, text),
        )
    except Exception:
        pass


atexit.register(alert_monitor.stop_alert_monitor)


init_db()
camera_manager.start_all()
_start_alerts()
# Shutdown limpio (F1+F3): al salir el proceso se detienen los workers de
# camara, el watchdog y el worker de notificaciones (cola+reloj horario) con
# join acotado -- sin threads huerfanos ni conteos por escribir.
atexit.register(camera_manager.stop_all)
atexit.register(notifications.stop_all)
# F4.5: detiene el worker de retencion de galeria (daemon) con join acotado.
atexit.register(stop_gallery_retention_worker)


@app.route("/login", methods=["GET", "POST"])
@limiter.limit(os.getenv("LOGIN_RATE_LIMIT", "5 per minute"))
def login():
    """F5.2/F5.5: sesion de cookie con Flask-Login + rate limit de 5/min por IP."""
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        user = auth.authenticate_user(username, password)
        if user is None:
            error = "Usuario o contrasena incorrectos."
        else:
            auth.log_user_in(user)
            _app_log.info("login ok: %s", user.username)
            nxt = request.args.get("next")
            # solo redirigir a rutas internas (evita open redirect)
            if nxt and nxt.startswith("/") and not nxt.startswith("//"):
                return redirect(nxt)
            return redirect(url_for("index"))
    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    username = current_user.username if current_user.is_authenticated else None
    auth.log_user_out()
    if username:
        _app_log.info("logout: %s", username)
    return redirect(url_for("login"))


@app.route("/")
def index():
    cameras = list_cameras()
    statuses = {}
    counts = {}
    for c in cameras:
        worker = camera_manager.get_worker(c["name"])
        statuses[c["name"]] = getattr(worker, "status", "detenida")
        counts[c["name"]] = {
            "in_count": getattr(worker, "in_count", 0),
            "out_count": getattr(worker, "out_count", 0),
        }
    return render_template("index.html", cameras=cameras, statuses=statuses, counts=counts)


@app.route("/cameras/add", methods=["POST"])
def add_camera_route():
    data = {
        "name": request.form["name"].strip(),
        "rtsp_url": request.form["rtsp_url"].strip(),
        "classes": request.form.get("classes", "0,2,3,5,7").strip() or "0,2,3,5,7",
        "conf_threshold": request.form.get("conf_threshold", "0.35").strip() or "0.35",
        "line_start": request.form.get("line_start", "0,300").strip() or "0,300",
        "line_end": request.form.get("line_end", "1280,300").strip() or "1280,300",
    }
    require_admin()
    add_camera(data)
    log_audit(current_user.username, "camera_add", "camera", data["name"])
    cam = get_camera(data["name"])
    camera_manager.start_camera(cam)
    return redirect(url_for("index"))


@app.route("/cameras/<name>/delete", methods=["POST"])
def delete_camera_route(name):
    require_admin()
    camera_manager.stop_camera(name)
    delete_camera(name)
    log_audit(current_user.username, "camera_delete", "camera", name)
    return redirect(url_for("index"))


@app.route("/live/<name>")
def live(name):
    cam = get_camera(name)
    if not cam:
        abort(404)
    # Resolucion nativa del frame: la UI calibra la linea de conteo en las
    # coordenadas que usa LineZone (frame nativo), no en las del JPEG del vivo
    # (que puede venir re-escalado a LIVE_MAX_WIDTH).
    worker = camera_manager.get_worker(name)
    frame_w = getattr(worker, "last_frame_w", 0) or 0
    frame_h = getattr(worker, "last_frame_h", 0) or 0
    return render_template("live.html", camera=cam, frame_w=frame_w, frame_h=frame_h)


@app.route("/cameras/<name>/line", methods=["POST"])
def update_line_route(name):
    cam = get_camera(name)
    if not cam:
        abort(404)
    data = request.get_json(force=True)
    try:
        x1, y1, x2, y2 = (int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"]))
    except (KeyError, ValueError, TypeError):
        return jsonify({"ok": False, "error": "coordenadas invalidas"}), 400

    line_start = f"{x1},{y1}"
    line_end = f"{x2},{y2}"
    update_camera_line(name, line_start, line_end)

    # reiniciar el worker para que tome la nueva linea de conteo
    updated_cam = get_camera(name)
    camera_manager.restart_camera(updated_cam)

    return jsonify({"ok": True, "line_start": line_start, "line_end": line_end})


@app.route("/stream/<name>")
def stream(name):
    """Stream MJPEG en vivo (F6): anotado en tiempo real.

    - Nunca se cachea (MJPEG es por definicion efimero): Cache-Control no-store.
    - X-Accel-Buffering: no -> impide que un proxy (nginx) bufferiza el flujo.
    - El generador courtejos cuando el cliente corta la conexion (GeneratorExit)
      en vez de dejar el hilo dando vueltas para siempre."""
    worker = camera_manager.get_worker(name)
    if not worker:
        abort(404)

    def generate():
        import time

        poll_interval = STREAM_POLL_INTERVAL
        last_sent = None
        try:
            while True:
                frame = worker.get_latest_jpeg()
                # solo reenvia si hay un frame nuevo -- evita gastar ancho
                # de banda repitiendo el mismo jpeg mientras no cambia nada
                if frame is not None and frame is not last_sent:
                    last_sent = frame
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(poll_interval)
        except GeneratorExit:
            # el cliente (navegador) corto la conexion -- salir prolijo en
            # vez de dejar el hilo dando vueltas para siempre
            return

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# F6: HLS streaming — sirve segmentos .m3u8 y .ts generados por hls_segmenter.
# MJPEG (/stream) queda como fallback de compatibilidad (F6.3: se conserva).

@app.route("/hls/<name>/stream.m3u8")
def hls_playlist(name):
    if not hls_segmenter.hls_enabled():
        abort(404)
    playlist = os.path.join(hls_segmenter.get_hls_dir(name), "stream.m3u8")
    if not os.path.isfile(playlist):
        abort(404)
    return send_file(
        playlist,
        mimetype="application/vnd.apple.mpegurl",
        download_name="stream.m3u8",
    )


@app.route("/hls/<name>/<segment>")
def hls_segment(name, segment):
    if not hls_segmenter.hls_enabled():
        abort(404)
    if not segment.endswith(".ts") or "/" in segment or ".." in segment:
        abort(400)
    seg_path = os.path.join(hls_segmenter.get_hls_dir(name), segment)
    if not os.path.isfile(seg_path):
        abort(404)
    return send_file(seg_path, mimetype="video/mp2t")


@app.route("/metrics")
def metrics_endpoint():
    """F7.1: Prometheus exposition format. Publico para scraping."""
    return Response(
        prom_metrics.collect_metrics(_APP_START_TIME),
        mimetype="text/plain; version=0.0.4; charset=utf-8",
    )


@app.route("/api/perf")
def api_perf():
    """Contadores de rendimiento por camara (F2, solo lectura, para benchmarks).

    frames_skipped = frames_available - frames_processed: frames nuevos que el
    bucle de deteccion no alcanzo a procesar (FRAME_SKIP o espera por el lock
    de inferencia). El vivo no se ve afectado: corre en su propio hilo."""
    perf = {}
    for c in list_cameras():
        w = camera_manager.get_worker(c["name"])
        available = getattr(w, "frames_available", 0)
        processed = getattr(w, "frames_processed", 0)
        count = getattr(w, "inference_count", 0)
        total_ms = getattr(w, "inference_total_ms", 0.0)
        perf[c["name"]] = {
            "status": getattr(w, "status", "detenida"),
            "frames_available": available,
            "frames_processed": processed,
            "frames_skipped": available - processed,
            "inference_count": count,
            "inference_last_ms": round(getattr(w, "last_inference_ms", 0.0), 3),
            "inference_total_ms": round(total_ms, 3),
            "inference_avg_ms": round(total_ms / count, 3) if count else 0.0,
            "in": getattr(w, "in_count", 0),
            "out": getattr(w, "out_count", 0),
        }
    return jsonify(perf)


@app.route("/health")
@app.route("/api/health")
def api_health():
    """F7.2/F8.2: health check para el orquestador / Docker / watchdog externo.

    Publico siempre (sin login) para que Docker HEALTHCHECK y el orquestador
    puedan consultarlo. Devuelve 200 cuando la BD responde; 503 con el mismo
    cuerpo cuando no. Es solo lectura y no arranca nada.

    Campos:
      - ok: la BD responde
      - uptime_s: segundos desde el arranque del proceso
      - python: version de Python (diagnostico)
      - threads: hilos vivos del proceso (threading.active_count)
      - users: cantidad de usuarios registrados (F5; si es 0 hay que bootstrap)
      - cameras: estado y contadores por camara
    """
    base = {
        "uptime_s": round(time.time() - _APP_START_TIME, 1),
        "python": platform.python_version(),
        "threads": threading.active_count(),
        "users": auth.user_count(),
    }
    db_status = {"ok": True, "error": None}
    status_code = 200
    try:
        cameras = list_cameras()
    except Exception as exc:  # noqa: BLE001 - el health check nunca puede tirar la app
        db_status = {"ok": False, "error": str(exc)}
        cameras = []
        status_code = 503

    status_data = {}
    for c in cameras:
        w = camera_manager.get_worker(c["name"])
        status_data[c["name"]] = {
            "status": getattr(w, "status", "detenida"),
            "in": getattr(w, "in_count", 0),
            "out": getattr(w, "out_count", 0),
        }

    payload = {**base, "ok": db_status["ok"], "db": db_status, "cameras": status_data}
    resp = jsonify(payload)
    resp.status_code = status_code
    return resp


@app.route("/gallery")
def gallery():
    class_filter = request.args.get("class") or None
    camera_filter = request.args.get("camera") or None
    date_from = request.args.get("date_from") or None
    date_to = request.args.get("date_to") or None
    detections = list_detections(
        class_name=class_filter, camera_name=camera_filter, date_from=date_from, date_to=date_to
    )
    return render_template(
        "gallery.html",
        detections=detections,
        classes=distinct_classes(),
        cameras=list_cameras(),
        selected_class=class_filter,
        selected_camera=camera_filter,
        date_from=date_from,
        date_to=date_to,
    )


@app.route("/gallery/image")
def gallery_image():
    path = request.args.get("path", "")
    gallery_dir = os.path.abspath(os.getenv("GALLERY_DIR", "/app/data/gallery"))
    full_path = os.path.abspath(path)
    if not full_path.startswith(gallery_dir) or not os.path.isfile(full_path):
        abort(404)
    return send_file(full_path)


@app.route("/gallery/export.csv")
def export_csv():
    class_filter = request.args.get("class") or None
    camera_filter = request.args.get("camera") or None
    date_from = request.args.get("date_from") or None
    date_to = request.args.get("date_to") or None
    detections = list_detections(
        class_name=class_filter, camera_name=camera_filter, date_from=date_from, date_to=date_to, limit=100000
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "camara", "clase", "tracker_id", "fecha_hora", "archivo_imagen"])
    for d in detections:
        writer.writerow(
            [d["id"], d["camera_name"], d["class_name"], d["tracker_id"], d["timestamp"], d["image_path"]]
        )

    mem = io.BytesIO(output.getvalue().encode("utf-8-sig"))  # BOM para que Excel abra bien los acentos
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="detecciones.csv",
    )


@app.route("/gallery/export.zip")
def export_zip():
    class_filter = request.args.get("class") or None
    camera_filter = request.args.get("camera") or None
    date_from = request.args.get("date_from") or None
    date_to = request.args.get("date_to") or None
    # tope para no generar un zip gigante en un solo pedido
    detections = list_detections(
        class_name=class_filter, camera_name=camera_filter, date_from=date_from, date_to=date_to, limit=500
    )

    # F4.6: se escribe a un archivo temporal en disco (no a un BytesIO en RAM)
    # para evitar el pico de memoria de volcar hasta 500 imagenes en un solo
    # zip. Mismo tope de 500. La respuesta se sirve por un generador que lee
    # el archivo y lo elimina en `finally` DESPUES de cerrar el handle (asi
    # funciona tambien en Windows, donde un archivo abierto no puede borrarse).
    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for d in detections:
                path = d["image_path"]
                if os.path.isfile(path):
                    arcname = f"{d['class_name']}/{os.path.basename(path)}"
                    zf.write(path, arcname)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    def generate():
        try:
            with open(tmp_path, "rb") as fh:
                while True:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    yield chunk
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return Response(
        generate(),
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=detecciones.zip"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("WEB_PORT", "8001")), threaded=True)
