import atexit
import csv
import io
import logging
import os
import zipfile

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file, url_for

import camera_manager
from db import (
    add_camera,
    delete_camera,
    distinct_classes,
    get_camera,
    init_db,
    list_cameras,
    list_detections,
    update_camera_line,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = Flask(__name__)

init_db()
camera_manager.start_all()
# Shutdown limpio (F1): al salir el proceso se detienen todos los workers y el
# watchdog con join acotado -- sin threads huerfanos ni conteos por escribir.
atexit.register(camera_manager.stop_all)


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
    add_camera(data)
    cam = get_camera(data["name"])
    camera_manager.start_camera(cam)
    return redirect(url_for("index"))


@app.route("/cameras/<name>/delete", methods=["POST"])
def delete_camera_route(name):
    camera_manager.stop_camera(name)
    delete_camera(name)
    return redirect(url_for("index"))


@app.route("/live/<name>")
def live(name):
    cam = get_camera(name)
    if not cam:
        abort(404)
    return render_template("live.html", camera=cam)


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
    worker = camera_manager.get_worker(name)
    if not worker:
        abort(404)

    def generate():
        import time

        last_sent = None
        try:
            while True:
                frame = worker.get_latest_jpeg()
                # solo reenvia si hay un frame nuevo -- evita gastar ancho
                # de banda repitiendo el mismo jpeg mientras no cambia nada
                if frame is not None and frame is not last_sent:
                    last_sent = frame
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.05)
        except GeneratorExit:
            # el cliente (navegador) corto la conexion -- salir prolijo en
            # vez de dejar el hilo dando vueltas para siempre
            return

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


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

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in detections:
            path = d["image_path"]
            if os.path.isfile(path):
                arcname = f"{d['class_name']}/{os.path.basename(path)}"
                zf.write(path, arcname)
    mem.seek(0)
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name="detecciones.zip",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("WEB_PORT", "8001")), threaded=True)
