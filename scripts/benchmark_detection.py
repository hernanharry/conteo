"""Benchmark reproducible de tuning de deteccion (Fase C, hardware debil CPU).

Dado un directorio con frames capturados de la camara (PNG/JPG), barre

    modelo x conf_threshold x imgsz

y mide, por configuracion:

  - recall "relativo" y falsos positivos (proxy) contra una referencia-union
    (merge por IoU+clase de TODAS las configs) sobre las clases visibles;
  - latencia mediana por forward (ms) con TORCH_THREADS en 2 y 4;
  - overlays por frame para comparacion visual humana (este flujo de auto-tuning
    no puede "ver" imagenes, asi que el ground truth visual lo confirma la
    persona con los overlays).

Escribe `result.json` + tabla por consola en el directorio de salida.

Uso:
    python scripts/benchmark_detection.py --frames DIR_FRAMES [--out DIR_SALIDA] [--models ...]

Ejecutar con el contenedor apagado (o en un venv aislado) para mediciones
limpias de latencia. Documenta el metodo detallado en docs/detection-tuning.md.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import cv2  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# Clases COCO por defecto de la app (0,2,3,5,7 = persona, auto, moto, bus, camion).
APPT_CLASSES = {0, 2, 3, 5, 7}

COCO = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    9: "traffic light", 10: "fire hydrant", 11: "stop sign",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", required=True, help="directorio con frames capturados")
    ap.add_argument("--out", default="bench_out", help="directorio de salida")
    ap.add_argument("--models", default="yolov8s.pt", help="modelos separados por coma")
    return ap.parse_args()


def find_frames(d):
    pats = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
    if not pats:
        raise SystemExit(f"no hay frames (.jpg/.png) en {d}")
    return pats


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    args = parse_args()
    frame_paths = find_frames(args.frames)
    frames = {os.path.basename(p): cv2.imread(p) for p in frame_paths}
    models_cfg = {m: m.split("/")[-1].replace(".pt", "") for m in args.models.split(",")}
    confs = [0.15, 0.25, 0.35]
    imgszs = [640, 960]
    threads = [2, 4]
    iou_match = 0.3

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    models = {m: YOLO(m) for m in models_cfg}

    print(f"frames: {list(frames)[:6]}{'...' if len(frames) > 6 else ''} ({len(frames)} total)")

    # ---- Barrido de detecciones (todas las clases, para ver las no tomadas) ----
    det_store = {}
    for mname in models_cfg:
        for imgsz in imgszs:
            for conf in confs:
                key = f"{models_cfg[mname]}@{imgsz}@{conf}"
                for fname, img in frames.items():
                    res = models[mname](img, verbose=False, conf=conf, imgsz=imgsz, classes=None)[0]
                    dets = []
                    for box in res.boxes:
                        dets.append((
                            int(box.cls[0]),
                            float(box.conf[0]),
                            tuple(float(v) for v in box.xyxy[0]),
                        ))
                    det_store.setdefault(key, {})[fname] = dets
            print(f"  detecciones OK: {mname} @ {imgsz} (confs {confs})", flush=True)

    # ---- Referencia (union de TODAS las configs, merge por IoU+clase) ----
    def build_ref(fname):
        ref = []
        for key in det_store:
            for cls, c, xyxy in det_store[key][fname]:
                hit = None
                for i, r in enumerate(ref):
                    if r[0] == cls and iou(xyxy, r[2]) >= iou_match:
                        hit = i
                        break
                if hit is None:
                    ref.append([cls, c, xyxy])
                elif c > ref[hit][1]:
                    ref[hit][1] = c
                    ref[hit][2] = xyxy
        return [tuple(r) for r in ref]

    ref_frame = {f: build_ref(f) for f in frames}

    # ---- Latencias (threads 2 vs 4): serie de forwards sobre el primer frame ----
    def latency(model, img, conf, imgsz, thr, n=7):
        torch.set_num_threads(thr)
        model(img, verbose=False, conf=conf, imgsz=imgsz)[0]  # warmup
        ts = []
        with torch.inference_mode():
            for _ in range(n):
                t0 = time.perf_counter()
                model(img, verbose=False, conf=conf, imgsz=imgsz)[0]
                ts.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(ts)

    lat = {}
    first_frame = frames[list(frames)[0]]
    for mname in models_cfg:
        for imgsz in imgszs:
            for thr in threads:
                k = f"{models_cfg[mname]}@{imgsz}@{thr}t"
                lat[k] = latency(models[mname], first_frame, 0.25, imgsz, thr)
                print(f"  latencia {k}: {lat[k]:.0f} ms", flush=True)

    # ---- Metricas por config (recall relativo vs referencia) ----
    def recall_fp(key):
        tp = fp = ref_tot = 0
        for fname in frames:
            ref = ref_frame[fname]
            ref_tot += len(ref)
            matched = set()
            for cls, c, xyxy in det_store[key][fname]:
                hit = None
                for i, r in enumerate(ref):
                    if i in matched:
                        continue
                    if r[0] == cls and iou(xyxy, r[2]) >= iou_match:
                        hit = i
                        break
                if hit is not None:
                    tp += 1
                    matched.add(hit)
                else:
                    fp += 1
        return tp, fp, ref_tot

    print("\n=== TABLA POR CONFIG (recall relativo sobre la referencia-union) ===")
    print(f"{'config':<18} {'det':>5} {'recall%':>8} {'fp':>4} {'tp':>4}")
    for key in det_store:
        total = sum(len(v) for v in det_store[key].values())
        tp, fp, rt = recall_fp(key)
        print(f"{key:<18} {total:>5} {100.0 * tp / rt if rt else 0:>7.1f}% {fp:>4} {tp:>4}")

    print("\n=== LATENCIA MEDIANA (ms/frame) ===")
    for k, v in lat.items():
        print(f"{k:<16} {v:>7.0f} ms")

    print("\n=== CLASES VISIBLES EN LA REFERENCIA ===")
    cls_counts = {}
    for fname in frames:
        for cls, c, _ in ref_frame[fname]:
            cls_counts.setdefault(cls, []).append(c)
    for cls in sorted(cls_counts):
        nota = "" if cls in APPT_CLASSES else "  <-- NO esta en la lista (0,2,3,5,7)"
        print(f"  {cls:>3} {COCO.get(cls, 'coco%d' % cls):<16} maxconf={max(cls_counts[cls]):.2f}{nota}")

    # ---- Overlays: config actual (s@640@0.35) vs propuesta (s@960@0.25) ----
    overlays = {
        "actual": {"model": "yolov8s.pt", "conf": 0.35, "imgsz": 640},
        "propuesto": {"model": "yolov8s.pt", "conf": 0.25, "imgsz": 960},
    }
    for cfg_name, cfg in overlays.items():
        m = models.get(cfg["model"])
        if m is None:
            continue
        for fname, img in frames.items():
            oimg = img.copy()
            res = m(oimg, verbose=False, conf=cfg["conf"], imgsz=cfg["imgsz"])[0]
            for box in res.boxes:
                cls, c = int(box.cls[0]), float(box.conf[0])
                if cls not in APPT_CLASSES:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                color = (0, 255, 0) if c >= cfg["conf"] else (0, 165, 255)
                cv2.rectangle(oimg, (x1, y1), (x2, y2), color, 2)
                cv2.putText(oimg, f"{COCO.get(cls, cls)} {c:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if oimg.shape[1] > 960:
                oimg = cv2.resize(oimg, (960, int(oimg.shape[0] * 960 / oimg.shape[1])))
            d = os.path.join(args.out, cfg_name)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, fname), oimg)

    out = {
        "const": {
            "frames": list(frames),
            "coco_classes_tomadas": sorted(APPT_CLASSES),
            "IOU_MATCH": iou_match,
        },
        "latencias_ms": lat,
        "detecciones_por_config": det_store,
        "referencia_por_frame": {f: [[c, cf, xy] for c, cf, xy in v] for f, v in ref_frame.items()},
    }
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nresultados en", args.out)


if __name__ == "__main__":
    main()
