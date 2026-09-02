# Tuning de detección (hardware débil, CPU-only)

Benchmark Fase A realizado sobre la cámara `Vereda` (RTSP 1920x1080@25fps,
host Intel i3-4330 Haswell 2 núcleos/4 hilos, 12 GB RAM, sin GPU).

## Conclusión

En esta máquina la mejor combinación medida de precisión/velocidad es
**YOLOv8s a `IMGSZ=960` con `conf_threshold=0.25` nativo (PyTorch)**. El
`imgsz` (no el umbral ni el modelo) es la palanca que más sube el recuerdo de
objetos lejanos/pequeños.

## Recuerdo relativo (39 objetos objetivo detectables en 6 frames reales)

| config | recall |
|---|---|
| s @ 640 @ 0.35 (config original) | 51.3% |
| s @ 640 @ 0.25 | 51.3% |
| **s @ 960 @ 0.25** | **69.2%** |
| s @ 960 @ 0.15 | 87.2% |

- Bajar `conf_threshold` solo (sin subir imgsz) NO cambió el recall: a 640 los
  objetos distantes escapan al modelo sin importar el umbral. Falsos positivos
  con conf 0.15: ~0 (proxy) — bajar conf es seguro.
- Clases extra detectadas a baja confianza (boat, potted plant, stop sign,
  traffic light) son ruido de escena; **no** agregarlas a la lista `0,2,3,5,7`.

## Throughput en CPU (mediana ms/frame)

| modelo @ imgsz | nativo (aislado, host) | en contenedor (aislado, Linux) | en app (medido /api/perf) |
|---|---|---|---|
| n @ 640 | 181 | — | — |
| n @ 960 | 404 | — | — |
| s @ 640 | 462 | — | ~1000-1300 (config original) |
| s @ 960 | 900 | 1042 (thr=2) / 1254 (thr=4) | ~1100-1400 (config final) |

Ajustes que redujeron el ~2x de contención del contenedor frente al benchmark
aislado (2 núcleos físicos):

- `TORCH_THREADS=2` (4 hilos sobresuscribían los 2 núcleos físicos; en
  contenedor 2 hilos resultó más rápido que 4).
- `LIVE_STREAM_FPS=8` y `LIVE_MAX_WIDTH=720`: menos CPU del render del vivo.
- `GRABBER_MAX_FPS=8` (nuevo): apaga la decodificación H264 1080p continua a
  25 fps, que competía con la inferencia.

## Runtimes alternativos RECHAZADOS (medidos en esta CPU)

| export | ms/frame @960 | veredicto |
|---|---|---|
| PyTorch nativo | **1042** | el más rápido |
| ONNX Runtime (estático 960) | ~1900 | +80% lento |
| OpenVINO IR (dinámico) | ~3900 | ~4x lento |
| OpenVINO IR (estático 960) | ~3480 | ~3.4x lento |

En un Haswell de 2013, los kernels CPU de OpenVINO/ONNX no superan a los de
torch nativo (oneDNN/MKL); por eso este repo NO agrega openvino/onnx al
contenedor. En CPUs modernas podría ser a favor, pero exigiría re-benchmark.

## Método reproducible

El barrido está incrustado y parametrizado en `scripts/benchmark_detection.py`:
capturar N frames reales del RTSP en un directorio y correr

```
python scripts/benchmark_detection.py --frames DIR_FRAMES --out DIR_SALIDA
```

El script barre modelo × conf (0.15/0.25/0.35) × imgsz (640/960), mide latencia
mediana por forward con TORCH_THREADS 2 y 4, arma una referencia-unión
(IoU≥0.3 por clase) sobre todas las configs para el recall relativo, y genera
overlays (`actual/` vs `propuesto/`) para revisión visual humana. Escribe
`result.json` y la tabla por consola.

> Nota: ejecutar con el contenedor apagado (o venv aislado) para latencias
> limpias; en la máquina real conviene además contrastar con `/api/perf`
> (cadencia en app, con contención del grabber/render).