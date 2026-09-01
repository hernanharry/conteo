FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# GAP-PROD-02 (resuelto en F8): torch/torchvision quedan PINNED a las versiones
# validadas en desarrollo (2.13.0 / 0.28.0 CPU). Antes sin pin, un build hoy o
# en 6 meses instalaba cualquier version de torch, rompiendo la
# reproducibilidad. El indice https://download.pytorch.org/whl/cpu entrega las
# variantes +cpu de esas versiones.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ .

# F7: health check del contenedor usa /api/health (publico, sin auth). Se usa
# python urllib en vez de curl para no instalar paquetes extra en la imagen.
# start-period: el modelo YOLO tarda en cargarse al arrancar.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/api/health', timeout=4)"

EXPOSE 8001

CMD ["python", "server.py"]