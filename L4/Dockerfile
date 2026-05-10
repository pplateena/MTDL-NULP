# CUDA 12.4 + cuDNN 9 — required by onnxruntime-gpu 1.20+
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.11 /usr/bin/python3 && ln -sf python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app/ ./app/
COPY model.onnx model.onnx.data ./


ENV MODEL_PATH=/app/model.onnx
ENV PROVIDER=cuda

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
