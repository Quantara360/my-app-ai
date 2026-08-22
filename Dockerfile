FROM python:3.11-slim

# OpenCV/DeepFace need these system libs even with the headless opencv build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch, installed separately from PyTorch's own CPU wheel index -
# the default PyPI build pulls in CUDA libraries this VPS (no GPU) never
# uses, ballooning both image size and build time for nothing. Only needed
# for DeepFace's Fasnet anti-spoofing (liveness) model - see
# FACE_LIVENESS_CHECK in face_service.py.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FACE_SERVICE_PORT=5050
EXPOSE 5050

CMD ["python", "face_service.py"]
