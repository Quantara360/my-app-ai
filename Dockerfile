FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall opencv-python -y || true
RUN pip install --no-cache-dir opencv-python-headless deepface tensorflow

COPY . .
EXPOSE 5050
CMD ["python", "face_service.py"]
