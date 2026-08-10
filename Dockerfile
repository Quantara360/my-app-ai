FROM python:3.11-slim

# OpenCV/DeepFace need these system libs even with the headless opencv build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FACE_SERVICE_PORT=5050
EXPOSE 5050

CMD ["python", "face_service.py"]
