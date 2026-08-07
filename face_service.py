"""
face_service.py
---------------
Lightweight Flask micro-service that provides two endpoints:

  POST /register   – Store a face embedding for a worker.
  POST /recognize  – Match a captured image against stored embeddings.
  GET  /health     – Liveness check.

This script is the bridge between the Laravel backend and DeepFace.
test.py can be imported or run standalone to verify DeepFace is working.

Usage:
  python face_service.py

Requirements (install in your venv):
  pip install flask deepface tf-keras pillow numpy
"""

import base64
import io
import json
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# DeepFace import – optional so the health endpoint works even before models
# are downloaded on first run.
# ---------------------------------------------------------------------------
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "face_data", "faces")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "face_data", "embeddings.json")

os.makedirs(FACES_DIR, exist_ok=True)

# Model / backend settings – change here if you want a different model.
MODEL_NAME = "Facenet512"
DETECTOR = "mtcnn"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.40          # lower  → stricter match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings() -> dict:
    """Return {worker_id: {name, embedding}} dict."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(data: dict) -> None:
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)


def base64_to_image_path(b64_string: str, filename: str) -> str:
    """Decode base64 image and save to FACES_DIR; return the saved path."""
    # Strip data-URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    path = os.path.join(FACES_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


def get_embedding(image_path: str) -> list:
    """Extract face embedding using DeepFace."""
    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True,
    )
    # represent() returns a list; take the first face
    return result[0]["embedding"]


def cosine_distance(vec_a: list, vec_b: list) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "deepface_available": DEEPFACE_AVAILABLE,
        "model": MODEL_NAME,
    })


@app.route("/register", methods=["POST"])
def register_face():
    """
    Body (JSON):
      {
        "worker_id": 42,
        "worker_name": "John Doe",
        "image_base64": "<base64 encoded JPG/PNG>"
      }

    Response:
      { "success": true, "worker_id": 42 }
    """
    if not DEEPFACE_AVAILABLE:
        return jsonify({"success": False, "error": "DeepFace not installed"}), 500

    data = request.get_json(force=True)
    worker_id = str(data.get("worker_id", ""))
    worker_name = data.get("worker_name", "Unknown")
    image_b64 = data.get("image_base64", "")

    if not worker_id or not image_b64:
        return jsonify({"success": False, "error": "worker_id and image_base64 are required"}), 400

    filename = f"worker_{worker_id}_{int(time.time())}.jpg"
    try:
        img_path = base64_to_image_path(image_b64, filename)
        embedding = get_embedding(img_path)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 422

    embeddings = load_embeddings()
    embeddings[worker_id] = {
        "name": worker_name,
        "embedding": embedding,
        "photo": filename,
    }
    save_embeddings(embeddings)

    return jsonify({"success": True, "worker_id": int(worker_id)})


@app.route("/recognize", methods=["POST"])
def recognize_face():
    """
    Body (JSON):
      { "image_base64": "<base64 encoded JPG/PNG>" }

    Response (match found):
      { "success": true, "matched": true, "worker_id": 42, "worker_name": "John Doe", "distance": 0.21 }

    Response (no match):
      { "success": true, "matched": false }
    """
    if not DEEPFACE_AVAILABLE:
        return jsonify({"success": False, "error": "DeepFace not installed"}), 500

    data = request.get_json(force=True)
    image_b64 = data.get("image_base64", "")

    if not image_b64:
        return jsonify({"success": False, "error": "image_base64 is required"}), 400

    filename = f"capture_{int(time.time())}.jpg"
    try:
        img_path = base64_to_image_path(image_b64, filename)
        probe_embedding = get_embedding(img_path)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    finally:
        # Remove temporary capture file
        if "img_path" in locals() and os.path.exists(img_path):
            os.remove(img_path)

    embeddings = load_embeddings()
    if not embeddings:
        return jsonify({"success": True, "matched": False, "reason": "No registered workers"})

    best_id = None
    best_distance = float("inf")

    for worker_id, info in embeddings.items():
        dist = cosine_distance(probe_embedding, info["embedding"])
        if dist < best_distance:
            best_distance = dist
            best_id = worker_id

    if best_distance <= THRESHOLD:
        matched_info = embeddings[best_id]
        return jsonify({
            "success": True,
            "matched": True,
            "worker_id": int(best_id),
            "worker_name": matched_info["name"],
            "distance": round(best_distance, 4),
        })

    return jsonify({"success": True, "matched": False, "distance": round(best_distance, 4)})


@app.route("/delete/<int:worker_id>", methods=["DELETE"])
def delete_face(worker_id: int):
    """Remove a worker's face registration."""
    embeddings = load_embeddings()
    key = str(worker_id)
    if key not in embeddings:
        return jsonify({"success": False, "error": "Worker not registered"}), 404

    info = embeddings.pop(key)
    save_embeddings(embeddings)

    # Remove stored photo
    photo_path = os.path.join(FACES_DIR, info.get("photo", ""))
    if os.path.exists(photo_path):
        os.remove(photo_path)

    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("FACE_SERVICE_PORT", 5050))
    print(f"[face_service] Starting on http://127.0.0.1:{port}")
    print(f"[face_service] DeepFace available: {DEEPFACE_AVAILABLE}")
    app.run(host="0.0.0.0", port=port, debug=False)
