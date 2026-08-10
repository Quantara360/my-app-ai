"""
face_service.py
---------------
Lightweight Flask micro-service that provides two endpoints:

  POST /register   – Store a face embedding for a worker.
  POST /recognize  – Match a captured image against stored embeddings.
  GET  /health     – Liveness check.

This script is the bridge between the Laravel backend and DeepFace.

Recognition model: ArcFace (via DeepFace's built-in "ArcFace" model_name).
ArcFace's angular-margin loss keeps different identities further apart in
embedding space than Facenet512 (previously used here), which matters
because /recognize does 1:N identification - scanning one probe embedding
against every registered worker - not just a single 1:1 comparison.

On top of the recognition model, every capture is run through quality
gates before it's turned into a stored or probe embedding:
  - pose check: rejects tilted/turned heads (assess_pose)
  - distance check: rejects faces that are too far/too close/cropped
    at the frame edge (assess_distance)
  - registration photos are additionally checked for blur/lighting
    (assess_sharpness / assess_brightness), since a noisy reference
    embedding poisons every future match against that worker
Garbage in -> garbage matches, so this is as big an accuracy lever as
the model choice itself.

Matches are also scored with a calibrated 0-100 confidence value
(find_confidence), instead of just a bare matched: true/false.

Usage:
  python face_service.py

Requirements (install in your venv):
  pip install flask deepface tf-keras pillow numpy opencv-python-headless
"""

import base64
import binascii
import json
import logging
import math
import os
import re
import time

import numpy as np
from flask import Flask, jsonify, request

logger = logging.getLogger("face_service")

# Verbose per-capture sharpness/brightness/crop-size logging, off by default -
# flip on with FACE_SERVICE_DEBUG=1 when tuning the quality thresholds.
DEBUG = os.environ.get("FACE_SERVICE_DEBUG") == "1"

# Reject payloads above this before they're even base64-decoded - a phone
# photo runs a few MB; anything wildly larger than that is either a bad
# client or abuse, not a real capture.
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB decoded

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
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + (1024 * 1024)  # + headroom for JSON/base64 overhead

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "face_data", "faces")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "face_data", "embeddings.json")

os.makedirs(FACES_DIR, exist_ok=True)

# Model / backend settings – change here if you want a different model.
MODEL_NAME = "ArcFace"
DETECTOR = "mtcnn"
DISTANCE_METRIC = "cosine"

# Pre-tuned for ArcFace + cosine. Not comparable to Facenet512's old 0.40.
# 0.68 was the original calibration value; the calibration data behind
# _CONFIDENCE_CONFIG shows genuine same-person distances ranging from
# ~0.13 (best observed) to ~0.59 (loosest still-genuine observed), so 0.68
# left headroom past the loosest real match. Tightened to 0.45 (attendance
# must be a close, high-confidence match, not just "closer than everyone
# else") to cut false-accepts, e.g. look-alikes or a coworker's registration
# photo lighting drifting into range - at the cost of occasionally asking a
# genuine worker to retake the photo in bad lighting/angle.
THRESHOLD = 0.45

# Minimum required gap (in cosine distance) between the best and
# second-best match before /recognize will commit to an identification.
# Nearest-neighbor matching alone can't distinguish "unambiguously this
# worker" from "won by a hair over a similar-looking coworker" - only this
# margin can. Engineering default, not derived from calibration data (unlike
# THRESHOLD above) - tighten it (raise the number) if mismatches between
# similar-looking people are still getting through; loosen it if genuine
# matches start getting rejected as "too close to call" too often.
MIN_MATCH_MARGIN = 0.05

# Toggle the pose/distance capture-quality gate off if it proves too strict
# for your camera setups.
ENFORCE_POSE_AND_DISTANCE = True

# ---------------------------------------------------------------------------
# Quality-control thresholds
# ---------------------------------------------------------------------------
MAX_ROLL_DEGREES = 20.0     # head tilt tolerance
MAX_YAW_RATIO = 0.2         # head turn tolerance (eyes off-center of face box)
MIN_FACE_RATIO = 0.03       # face bbox area / image area, below = too far
MIN_FACE_WIDTH = 50         # px, below = too far / too low-res
BLUR_THRESHOLD = 100.0      # Laplacian variance, below = too blurry
MAX_DETECT_DIM = 960        # px, longest side - phone photos come in at several
# thousand px; MTCNN's coarse-to-fine detection cost scales with pixel count,
# and that (not the ArcFace embedding step, which is constant-cost) is what
# was making /recognize and /register take 13-17s on CPU. Downscaling before
# detection cuts that cost by roughly (original/960)^2 with no accuracy loss -
# faces stay far above MIN_FACE_WIDTH/MIN_FACE_RATIO at normal framing
# distances, and the sharpness/pose checks run on the same downscaled image
# consistently.
DARK_THRESHOLD = 60.0       # mean grayscale intensity, below = too dark
BRIGHT_THRESHOLD = 200.0    # mean grayscale intensity, above = overexposed

# ---------------------------------------------------------------------------
# ArcFace confidence-score regression constants (pre-fit logistic regression
# mapping an ArcFace cosine distance to a calibrated 0-100 confidence score)
# ---------------------------------------------------------------------------
_CONFIDENCE_CONFIG = {
    "w": -6.586150423868342,
    "b": 2.2617127737265186,
    "normalizer": 1.22267,
    "denorm_max_true": 82.32544721731948,
    "denorm_min_true": 28.687745855666044,
    "denorm_max_false": 17.210359124614925,
    "denorm_min_false": 1.306796127659645,
}


# ---------------------------------------------------------------------------
# Quality assessment helpers
# ---------------------------------------------------------------------------

def assess_sharpness(img_bgr: np.ndarray) -> float:
    """Laplacian-variance sharpness estimate. Lower = blurrier."""
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_brightness(img_bgr: np.ndarray) -> float:
    """Mean grayscale pixel intensity (0-255)."""
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return float(gray.mean())


def assess_pose(facial_area: dict) -> dict:
    """
    Estimate whether a detected face is roughly front-facing, using the eye
    landmarks DeepFace's detector provides.
    Returns {'is_frontal': bool, 'reason': str or None}.
    """
    left_eye = facial_area.get("left_eye")
    right_eye = facial_area.get("right_eye")
    w = facial_area.get("w") or 0

    if left_eye is None or right_eye is None or not w:
        # Not enough signal to judge pose - report as frontal to avoid
        # false rejections rather than blocking a legitimate capture.
        return {"is_frontal": True, "reason": None}

    lx, ly = left_eye
    rx, ry = right_eye

    roll_degrees = float(np.degrees(np.arctan2(ly - ry, lx - rx)))
    eyes_center_x = (lx + rx) / 2
    face_center_x = facial_area.get("x", 0) + w / 2
    yaw_ratio = abs(eyes_center_x - face_center_x) / w

    reasons = []
    if abs(roll_degrees) > MAX_ROLL_DEGREES:
        reasons.append(
            f"head is tilted about {abs(roll_degrees):.0f} degrees - "
            "keep it level and look straight at the camera"
        )
    if yaw_ratio > MAX_YAW_RATIO:
        reasons.append("face is turned to the side - look directly at the camera")

    return {"is_frontal": len(reasons) == 0, "reason": "; ".join(reasons) or None}


def assess_distance(facial_area: dict, img_width: int, img_height: int) -> dict:
    """
    Estimate whether the detected face was photographed at a workable
    distance from the camera. Returns {'is_good_distance': bool, 'reason': str or None}.
    """
    x = facial_area.get("x") or 0
    y = facial_area.get("y") or 0
    w = facial_area.get("w") or 0
    h = facial_area.get("h") or 0

    img_area = img_width * img_height
    face_ratio = (w * h) / img_area if img_area else 0.0

    reasons = []
    if w < MIN_FACE_WIDTH or h < MIN_FACE_WIDTH or face_ratio < MIN_FACE_RATIO:
        reasons.append(
            "face is too small in the frame - move closer to the camera for an accurate match"
        )

    touches_edge = x <= 1 or y <= 1 or (x + w) >= img_width - 1 or (y + h) >= img_height - 1
    if touches_edge and face_ratio > 0.5:
        reasons.append(
            "face is too close to the camera and is being cropped by the frame edge - "
            "move back a little"
        )

    return {"is_good_distance": len(reasons) == 0, "reason": "; ".join(reasons) or None}


def check_capture_quality(face_obj: dict, img_width: int, img_height: int, check_sharpness: bool = False) -> str:
    """
    Run pose/distance (and optionally blur/lighting) checks against a single
    DeepFace.extract_faces() result. Returns a reason string if the capture
    should be rejected, or None if it passes.
    """
    facial_area = face_obj["facial_area"]

    if ENFORCE_POSE_AND_DISTANCE:
        pose = assess_pose(facial_area)
        distance = assess_distance(facial_area, img_width, img_height)
        reasons = [r for r in (pose["reason"], distance["reason"]) if r]
        if reasons:
            return "; ".join(reasons)

    if check_sharpness:
        # face_obj["face"] is RGB, float [0, 1] (DeepFace convention) - convert
        # back to BGR uint8 for the OpenCV-based sharpness/brightness checks.
        face_bgr = (face_obj["face"][:, :, ::-1] * 255).astype(np.uint8)
        sharpness = assess_sharpness(face_bgr)
        brightness = assess_brightness(face_bgr)
        if DEBUG:
            print(f"[face_service] DEBUG crop_shape={face_bgr.shape} sharpness={sharpness:.1f} "
                  f"(threshold={BLUR_THRESHOLD}) brightness={brightness:.1f} "
                  f"(dark<{DARK_THRESHOLD}, bright>{BRIGHT_THRESHOLD})")
        reasons = []
        if sharpness < BLUR_THRESHOLD:
            reasons.append("photo is too blurry - hold the camera steady and retake")
        if brightness < DARK_THRESHOLD:
            reasons.append("photo is too dark - move to better lighting")
        elif brightness > BRIGHT_THRESHOLD:
            reasons.append("photo is overexposed - avoid strong backlight/glare")
        if reasons:
            return "; ".join(reasons)

    return None


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1 + ez)


def find_confidence(distance: float, verified: bool) -> float:
    """
    Map an ArcFace cosine distance to a calibrated 0-100 confidence score
    using a pre-fit logistic regression. Same-person pairs land in
    [51, 100], different-person pairs in [0, 49].
    """
    if distance <= 0:
        return 100.0 if verified else 0.0

    cfg = _CONFIDENCE_CONFIG
    d = distance / cfg["normalizer"] if cfg["normalizer"] > 1 else distance
    confidence = 100 * _sigmoid(cfg["w"] * d + cfg["b"])

    if verified:
        min_o, max_o = cfg["denorm_min_true"], cfg["denorm_max_true"]
        min_t, max_t = max(51, min_o), 100
    else:
        min_o, max_o = cfg["denorm_min_false"], cfg["denorm_max_false"]
        min_t, max_t = 0, min(49, int(max_o))

    scaled = ((confidence - min_o) / (max_o - min_o)) * (max_t - min_t) + min_t

    if verified and scaled < 51:
        scaled = 51
    elif not verified and scaled > 49:
        scaled = 49

    return round(max(0.0, min(100.0, scaled)), 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings() -> dict:
    """Return {worker_id: {name, embedding, model, photo}} dict."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(data: dict) -> None:
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)


def base64_to_image_path(b64_string: str, filename: str) -> str:
    """Decode base64 image and save to FACES_DIR; return the saved path.

    Raises ValueError (caller turns this into a 400/422, not a 500) on
    malformed base64 or an oversized payload - both are client mistakes,
    not server bugs.
    """
    # Strip data-URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(b64_string, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"image_base64 is not valid base64 image data: {exc}") from exc
    if not image_bytes:
        raise ValueError("image_base64 decoded to an empty file")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"image is too large ({len(image_bytes) / 1_000_000:.1f} MB, "
            f"max {MAX_IMAGE_BYTES / 1_000_000:.0f} MB)"
        )
    # filename is caller-controlled (worker_id / timestamp) - resolve and
    # confirm the write stays inside FACES_DIR before touching disk.
    path = os.path.join(FACES_DIR, filename)
    if os.path.commonpath([os.path.abspath(path), os.path.abspath(FACES_DIR)]) != os.path.abspath(FACES_DIR):
        raise ValueError("invalid filename")
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


def get_embedding(image_path: str, check_sharpness: bool = False) -> list:
    """
    Extract a face embedding using DeepFace + ArcFace, gated by pose,
    distance-from-camera, and (for registration) blur/lighting checks.

    Face detection (MTCNN) runs exactly once here. The aligned crop that
    comes back from extract_faces() is reused directly as the input to
    represent() via detector_backend="skip", instead of handing represent()
    the original full image and letting it re-run MTCNN from scratch.
    That second detection pass used to roughly double the wall-clock time
    of every /recognize and /register call for zero accuracy benefit -
    verified empirically that the single-detect and double-detect paths
    produce embeddings with cosine similarity > 0.9999.

    Raises ValueError with a human-readable reason if the capture fails a
    quality check or no face is found.
    """
    from PIL import Image
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        if max(w0, h0) > MAX_DETECT_DIM:
            scale = MAX_DETECT_DIM / max(w0, h0)
            im = im.resize((max(1, round(w0 * scale)), max(1, round(h0 * scale))), Image.LANCZOS)
            im.save(image_path, quality=90)
        img_width, img_height = im.size

    face_objs = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend=DETECTOR,
        enforce_detection=True,
        align=True,
    )

    # Largest detected face first (most likely the intended subject).
    face_objs = sorted(
        face_objs,
        key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"],
        reverse=True,
    )
    best_face = face_objs[0]
    reject_reason = check_capture_quality(best_face, img_width, img_height, check_sharpness)
    if reject_reason:
        raise ValueError(reject_reason)

    # face_obj["face"] is RGB, float [0, 1] (DeepFace convention) - convert
    # back to BGR uint8, the format DeepFace expects for a raw array input.
    face_bgr = (best_face["face"][:, :, ::-1] * 255).astype(np.uint8)
    result = DeepFace.represent(
        img_path=face_bgr,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
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
        "detector": DETECTOR,
        "threshold": THRESHOLD,
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

    data = request.get_json(force=True, silent=True) or {}
    worker_id = str(data.get("worker_id", "")).strip()
    worker_name = str(data.get("worker_name") or "Unknown").strip()[:255]
    image_b64 = data.get("image_base64", "")

    if not worker_id or not re.fullmatch(r"[0-9]+", worker_id):
        return jsonify({"success": False, "error": "worker_id must be a positive integer"}), 400
    if not image_b64 or not isinstance(image_b64, str):
        return jsonify({"success": False, "error": "image_base64 is required"}), 400

    filename = f"worker_{worker_id}_{int(time.time())}.jpg"
    img_path = None
    try:
        img_path = base64_to_image_path(image_b64, filename)
        # Registration photos are the reference for every future match, so
        # also gate on blur/lighting here (not just at recognize time).
        embedding = get_embedding(img_path, check_sharpness=True)
    except (ValueError, OSError) as exc:
        # Expected, client-facing: bad/corrupt image data or a failed quality gate.
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"success": False, "error": str(exc)}), 422
    except Exception:
        # Unexpected: don't leak internals to the client, and don't leave an
        # orphaned photo behind for a registration that never completed.
        logger.exception("register_face: unexpected error for worker_id=%s", worker_id)
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"success": False, "error": "Internal error while processing the photo"}), 500

    embeddings = load_embeddings()
    embeddings[worker_id] = {
        "name": worker_name,
        "embedding": embedding,
        "model": MODEL_NAME,
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
      { "success": true, "matched": true, "worker_id": 42, "worker_name": "John Doe",
        "distance": 0.21, "confidence": 87.4 }

    Response (no match):
      { "success": true, "matched": false, "distance": 0.71, "confidence": 12.3 }
    """
    if not DEEPFACE_AVAILABLE:
        return jsonify({"success": False, "error": "DeepFace not installed"}), 500

    data = request.get_json(force=True, silent=True) or {}
    image_b64 = data.get("image_base64", "")

    if not image_b64 or not isinstance(image_b64, str):
        return jsonify({"success": False, "error": "image_base64 is required"}), 400

    filename = f"capture_{int(time.time())}.jpg"
    img_path = None
    try:
        img_path = base64_to_image_path(image_b64, filename)
        # Gate scans on blur/lighting too, not just pose/distance - a dark or
        # blurry probe photo produces an unreliable embedding and should be
        # rejected with a clear reason rather than risk a false match/no-match.
        probe_embedding = get_embedding(img_path, check_sharpness=True)
    except (ValueError, OSError) as exc:
        # Expected, client-facing: bad/corrupt image data or a failed quality gate.
        return jsonify({"success": False, "error": str(exc)}), 422
    except Exception:
        # Unexpected: don't leak internals to the client.
        logger.exception("recognize_face: unexpected error")
        return jsonify({"success": False, "error": "Internal error while processing the photo"}), 500
    finally:
        # Remove temporary capture file
        if img_path and os.path.exists(img_path):
            os.remove(img_path)

    embeddings = load_embeddings()

    # Only compare against embeddings produced by the current model - a
    # stored vector tagged with a different (or missing) model isn't in the
    # same embedding space as the probe and would produce a meaningless
    # distance if compared directly.
    usable = {
        worker_id: info
        for worker_id, info in embeddings.items()
        if info.get("model") == MODEL_NAME
    }
    if not usable:
        reason = "No registered workers" if not embeddings else (
            "No workers registered under the current model "
            f"('{MODEL_NAME}') - re-register workers to use this service."
        )
        return jsonify({"success": True, "matched": False, "reason": reason})

    # Pure nearest-neighbor over every registered worker - track the top TWO,
    # not just the winner, so we can tell "clearly this person" apart from
    # "barely edged out a similar-looking coworker".
    best_id = None
    best_distance = float("inf")
    second_best_distance = float("inf")

    for worker_id, info in usable.items():
        dist = cosine_distance(probe_embedding, info["embedding"])
        if dist < best_distance:
            second_best_distance = best_distance
            best_distance = dist
            best_id = worker_id
        elif dist < second_best_distance:
            second_best_distance = dist

    verified = best_distance <= THRESHOLD
    confidence = find_confidence(best_distance, verified)

    # Two people who genuinely look alike land close together in embedding
    # space. Distance alone can't tell "unambiguously this worker" apart
    # from "won by a hair over a similar-looking coworker, could easily flip
    # on the next scan's lighting/angle" - only the GAP to the runner-up
    # can. Below MIN_MATCH_MARGIN, refuse to guess rather than silently
    # attributing attendance to whichever one edged out by noise.
    ambiguous = verified and (second_best_distance - best_distance) < MIN_MATCH_MARGIN

    if verified and not ambiguous:
        matched_info = usable[best_id]
        return jsonify({
            "success": True,
            "matched": True,
            "worker_id": int(best_id),
            "worker_name": matched_info["name"],
            "distance": round(best_distance, 4),
            "confidence": confidence,
        })

    if ambiguous:
        return jsonify({
            "success": True,
            "matched": False,
            "distance": round(best_distance, 4),
            "confidence": confidence,
            "reason": (
                "Match too close to call between two or more registered workers - "
                "retake with the face centered, well-lit, and looking straight at "
                "the camera, or verify manually."
            ),
        })

    return jsonify({
        "success": True,
        "matched": False,
        "distance": round(best_distance, 4),
        "confidence": confidence,
    })


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
    # 0.0.0.0 by default so the service is reachable from sibling containers
    # (backend -> http://ai:5050) per the deployment guide. Override with
    # FACE_SERVICE_HOST=127.0.0.1 for local-only runs outside Docker.
    host = os.environ.get("FACE_SERVICE_HOST", "0.0.0.0")
    print(f"[face_service] Starting on http://{host}:{port}")
    print(f"[face_service] DeepFace available: {DEEPFACE_AVAILABLE}, model: {MODEL_NAME}")
    app.run(host=host, port=port, debug=False)
