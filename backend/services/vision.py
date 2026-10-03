"""Image wildfire classifier: real EfficientNet engine if models/vision.pt exists, else mock."""
import base64
import hashlib
import io

import logging

import numpy as np
from PIL import Image

from ..ml.paths import VISION_MODEL
from ..schemas import ImagePrediction

log = logging.getLogger(__name__)


def mock_predict_image(data: bytes) -> ImagePrediction:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((512, 512))
    seed = int(hashlib.md5(data).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    confidence = float(rng.uniform(0.6, 0.98))
    label = "Wildfire" if rng.random() > 0.5 else "No Wildfire"

    # fake Grad-CAM: a few gaussian blobs
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    cam = np.zeros((h, w))
    for _ in range(3):
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        s = rng.uniform(0.08, 0.2) * max(w, h)
        cam += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s**2)))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    heat = np.stack([cam, np.clip(cam * 2 - 1, 0, 1) * 0.8, np.zeros_like(cam)], -1) * 255
    alpha = (cam[..., None] * 0.6)
    out = np.asarray(img) * (1 - alpha) + heat * alpha
    buf = io.BytesIO()
    Image.fromarray(out.astype("uint8")).save(buf, format="PNG")

    return ImagePrediction(
        label=label,
        confidence=round(confidence, 3),
        heatmap_png_b64=base64.b64encode(buf.getvalue()).decode(),
    )


_engine = None
_engine_tried = False


def predict_image(data: bytes) -> ImagePrediction:
    global _engine, _engine_tried
    if not _engine_tried:
        _engine_tried = True
        if VISION_MODEL.exists():
            from ..ml.vision_engine import VisionEngine

            _engine = VisionEngine()
        else:
            log.warning("%s missing, using mock vision predictions", VISION_MODEL)
    return _engine.predict(data) if _engine else mock_predict_image(data)
