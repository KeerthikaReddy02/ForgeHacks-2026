"""Image wildfire classifier: real EfficientNet engine if models/vision.pt exists, else mock."""
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

    return ImagePrediction(
        label=label,
        confidence=round(confidence, 3),
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
