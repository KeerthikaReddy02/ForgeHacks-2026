"""EfficientNet-B0 wildfire classifier with Grad-CAM, loaded lazily from models/vision.pt."""
import base64
import io
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from ..schemas import ImagePrediction
from .paths import VISION_MODEL

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _is_fire(name: str) -> bool:
    n = name.lower().replace("_", "").replace(" ", "")
    return "fire" in n and not n.startswith(("no", "non"))


class VisionEngine:
    def __init__(self, path=VISION_MODEL):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.classes = ck["classes"]
        size = ck["img_size"]
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, len(self.classes))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                      transforms.Normalize(MEAN, STD)])
        self._lock = threading.Lock()  # hooks are per-model state
        self._acts = self._grads = None
        layer = self.model.features[-1]
        layer.register_forward_hook(self._save_act)

    def _save_act(self, _m, _i, out):
        self._acts = out
        out.register_hook(lambda g: setattr(self, "_grads", g))

    def predict(self, data: bytes) -> ImagePrediction:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((512, 512))
        x = self.tf(img).unsqueeze(0)
        with self._lock, torch.enable_grad():
            logits = self.model(x)
            probs = logits.softmax(1)[0].detach()
            k = int(probs.argmax())
            self.model.zero_grad()
            logits[0, k].backward()
            w = self._grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * self._acts).sum(1, keepdim=True)).detach()
        cam = F.interpolate(cam, size=img.size[::-1], mode="bilinear", align_corners=False)[0, 0].numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        fire = _is_fire(self.classes[k])
        return ImagePrediction(
            label="Wildfire" if fire else "No Wildfire",
            confidence=round(float(probs[k]), 4),
            heatmap_png_b64=_overlay(img, cam),
            mock=False,
        )


def _overlay(img: Image.Image, cam: np.ndarray) -> str:
    heat = np.stack([np.clip(cam * 2, 0, 1), np.clip(cam * 2 - 1, 0, 1) * 0.8, np.zeros_like(cam)], -1) * 255
    alpha = cam[..., None] * 0.6
    out = np.asarray(img) * (1 - alpha) + heat * alpha
    buf = io.BytesIO()
    Image.fromarray(out.astype("uint8")).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
