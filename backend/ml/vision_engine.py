"""EfficientNet-B0 wildfire classifier loaded lazily from models/vision.pt."""
import io

import torch
from PIL import Image, ImageFile
from torchvision import models, transforms

from ..schemas import ImagePrediction
from .paths import VISION_MODEL

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few dataset/user images are cut off

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _is_fire(name: str) -> bool:
    n = name.lower().replace("_", "").replace(" ", "")
    return "fire" in n and not n.startswith(("no", "non"))


class VisionEngine:
    """Loads either checkpoint format:
    - {"state_dict", "classes", "img_size", "arch"?}  (EfficientNet-B0 default)
    - a bare ResNet-18 state dict (the notebook's satellite_image_classification_model.pth, 350px)
    """

    def __init__(self, path=VISION_MODEL):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        mean, std = MEAN, STD
        if "state_dict" in ck:
            arch, self.classes, size, state = ck.get("arch", "efficientnet_b0"), ck["classes"], ck["img_size"], ck["state_dict"]
        else:
            # the notebook trained on raw ToTensor() pixels (no ImageNet normalisation)
            arch, self.classes, size, state = "resnet18", ["nowildfire", "wildfire"], 350, ck
            mean, std = [0.0] * 3, [1.0] * 3
        if arch == "resnet18":
            self.model = models.resnet18(weights=None)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        else:
            self.model = models.efficientnet_b0(weights=None)
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, len(self.classes))
        self.arch = arch
        self.model.load_state_dict(state)
        self.model.eval()
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])

    def predict(self, data: bytes) -> ImagePrediction:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        x = self.tf(img).unsqueeze(0)
        with torch.no_grad():
            probs = self.model(x).softmax(1)[0]
        k = int(probs.argmax())

        fire = _is_fire(self.classes[k])
        return ImagePrediction(
            label="Wildfire" if fire else "No Wildfire",
            confidence=round(float(probs[k]), 4),
            mock=False,
        )

