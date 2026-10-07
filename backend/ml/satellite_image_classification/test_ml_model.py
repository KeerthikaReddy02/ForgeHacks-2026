import os
import torch
from PIL import Image
from torchvision import transforms, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "satellite_image_classification_model.pth")
IMAGE_PATH = os.path.join(BASE_DIR, "test_nowildfire.jpg")

# 1. Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18()  # Match your architecture
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

class_names = ['No wild fire', 'wildfire']

# 3. Transform & Predict
transform = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        idx = torch.argmax(probs).item()
        
    print(f"Result: {class_names[idx]} ({probs[idx].item() * 100:.1f}%)")

predict(IMAGE_PATH)