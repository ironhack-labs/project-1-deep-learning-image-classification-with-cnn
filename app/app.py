import os
from pathlib import Path
from PIL import Image

from flask import Flask, request, render_template
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.config import Config
from src.model_tl import build_resnet18

# CIFAR-10 classes
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ImageNet normalization (used for transfer learning model)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = Path("app/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Preprocessing for ResNet18
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_model():
    cfg = Config()
    device = torch.device(cfg.device)

    model = build_resnet18(num_classes=cfg.num_classes, freeze_backbone=False).to(device)

    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_best.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device, ckpt_path

# Load model once at startup
model, device, ckpt_path = load_model()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", results=None, ckpt_path=ckpt_path)

@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("images")
    results = []

    if not files:
        return render_template("index.html", results=[], ckpt_path=ckpt_path)

    for file in files:
        if file.filename == "":
            continue

        save_path = UPLOAD_FOLDER / file.filename
        file.save(save_path)

        img = Image.open(save_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu()

        top_probs, top_indices = torch.topk(probs, 5)

        top5 = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            top5.append((CIFAR10_CLASSES[idx], prob))

        all_probs = [(CIFAR10_CLASSES[i], float(probs[i])) for i in range(len(CIFAR10_CLASSES))]
        all_probs = sorted(all_probs, key=lambda x: x[1], reverse=True)

        results.append({
            "filename": file.filename,
            "top5": top5,
            "all_probs": all_probs
        })

    return render_template("index.html", results=results, ckpt_path=ckpt_path)

if __name__ == "__main__":
    app.run(debug=True)