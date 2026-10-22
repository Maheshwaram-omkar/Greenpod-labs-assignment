"""Predict an orange image class for one image."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from src.dataset import DEFAULT_CLASSES
from src.dl_model import load_model


def predict(image_path, model, device, classes=DEFAULT_CLASSES):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with Image.open(image_path) as image:
        x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0]
    label = int(probabilities.argmax())
    return classes[label], float(probabilities[label]), {c: float(p) for c, p in zip(classes, probabilities)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", default="models/mobilenetv2_best.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); model = load_model(args.checkpoint, device=args.device)
    label, confidence, scores = predict(args.image, model, args.device)
    print(f"Prediction: {label} (softmax score: {confidence:.3f})")
    print("Class scores:", ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    print("Scores are model confidence indicators, not calibrated probabilities.")


if __name__ == "__main__": main()
