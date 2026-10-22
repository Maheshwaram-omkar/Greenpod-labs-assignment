"""MobileNetV2 transfer-learning classifier and training/evaluation helpers."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def build_model(num_classes=4, pretrained=True):
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def train_model(model, train_loader, val_loader, epochs=8, lr=1e-3, device=None,
                save_path="models/mobilenetv2_best.pth"):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    # Freeze pretrained convolutional features for a stable, modest-compute baseline.
    for p in model.features.parameters(): p.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    best_loss = float("inf"); history = []
    for epoch in range(epochs):
        model.train(); train_loss = 0.0; correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y); loss.backward(); optimizer.step()
            train_loss += loss.item() * y.size(0); correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
        model.eval(); val_loss = 0.0; val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device); logits = model(x)
                val_loss += criterion(logits, y).item() * y.size(0)
                val_correct += (logits.argmax(1) == y).sum().item(); val_total += y.size(0)
        row = {"epoch": epoch + 1, "train_loss": train_loss / total, "train_accuracy": correct / total,
               "val_loss": val_loss / val_total, "val_accuracy": val_correct / val_total}
        history.append(row); print(row)
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]; path = Path(save_path); path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "num_classes": model.classifier[1].out_features}, path)
    return model, history


def load_model(checkpoint, num_classes=4, device="cpu"):
    model = build_model(num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state.get("state_dict", state)); model.to(device).eval()
    return model
