"""Classical image baseline using color statistics, HSV histograms, and LBP."""
from __future__ import annotations
import numpy as np
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import local_binary_pattern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .dataset import load_rgb


def extract_features(image, size=(128, 128), lbp_points=24, lbp_radius=3):
    image = np.asarray(image, dtype=np.float32)
    if image.max() > 1: image = image / 255.0
    from PIL import Image
    image = np.asarray(Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).resize(size)) / 255.0
    hsv = rgb2hsv(image); gray = rgb2gray(image)
    parts = [image.mean((0, 1)), image.std((0, 1))]
    for channel in range(3):
        hist, _ = np.histogram(hsv[:, :, channel], bins=16, range=(0, 1), density=True)
        parts.append(hist.astype(np.float32))
    lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=lbp_points + 2, range=(0, lbp_points + 2), density=True)
    parts.append(hist.astype(np.float32))
    return np.concatenate(parts).astype(np.float32)


def features_from_paths(paths):
    return np.stack([extract_features(load_rgb(path, size=(128, 128))) for path in paths])


def make_classical_models(seed=42):
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", class_weight="balanced")),
        "Random forest": RandomForestClassifier(n_estimators=400, class_weight="balanced",
            random_state=seed, n_jobs=-1, min_samples_leaf=2),
    }
