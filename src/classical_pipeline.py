"""Training and validation workflow for the classical orange classifier."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .classical_model import features_from_paths, make_classical_models
from .dataset import DEFAULT_CLASSES, discover_images, train_validation_split


def run_classical_pipeline(project_root: str | Path) -> pd.DataFrame:
    """Fit classical models, select on validation data, and save test features."""
    project_root = Path(project_root)
    data_root = project_root / "data" / "raw"
    train_root, test_root = data_root / "train", data_root / "test"
    missing = [
        f"{split.name}/{label}"
        for split in (train_root, test_root)
        for label in DEFAULT_CLASSES
        if not (split / label).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing orange dataset folders under {data_root}: {', '.join(missing)}"
        )

    train_records = discover_images(train_root)
    test_records = discover_images(test_root)
    train_indices, val_indices = train_validation_split(
        train_records, val_size=0.15, seed=42
    )
    train_labels = np.array([label for _, label in train_records])
    test_labels = np.array([label for _, label in test_records])
    train_features = features_from_paths([str(path) for path, _ in train_records])
    test_features = features_from_paths([str(path) for path, _ in test_records])

    results = []
    for name, model in make_classical_models().items():
        model.fit(train_features[train_indices], train_labels[train_indices])
        predictions = model.predict(train_features[val_indices])
        results.append(
            {
                "model": name,
                "val_accuracy": accuracy_score(
                    train_labels[val_indices], predictions
                ),
                "val_macro_f1": f1_score(
                    train_labels[val_indices], predictions, average="macro"
                ),
            }
        )

    results_frame = pd.DataFrame(results).sort_values(
        "val_macro_f1", ascending=False
    )
    best_name = results_frame.iloc[0]["model"]
    best_model = make_classical_models()[best_name]
    best_model.fit(train_features, train_labels)
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "features": test_features,
            "labels": test_labels,
            "class_names": list(DEFAULT_CLASSES),
        },
        processed_dir / "orange_test_features.joblib",
    )
    joblib.dump(
        {
            "model": best_model,
            "class_names": list(DEFAULT_CLASSES),
            "feature_version": "rgb-hsv-lbp-v1",
        },
        project_root / "models" / "classical_best.joblib",
    )
    print("Selected using validation macro F1:", best_name)
    print("Saved held-out test features and labels for notebook 04.")
    return results_frame
