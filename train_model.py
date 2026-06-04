from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from feature_extractor import FEATURE_NAMES, extract_features_from_image, read_image, allowed_extensions

CLASS_NAMES = ["Ripe", "Unripe", "Old", "Damaged"]
BINARY_MAPPING = {"Ripe": "Good Quality", "Unripe": "Low Quality", "Old": "Low Quality", "Damaged": "Low Quality"}


def iter_images(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower().lstrip('.') in allowed_extensions():
            yield path


def load_dataset(dataset_root: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    X: List[np.ndarray] = []
    y: List[str] = []
    counts: Dict[str, int] = {}
    for label in CLASS_NAMES:
        class_dir = dataset_root / label
        if not class_dir.exists():
            counts[label] = 0
            continue
        count = 0
        for image_path in iter_images(class_dir):
            try:
                image = read_image(str(image_path))
                features, _ = extract_features_from_image(image)
                X.append(features)
                y.append(label)
                count += 1
            except Exception:
                continue
        counts[label] = count
    if not X:
        raise ValueError("No valid images were loaded from the dataset path.")
    return np.vstack(X), np.array(y), counts


def build_demo_dataset(samples_per_class: int = 120) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(42)
    # Approximate feature centroids for the four classes.
    centroids = {
        "Ripe": np.array([205, 70, 55, 26, 22, 20, 8, 180, 190, 5, 28, 25, 150, 0.42, 0.18, 0.83, 0.03, 0.56, 0.72, 0.87, 0.96, 0.84, 0.04, 0.09, 82, 0.38], dtype=np.float32),
        "Unripe": np.array([125, 105, 65, 22, 18, 18, 28, 120, 160, 7, 24, 22, 170, 0.37, 0.15, 0.76, 0.02, 0.54, 0.69, 0.80, 0.98, 0.80, 0.05, 0.10, 12, 0.98], dtype=np.float32),
        "Old": np.array([150, 78, 55, 34, 28, 24, 10, 90, 125, 9, 36, 35, 260, 0.28, 0.11, 0.70, 0.02, 0.49, 0.74, 0.75, 1.00, 0.76, 0.18, 0.17, 38, 0.62], dtype=np.float32),
        "Damaged": np.array([170, 72, 60, 36, 32, 28, 11, 130, 150, 11, 34, 32, 330, 0.24, 0.10, 0.68, 0.01, 0.45, 0.88, 0.58, 1.08, 0.69, 0.24, 0.25, 48, 0.70], dtype=np.float32),
    }
    X, y = [], []
    for label, centroid in centroids.items():
        noise = rng.normal(0, 1.0, size=(samples_per_class, centroid.shape[0])).astype(np.float32)
        scale = np.maximum(np.abs(centroid) * 0.08, 0.02).astype(np.float32)
        samples = centroid + noise * scale
        X.append(samples)
        y.extend([label] * samples_per_class)
    counts = {label: samples_per_class for label in CLASS_NAMES}
    return np.vstack(X), np.array(y), counts


def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    metrics = {
        "validation_accuracy": accuracy,
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
    }
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train tomato quality model from Kaggle-style folders.")
    parser.add_argument("--dataset", type=str, default="", help="Path to dataset root containing Ripe, Unripe, Old, Damaged folders.")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save the trained model and metadata.")
    parser.add_argument("--force-demo", action="store_true", help="Build a demo model using synthetic data.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "tomato_quality_model.joblib"
    metadata_path = model_dir / "metadata.json"

    demo = args.force_demo
    dataset_root = Path(args.dataset) if args.dataset else None

    if not demo and dataset_root and dataset_root.exists():
        X, y, counts = load_dataset(dataset_root)
        dataset_source = str(dataset_root.resolve())
    else:
        X, y, counts = build_demo_dataset()
        dataset_source = "demo_synthetic_dataset"
        demo = True

    model, metrics = train_model(X, y)
    joblib.dump(model, model_path)

    metadata = {
        "class_names": CLASS_NAMES,
        "binary_mapping": BINARY_MAPPING,
        "feature_names": FEATURE_NAMES,
        "dataset_source": dataset_source,
        "dataset_counts": counts,
        "demo_model": demo,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Model saved to:", model_path)
    print("Metadata saved to:", metadata_path)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
