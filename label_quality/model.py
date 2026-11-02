"""Embed images with a pretrained network and fit a simple classifier on top."""

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.models import ResNet18_Weights, resnet18


def embed(paths: list, batch_size: int = 32) -> np.ndarray:
    weights = ResNet18_Weights.IMAGENET1K_V1
    network = resnet18(weights=weights)
    network.fc = torch.nn.Identity()
    network.eval()
    preprocess = weights.transforms()

    vectors = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    batch.append(preprocess(image.convert("RGB")))
            vectors.append(network(torch.stack(batch)).numpy())
    return np.concatenate(vectors)


def classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=1.0)


def out_of_fold_probabilities(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Predict each image with a model that never saw it or its duplicates."""
    probabilities = np.zeros(len(y))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in folds.split(x, y, groups):
        probabilities[test] = classifier().fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    return probabilities


def group_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple:
    """Hold out about 20%, keeping every duplicate group on one side."""
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train, test = next(folds.split(np.zeros(len(y)), y, groups))
    return train, test


def evaluate(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    predicted = classifier().fit(x[train], y[train]).predict(x[test])
    return {
        "accuracy": float(accuracy_score(y[test], predicted)),
        "f1": float(f1_score(y[test], predicted)),
        "test_images": int(len(test)),
    }
