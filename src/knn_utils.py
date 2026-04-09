from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def standardize_feature_matrices(
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    mean = np.mean(train_vectors, axis=0, dtype=np.float64)
    std = np.std(train_vectors, axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    train_scaled = ((train_vectors - mean) / std).astype(np.float32)
    test_scaled = ((test_vectors - mean) / std).astype(np.float32)
    scaler = {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }
    return train_scaled, test_scaled, scaler


def choose_knn_label(neighbor_labels: list[str], neighbor_distances: list[float]) -> str:
    label_counts = Counter(neighbor_labels)
    max_count = max(label_counts.values())
    candidates = [label for label, count in label_counts.items() if count == max_count]
    if len(candidates) == 1:
        return candidates[0]

    distance_by_label: dict[str, float] = {}
    for label in candidates:
        total_distance = sum(
            distance
            for candidate_label, distance in zip(neighbor_labels, neighbor_distances)
            if candidate_label == label
        )
        distance_by_label[label] = total_distance
    return sorted(candidates, key=lambda label: (distance_by_label[label], label))[0]


def knn_predict(
    train_vectors: np.ndarray,
    train_labels: list[str],
    test_vectors: np.ndarray,
    *,
    k: int,
) -> list[dict[str, Any]]:
    if train_vectors.size == 0:
        raise ValueError("Training vectors are empty.")
    effective_k = max(1, min(int(k), len(train_labels)))
    predictions: list[dict[str, Any]] = []

    for test_vector in test_vectors:
        distances = np.linalg.norm(train_vectors - test_vector, axis=1)
        neighbor_indices = np.argsort(distances)[:effective_k]
        neighbor_labels = [train_labels[index] for index in neighbor_indices]
        neighbor_distances = [float(distances[index]) for index in neighbor_indices]
        predicted_label = choose_knn_label(neighbor_labels, neighbor_distances)
        predictions.append(
            {
                "predicted_label": predicted_label,
                "neighbor_labels": neighbor_labels,
                "neighbor_distances": neighbor_distances,
            }
        )
    return predictions


def compute_accuracy(true_labels: list[str], predicted_labels: list[str]) -> float:
    if not true_labels:
        return 0.0
    correct = sum(
        1
        for true_label, predicted_label in zip(true_labels, predicted_labels)
        if true_label == predicted_label
    )
    return float(correct) / float(len(true_labels))
