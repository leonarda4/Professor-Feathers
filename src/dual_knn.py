from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from features.feature_core import SampleFeatureParts
from features.feature_spaces import (
    OnlineFeatureNormalizer,
    build_keyword_matrix,
    build_keyword_vector,
    build_person_keyword_matrix,
    build_person_keyword_vector,
)
from knn_utils import compute_accuracy, knn_predict, standardize_feature_matrices


def _mean_k_neighbor_distance(
    train_vectors_scaled: np.ndarray,
    query_vector_scaled: np.ndarray,
    *,
    k: int,
) -> float:
    distances = np.linalg.norm(train_vectors_scaled - query_vector_scaled.reshape(1, -1), axis=1)
    effective_k = max(1, min(int(k), int(distances.size)))
    nearest_distances = np.sort(distances)[:effective_k]
    return float(np.mean(nearest_distances, dtype=np.float64))


@dataclass
class KeywordClassifier:
    k: int = 5
    _train_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    _train_labels: list[str] = field(default_factory=list)
    _scaler: dict[str, list[float]] = field(default_factory=dict)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, parts: list[SampleFeatureParts]) -> "KeywordClassifier":
        raw, labels = build_keyword_matrix(parts)
        if raw.size == 0:
            raise ValueError("Need at least one keyword sample.")
        scaled, _, self._scaler = standardize_feature_matrices(raw, raw)
        self._train_vectors = scaled
        self._train_labels = labels
        self._fitted = True
        return self

    def transform_vector(self, parts: SampleFeatureParts) -> np.ndarray:
        self._check_fitted()
        raw = build_keyword_vector(parts)
        mean = np.asarray(self._scaler["mean"], dtype=np.float32)
        std = np.asarray(self._scaler["std"], dtype=np.float32)
        return ((raw - mean) / std).astype(np.float32)

    def predict(self, parts: SampleFeatureParts | list[SampleFeatureParts]) -> dict[str, Any] | list[dict[str, Any]]:
        self._check_fitted()
        single = isinstance(parts, SampleFeatureParts)
        items = [parts] if single else parts
        scaled = np.stack([self.transform_vector(item) for item in items]).astype(np.float32)
        results = knn_predict(self._train_vectors, self._train_labels, scaled, k=self.k)
        return results[0] if single else results

    def mean_neighbor_distance(self, parts: SampleFeatureParts) -> float:
        vec = self.transform_vector(parts)
        return _mean_k_neighbor_distance(self._train_vectors, vec, k=self.k)

    def evaluate(self, parts: list[SampleFeatureParts], true_labels: list[str]) -> float:
        results = self.predict(parts)
        preds = [item["predicted_label"] for item in results]
        return compute_accuracy(true_labels, preds)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")


@dataclass
class PersonKeywordClassifier:
    target_label: str = "target"
    negative_label: str = "other"
    k: int = 3
    _train_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    _train_labels: list[str] = field(default_factory=list)
    _normalizer: OnlineFeatureNormalizer = field(default_factory=OnlineFeatureNormalizer)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        sample_parts: list[SampleFeatureParts],
        *,
        label_map: dict[str, str],
        delta_map: dict[str, np.ndarray] | None = None,
    ) -> "PersonKeywordClassifier":
        raw, labels = build_person_keyword_matrix(sample_parts, label_map=label_map, delta_map=delta_map)
        if raw.size == 0:
            raise ValueError("Need at least one person-keyword sample.")
        self._normalizer.fit(raw)
        self._train_vectors = np.stack([self._normalizer.transform(v) for v in raw]).astype(np.float32)
        self._train_labels = labels
        self._fitted = True
        return self

    def transform_vector(
        self,
        parts: SampleFeatureParts,
        *,
        delta_mfcc_mean: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_fitted()
        raw = build_person_keyword_vector(parts, delta_mfcc_mean=delta_mfcc_mean)
        return self._normalizer.transform(raw)

    def predict(
        self,
        parts: SampleFeatureParts | list[SampleFeatureParts],
        *,
        delta_mfcc_mean: np.ndarray | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        self._check_fitted()
        single = isinstance(parts, SampleFeatureParts)
        items = [parts] if single else parts
        scaled = np.stack([
            self.transform_vector(item, delta_mfcc_mean=delta_mfcc_mean) for item in items
        ]).astype(np.float32)
        results = knn_predict(self._train_vectors, self._train_labels, scaled, k=self.k)
        return results[0] if single else results

    def mean_neighbor_distance(
        self,
        parts: SampleFeatureParts,
        *,
        delta_mfcc_mean: np.ndarray | None = None,
    ) -> float:
        vec = self.transform_vector(parts, delta_mfcc_mean=delta_mfcc_mean)
        return _mean_k_neighbor_distance(self._train_vectors, vec, k=self.k)

    def add_sample(
        self,
        parts: SampleFeatureParts,
        label: str,
        *,
        delta_mfcc_mean: np.ndarray | None = None,
    ) -> None:
        raw = build_person_keyword_vector(parts, delta_mfcc_mean=delta_mfcc_mean)
        self._normalizer.update(raw)
        scaled = self._normalizer.transform(raw).reshape(1, -1)
        if not self._fitted or self._train_vectors.size == 0:
            self._train_vectors = scaled.astype(np.float32)
            self._train_labels = [label]
            self._fitted = True
        else:
            self._train_vectors = np.vstack([self._train_vectors, scaled]).astype(np.float32)
            self._train_labels.append(label)

    def evaluate(
        self,
        sample_parts: list[SampleFeatureParts],
        true_labels: list[str],
        *,
        delta_map: dict[str, np.ndarray] | None = None,
    ) -> float:
        self._check_fitted()
        results = []
        for item in sample_parts:
            delta = None if delta_map is None else delta_map.get(item.record.sample_id)
            results.append(self.predict(item, delta_mfcc_mean=delta))
        preds = [item["predicted_label"] for item in results]
        return compute_accuracy(true_labels, preds)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
