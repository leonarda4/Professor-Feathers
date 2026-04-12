from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.svm import SVC

from features.feature_core import SampleFeatureParts
from features.feature_spaces import build_keyword_matrix, build_keyword_vector
from knn_utils import compute_accuracy, standardize_feature_matrices


@dataclass
class BaseSVMClassifier:
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str | float = "scale"
    probability: bool = False
    class_weight: str | dict[str, float] | None = None
    _model: SVC | None = field(default=None, init=False, repr=False)
    _scaler: dict[str, list[float]] = field(default_factory=dict)
    _train_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    _train_labels: list[str] = field(default_factory=list)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, parts: list[SampleFeatureParts]) -> "BaseSVMClassifier":
        raw, labels = build_keyword_matrix(parts)
        if raw.size == 0:
            raise ValueError("Need at least one keyword sample.")
        scaled, _, self._scaler = standardize_feature_matrices(raw, raw)
        self._model = SVC(
            kernel=self.kernel,
            C=float(self.C),
            gamma=self.gamma,
            probability=bool(self.probability),
            class_weight=self.class_weight,
        )
        self._model.fit(scaled, np.asarray(labels))
        self._train_vectors = scaled.astype(np.float32)
        self._train_labels = [str(x) for x in labels]
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
        labels = [str(x) for x in self._model.predict(scaled)]
        outputs: list[dict[str, Any]] = []
        for idx, label in enumerate(labels):
            row: dict[str, Any] = {"predicted_label": label}
            if hasattr(self._model, "decision_function"):
                decision = np.asarray(self._model.decision_function(scaled[idx : idx + 1]), dtype=np.float64).reshape(-1)
                row["decision_value"] = decision.tolist()
            if self.probability and hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(scaled[idx : idx + 1])[0]
                row["class_probabilities"] = {
                    str(cls): float(prob) for cls, prob in zip(self._model.classes_, probs)
                }
            outputs.append(row)
        return outputs[0] if single else outputs

    def decision_margin(self, parts: SampleFeatureParts) -> float:
        self._check_fitted()
        vec = self.transform_vector(parts).reshape(1, -1)
        decision = np.asarray(self._model.decision_function(vec), dtype=np.float64).reshape(-1)
        if decision.size == 0:
            return 0.0
        if decision.size == 1:
            return float(abs(decision[0]))
        top2 = np.sort(decision)[-2:]
        return float(top2[-1] - top2[-2])

    def evaluate(self, parts: list[SampleFeatureParts], true_labels: list[str]) -> float:
        results = self.predict(parts)
        predicted_labels = [item["predicted_label"] for item in results]
        return compute_accuracy(true_labels, predicted_labels)

    def _check_fitted(self) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit() before predict().")
