from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from features.feature_core import SampleFeatureParts, compute_mfcc


@dataclass
class OnlineFeatureNormalizer:
    mean: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    std: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    _n: int = field(default=0, init=False, repr=False)
    _M2: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64), repr=False)

    def fit(self, X: np.ndarray) -> "OnlineFeatureNormalizer":
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array.")
        self.mean = np.mean(X, axis=0, dtype=np.float64).astype(np.float32)
        std64 = np.std(X, axis=0, dtype=np.float64)
        std64[std64 < 1e-8] = 1.0
        self.std = std64.astype(np.float32)
        self._n = int(X.shape[0])
        self._M2 = (np.var(X, axis=0, dtype=np.float64) * X.shape[0]).astype(np.float64)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if self._n == 0:
            self.mean = x.astype(np.float32)
            self._M2 = np.zeros_like(x, dtype=np.float64)
            self._n = 1
            self.std = np.ones_like(self.mean, dtype=np.float32)
            return
        self._n += 1
        delta = x - self.mean.astype(np.float64)
        self.mean = (self.mean.astype(np.float64) + delta / self._n).astype(np.float32)
        delta2 = x - self.mean.astype(np.float64)
        self._M2 += delta * delta2
        variance = self._M2 / max(self._n - 1, 1)
        self.std = np.where(variance < 1e-16, 1.0, np.sqrt(variance)).astype(np.float32)


def compute_delta_mfcc(mfcc: np.ndarray, window: int = 2) -> np.ndarray:
    if mfcc.ndim != 2 or mfcc.shape[0] == 0:
        return np.zeros_like(mfcc)
    T, n = mfcc.shape
    denom = 2.0 * sum(t * t for t in range(1, window + 1))
    delta = np.zeros_like(mfcc, dtype=np.float32)
    for t in range(T):
        num = np.zeros(n, dtype=np.float32)
        for k in range(1, window + 1):
            num += k * (mfcc[min(t + k, T - 1)] - mfcc[max(t - k, 0)])
        delta[t] = num / max(denom, 1e-12)
    return delta


def compute_delta_mfcc_mean(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    mfcc = compute_mfcc(samples, sample_rate=sample_rate)
    delta = compute_delta_mfcc(mfcc)
    if delta.size == 0:
        return np.zeros(13, dtype=np.float32)
    return np.mean(delta, axis=0, dtype=np.float64).astype(np.float32)


def build_keyword_vector(parts: SampleFeatureParts) -> np.ndarray:
    return np.concatenate([
        parts.mfcc_mean.astype(np.float32),
        parts.mfcc_std.astype(np.float32),
    ]).astype(np.float32)


def build_person_keyword_vector(
    parts: SampleFeatureParts,
    *,
    delta_mfcc_mean: np.ndarray | None = None,
) -> np.ndarray:
    voiced = parts.voiced_f0.astype(np.float64)
    f0_mean = float(np.mean(voiced)) if voiced.size > 0 else 0.0
    f0_std = float(np.std(voiced)) if voiced.size > 0 else 0.0
    pieces = [
        parts.mfcc_mean.astype(np.float32),
        parts.mfcc_std.astype(np.float32),
        np.asarray([f0_mean, f0_std, parts.voiced_ratio], dtype=np.float32),
    ]
    if delta_mfcc_mean is not None:
        pieces.append(np.asarray(delta_mfcc_mean, dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32)


def build_keyword_matrix(sample_parts: list[SampleFeatureParts]) -> tuple[np.ndarray, list[str]]:
    if not sample_parts:
        return np.zeros((0, 0), dtype=np.float32), []
    vectors = [build_keyword_vector(item) for item in sample_parts]
    labels = [item.record.keyword for item in sample_parts]
    return np.stack(vectors).astype(np.float32), labels


def build_person_keyword_matrix(
    sample_parts: list[SampleFeatureParts],
    *,
    label_map: dict[str, str],
    delta_map: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if not sample_parts:
        return np.zeros((0, 0), dtype=np.float32), []
    vectors = []
    labels = []
    for item in sample_parts:
        delta = None if delta_map is None else delta_map.get(item.record.sample_id)
        vectors.append(build_person_keyword_vector(item, delta_mfcc_mean=delta))
        labels.append(label_map[item.record.sample_id])
    return np.stack(vectors).astype(np.float32), labels