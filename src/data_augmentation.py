from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from audio import read_wav
from features.feature_core import SampleFeatureParts, extract_feature_parts
from storage import SampleRecord


@dataclass(frozen=True)
class AugmentationConfig:
    copies_per_sample: int = 3
    gain_min: float = 0.85
    gain_max: float = 1.15
    noise_probability: float = 0.8
    noise_std_min: float = 0.0015
    noise_std_max: float = 0.0100
    time_shift_ms: float = 60.0


@dataclass(frozen=True)
class AugmentedSample:
    record: SampleRecord
    samples: np.ndarray
    sample_rate: int


def _normalize_audio(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x = x / peak
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


def apply_gain(samples: np.ndarray, gain: float) -> np.ndarray:
    return _normalize_audio(np.asarray(samples, dtype=np.float32) * float(gain))


def add_white_noise(samples: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    noise = rng.normal(0.0, float(noise_std), size=x.shape).astype(np.float32)
    return _normalize_audio(x + noise)


def time_shift(samples: np.ndarray, shift_samples: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0 or shift_samples == 0:
        return x.astype(np.float32, copy=True)
    y = np.zeros_like(x, dtype=np.float32)
    if shift_samples > 0:
        shift = min(int(shift_samples), int(x.size))
        y[shift:] = x[:-shift]
    else:
        shift = min(int(-shift_samples), int(x.size))
        y[:-shift] = x[shift:]
    return y


def augment_waveform(
    samples: np.ndarray,
    sample_rate: int,
    *,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> np.ndarray:
    augmented = np.asarray(samples, dtype=np.float32).copy()
    gain = float(rng.uniform(config.gain_min, config.gain_max))
    augmented = apply_gain(augmented, gain)
    max_shift = int(round(float(config.time_shift_ms) * float(sample_rate) / 1000.0))
    if max_shift > 0:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        augmented = time_shift(augmented, shift)
    if float(rng.random()) < float(config.noise_probability):
        noise_std = float(rng.uniform(config.noise_std_min, config.noise_std_max))
        augmented = add_white_noise(augmented, noise_std, rng)
    return _normalize_audio(augmented)


def build_augmented_sample_record(record: SampleRecord, *, copy_index: int) -> SampleRecord:
    return SampleRecord(
        sample_id=f"{record.sample_id}_aug{copy_index:02d}",
        keyword=record.keyword,
        path=f"{record.path}#aug{copy_index:02d}",
        sample_rate=record.sample_rate,
        duration_ms=record.duration_ms,
        timestamp=record.timestamp,
        session_id=record.session_id,
        num_samples=record.num_samples,
        status=record.status,
    )


def build_augmented_feature_parts(
    sample_parts: Iterable[SampleFeatureParts],
    *,
    project_root: Path,
    config: AugmentationConfig | None = None,
    seed: int = 42,
    include_original: bool = True,
) -> list[SampleFeatureParts]:
    cfg = config or AugmentationConfig()
    rng = np.random.default_rng(int(seed))
    project_root = Path(project_root).resolve()
    output: list[SampleFeatureParts] = []

    for item in sample_parts:
        if include_original:
            output.append(item)
        sample_path = project_root / item.record.path
        if not sample_path.exists():
            continue
        samples, sample_rate = read_wav(sample_path)
        for copy_index in range(1, int(cfg.copies_per_sample) + 1):
            augmented_samples = augment_waveform(samples, sample_rate, rng=rng, config=cfg)
            augmented_record = build_augmented_sample_record(item.record, copy_index=copy_index)
            output.append(extract_feature_parts(augmented_record, augmented_samples, sample_rate))
    return output
