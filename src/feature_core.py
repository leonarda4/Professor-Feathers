from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from storage import SampleRecord


@dataclass
class SampleFeatures:
    record: SampleRecord
    vector: np.ndarray


@dataclass
class SampleFeatureParts:
    record: SampleRecord
    mfcc_mean: np.ndarray
    mfcc_std: np.ndarray
    voiced_f0: np.ndarray
    voiced_ratio: float


@dataclass
class PcaResult:
    coordinates: np.ndarray
    explained_variance_ratio: np.ndarray


FEATURE_VARIANTS = ("f0_mean_std", "no_f0")


def frame_signal(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.zeros((0, frame_size), dtype=np.float32)
    frame_size = max(1, int(frame_size))
    hop_size = max(1, int(hop_size))
    if samples.size < frame_size:
        padded = np.zeros(frame_size, dtype=np.float32)
        padded[: samples.size] = samples
        return padded.reshape(1, -1)
    frame_count = 1 + int(math.ceil((samples.size - frame_size) / float(hop_size)))
    padded_length = frame_size + hop_size * max(0, frame_count - 1)
    padded = np.zeros(padded_length, dtype=np.float32)
    padded[: samples.size] = samples
    frames = []
    for start in range(0, padded_length - frame_size + 1, hop_size):
        frames.append(padded[start : start + frame_size].copy())
    return np.asarray(frames, dtype=np.float32)


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(mels: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mels) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 26,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    fmax = float(fmax or (sample_rate / 2.0))
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / float(sample_rate)).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left = int(np.clip(bins[index], 0, filters.shape[1] - 1))
        center = int(np.clip(bins[index + 1], left + 1, filters.shape[1] - 1))
        right = int(np.clip(bins[index + 2], center + 1, filters.shape[1]))
        if center > left:
            filters[index, left:center] = (
                np.arange(left, center, dtype=np.float32) - float(left)
            ) / float(center - left)
        if right > center:
            filters[index, center:right] = (
                float(right) - np.arange(center, right, dtype=np.float32)
            ) / float(right - center)
    return filters


def dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    basis = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    scale = math.sqrt(2.0 / float(n_mels))
    for coeff in range(n_mfcc):
        for mel_index in range(n_mels):
            basis[coeff, mel_index] = math.cos(
                math.pi * coeff * (mel_index + 0.5) / float(n_mels)
            )
    basis *= scale
    basis[0, :] *= math.sqrt(0.5)
    return basis


def estimate_f0_track(
    samples: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    f0_min: float = 80.0,
    f0_max: float = 400.0,
    min_periodicity: float = 0.30,
) -> np.ndarray:
    frames = frame_signal(samples, frame_size=frame_size, hop_size=hop_size)
    if frames.size == 0:
        return np.zeros(0, dtype=np.float32)
    window = np.hanning(frame_size).astype(np.float32)
    min_lag = max(1, int(sample_rate / float(f0_max)))
    max_lag = max(min_lag + 1, int(sample_rate / float(f0_min)))
    track: list[float] = []
    for frame in frames:
        centered = (frame - np.mean(frame, dtype=np.float64)).astype(np.float32, copy=False)
        weighted = centered * window
        energy = float(np.dot(weighted, weighted))
        if energy <= 1e-6:
            track.append(0.0)
            continue
        autocorr = np.correlate(weighted, weighted, mode="full")[frame_size - 1 :]
        autocorr /= max(float(autocorr[0]), 1e-12)
        upper = min(max_lag, autocorr.size)
        if upper <= min_lag:
            track.append(0.0)
            continue
        candidate = autocorr[min_lag:upper]
        peak_offset = int(np.argmax(candidate))
        peak_value = float(candidate[peak_offset])
        if peak_value < min_periodicity:
            track.append(0.0)
            continue
        lag = min_lag + peak_offset
        track.append(float(sample_rate) / float(lag))
    return np.asarray(track, dtype=np.float32)


def compute_mfcc(
    samples: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32)
    samples = samples - np.mean(samples, dtype=np.float64)
    emphasized = samples.copy()
    if emphasized.size > 1:
        emphasized[1:] = emphasized[1:] - 0.97 * emphasized[:-1]
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_size = max(1, int(sample_rate * hop_ms / 1000.0))
    frames = frame_signal(emphasized, frame_size=frame_size, hop_size=hop_size)
    if frames.size == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32)
    window = np.hamming(frame_size).astype(np.float32)
    n_fft = 1
    while n_fft < frame_size:
        n_fft *= 2
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2) / float(n_fft)
    filters = mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.maximum(power @ filters.T, 1e-10)
    log_mel = np.log(mel_energy)
    basis = dct_basis(n_mfcc=n_mfcc, n_mels=n_mels)
    return log_mel @ basis.T


def extract_feature_parts(
    record: SampleRecord,
    samples: np.ndarray,
    sample_rate: int,
) -> SampleFeatureParts:
    mfcc = compute_mfcc(samples, sample_rate=sample_rate)
    mfcc_mean = np.mean(mfcc, axis=0, dtype=np.float64) if mfcc.size else np.zeros(13, dtype=np.float64)
    mfcc_std = np.std(mfcc, axis=0, dtype=np.float64) if mfcc.size else np.zeros(13, dtype=np.float64)
    frame_size = max(1, int(sample_rate * 0.040))
    hop_size = max(1, int(sample_rate * 0.010))
    f0_track = estimate_f0_track(
        samples,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
    )
    voiced_f0 = f0_track[f0_track > 0.0]
    voiced_ratio = float(voiced_f0.size) / float(max(1, f0_track.size))
    return SampleFeatureParts(
        record=record,
        mfcc_mean=np.asarray(mfcc_mean, dtype=np.float32),
        mfcc_std=np.asarray(mfcc_std, dtype=np.float32),
        voiced_f0=np.asarray(voiced_f0, dtype=np.float32),
        voiced_ratio=voiced_ratio,
    )


def _default_record_for_summary(keyword: str = "keyword") -> SampleRecord:
    return SampleRecord(
        sample_id="sample",
        keyword=keyword,
        path="",
        sample_rate=0,
        duration_ms=0,
        timestamp="",
        session_id="",
        num_samples=0,
        status="accepted",
    )


def split_keyword_label(keyword: str) -> tuple[str, str]:
    stripped = keyword.strip()
    match = re.match(r"^(.*?)(?:[_-]?(\d+))?$", stripped)
    if not match:
        return stripped, "speaker1"
    base_keyword = (match.group(1) or stripped).rstrip("_-")
    suffix = match.group(2)
    if suffix is None:
        return base_keyword or stripped, "speaker1"
    return base_keyword, f"speaker{suffix}"


def build_feature_vector_from_parts(
    item: SampleFeatureParts,
    *,
    variant: str = "f0_mean_std",
) -> np.ndarray:
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unsupported feature variant: {variant}")

    base_vector = [
        item.mfcc_mean.astype(np.float32, copy=False),
        item.mfcc_std.astype(np.float32, copy=False),
        np.asarray([item.voiced_ratio], dtype=np.float32),
    ]
    if variant == "no_f0":
        return np.concatenate(base_vector).astype(np.float32)

    voiced_f0 = item.voiced_f0.astype(np.float64, copy=False)
    if voiced_f0.size > 0:
        mean_f0 = float(np.mean(voiced_f0, dtype=np.float64))
        std_f0 = float(np.std(voiced_f0, dtype=np.float64))
    else:
        mean_f0 = 0.0
        std_f0 = 0.0
    return np.concatenate(
        [
            item.mfcc_mean.astype(np.float32, copy=False),
            item.mfcc_std.astype(np.float32, copy=False),
            np.asarray([mean_f0, std_f0, item.voiced_ratio], dtype=np.float32),
        ]
    ).astype(np.float32)


def build_feature_matrix(
    sample_parts: list[SampleFeatureParts],
    *,
    variant: str = "f0_mean_std",
) -> tuple[np.ndarray, list[str]]:
    if not sample_parts:
        return np.zeros((0, 0), dtype=np.float32), []

    vectors = [
        build_feature_vector_from_parts(
            item,
            variant=variant,
        )
        for item in sample_parts
    ]
    labels = [item.record.keyword for item in sample_parts]
    return np.stack(vectors).astype(np.float32), labels


def build_sample_features(
    sample_parts: list[SampleFeatureParts],
    *,
    variant: str = "f0_mean_std",
) -> list[SampleFeatures]:
    return [
        SampleFeatures(
            record=item.record,
            vector=build_feature_vector_from_parts(
                item,
                variant=variant,
            ),
        )
        for item in sample_parts
    ]


def summarize_feature_vector(
    samples: np.ndarray,
    sample_rate: int,
    *,
    keyword: str = "keyword",
    variant: str = "f0_mean_std",
) -> np.ndarray:
    parts = extract_feature_parts(_default_record_for_summary(keyword=keyword), samples, sample_rate)
    return build_feature_vector_from_parts(parts, variant=variant)
