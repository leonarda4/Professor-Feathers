from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from audio import read_wav
from config import load_app_config
from storage import SampleRecord, read_manifest_records


@dataclass
class SampleFeatures:
    record: SampleRecord
    vector: np.ndarray


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
    fmax: Optional[float] = None,
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


def estimate_f0_contour(
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
    contour: list[float] = []
    for frame in frames:
        centered = (frame - np.mean(frame, dtype=np.float64)).astype(np.float32, copy=False)
        weighted = centered * window
        energy = float(np.dot(weighted, weighted))
        if energy <= 1e-6:
            contour.append(0.0)
            continue
        autocorr = np.correlate(weighted, weighted, mode="full")[frame_size - 1 :]
        autocorr /= max(float(autocorr[0]), 1e-12)
        upper = min(max_lag, autocorr.size)
        if upper <= min_lag:
            contour.append(0.0)
            continue
        candidate = autocorr[min_lag:upper]
        peak_offset = int(np.argmax(candidate))
        peak_value = float(candidate[peak_offset])
        if peak_value < min_periodicity:
            contour.append(0.0)
            continue
        lag = min_lag + peak_offset
        contour.append(float(sample_rate) / float(lag))
    return np.asarray(contour, dtype=np.float32)


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


def summarize_feature_vector(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    mfcc = compute_mfcc(samples, sample_rate=sample_rate)
    frame_size = max(1, int(sample_rate * 0.040))
    hop_size = max(1, int(sample_rate * 0.010))
    f0_contour = estimate_f0_contour(
        samples,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
    )
    mfcc_mean = np.mean(mfcc, axis=0, dtype=np.float64) if mfcc.size else np.zeros(13, dtype=np.float64)
    mfcc_std = np.std(mfcc, axis=0, dtype=np.float64) if mfcc.size else np.zeros(13, dtype=np.float64)
    voiced_f0 = f0_contour[f0_contour > 0.0]
    mean_f0 = float(np.mean(voiced_f0, dtype=np.float64)) if voiced_f0.size else 0.0
    std_f0 = float(np.std(voiced_f0, dtype=np.float64)) if voiced_f0.size else 0.0
    voiced_ratio = float(voiced_f0.size) / float(max(1, f0_contour.size))
    return np.concatenate(
        [
            np.asarray(mfcc_mean, dtype=np.float32),
            np.asarray(mfcc_std, dtype=np.float32),
            np.asarray([mean_f0, std_f0, voiced_ratio], dtype=np.float32),
        ]
    ).astype(np.float32)


def load_sample_features(project_root: Path, manifest_path: Path, keyword: Optional[str] = None) -> list[SampleFeatures]:
    project_root = Path(project_root).resolve()
    records = read_manifest_records(manifest_path, keyword=keyword)
    features: list[SampleFeatures] = []
    for record in records:
        sample_path = project_root / record.path
        if not sample_path.exists():
            continue
        samples, sample_rate = read_wav(sample_path)
        vector = summarize_feature_vector(samples, sample_rate=sample_rate)
        features.append(SampleFeatures(record=record, vector=vector))
    return features


def compute_feature_space(feature_vectors: np.ndarray, n_dims: int = 2) -> np.ndarray:
    vectors = np.asarray(feature_vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("feature_vectors must be a 2D array.")
    if vectors.shape[0] == 0:
        return np.zeros((0, n_dims), dtype=np.float32)
    centered = vectors - np.mean(vectors, axis=0, keepdims=True, dtype=np.float64)
    scale = np.std(centered, axis=0, keepdims=True, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    normalized = centered / scale
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    components = vt[:n_dims].T
    return (normalized @ components).astype(np.float32)


def plot_feature_space(
    sample_features: list[SampleFeatures],
    coordinates: np.ndarray,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
    annotate: bool = False,
) -> Path | None:
    if not sample_features:
        raise ValueError("No samples available to plot.")
    keywords = [item.record.keyword for item in sample_features]
    unique_keywords = sorted(set(keywords))
    figure, axis = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")
    for index, keyword in enumerate(unique_keywords):
        mask = [item.record.keyword == keyword for item in sample_features]
        xs = coordinates[mask, 0]
        ys = coordinates[mask, 1]
        axis.scatter(xs, ys, label=keyword, s=70, alpha=0.85, color=cmap(index % 10))
    if annotate:
        for point, item in zip(coordinates, sample_features):
            axis.annotate(item.record.sample_id, (point[0], point[1]), fontsize=8, alpha=0.7)
    axis.set_title("MFCC + F0 Feature Space")
    axis.set_xlabel("Principal Component 1")
    axis.set_ylabel("Principal Component 2")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    saved_path: Path | None = None
    if output_path is not None:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
        saved_path = output_path
    if show:
        plt.show()
    else:
        plt.close(figure)
    return saved_path


def build_default_output_path(project_root: Path) -> Path:
    return (Path(project_root).resolve() / "data" / "features" / "feature_space.png").resolve()


def build_feature_plot(
    *,
    project_root: Path,
    config_path: Optional[str] = None,
    keyword: Optional[str] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
    annotate: bool = False,
) -> Path | None:
    config = load_app_config(config_path=config_path)
    project_root = Path(project_root).resolve()
    manifest_path = (project_root / config.storage.manifest_path).resolve()
    sample_features = load_sample_features(project_root, manifest_path, keyword=keyword)
    if not sample_features:
        raise RuntimeError("No saved samples found for feature extraction.")
    vectors = np.vstack([item.vector for item in sample_features]).astype(np.float32)
    coordinates = compute_feature_space(vectors)
    return plot_feature_space(
        sample_features,
        coordinates,
        output_path=output_path,
        show=show,
        annotate=annotate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract MFCC + F0 features and plot a 2D feature space.")
    parser.add_argument("--project-root", default=".", help="Project root containing src/ and data/.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    parser.add_argument("--keyword", default=None, help="Optional keyword filter.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. Defaults to data/features/feature_space.png.",
    )
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a window.")
    parser.add_argument("--annotate", action="store_true", help="Label points with sample IDs.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output).resolve() if args.output else build_default_output_path(project_root)
    saved_path = build_feature_plot(
        project_root=project_root,
        config_path=args.config,
        keyword=args.keyword,
        output_path=output_path,
        show=not args.no_show,
        annotate=args.annotate,
    )
    if saved_path is not None:
        print(f"Saved feature space plot to {saved_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
