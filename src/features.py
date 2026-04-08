from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from audio import read_wav
from config import load_app_config
from storage import SampleRecord, read_manifest_records, relative_to_root


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


@dataclass
class LdaResult:
    coordinates: np.ndarray
    explained_discriminant_ratio: np.ndarray


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
    f0_contour = estimate_f0_contour(
        samples,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
    )
    voiced_f0 = f0_contour[f0_contour > 0.0]
    voiced_ratio = float(voiced_f0.size) / float(max(1, f0_contour.size))
    return SampleFeatureParts(
        record=record,
        mfcc_mean=np.asarray(mfcc_mean, dtype=np.float32),
        mfcc_std=np.asarray(mfcc_std, dtype=np.float32),
        voiced_f0=np.asarray(voiced_f0, dtype=np.float32),
        voiced_ratio=voiced_ratio,
    )


def _speaker_from_keyword(keyword: str) -> str:
    return split_keyword_label(keyword)[1]


def _base_keyword_from_keyword(keyword: str) -> str:
    return split_keyword_label(keyword)[0]


def _compute_speaker_f0_stats(
    sample_parts: list[SampleFeatureParts],
) -> dict[str, tuple[float, float]]:
    grouped_voiced_f0: dict[str, list[np.ndarray]] = {}
    for item in sample_parts:
        speaker = _speaker_from_keyword(item.record.keyword)
        if item.voiced_f0.size == 0:
            continue
        grouped_voiced_f0.setdefault(speaker, []).append(item.voiced_f0.astype(np.float64, copy=False))

    speaker_stats: dict[str, tuple[float, float]] = {}
    for speaker, values in grouped_voiced_f0.items():
        concatenated = np.concatenate(values) if values else np.zeros(0, dtype=np.float64)
        if concatenated.size == 0:
            speaker_stats[speaker] = (0.0, 1.0)
            continue
        mean_f0 = float(np.mean(concatenated, dtype=np.float64))
        std_f0 = float(np.std(concatenated, dtype=np.float64))
        speaker_stats[speaker] = (mean_f0, std_f0 if std_f0 > 1e-6 else 1.0)
    return speaker_stats


def _vectorize_feature_parts(
    item: SampleFeatureParts,
    speaker_f0_stats: dict[str, tuple[float, float]],
) -> np.ndarray:
    speaker = _speaker_from_keyword(item.record.keyword)
    speaker_mean_f0, speaker_std_f0 = speaker_f0_stats.get(speaker, (0.0, 1.0))
    if item.voiced_f0.size > 0:
        normalized_f0 = (item.voiced_f0.astype(np.float64) - speaker_mean_f0) / max(speaker_std_f0, 1e-6)
        mean_f0 = float(np.mean(normalized_f0, dtype=np.float64))
        std_f0 = float(np.std(normalized_f0, dtype=np.float64))
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


def build_sample_features(sample_parts: list[SampleFeatureParts]) -> list[SampleFeatures]:
    speaker_f0_stats = _compute_speaker_f0_stats(sample_parts)
    return [
        SampleFeatures(record=item.record, vector=_vectorize_feature_parts(item, speaker_f0_stats))
        for item in sample_parts
    ]


def load_sample_feature_parts(
    project_root: Path,
    manifest_path: Path,
    keyword: Optional[str] = None,
) -> list[SampleFeatureParts]:
    project_root = Path(project_root).resolve()
    records = read_manifest_records(manifest_path, keyword=keyword)
    sample_parts: list[SampleFeatureParts] = []
    for record in records:
        sample_path = project_root / record.path
        if not sample_path.exists():
            continue
        samples, sample_rate = read_wav(sample_path)
        sample_parts.append(extract_feature_parts(record, samples, sample_rate=sample_rate))
    return sample_parts


def load_sample_features(
    project_root: Path,
    manifest_path: Path,
    keyword: Optional[str] = None,
) -> list[SampleFeatures]:
    return build_sample_features(load_sample_feature_parts(project_root, manifest_path, keyword=keyword))


def _build_scanned_record(
    *,
    project_root: Path,
    samples_root: Path,
    sample_path: Path,
    sample_rate: int,
    num_samples: int,
) -> SampleRecord:
    relative_path = sample_path.resolve().relative_to(samples_root.resolve())
    parts = relative_path.parts
    if len(parts) >= 2:
        keyword = parts[0].strip()
    else:
        keyword = sample_path.parent.name.strip() or "keyword"
    if len(parts) >= 3:
        session_id = parts[1].strip() or "manual"
    else:
        session_id = "manual"
    try:
        record_path = relative_to_root(project_root, sample_path)
    except ValueError:
        record_path = str(sample_path.resolve())
    return SampleRecord(
        sample_id=sample_path.stem,
        keyword=keyword,
        path=record_path,
        sample_rate=int(sample_rate),
        duration_ms=int(round((num_samples / float(sample_rate)) * 1000.0)) if sample_rate > 0 else 0,
        timestamp="folder-scan",
        session_id=session_id,
        num_samples=int(num_samples),
        status="accepted",
    )


def load_sample_feature_parts_from_root(
    project_root: Path,
    samples_root: Path,
    keyword: Optional[str] = None,
) -> list[SampleFeatureParts]:
    project_root = Path(project_root).resolve()
    samples_root = Path(samples_root).resolve()
    if not samples_root.exists():
        raise FileNotFoundError(f"Sample root does not exist: {samples_root}")
    wav_paths = sorted(
        path
        for path in samples_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    sample_parts: list[SampleFeatureParts] = []
    for sample_path in wav_paths:
        samples, sample_rate = read_wav(sample_path)
        record = _build_scanned_record(
            project_root=project_root,
            samples_root=samples_root,
            sample_path=sample_path,
            sample_rate=sample_rate,
            num_samples=int(np.asarray(samples).size),
        )
        if keyword and record.keyword != keyword:
            continue
        sample_parts.append(extract_feature_parts(record, samples, sample_rate=sample_rate))
    return sample_parts


def load_sample_features_from_root(
    project_root: Path,
    samples_root: Path,
    keyword: Optional[str] = None,
) -> list[SampleFeatures]:
    return build_sample_features(load_sample_feature_parts_from_root(project_root, samples_root, keyword=keyword))


def split_keyword_label(keyword: str) -> tuple[str, str]:
    match = re.match(r"^(.*?)(\d+)?$", keyword.strip())
    if not match:
        return keyword, "speaker1"
    base_keyword = match.group(1) or keyword
    suffix = match.group(2)
    if suffix is None:
        return base_keyword, "speaker1"
    return base_keyword, f"speaker{suffix}"


def _standardize_feature_vectors(feature_vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(feature_vectors, dtype=np.float32)
    centered = vectors - np.mean(vectors, axis=0, keepdims=True, dtype=np.float64)
    scale = np.std(centered, axis=0, keepdims=True, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return centered / scale


def compute_pca(feature_vectors: np.ndarray, n_dims: int = 3) -> PcaResult:
    vectors = np.asarray(feature_vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("feature_vectors must be a 2D array.")
    if vectors.shape[0] == 0:
        return PcaResult(
            coordinates=np.zeros((0, n_dims), dtype=np.float32),
            explained_variance_ratio=np.zeros(0, dtype=np.float32),
        )
    normalized = _standardize_feature_vectors(vectors)
    _, singular_values, vt = np.linalg.svd(normalized, full_matrices=False)
    actual_dims = min(n_dims, vt.shape[0])
    components = vt[:actual_dims].T
    variances = (singular_values ** 2) / max(1, normalized.shape[0] - 1)
    variance_ratio = variances / max(float(np.sum(variances)), 1e-12)
    coordinates = (normalized @ components).astype(np.float32)
    if actual_dims < n_dims:
        padded = np.zeros((coordinates.shape[0], n_dims), dtype=np.float32)
        padded[:, :actual_dims] = coordinates
        coordinates = padded
    return PcaResult(
        coordinates=coordinates,
        explained_variance_ratio=np.asarray(variance_ratio, dtype=np.float32),
    )


def compute_lda(
    feature_vectors: np.ndarray,
    labels: list[str] | np.ndarray,
    n_dims: int = 3,
) -> LdaResult:
    vectors = np.asarray(feature_vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("feature_vectors must be a 2D array.")
    if vectors.shape[0] == 0:
        return LdaResult(
            coordinates=np.zeros((0, n_dims), dtype=np.float32),
            explained_discriminant_ratio=np.zeros(0, dtype=np.float32),
        )

    labels_array = np.asarray(labels)
    if labels_array.shape[0] != vectors.shape[0]:
        raise ValueError("labels must have the same number of rows as feature_vectors.")

    unique_labels = [str(label) for label in sorted(set(labels_array.tolist()))]
    max_discriminants = max(0, min(vectors.shape[1], len(unique_labels) - 1))
    if max_discriminants == 0:
        return LdaResult(
            coordinates=np.zeros((vectors.shape[0], n_dims), dtype=np.float32),
            explained_discriminant_ratio=np.zeros(0, dtype=np.float32),
        )

    normalized = _standardize_feature_vectors(vectors)
    overall_mean = np.mean(normalized, axis=0, dtype=np.float64)
    feature_dim = normalized.shape[1]
    within_scatter = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    between_scatter = np.zeros((feature_dim, feature_dim), dtype=np.float64)

    for label in unique_labels:
        class_vectors = normalized[labels_array == label]
        if class_vectors.size == 0:
            continue
        class_mean = np.mean(class_vectors, axis=0, dtype=np.float64)
        centered = class_vectors - class_mean
        within_scatter += centered.T @ centered
        mean_delta = (class_mean - overall_mean).reshape(-1, 1)
        between_scatter += class_vectors.shape[0] * (mean_delta @ mean_delta.T)

    regularized_within = within_scatter + (1e-6 * np.eye(feature_dim, dtype=np.float64))
    matrix = np.linalg.pinv(regularized_within) @ between_scatter
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 1e-8
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]

    actual_dims = min(n_dims, eigenvectors.shape[1], max_discriminants)
    if actual_dims == 0:
        return LdaResult(
            coordinates=np.zeros((vectors.shape[0], n_dims), dtype=np.float32),
            explained_discriminant_ratio=np.zeros(0, dtype=np.float32),
        )

    components = eigenvectors[:, :actual_dims]
    coordinates = (normalized @ components).astype(np.float32)
    if actual_dims < n_dims:
        padded = np.zeros((coordinates.shape[0], n_dims), dtype=np.float32)
        padded[:, :actual_dims] = coordinates
        coordinates = padded

    discriminant_ratio = eigenvalues / max(float(np.sum(eigenvalues)), 1e-12)
    return LdaResult(
        coordinates=coordinates,
        explained_discriminant_ratio=np.asarray(discriminant_ratio, dtype=np.float32),
    )


def _pc_label(index: int, variance_ratio: np.ndarray) -> str:
    if index < variance_ratio.size:
        return f"PC{index + 1} ({variance_ratio[index] * 100.0:.1f}%)"
    return f"PC{index + 1}"


def _ld_label(index: int, discriminant_ratio: np.ndarray) -> str:
    if index < discriminant_ratio.size:
        return f"LD{index + 1} ({discriminant_ratio[index] * 100.0:.1f}%)"
    return f"LD{index + 1}"


def _scatter_by_exact_keyword(
    axis,
    sample_features: list[SampleFeatures],
    coordinates: np.ndarray,
) -> None:
    keywords = [item.record.keyword for item in sample_features]
    unique_keywords = sorted(set(keywords))
    cmap = plt.get_cmap("tab10")
    for index, keyword in enumerate(unique_keywords):
        mask = [item.record.keyword == keyword for item in sample_features]
        xs = coordinates[mask, 0]
        ys = coordinates[mask, 1]
        axis.scatter(xs, ys, label=keyword, s=70, alpha=0.85, color=cmap(index % 10))
    axis.set_title("Exact Labels: PC1 vs PC2")
    axis.legend()


def _scatter_by_base_keyword_and_speaker(
    axis,
    sample_features: list[SampleFeatures],
    coordinates: np.ndarray,
    x_pc_index: int,
    y_pc_index: int,
) -> None:
    parsed = [split_keyword_label(item.record.keyword) for item in sample_features]
    base_keywords = sorted({base for base, _ in parsed})
    speakers = sorted({speaker for _, speaker in parsed})
    color_map = {keyword: plt.get_cmap("tab10")(index % 10) for index, keyword in enumerate(base_keywords)}
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_map = {speaker: marker_cycle[index % len(marker_cycle)] for index, speaker in enumerate(speakers)}

    for base_keyword in base_keywords:
        for speaker in speakers:
            mask = [
                parsed[index][0] == base_keyword and parsed[index][1] == speaker
                for index in range(len(sample_features))
            ]
            if not any(mask):
                continue
            xs = coordinates[mask, x_pc_index]
            ys = coordinates[mask, y_pc_index]
            axis.scatter(
                xs,
                ys,
                s=70,
                alpha=0.85,
                color=color_map[base_keyword],
                marker=marker_map[speaker],
            )

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[keyword], label=keyword, markersize=8)
        for keyword in base_keywords
    ]
    speaker_handles = [
        Line2D([0], [0], marker=marker_map[speaker], linestyle="", color="black", label=speaker, markersize=8)
        for speaker in speakers
    ]
    legend1 = axis.legend(handles=color_handles, title="Base Keyword", loc="best")
    axis.add_artist(legend1)
    axis.legend(handles=speaker_handles, title="Speaker", loc="lower right")


def _plot_label_centroids(axis, sample_features: list[SampleFeatures], coordinates: np.ndarray) -> None:
    keywords = sorted({item.record.keyword for item in sample_features})
    cmap = plt.get_cmap("tab10")
    for index, keyword in enumerate(keywords):
        mask = [item.record.keyword == keyword for item in sample_features]
        centroid = np.mean(coordinates[mask, :2], axis=0, dtype=np.float64)
        axis.scatter(
            centroid[0],
            centroid[1],
            s=120,
            alpha=0.9,
            color=cmap(index % 10),
        )
        axis.annotate(keyword, (centroid[0], centroid[1]), fontsize=9, alpha=0.85)
    axis.set_title("Label Centroids: PC1 vs PC2")


def _plot_all_pc_heatmap(axis, sample_features: list[SampleFeatures], coordinates: np.ndarray):
    if coordinates.ndim != 2 or coordinates.shape[0] == 0 or coordinates.shape[1] == 0:
        axis.set_visible(False)
        return None

    order = sorted(
        range(len(sample_features)),
        key=lambda index: (
            sample_features[index].record.keyword,
            sample_features[index].record.session_id,
            sample_features[index].record.sample_id,
        ),
    )
    ordered_features = [sample_features[index] for index in order]
    ordered_coordinates = coordinates[order, :]

    max_abs = float(np.max(np.abs(ordered_coordinates))) if ordered_coordinates.size else 1.0
    max_abs = max(max_abs, 1e-6)
    image = axis.imshow(
        ordered_coordinates,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    pc_count = ordered_coordinates.shape[1]
    tick_step = max(1, int(math.ceil(pc_count / 12.0)))
    xticks = list(range(0, pc_count, tick_step))
    if (pc_count - 1) not in xticks:
        xticks.append(pc_count - 1)
    axis.set_xticks(xticks)
    axis.set_xticklabels([f"PC{index + 1}" for index in xticks], rotation=0)

    ytick_positions: list[float] = []
    ytick_labels: list[str] = []
    group_start = 0
    current_keyword = ordered_features[0].record.keyword
    for index, item in enumerate(ordered_features[1:], start=1):
        if item.record.keyword == current_keyword:
            continue
        ytick_positions.append((group_start + index - 1) / 2.0)
        ytick_labels.append(current_keyword)
        axis.axhline(index - 0.5, color="black", linewidth=0.6, alpha=0.35)
        group_start = index
        current_keyword = item.record.keyword
    ytick_positions.append((group_start + len(ordered_features) - 1) / 2.0)
    ytick_labels.append(current_keyword)

    axis.set_yticks(ytick_positions)
    axis.set_yticklabels(ytick_labels)
    axis.set_xlabel("Principal Component")
    axis.set_ylabel("Samples grouped by keyword")
    axis.set_title("All-PC Score Heatmap")
    return image


def _plot_lda_strength(axis, lda: LdaResult) -> None:
    if lda.explained_discriminant_ratio.size == 0:
        axis.text(0.5, 0.5, "LDA unavailable", ha="center", va="center", fontsize=12)
        axis.set_axis_off()
        return
    count = lda.explained_discriminant_ratio.size
    x = np.arange(1, count + 1)
    axis.bar(x, lda.explained_discriminant_ratio * 100.0, color="#72B7B2", alpha=0.9)
    axis.set_title("LDA Discriminant Strength")
    axis.set_xlabel("Linear Discriminant")
    axis.set_ylabel("Relative Strength (%)")
    axis.set_xticks(x)
    axis.grid(axis="y", alpha=0.25)


def _plot_lda_ld1_by_keyword(axis, sample_features: list[SampleFeatures], coordinates: np.ndarray) -> None:
    parsed = [split_keyword_label(item.record.keyword) for item in sample_features]
    base_keywords = sorted({base for base, _ in parsed})
    color_map = {keyword: plt.get_cmap("tab10")(index % 10) for index, keyword in enumerate(base_keywords)}
    rng = np.random.default_rng(0)

    for index, keyword in enumerate(base_keywords):
        values = coordinates[[base == keyword for base, _ in parsed], 0]
        if values.size == 0:
            continue
        x = np.full(values.shape[0], index, dtype=np.float32)
        x = x + rng.uniform(-0.08, 0.08, size=values.shape[0]).astype(np.float32)
        axis.scatter(x, values, s=32, alpha=0.75, color=color_map[keyword], label=keyword)

    axis.set_xticks(np.arange(len(base_keywords)))
    axis.set_xticklabels(base_keywords)
    axis.set_ylabel("LD1")
    axis.set_title("LDA: LD1 by Base Keyword")
    axis.grid(alpha=0.25)


def _plot_unavailable(axis, title: str, message: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)


def plot_feature_dashboard(
    sample_features: list[SampleFeatures],
    pca: PcaResult,
    lda: Optional[LdaResult] = None,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
    annotate: bool = False,
) -> Path | None:
    if not sample_features:
        raise ValueError("No samples available to plot.")
    pca_coordinates = pca.coordinates
    variance_ratio = pca.explained_variance_ratio
    all_pc_coordinates = pca_coordinates[:, : variance_ratio.size] if variance_ratio.size else pca_coordinates[:, :0]
    lda_coordinates = None if lda is None else lda.coordinates
    figure = plt.figure(figsize=(20, 19))
    grid = figure.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 0.9, 1.15])
    axes = np.empty((3, 3), dtype=object)
    for row in range(2):
        for column in range(3):
            axes[row, column] = figure.add_subplot(grid[row, column])
    for column in range(3):
        axes[2, column] = figure.add_subplot(grid[2, column])
    heatmap_axis = figure.add_subplot(grid[3, :])

    _scatter_by_exact_keyword(axes[0, 0], sample_features, pca_coordinates)
    _scatter_by_base_keyword_and_speaker(axes[0, 1], sample_features, pca_coordinates, 0, 1)
    axes[0, 1].set_title("Base Keyword + Speaker: PC1 vs PC2")
    _scatter_by_base_keyword_and_speaker(axes[0, 2], sample_features, pca_coordinates, 0, 2)
    axes[0, 2].set_title("Base Keyword + Speaker: PC1 vs PC3")
    _scatter_by_base_keyword_and_speaker(axes[1, 0], sample_features, pca_coordinates, 1, 2)
    axes[1, 0].set_title("Base Keyword + Speaker: PC2 vs PC3")
    _plot_label_centroids(axes[1, 1], sample_features, pca_coordinates)

    num_components = variance_ratio.size
    scree_x = np.arange(1, num_components + 1)
    axes[1, 2].bar(scree_x, variance_ratio[:num_components] * 100.0, color="#4C78A8", alpha=0.85)
    cumulative = np.cumsum(variance_ratio[:num_components]) * 100.0
    cumulative_axis = axes[1, 2].twinx()
    cumulative_axis.plot(scree_x, cumulative, color="#E45756", marker="o", markersize=3, linewidth=1.5)
    cumulative_axis.set_ylabel("Cumulative Variance (%)")
    cumulative_axis.set_ylim(0.0, 100.0)
    axes[1, 2].set_title("Explained Variance (All PCs)")
    axes[1, 2].set_xlabel("Principal Component")
    axes[1, 2].set_ylabel("Variance Explained (%)")
    tick_step = max(1, int(math.ceil(num_components / 12.0)))
    xticks = list(scree_x[::tick_step])
    if scree_x.size and int(scree_x[-1]) not in xticks:
        xticks.append(int(scree_x[-1]))
    axes[1, 2].set_xticks(xticks)
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].legend(
        handles=[
            Line2D([0], [0], color="#4C78A8", linewidth=6, label="Per-PC"),
            Line2D([0], [0], color="#E45756", marker="o", linewidth=1.5, label="Cumulative"),
        ],
        loc="best",
    )

    heatmap_image = _plot_all_pc_heatmap(heatmap_axis, sample_features, all_pc_coordinates)
    if heatmap_image is not None:
        figure.colorbar(heatmap_image, ax=heatmap_axis, fraction=0.02, pad=0.01, label="PC score")

    if lda is None or lda_coordinates is None or lda.explained_discriminant_ratio.size == 0:
        _plot_unavailable(
            axes[2, 0],
            "LDA: Base Keyword + Speaker",
            "Need at least two base keywords\nwith non-degenerate features.",
        )
        _plot_unavailable(
            axes[2, 1],
            "LDA: Base Keyword + Speaker",
            "No supervised discriminants\navailable for this selection.",
        )
        _plot_unavailable(axes[2, 2], "LDA Discriminant Strength", "Unavailable")
    else:
        _scatter_by_base_keyword_and_speaker(axes[2, 0], sample_features, lda_coordinates, 0, 1)
        axes[2, 0].set_title("LDA: Base Keyword + Speaker: LD1 vs LD2")
        if lda.explained_discriminant_ratio.size == 1:
            _plot_lda_ld1_by_keyword(axes[2, 1], sample_features, lda_coordinates)
        elif lda_coordinates.shape[1] >= 3 and lda.explained_discriminant_ratio.size >= 3:
            _scatter_by_base_keyword_and_speaker(axes[2, 1], sample_features, lda_coordinates, 0, 2)
            axes[2, 1].set_title("LDA: Base Keyword + Speaker: LD1 vs LD3")
        else:
            _scatter_by_base_keyword_and_speaker(axes[2, 1], sample_features, lda_coordinates, 0, 1)
            axes[2, 1].set_title("LDA: Base Keyword + Speaker: LD1 vs LD2")
        _plot_lda_strength(axes[2, 2], lda)

    for axis, x_pc_index, y_pc_index in (
        (axes[0, 0], 0, 1),
        (axes[0, 1], 0, 1),
        (axes[0, 2], 0, 2),
        (axes[1, 0], 1, 2),
        (axes[1, 1], 0, 1),
    ):
        axis.set_xlabel(_pc_label(x_pc_index, variance_ratio))
        axis.set_ylabel(_pc_label(y_pc_index, variance_ratio))
        axis.grid(alpha=0.25)

    if lda is not None and lda_coordinates is not None and lda.explained_discriminant_ratio.size > 0:
        axes[2, 0].set_xlabel(_ld_label(0, lda.explained_discriminant_ratio))
        axes[2, 0].set_ylabel(_ld_label(1, lda.explained_discriminant_ratio))
        axes[2, 0].grid(alpha=0.25)
    if lda is not None and lda_coordinates is not None and lda.explained_discriminant_ratio.size > 1:
        axes[2, 1].set_xlabel(_ld_label(0, lda.explained_discriminant_ratio))
        axes[2, 1].set_ylabel(
            _ld_label(2 if lda.explained_discriminant_ratio.size >= 3 else 1, lda.explained_discriminant_ratio)
        )
        axes[2, 1].grid(alpha=0.25)

    if annotate:
        for point, item in zip(pca_coordinates, sample_features):
            axes[0, 0].annotate(item.record.sample_id, (point[0], point[1]), fontsize=7, alpha=0.65)

    figure.suptitle(
        "MFCC + Speaker-Normalized F0 PCA/LDA Dashboard with All-PC Overview",
        fontsize=18,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
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
    samples_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
    annotate: bool = False,
) -> Path | None:
    config = load_app_config(config_path=config_path)
    project_root = Path(project_root).resolve()
    if samples_root is not None:
        sample_features = load_sample_features_from_root(
            project_root,
            Path(samples_root).resolve(),
            keyword=keyword,
        )
    else:
        manifest_path = (project_root / config.storage.manifest_path).resolve()
        sample_features = load_sample_features(project_root, manifest_path, keyword=keyword)
    if not sample_features:
        raise RuntimeError("No saved samples found for feature extraction.")
    vectors = np.vstack([item.vector for item in sample_features]).astype(np.float32)
    pca = compute_pca(vectors, n_dims=max(3, min(vectors.shape[0], vectors.shape[1])))
    lda_labels = [_base_keyword_from_keyword(item.record.keyword) for item in sample_features]
    lda = compute_lda(vectors, lda_labels, n_dims=3)
    return plot_feature_dashboard(
        sample_features,
        pca,
        lda=lda,
        output_path=output_path,
        show=show,
        annotate=annotate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract MFCC + speaker-normalized F0 features and plot PCA/LDA feature spaces."
    )
    parser.add_argument("--project-root", default=".", help="Project root containing src/ and data/.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    parser.add_argument("--keyword", default=None, help="Optional keyword filter.")
    parser.add_argument(
        "--samples-root",
        default=None,
        help=(
            "Optional folder tree of WAV files to plot instead of using the manifest. "
            "Expected layout: <samples-root>/<keyword>/<session-id>/sample.wav or "
            "<samples-root>/<keyword>/sample.wav."
        ),
    )
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
        samples_root=Path(args.samples_root).resolve() if args.samples_root else None,
        output_path=output_path,
        show=not args.no_show,
        annotate=args.annotate,
    )
    if saved_path is not None:
        print(f"Saved feature space plot to {saved_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
