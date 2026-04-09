from __future__ import annotations

import queue
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio import MicrophoneStream, describe_default_input
from collect import (
    build_endpointer,
    build_session_id,
    drain_actions,
    save_utterance,
    start_hotkey_listener,
)
from config import load_app_config
from features import (
    F0_CONTOUR_POINTS,
    build_sample_features,
    extract_feature_parts,
    fit_speaker_f0_statistics,
    load_sample_feature_parts_from_root,
    resample_f0_contour,
    split_keyword_label,
)
from run_knn_classifier import choose_knn_label, knn_predict, standardize_feature_matrices
from storage import SampleRecord
from storage import ensure_storage


CONFIG_PATH = None
SOURCE_ROOT = PROJECT_ROOT / "data" / "live"
FEATURE_VARIANT = "f0_mean_std"  # "f0_mean_std", "f0_contour", or "no_f0"
K = 6
SPEAKER_ID = None  # Optional known speaker id such as "speaker3"; otherwise global F0 stats are used.
UNKNOWN_DISTANCE_THRESHOLD = None  # Set a float to override auto-thresholding.
UNKNOWN_DISTANCE_PERCENTILE = 99.0
UNKNOWN_DISTANCE_MARGIN = 1.10
UNKNOWN_BASE_DISTANCE_PERCENTILE = 99.0
UNKNOWN_BASE_DISTANCE_MARGIN = 1.10
UNKNOWN_MIN_BASE_VOTE_RATIO = 0.67
LEARN_DANCE_KEY = "d"
LEARN_SING_KEY = "s"
QUIT_KEY = None
DRY_RUN = False


@dataclass
class LiveKnnModel:
    source_root: Path
    feature_variant: str
    k: int
    train_labels: list[str]
    train_vectors_scaled: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    f0_statistics: Optional[dict[str, Any]]
    unknown_distance_threshold: float
    unknown_distance_percentile: float
    unknown_distance_margin: float
    base_keyword_thresholds: dict[str, float]
    unknown_base_distance_percentile: float
    unknown_base_distance_margin: float
    min_base_keyword_vote_ratio: float


@dataclass
class LearningBatch:
    keyword: str
    action_prefix: str
    session_id: str
    total: int
    remaining: int


def _build_live_record() -> SampleRecord:
    return SampleRecord(
        sample_id="live_sample",
        keyword="live",
        path="",
        sample_rate=0,
        duration_ms=0,
        timestamp="live",
        session_id="live",
        num_samples=0,
        status="accepted",
    )


def _standardize_single_vector(vector: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((vector.astype(np.float64) - mean) / std).astype(np.float32)


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


def _auto_unknown_distance_threshold(
    train_vectors_scaled: np.ndarray,
    *,
    k: int,
    percentile: float,
    margin: float,
) -> float:
    if train_vectors_scaled.shape[0] <= 1:
        return float("inf")
    percentile = float(np.clip(percentile, 0.0, 100.0))
    margin = max(1.0, float(margin))
    mean_neighbor_distances: list[float] = []
    effective_k = max(1, min(int(k), train_vectors_scaled.shape[0] - 1))
    for index in range(train_vectors_scaled.shape[0]):
        distances = np.linalg.norm(train_vectors_scaled - train_vectors_scaled[index], axis=1)
        distances[index] = np.inf
        nearest_distances = np.sort(distances)[:effective_k]
        mean_neighbor_distances.append(float(np.mean(nearest_distances, dtype=np.float64)))
    baseline = float(np.percentile(np.asarray(mean_neighbor_distances, dtype=np.float64), percentile))
    return baseline * margin


def _auto_base_keyword_thresholds(
    train_vectors_scaled: np.ndarray,
    train_labels: list[str],
    *,
    k: int,
    percentile: float,
    margin: float,
) -> dict[str, float]:
    percentile = float(np.clip(percentile, 0.0, 100.0))
    margin = max(1.0, float(margin))
    grouped_indices: dict[str, list[int]] = {}
    for index, label in enumerate(train_labels):
        base_keyword = split_keyword_label(label)[0]
        grouped_indices.setdefault(base_keyword, []).append(index)

    thresholds: dict[str, float] = {}
    for base_keyword, indices in grouped_indices.items():
        if len(indices) <= 1:
            thresholds[base_keyword] = float("inf")
            continue
        effective_k = max(1, min(int(k), len(indices) - 1))
        mean_neighbor_distances: list[float] = []
        group_vectors = train_vectors_scaled[np.asarray(indices)]
        for local_index in range(len(indices)):
            distances = np.linalg.norm(group_vectors - group_vectors[local_index], axis=1)
            distances[local_index] = np.inf
            nearest_distances = np.sort(distances)[:effective_k]
            mean_neighbor_distances.append(float(np.mean(nearest_distances, dtype=np.float64)))
        baseline = float(np.percentile(np.asarray(mean_neighbor_distances, dtype=np.float64), percentile))
        thresholds[base_keyword] = baseline * margin
    return thresholds


def _choose_base_keyword_from_neighbors(
    neighbor_labels: list[str],
    neighbor_distances: list[float],
) -> tuple[str, float]:
    if not neighbor_labels:
        return "", 0.0
    base_counts: Counter[str] = Counter()
    base_total_distance: dict[str, float] = {}
    for label, distance in zip(neighbor_labels, neighbor_distances):
        base_keyword = split_keyword_label(label)[0]
        base_counts[base_keyword] += 1
        base_total_distance[base_keyword] = base_total_distance.get(base_keyword, 0.0) + float(distance)
    ranked = sorted(
        base_counts,
        key=lambda base_keyword: (
            -base_counts[base_keyword],
            base_total_distance[base_keyword],
            base_keyword,
        ),
    )
    predicted_base_keyword = ranked[0]
    vote_ratio = float(base_counts[predicted_base_keyword]) / float(len(neighbor_labels))
    return predicted_base_keyword, vote_ratio


def _choose_exact_label_from_base(
    neighbor_labels: list[str],
    neighbor_distances: list[float],
    base_keyword: str,
) -> str:
    candidate_labels = [
        label
        for label in neighbor_labels
        if split_keyword_label(label)[0] == base_keyword
    ]
    candidate_distances = [
        distance
        for label, distance in zip(neighbor_labels, neighbor_distances)
        if split_keyword_label(label)[0] == base_keyword
    ]
    if not candidate_labels:
        return ""
    return choose_knn_label(candidate_labels, candidate_distances)


def _build_no_f0_vector(sample_parts) -> np.ndarray:
    return np.concatenate(
        [
            sample_parts.mfcc_mean.astype(np.float32, copy=False),
            sample_parts.mfcc_std.astype(np.float32, copy=False),
            np.asarray([sample_parts.voiced_ratio], dtype=np.float32),
        ]
    ).astype(np.float32)


def _build_mean_std_vector(sample_parts) -> np.ndarray:
    voiced_f0 = sample_parts.voiced_f0.astype(np.float64, copy=False)
    if voiced_f0.size > 0:
        mean_f0 = float(np.mean(voiced_f0, dtype=np.float64))
        std_f0 = float(np.std(voiced_f0, dtype=np.float64))
    else:
        mean_f0 = 0.0
        std_f0 = 0.0
    return np.concatenate(
        [
            sample_parts.mfcc_mean.astype(np.float32, copy=False),
            sample_parts.mfcc_std.astype(np.float32, copy=False),
            np.asarray([mean_f0, std_f0, sample_parts.voiced_ratio], dtype=np.float32),
        ]
    ).astype(np.float32)


def _lookup_live_f0_stats(
    f0_statistics: dict[str, Any],
    *,
    speaker_id: Optional[str],
) -> tuple[float, float]:
    global_stats = dict(f0_statistics.get("global", {}))
    global_mean_f0 = float(global_stats.get("mean_f0", 0.0))
    global_std_f0 = float(global_stats.get("std_f0", 1.0))
    if global_std_f0 <= 1e-6:
        global_std_f0 = 1.0
    if not speaker_id:
        return global_mean_f0, global_std_f0
    speaker_stats = dict(f0_statistics.get("speakers", {})).get(speaker_id)
    if not isinstance(speaker_stats, dict):
        return global_mean_f0, global_std_f0
    mean_f0 = float(speaker_stats.get("mean_f0", global_mean_f0))
    std_f0 = float(speaker_stats.get("std_f0", global_std_f0))
    if std_f0 <= 1e-6:
        std_f0 = global_std_f0
    return mean_f0, std_f0


def _build_contour_vector(
    sample_parts,
    *,
    f0_statistics: dict[str, Any],
    speaker_id: Optional[str],
) -> np.ndarray:
    resampled_f0, fully_unvoiced = resample_f0_contour(sample_parts.f0_contour, n_points=F0_CONTOUR_POINTS)
    if fully_unvoiced:
        normalized_contour = np.zeros(F0_CONTOUR_POINTS, dtype=np.float32)
    else:
        mean_f0, std_f0 = _lookup_live_f0_stats(f0_statistics, speaker_id=speaker_id)
        normalized_contour = (
            (resampled_f0.astype(np.float64) - mean_f0) / max(std_f0, 1e-6)
        ).astype(np.float32)
    return np.concatenate(
        [
            sample_parts.mfcc_mean.astype(np.float32, copy=False),
            sample_parts.mfcc_std.astype(np.float32, copy=False),
            np.asarray([sample_parts.voiced_ratio], dtype=np.float32),
            normalized_contour.astype(np.float32, copy=False),
        ]
    ).astype(np.float32)


def _build_live_feature_vector(
    samples: np.ndarray,
    sample_rate: int,
    *,
    feature_variant: str,
    f0_statistics: Optional[dict[str, Any]],
    speaker_id: Optional[str],
) -> np.ndarray:
    sample_parts = extract_feature_parts(_build_live_record(), samples, sample_rate)
    if feature_variant == "no_f0":
        return _build_no_f0_vector(sample_parts)
    if feature_variant == "f0_mean_std":
        return _build_mean_std_vector(sample_parts)
    if feature_variant == "f0_contour":
        if f0_statistics is None:
            raise ValueError("f0_statistics are required for the f0_contour variant.")
        return _build_contour_vector(sample_parts, f0_statistics=f0_statistics, speaker_id=speaker_id)
    raise ValueError(f"Unsupported feature variant: {feature_variant}")


def prepare_live_knn_model(
    *,
    project_root: Path,
    source_root: Path,
    feature_variant: str,
    k: int,
) -> LiveKnnModel:
    sample_parts = load_sample_feature_parts_from_root(project_root, source_root)
    if not sample_parts:
        raise ValueError(f"No training samples found under {source_root}")

    if feature_variant == "f0_contour":
        f0_statistics = fit_speaker_f0_statistics(sample_parts)
        train_features = build_sample_features(sample_parts, f0_statistics=f0_statistics)
        train_vectors = np.stack([item.vector for item in train_features]).astype(np.float32)
        train_labels = [item.record.keyword for item in train_features]
    elif feature_variant == "f0_mean_std":
        f0_statistics = None
        train_vectors = np.stack([_build_mean_std_vector(item) for item in sample_parts]).astype(np.float32)
        train_labels = [item.record.keyword for item in sample_parts]
    elif feature_variant == "no_f0":
        f0_statistics = None
        train_vectors = np.stack([_build_no_f0_vector(item) for item in sample_parts]).astype(np.float32)
        train_labels = [item.record.keyword for item in sample_parts]
    else:
        raise ValueError(f"Unsupported feature variant: {feature_variant}")

    train_vectors_scaled, _, scaler = standardize_feature_matrices(train_vectors, train_vectors)
    if UNKNOWN_DISTANCE_THRESHOLD is None:
        unknown_distance_threshold = _auto_unknown_distance_threshold(
            train_vectors_scaled,
            k=max(1, int(k)),
            percentile=UNKNOWN_DISTANCE_PERCENTILE,
            margin=UNKNOWN_DISTANCE_MARGIN,
        )
    else:
        unknown_distance_threshold = float(UNKNOWN_DISTANCE_THRESHOLD)
    return LiveKnnModel(
        source_root=source_root,
        feature_variant=feature_variant,
        k=max(1, int(k)),
        train_labels=train_labels,
        train_vectors_scaled=train_vectors_scaled,
        scaler_mean=np.asarray(scaler["mean"], dtype=np.float64),
        scaler_std=np.asarray(scaler["std"], dtype=np.float64),
        f0_statistics=f0_statistics,
        unknown_distance_threshold=float(unknown_distance_threshold),
        unknown_distance_percentile=float(UNKNOWN_DISTANCE_PERCENTILE),
        unknown_distance_margin=float(UNKNOWN_DISTANCE_MARGIN),
        base_keyword_thresholds=_auto_base_keyword_thresholds(
            train_vectors_scaled,
            train_labels,
            k=max(1, int(k)),
            percentile=UNKNOWN_BASE_DISTANCE_PERCENTILE,
            margin=UNKNOWN_BASE_DISTANCE_MARGIN,
        ),
        unknown_base_distance_percentile=float(UNKNOWN_BASE_DISTANCE_PERCENTILE),
        unknown_base_distance_margin=float(UNKNOWN_BASE_DISTANCE_MARGIN),
        min_base_keyword_vote_ratio=float(UNKNOWN_MIN_BASE_VOTE_RATIO),
    )


def predict_live_keyword(
    model: LiveKnnModel,
    samples: np.ndarray,
    sample_rate: int,
    *,
    speaker_id: Optional[str],
) -> dict[str, Any]:
    feature_vector = _build_live_feature_vector(
        samples,
        sample_rate,
        feature_variant=model.feature_variant,
        f0_statistics=model.f0_statistics,
        speaker_id=speaker_id,
    )
    scaled_vector = _standardize_single_vector(feature_vector, model.scaler_mean, model.scaler_std)
    mean_neighbor_distance = _mean_k_neighbor_distance(
        model.train_vectors_scaled,
        scaled_vector,
        k=model.k,
    )
    prediction = knn_predict(
        model.train_vectors_scaled,
        model.train_labels,
        scaled_vector.reshape(1, -1),
        k=model.k,
    )[0]
    neighbor_labels = list(prediction["neighbor_labels"])
    neighbor_distances = list(prediction["neighbor_distances"])
    keyword, base_vote_ratio = _choose_base_keyword_from_neighbors(neighbor_labels, neighbor_distances)
    exact_label = _choose_exact_label_from_base(neighbor_labels, neighbor_distances, keyword)
    base_neighbor_distances = [
        float(distance)
        for label, distance in zip(neighbor_labels, neighbor_distances)
        if split_keyword_label(label)[0] == keyword
    ]
    mean_base_neighbor_distance = (
        float(np.mean(np.asarray(base_neighbor_distances, dtype=np.float64)))
        if base_neighbor_distances
        else float("inf")
    )
    base_distance_threshold = float(model.base_keyword_thresholds.get(keyword, float("inf")))
    is_known = (
        mean_neighbor_distance <= model.unknown_distance_threshold
        and base_vote_ratio >= model.min_base_keyword_vote_ratio
        and mean_base_neighbor_distance <= base_distance_threshold
    )
    return {
        "exact_label": exact_label if is_known else None,
        "keyword": keyword if is_known else None,
        "closest_exact_label": exact_label,
        "closest_keyword": keyword,
        "is_known": bool(is_known),
        "mean_neighbor_distance": float(mean_neighbor_distance),
        "unknown_distance_threshold": float(model.unknown_distance_threshold),
        "base_vote_ratio": float(base_vote_ratio),
        "min_base_keyword_vote_ratio": float(model.min_base_keyword_vote_ratio),
        "mean_base_neighbor_distance": float(mean_base_neighbor_distance),
        "base_distance_threshold": float(base_distance_threshold),
        "neighbor_labels": neighbor_labels,
        "neighbor_distances": neighbor_distances,
    }


def _next_synthetic_keyword(source_root: Path, prefix: str) -> str:
    source_root = Path(source_root).resolve()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_serial = 0
    for child in source_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name.strip())
        if not match:
            continue
        max_serial = max(max_serial, int(match.group(1)))
    return f"{prefix}_{max_serial + 1:03d}"


def _begin_learning_batch(
    *,
    source_root: Path,
    action_prefix: str,
    batch_size: int,
) -> LearningBatch:
    keyword = _next_synthetic_keyword(source_root, action_prefix)
    return LearningBatch(
        keyword=keyword,
        action_prefix=action_prefix,
        session_id=build_session_id(),
        total=max(1, int(batch_size)),
        remaining=max(1, int(batch_size)),
    )


def _reload_model(model: LiveKnnModel) -> LiveKnnModel:
    print("Rebuilding classifier with the updated dataset...")
    refreshed = prepare_live_knn_model(
        project_root=PROJECT_ROOT,
        source_root=model.source_root,
        feature_variant=model.feature_variant,
        k=model.k,
    )
    print(
        f"Classifier updated. Training samples={len(refreshed.train_labels)}, "
        f"feature_dim={refreshed.train_vectors_scaled.shape[1]}."
    )
    return refreshed


def _print_parrot_action(exact_label: str) -> None:
    if exact_label.startswith("d_"):
        print("Parrot is dancing")
    elif exact_label.startswith("s_"):
        print("Parrot is singing")


def run_live_classification(
    *,
    model: LiveKnnModel,
    config,
    project_root: Path,
    speaker_id: Optional[str],
) -> int:
    sample_rate = int(config.audio.sample_rate)
    endpointer = build_endpointer(config)
    raw_dir, manifest_path = ensure_storage(
        project_root=project_root,
        data_dir=config.storage.data_dir,
        manifest_path=config.storage.manifest_path,
    )
    hotkey_events: "queue.Queue[str]" = queue.Queue()
    listener = start_hotkey_listener(
        hotkey_events,
        arm_key=None,
        quit_key=config.collection.quit_key,
        extra_actions={
            LEARN_DANCE_KEY: "learn_dance",
            LEARN_SING_KEY: "learn_sing",
        },
    )
    current_batch: Optional[LearningBatch] = None

    default_device = describe_default_input()
    if default_device:
        print(
            f"Using input device {default_device.index}: {default_device.name} "
            f"({default_device.default_samplerate:.0f} Hz default)"
        )
    print(
        f"Live keyword classification is ready. Feature variant={model.feature_variant}, "
        f"k={model.k}, training_samples={len(model.train_labels)}."
    )
    print(
        f"Unknown-command threshold: mean {model.k}-NN distance <= "
        f"{model.unknown_distance_threshold:.3f}"
    )
    print(
        f"Base-cluster threshold: vote ratio >= {model.min_base_keyword_vote_ratio:.2f}, "
        f"plus per-keyword distance gate."
    )
    if model.feature_variant == "f0_contour":
        if speaker_id:
            print(f"Using contour normalization stats for {speaker_id} when available.")
        else:
            print("Using global contour normalization stats for the live speaker.")
    print(
        "Always listening for the next word."
    )
    print(
        f"Press {LEARN_DANCE_KEY} to record a new dance-command batch, "
        f"{LEARN_SING_KEY} to record a new sing-command batch, and "
        f"{config.collection.quit_key} to quit."
    )

    try:
        with MicrophoneStream(
            sample_rate=sample_rate,
            channels=config.audio.channels,
            block_ms=config.audio.block_ms,
            device=config.audio.device,
        ) as microphone:
            endpointer.arm()
            while True:
                for status_message in microphone.pop_status_messages():
                    print(f"audio-status: {status_message}")
                for action in drain_actions(hotkey_events):
                    if action == "quit":
                        print("Stopping live classification.")
                        return 0
                    if current_batch is not None:
                        print(
                            f"Ignoring '{action}' while learning batch {current_batch.keyword} is in progress."
                        )
                        continue
                    if action == "learn_dance":
                        current_batch = _begin_learning_batch(
                            source_root=model.source_root,
                            action_prefix="d",
                            batch_size=config.collection.batch_size,
                        )
                        endpointer.arm()
                        print("")
                        print(
                            f"Learning new dance command {current_batch.keyword}. "
                            f"Speak {current_batch.total} examples."
                        )
                        continue
                    if action == "learn_sing":
                        current_batch = _begin_learning_batch(
                            source_root=model.source_root,
                            action_prefix="s",
                            batch_size=config.collection.batch_size,
                        )
                        endpointer.arm()
                        print("")
                        print(
                            f"Learning new sing command {current_batch.keyword}. "
                            f"Speak {current_batch.total} examples."
                        )
                        continue

                chunk = microphone.read_block(timeout=0.1)
                if chunk is None:
                    continue

                for event in endpointer.process_chunk(chunk):
                    if event.kind == "discarded":
                        if current_batch is not None:
                            print(
                                f"Discarded short noise while learning {current_batch.keyword} "
                                f"({event.speech_ms} ms speech, {event.trailing_silence_ms} ms trailing silence)."
                            )
                        continue
                    if event.kind != "utterance" or event.audio is None or event.audio.size == 0:
                        continue

                    if current_batch is not None:
                        record = save_utterance(
                            event.audio,
                            project_root=project_root,
                            raw_dir=raw_dir,
                            manifest_path=manifest_path,
                            keyword=current_batch.keyword,
                            session_id=current_batch.session_id,
                            sample_rate=sample_rate,
                        )
                        current_batch.remaining -= 1
                        completed = current_batch.total - current_batch.remaining
                        print(
                            f"Saved {record.sample_id} to {record.path} "
                            f"for {current_batch.keyword} ({completed}/{current_batch.total})"
                        )
                        if current_batch.remaining > 0:
                            endpointer.arm()
                            continue

                        print(
                            f"Finished learning batch {current_batch.keyword}. "
                            "Updating classifier."
                        )
                        model = _reload_model(model)
                        current_batch = None
                        endpointer.arm()
                        print("Continuous listening resumed.")
                        print("")
                        continue

                    prediction = predict_live_keyword(
                        model,
                        event.audio,
                        sample_rate,
                        speaker_id=speaker_id,
                    )
                    neighbors = ", ".join(
                        f"{label} ({distance:.3f})"
                        for label, distance in zip(
                            prediction["neighbor_labels"],
                            prediction["neighbor_distances"],
                        )
                    )
                    print("")
                    if prediction["is_known"]:
                        print(f"Predicted exact label: {prediction['exact_label']}")
                        print(f"Predicted keyword: {prediction['keyword']}")
                        _print_parrot_action(prediction["exact_label"])
                    else:
                        print("Not known command. Parrot is ignoring")
                        print(
                            f"Closest known label was {prediction['closest_exact_label']} "
                            f"at mean {model.k}-NN distance {prediction['mean_neighbor_distance']:.3f} "
                            f"(threshold {prediction['unknown_distance_threshold']:.3f})"
                        )
                        print(
                            f"Base vote ratio was {prediction['base_vote_ratio']:.2f} "
                            f"(required {prediction['min_base_keyword_vote_ratio']:.2f}), "
                            f"base distance {prediction['mean_base_neighbor_distance']:.3f} "
                            f"(threshold {prediction['base_distance_threshold']:.3f})"
                        )
                    print(f"Nearest neighbors: {neighbors}")
                    print("Ready for the next word.")
                    print("")
                    endpointer.arm()
    finally:
        listener.stop()


def main() -> int:
    config = load_app_config(config_path=CONFIG_PATH)
    if QUIT_KEY:
        config.collection.quit_key = QUIT_KEY

    model = prepare_live_knn_model(
        project_root=PROJECT_ROOT,
        source_root=Path(SOURCE_ROOT).resolve(),
        feature_variant=FEATURE_VARIANT,
        k=K,
    )

    if DRY_RUN:
        print(
            f"Dry run complete. feature_variant={model.feature_variant}, "
            f"k={model.k}, source_root={model.source_root}, "
            f"training_samples={len(model.train_labels)}, "
            f"feature_dim={model.train_vectors_scaled.shape[1]}, "
            f"unknown_threshold={model.unknown_distance_threshold:.3f}"
        )
        return 0

    return run_live_classification(
        model=model,
        config=config,
        project_root=PROJECT_ROOT,
        speaker_id=SPEAKER_ID,
    )


if __name__ == "__main__":
    raise SystemExit(main())
