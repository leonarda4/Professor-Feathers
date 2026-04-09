from __future__ import annotations

import queue
import re
import sys
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
    RecordingBatch,
    build_endpointer,
    drain_actions,
    save_batch_utterance,
    start_recording_batch,
    start_hotkey_listener,
)
from config import load_app_config
from features import (
    build_feature_matrix,
    build_feature_vector_from_parts,
    extract_feature_parts,
    load_sample_feature_parts_from_root,
)
from knn_utils import choose_knn_label, knn_predict, standardize_feature_matrices
from storage import SampleRecord, ensure_storage


# Edit these values before running from the IDE.
CONFIG_PATH = None
SOURCE_ROOT = PROJECT_ROOT / "data" / "live"
FEATURE_VARIANT = "f0_mean_std"  # "f0_mean_std", "f0_contour", or "no_f0"
K = 6
SPEAKER_ID = None  # Optional known speaker id such as "speaker3".
UNKNOWN_DISTANCE_THRESHOLD = None  # Set a float to override auto-thresholding.
UNKNOWN_DISTANCE_PERCENTILE = 99.0
UNKNOWN_DISTANCE_MARGIN = 1.10
UNKNOWN_LABEL_DISTANCE_PERCENTILE = 99.0
UNKNOWN_LABEL_DISTANCE_MARGIN = 1.10
UNKNOWN_MIN_LABEL_VOTE_RATIO = 0.67
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
    label_thresholds: dict[str, float]
    min_label_vote_ratio: float


@dataclass
class CommandPrediction:
    is_known: bool
    predicted_label: Optional[str]
    closest_label: str
    mean_neighbor_distance: float
    unknown_distance_threshold: float
    label_vote_ratio: float
    min_label_vote_ratio: float
    mean_label_neighbor_distance: float
    label_distance_threshold: float
    neighbor_labels: list[str]
    neighbor_distances: list[float]


# Classification model


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
    effective_k = max(1, min(int(k), train_vectors_scaled.shape[0] - 1))
    mean_neighbor_distances: list[float] = []
    for index in range(train_vectors_scaled.shape[0]):
        distances = np.linalg.norm(train_vectors_scaled - train_vectors_scaled[index], axis=1)
        distances[index] = np.inf
        nearest_distances = np.sort(distances)[:effective_k]
        mean_neighbor_distances.append(float(np.mean(nearest_distances, dtype=np.float64)))
    baseline = float(
        np.percentile(
            np.asarray(mean_neighbor_distances, dtype=np.float64),
            float(np.clip(percentile, 0.0, 100.0)),
        )
    )
    return baseline * max(1.0, float(margin))


def _auto_label_thresholds(
    train_vectors_scaled: np.ndarray,
    train_labels: list[str],
    *,
    k: int,
    percentile: float,
    margin: float,
) -> dict[str, float]:
    grouped_indices: dict[str, list[int]] = {}
    for index, label in enumerate(train_labels):
        grouped_indices.setdefault(label, []).append(index)

    thresholds: dict[str, float] = {}
    for label, indices in grouped_indices.items():
        if len(indices) <= 1:
            thresholds[label] = float("inf")
            continue
        group_vectors = train_vectors_scaled[np.asarray(indices)]
        effective_k = max(1, min(int(k), len(indices) - 1))
        mean_neighbor_distances: list[float] = []
        for local_index in range(len(indices)):
            distances = np.linalg.norm(group_vectors - group_vectors[local_index], axis=1)
            distances[local_index] = np.inf
            nearest_distances = np.sort(distances)[:effective_k]
            mean_neighbor_distances.append(float(np.mean(nearest_distances, dtype=np.float64)))
        baseline = float(
            np.percentile(
                np.asarray(mean_neighbor_distances, dtype=np.float64),
                float(np.clip(percentile, 0.0, 100.0)),
            )
        )
        thresholds[label] = baseline * max(1.0, float(margin))
    return thresholds


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

    train_vectors, train_labels, f0_statistics = build_feature_matrix(
        sample_parts,
        variant=feature_variant,
    )
    train_vectors_scaled, _, scaler = standardize_feature_matrices(train_vectors, train_vectors)
    unknown_distance_threshold = (
        float(UNKNOWN_DISTANCE_THRESHOLD)
        if UNKNOWN_DISTANCE_THRESHOLD is not None
        else _auto_unknown_distance_threshold(
            train_vectors_scaled,
            k=max(1, int(k)),
            percentile=UNKNOWN_DISTANCE_PERCENTILE,
            margin=UNKNOWN_DISTANCE_MARGIN,
        )
    )
    return LiveKnnModel(
        source_root=Path(source_root).resolve(),
        feature_variant=feature_variant,
        k=max(1, int(k)),
        train_labels=train_labels,
        train_vectors_scaled=train_vectors_scaled,
        scaler_mean=np.asarray(scaler["mean"], dtype=np.float64),
        scaler_std=np.asarray(scaler["std"], dtype=np.float64),
        f0_statistics=f0_statistics,
        unknown_distance_threshold=float(unknown_distance_threshold),
        label_thresholds=_auto_label_thresholds(
            train_vectors_scaled,
            train_labels,
            k=max(1, int(k)),
            percentile=UNKNOWN_LABEL_DISTANCE_PERCENTILE,
            margin=UNKNOWN_LABEL_DISTANCE_MARGIN,
        ),
        min_label_vote_ratio=float(UNKNOWN_MIN_LABEL_VOTE_RATIO),
    )


def _build_live_command_vector(
    samples: np.ndarray,
    sample_rate: int,
    *,
    model: LiveKnnModel,
    speaker_id: Optional[str],
) -> np.ndarray:
    sample_parts = extract_feature_parts(_build_live_record(), samples, sample_rate)
    return build_feature_vector_from_parts(
        sample_parts,
        variant=model.feature_variant,
        f0_statistics=model.f0_statistics,
        speaker_id=speaker_id,
    )


def predict_live_command(
    model: LiveKnnModel,
    samples: np.ndarray,
    sample_rate: int,
    *,
    speaker_id: Optional[str],
) -> CommandPrediction:
    feature_vector = _build_live_command_vector(
        samples,
        sample_rate,
        model=model,
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
    closest_label = choose_knn_label(neighbor_labels, neighbor_distances)
    label_vote_count = sum(1 for label in neighbor_labels if label == closest_label)
    label_vote_ratio = float(label_vote_count) / float(len(neighbor_labels)) if neighbor_labels else 0.0
    label_neighbor_distances = [
        float(distance)
        for label, distance in zip(neighbor_labels, neighbor_distances)
        if label == closest_label
    ]
    mean_label_neighbor_distance = (
        float(np.mean(np.asarray(label_neighbor_distances, dtype=np.float64)))
        if label_neighbor_distances
        else float("inf")
    )
    label_distance_threshold = float(model.label_thresholds.get(closest_label, float("inf")))
    is_known = (
        mean_neighbor_distance <= model.unknown_distance_threshold
        and label_vote_ratio >= model.min_label_vote_ratio
        and mean_label_neighbor_distance <= label_distance_threshold
    )
    return CommandPrediction(
        is_known=bool(is_known),
        predicted_label=closest_label if is_known else None,
        closest_label=closest_label,
        mean_neighbor_distance=float(mean_neighbor_distance),
        unknown_distance_threshold=float(model.unknown_distance_threshold),
        label_vote_ratio=float(label_vote_ratio),
        min_label_vote_ratio=float(model.min_label_vote_ratio),
        mean_label_neighbor_distance=float(mean_label_neighbor_distance),
        label_distance_threshold=float(label_distance_threshold),
        neighbor_labels=neighbor_labels,
        neighbor_distances=neighbor_distances,
    )


# Recording and learning


def _next_synthetic_command_label(source_root: Path, prefix: str) -> str:
    source_root = Path(source_root).resolve()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_serial = 0
    for child in source_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name.strip())
        if match:
            max_serial = max(max_serial, int(match.group(1)))
    return f"{prefix}_{max_serial + 1:03d}"


def _begin_learning_batch(
    *,
    source_root: Path,
    action_prefix: str,
    batch_size: int,
) -> RecordingBatch:
    return start_recording_batch(
        _next_synthetic_command_label(source_root, action_prefix),
        batch_size=batch_size,
    )


def _reload_model(model: LiveKnnModel) -> LiveKnnModel:
    print("Rebuilding classifier with the updated command set...")
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


def _print_parrot_action(command_label: str) -> None:
    if command_label.startswith("d_"):
        print("Parrot is dancing")
    elif command_label.startswith("s_"):
        print("Parrot is singing")


def _print_learning_prompt(batch: RecordingBatch) -> None:
    print("")
    if batch.keyword.startswith("d_"):
        print(f"Learning new dance command {batch.keyword}. Speak {batch.total} examples.")
    else:
        print(f"Learning new sing command {batch.keyword}. Speak {batch.total} examples.")


def _print_prediction(prediction: CommandPrediction, *, k: int) -> None:
    neighbors = ", ".join(
        f"{label} ({distance:.3f})"
        for label, distance in zip(prediction.neighbor_labels, prediction.neighbor_distances)
    )
    print("")
    if prediction.is_known and prediction.predicted_label is not None:
        print(f"Predicted command: {prediction.predicted_label}")
        _print_parrot_action(prediction.predicted_label)
    else:
        print("Not known command. Parrot is ignoring")
        print(
            f"Closest known label was {prediction.closest_label} "
            f"at mean {k}-NN distance {prediction.mean_neighbor_distance:.3f} "
            f"(threshold {prediction.unknown_distance_threshold:.3f})"
        )
        print(
            f"Command vote ratio was {prediction.label_vote_ratio:.2f} "
            f"(required {prediction.min_label_vote_ratio:.2f}), "
            f"command distance {prediction.mean_label_neighbor_distance:.3f} "
            f"(threshold {prediction.label_distance_threshold:.3f})"
        )
    print(f"Nearest neighbors: {neighbors}")
    print("Ready for the next word.")
    print("")


# Runtime loop


def run_live_classification(
    *,
    model: LiveKnnModel,
    config,
    project_root: Path,
    speaker_id: Optional[str],
) -> int:
    sample_rate = int(config.audio.sample_rate)
    endpointer = build_endpointer(config)
    _, manifest_path = ensure_storage(
        project_root=project_root,
        data_dir=config.storage.data_dir,
        manifest_path=config.storage.manifest_path,
    )
    command_root = Path(model.source_root).resolve()
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
    current_batch: Optional[RecordingBatch] = None

    default_device = describe_default_input()
    if default_device:
        print(
            f"Using input device {default_device.index}: {default_device.name} "
            f"({default_device.default_samplerate:.0f} Hz default)"
        )
    print(
        f"Live command classification is ready. Feature variant={model.feature_variant}, "
        f"k={model.k}, training_samples={len(model.train_labels)}."
    )
    print(
        f"Unknown-command threshold: mean {model.k}-NN distance <= "
        f"{model.unknown_distance_threshold:.3f}"
    )
    print(
        f"Command-cluster threshold: vote ratio >= {model.min_label_vote_ratio:.2f}, "
        "plus per-command distance gate."
    )
    print("Always listening for the next word.")
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
                            source_root=command_root,
                            action_prefix="d",
                            batch_size=config.collection.batch_size,
                        )
                        _print_learning_prompt(current_batch)
                        endpointer.arm()
                    elif action == "learn_sing":
                        current_batch = _begin_learning_batch(
                            source_root=command_root,
                            action_prefix="s",
                            batch_size=config.collection.batch_size,
                        )
                        _print_learning_prompt(current_batch)
                        endpointer.arm()

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
                        record, completed, is_complete = save_batch_utterance(
                            event.audio,
                            project_root=project_root,
                            raw_dir=command_root,
                            manifest_path=manifest_path,
                            batch=current_batch,
                            sample_rate=sample_rate,
                        )
                        print(
                            f"Saved {record.sample_id} to {record.path} "
                            f"for {current_batch.keyword} ({completed}/{current_batch.total})"
                        )
                        if not is_complete:
                            endpointer.arm()
                            continue

                        print(
                            f"Finished learning batch {current_batch.keyword}. Updating classifier."
                        )
                        model = _reload_model(model)
                        current_batch = None
                        endpointer.arm()
                        print("Continuous listening resumed.")
                        print("")
                        continue

                    prediction = predict_live_command(
                        model,
                        event.audio,
                        sample_rate,
                        speaker_id=speaker_id,
                    )
                    _print_prediction(prediction, k=model.k)
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
            f"unknown_threshold={model.unknown_distance_threshold:.3f}, "
            f"label_thresholds={len(model.label_thresholds)}"
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
