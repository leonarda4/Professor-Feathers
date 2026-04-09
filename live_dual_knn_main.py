from __future__ import annotations

import queue
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio import MicrophoneStream, describe_default_input, read_wav
from collect import (
    RecordingBatch,
    build_endpointer,
    drain_actions,
    save_batch_utterance,
    start_recording_batch,
    start_hotkey_listener,
)
from config import load_app_config
from dual_knn import KeywordClassifier, PersonKeywordClassifier
from features.feature_core import extract_feature_parts
from features.feature_loading import load_sample_feature_parts_from_root
from features.feature_spaces import compute_delta_mfcc_mean
from storage import SampleRecord, ensure_storage


CONFIG_PATH = None
BASE_KEYWORD_SOURCE_ROOT = PROJECT_ROOT / "data" / "base_keywords"
DYNAMIC_KEYWORD_SOURCE_ROOT = PROJECT_ROOT / "data" / "dynamic_keywords"
BASE_K = 5
DYNAMIC_K = 3
BASE_UNKNOWN_DISTANCE_THRESHOLD = None
BASE_UNKNOWN_DISTANCE_PERCENTILE = 95.0
BASE_UNKNOWN_DISTANCE_MARGIN = 1.10
DYNAMIC_UNKNOWN_DISTANCE_THRESHOLD = None
DYNAMIC_UNKNOWN_DISTANCE_PERCENTILE = 99.0
DYNAMIC_UNKNOWN_DISTANCE_MARGIN = 1.10
BASE_MIN_LABEL_VOTE_RATIO = 0.80
LEARN_DANCE_KEY = "d"
LEARN_SING_KEY = "s"
QUIT_KEY = None
DRY_RUN = False
USE_DELTA_FOR_DYNAMIC_KNN = True
DANCE_ACTION_LABEL = "dance"
SING_ACTION_LABEL = "sing"
DANCE_PREFIX = "dance"
SING_PREFIX = "sing"


@dataclass
class BasePrediction:
    is_known: bool
    predicted_label: Optional[str]
    closest_label: str
    mean_neighbor_distance: float
    unknown_distance_threshold: float
    label_vote_ratio: float
    min_label_vote_ratio: float
    neighbor_labels: list[str]
    neighbor_distances: list[float]


@dataclass
class DynamicPrediction:
    is_known: bool
    action_label: Optional[str]
    closest_label: str
    mean_neighbor_distance: float
    unknown_distance_threshold: float
    neighbor_labels: list[str]
    neighbor_distances: list[float]


@dataclass
class DualLiveModel:
    base_source_root: Path
    dynamic_source_root: Path
    base_classifier: KeywordClassifier
    dynamic_classifier: PersonKeywordClassifier | None
    base_unknown_distance_threshold: float
    dynamic_unknown_distance_threshold: float
    dynamic_uses_delta: bool


@dataclass
class LearningContext:
    action_label: str
    batch: RecordingBatch


def _build_live_record(keyword: str = "live") -> SampleRecord:
    return SampleRecord(
        sample_id="live_sample",
        keyword=keyword,
        path="",
        sample_rate=0,
        duration_ms=0,
        timestamp="live",
        session_id="live",
        num_samples=0,
        status="accepted",
    )


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


def _infer_action_label(keyword: str) -> str:
    cleaned = keyword.strip().lower()
    if cleaned.startswith(f"{DANCE_PREFIX}_"):
        return DANCE_ACTION_LABEL
    if cleaned.startswith(f"{SING_PREFIX}_"):
        return SING_ACTION_LABEL
    return cleaned


def _build_dynamic_label_map(sample_parts) -> dict[str, str]:
    return {item.record.sample_id: _infer_action_label(item.record.keyword) for item in sample_parts}


def _build_dynamic_delta_map(sample_parts, project_root: Path) -> dict[str, np.ndarray]:
    if not USE_DELTA_FOR_DYNAMIC_KNN:
        return {}
    delta_map: dict[str, np.ndarray] = {}
    for item in sample_parts:
        sample_path = project_root / item.record.path
        if not sample_path.exists():
            continue
        samples, sample_rate = read_wav(sample_path)
        delta_map[item.record.sample_id] = compute_delta_mfcc_mean(samples, sample_rate)
    return delta_map


def prepare_dual_live_model(
    *,
    project_root: Path,
    base_source_root: Path,
    dynamic_source_root: Path,
) -> DualLiveModel:
    base_parts = load_sample_feature_parts_from_root(project_root, base_source_root)
    if not base_parts:
        raise ValueError(f"No base keyword samples found under {base_source_root}")
    base_classifier = KeywordClassifier(k=BASE_K).fit(base_parts)
    base_unknown_distance_threshold = (
        float(BASE_UNKNOWN_DISTANCE_THRESHOLD)
        if BASE_UNKNOWN_DISTANCE_THRESHOLD is not None
        else _auto_unknown_distance_threshold(
            base_classifier._train_vectors,
            k=base_classifier.k,
            percentile=BASE_UNKNOWN_DISTANCE_PERCENTILE,
            margin=BASE_UNKNOWN_DISTANCE_MARGIN,
        )
    )

    dynamic_parts = []
    if Path(dynamic_source_root).exists():
        dynamic_parts = load_sample_feature_parts_from_root(project_root, dynamic_source_root)
    dynamic_classifier: PersonKeywordClassifier | None = None
    dynamic_unknown_distance_threshold = float("inf")
    if dynamic_parts:
        label_map = _build_dynamic_label_map(dynamic_parts)
        delta_map = _build_dynamic_delta_map(dynamic_parts, project_root)
        dynamic_classifier = PersonKeywordClassifier(
            target_label=DANCE_ACTION_LABEL,
            negative_label=SING_ACTION_LABEL,
            k=DYNAMIC_K,
        ).fit(dynamic_parts, label_map=label_map, delta_map=delta_map if delta_map else None)
        dynamic_unknown_distance_threshold = (
            float(DYNAMIC_UNKNOWN_DISTANCE_THRESHOLD)
            if DYNAMIC_UNKNOWN_DISTANCE_THRESHOLD is not None
            else _auto_unknown_distance_threshold(
                dynamic_classifier._train_vectors,
                k=dynamic_classifier.k,
                percentile=DYNAMIC_UNKNOWN_DISTANCE_PERCENTILE,
                margin=DYNAMIC_UNKNOWN_DISTANCE_MARGIN,
            )
        )

    return DualLiveModel(
        base_source_root=Path(base_source_root).resolve(),
        dynamic_source_root=Path(dynamic_source_root).resolve(),
        base_classifier=base_classifier,
        dynamic_classifier=dynamic_classifier,
        base_unknown_distance_threshold=float(base_unknown_distance_threshold),
        dynamic_unknown_distance_threshold=float(dynamic_unknown_distance_threshold),
        dynamic_uses_delta=bool(USE_DELTA_FOR_DYNAMIC_KNN),
    )


def predict_base_command(
    model: DualLiveModel,
    samples: np.ndarray,
    sample_rate: int,
) -> BasePrediction:
    parts = extract_feature_parts(_build_live_record("live_base"), samples, sample_rate)
    prediction = model.base_classifier.predict(parts)
    mean_neighbor_distance = model.base_classifier.mean_neighbor_distance(parts)
    neighbor_labels = list(prediction["neighbor_labels"])
    neighbor_distances = list(prediction["neighbor_distances"])
    closest_label = str(prediction["predicted_label"])
    label_vote_count = sum(1 for label in neighbor_labels if label == closest_label)
    label_vote_ratio = (
        float(label_vote_count) / float(len(neighbor_labels))
        if neighbor_labels
        else 0.0
    )

    is_known = (
            mean_neighbor_distance <= model.base_unknown_distance_threshold
            and label_vote_ratio >= BASE_MIN_LABEL_VOTE_RATIO
    )
    return BasePrediction(
        is_known=bool(is_known),
        predicted_label=closest_label if is_known else None,
        closest_label=closest_label,
        mean_neighbor_distance=float(mean_neighbor_distance),
        unknown_distance_threshold=float(model.base_unknown_distance_threshold),
        label_vote_ratio=float(label_vote_ratio),
        min_label_vote_ratio=float(BASE_MIN_LABEL_VOTE_RATIO),
        neighbor_labels=neighbor_labels,
        neighbor_distances=neighbor_distances,
    )


def predict_dynamic_command(
    model: DualLiveModel,
    samples: np.ndarray,
    sample_rate: int,
) -> DynamicPrediction:
    if model.dynamic_classifier is None:
        return DynamicPrediction(
            is_known=False,
            action_label=None,
            closest_label="none",
            mean_neighbor_distance=float("inf"),
            unknown_distance_threshold=float(model.dynamic_unknown_distance_threshold),
            neighbor_labels=[],
            neighbor_distances=[],
        )
    parts = extract_feature_parts(_build_live_record("live_dynamic"), samples, sample_rate)
    delta = compute_delta_mfcc_mean(samples, sample_rate) if model.dynamic_uses_delta else None
    prediction = model.dynamic_classifier.predict(parts, delta_mfcc_mean=delta)
    mean_neighbor_distance = model.dynamic_classifier.mean_neighbor_distance(parts, delta_mfcc_mean=delta)
    closest_label = str(prediction["predicted_label"])
    is_known = mean_neighbor_distance <= model.dynamic_unknown_distance_threshold
    return DynamicPrediction(
        is_known=bool(is_known),
        action_label=closest_label if is_known else None,
        closest_label=closest_label,
        mean_neighbor_distance=float(mean_neighbor_distance),
        unknown_distance_threshold=float(model.dynamic_unknown_distance_threshold),
        neighbor_labels=list(prediction["neighbor_labels"]),
        neighbor_distances=list(prediction["neighbor_distances"]),
    )


def _next_anonymous_keyword(source_root: Path, prefix: str) -> str:
    source_root = Path(source_root).resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_serial = 0
    for child in source_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name.strip())
        if match:
            max_serial = max(max_serial, int(match.group(1)))
    return f"{prefix}_{max_serial + 1:03d}"


def _begin_learning_batch(*, source_root: Path, action_label: str, batch_size: int) -> RecordingBatch:
    prefix = DANCE_PREFIX if action_label == DANCE_ACTION_LABEL else SING_PREFIX
    return start_recording_batch(
        _next_anonymous_keyword(source_root, prefix),
        batch_size=batch_size,
    )


def _reload_model(model: DualLiveModel) -> DualLiveModel:
    print("Rebuilding classifiers with the updated datasets...")
    refreshed = prepare_dual_live_model(
        project_root=PROJECT_ROOT,
        base_source_root=model.base_source_root,
        dynamic_source_root=model.dynamic_source_root,
    )
    dynamic_count = 0 if refreshed.dynamic_classifier is None else len(refreshed.dynamic_classifier._train_labels)
    dynamic_dim = 0 if refreshed.dynamic_classifier is None or refreshed.dynamic_classifier._train_vectors.size == 0 else refreshed.dynamic_classifier._train_vectors.shape[1]
    print(
        f"Classifiers updated. base_samples={len(refreshed.base_classifier._train_labels)}, "
        f"base_dim={refreshed.base_classifier._train_vectors.shape[1]}, "
        f"dynamic_samples={dynamic_count}, dynamic_dim={dynamic_dim}."
    )
    return refreshed


def _perform_action(action_label: str) -> None:
    if action_label == DANCE_ACTION_LABEL:
        print("Parrot is dancing")
    elif action_label == SING_ACTION_LABEL:
        print("Parrot is singing")


def _print_base_prediction(prediction: BasePrediction, *, k: int) -> None:
    neighbors = ", ".join(
        f"{label} ({distance:.3f})"
        for label, distance in zip(prediction.neighbor_labels, prediction.neighbor_distances)
    )
    print("")
    if prediction.is_known and prediction.predicted_label is not None:
        print(f"Base KNN prediction: {prediction.predicted_label}")
    else:
        print("Base KNN rejected the utterance.")
        print(
            f"Closest base label was {prediction.closest_label} at mean {k}-NN distance "
            f"{prediction.mean_neighbor_distance:.3f} "
            f"(threshold {prediction.unknown_distance_threshold:.3f})"
        )
    print(f"Nearest base neighbors: {neighbors}")


def _print_dynamic_prediction(prediction: DynamicPrediction, *, k: int) -> None:
    neighbors = ", ".join(
        f"{label} ({distance:.3f})"
        for label, distance in zip(prediction.neighbor_labels, prediction.neighbor_distances)
    )
    if prediction.is_known and prediction.action_label is not None:
        print(f"Dynamic KNN action: {prediction.action_label}")
    else:
        print("Dynamic KNN rejected the utterance. No action triggered.")
        print(
            f"Closest dynamic action was {prediction.closest_label} at mean {k}-NN distance "
            f"{prediction.mean_neighbor_distance:.3f} "
            f"(threshold {prediction.unknown_distance_threshold:.3f})"
        )
    print(f"Nearest dynamic neighbors: {neighbors}")
    print("Ready for the next word.")
    print("")


def run_live_dual_classification(
    *,
    model: DualLiveModel,
    config,
    project_root: Path,
) -> int:
    sample_rate = int(config.audio.sample_rate)
    endpointer = build_endpointer(config)
    _, manifest_path = ensure_storage(
        project_root=project_root,
        data_dir=config.storage.data_dir,
        manifest_path=config.storage.manifest_path,
    )
    dynamic_root = Path(model.dynamic_source_root).resolve()
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
    current_learning: Optional[LearningContext] = None

    default_device = describe_default_input()
    if default_device:
        print(
            f"Using input device {default_device.index}: {default_device.name} "
            f"({default_device.default_samplerate:.0f} Hz default)"
        )
    dynamic_count = 0 if model.dynamic_classifier is None else len(model.dynamic_classifier._train_labels)
    print(
        f"Dual KNN live classification is ready. base_k={model.base_classifier.k}, dynamic_k={DYNAMIC_K}, "
        f"base_training_samples={len(model.base_classifier._train_labels)}, dynamic_training_samples={dynamic_count}."
    )
    print(
        f"Base unknown threshold: mean {model.base_classifier.k}-NN distance <= "
        f"{model.base_unknown_distance_threshold:.3f}"
    )
    print(
        f"Dynamic unknown threshold: mean {DYNAMIC_K}-NN distance <= "
        f"{model.dynamic_unknown_distance_threshold:.3f}"
    )
    print("System is waiting for either a spoken word or a key press.")
    print(
        f"Press {LEARN_DANCE_KEY} to teach a new anonymous keyword mapped to dance, "
        f"{LEARN_SING_KEY} to teach one mapped to sing, and {config.collection.quit_key} to quit."
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
                        print("Stopping live dual classification.")
                        return 0
                    if current_learning is not None:
                        print(
                            f"Ignoring '{action}' while learning batch {current_learning.batch.keyword} is in progress."
                        )
                        continue
                    if action == "learn_dance":
                        batch = _begin_learning_batch(
                            source_root=dynamic_root,
                            action_label=DANCE_ACTION_LABEL,
                            batch_size=config.collection.batch_size,
                        )
                        current_learning = LearningContext(action_label=DANCE_ACTION_LABEL, batch=batch)
                        print("")
                        print(
                            f"Learning new anonymous keyword {batch.keyword} mapped to {DANCE_ACTION_LABEL}. "
                            f"Speak {batch.total} examples."
                        )
                        endpointer.arm()
                    elif action == "learn_sing":
                        batch = _begin_learning_batch(
                            source_root=dynamic_root,
                            action_label=SING_ACTION_LABEL,
                            batch_size=config.collection.batch_size,
                        )
                        current_learning = LearningContext(action_label=SING_ACTION_LABEL, batch=batch)
                        print("")
                        print(
                            f"Learning new anonymous keyword {batch.keyword} mapped to {SING_ACTION_LABEL}. "
                            f"Speak {batch.total} examples."
                        )
                        endpointer.arm()

                chunk = microphone.read_block(timeout=0.1)
                if chunk is None:
                    continue

                for event in endpointer.process_chunk(chunk):
                    if event.kind == "discarded":
                        if current_learning is not None:
                            print(
                                f"Discarded short noise while learning {current_learning.batch.keyword} "
                                f"({event.speech_ms} ms speech, {event.trailing_silence_m} ms trailing silence)."
                            )
                        continue
                    if event.kind != "utterance" or event.audio is None or event.audio.size == 0:
                        continue

                    if current_learning is not None:
                        record, completed, is_complete = save_batch_utterance(
                            event.audio,
                            project_root=project_root,
                            raw_dir=dynamic_root,
                            manifest_path=manifest_path,
                            batch=current_learning.batch,
                            sample_rate=sample_rate,
                        )
                        print(
                            f"Saved {record.sample_id} to {record.path} for {current_learning.batch.keyword} "
                            f"mapped to {current_learning.action_label} ({completed}/{current_learning.batch.total})"
                        )
                        if not is_complete:
                            endpointer.arm()
                            continue
                        print(
                            f"Finished learning batch {current_learning.batch.keyword} for action {current_learning.action_label}. "
                            f"Updating dynamic classifier."
                        )
                        model = _reload_model(model)
                        current_learning = None
                        endpointer.arm()
                        print("Continuous listening resumed.")
                        print("")
                        continue

                    base_prediction = predict_base_command(model, event.audio, sample_rate)
                    _print_base_prediction(base_prediction, k=model.base_classifier.k)
                    if base_prediction.is_known and base_prediction.predicted_label is not None:
                        _perform_action(base_prediction.predicted_label)
                        print("Ready for the next word.")
                        print("")
                        endpointer.arm()
                        continue

                    dynamic_prediction = predict_dynamic_command(model, event.audio, sample_rate)
                    _print_dynamic_prediction(dynamic_prediction, k=DYNAMIC_K)
                    if dynamic_prediction.is_known and dynamic_prediction.action_label is not None:
                        _perform_action(dynamic_prediction.action_label)
                    endpointer.arm()
    finally:
        listener.stop()


def main() -> int:
    config = load_app_config(config_path=CONFIG_PATH)
    if QUIT_KEY:
        config.collection.quit_key = QUIT_KEY

    model = prepare_dual_live_model(
        project_root=PROJECT_ROOT,
        base_source_root=Path(BASE_KEYWORD_SOURCE_ROOT).resolve(),
        dynamic_source_root=Path(DYNAMIC_KEYWORD_SOURCE_ROOT).resolve(),
    )
    if DRY_RUN:
        dynamic_count = 0 if model.dynamic_classifier is None else len(model.dynamic_classifier._train_labels)
        dynamic_dim = 0 if model.dynamic_classifier is None or model.dynamic_classifier._train_vectors.size == 0 else model.dynamic_classifier._train_vectors.shape[1]
        print(
            f"Dry run complete. base_k={model.base_classifier.k}, dynamic_k={DYNAMIC_K}, "
            f"base_source_root={model.base_source_root}, dynamic_source_root={model.dynamic_source_root}, "
            f"base_training_samples={len(model.base_classifier._train_labels)}, dynamic_training_samples={dynamic_count}, "
            f"base_dim={model.base_classifier._train_vectors.shape[1]}, dynamic_dim={dynamic_dim}, "
            f"base_unknown_threshold={model.base_unknown_distance_threshold:.3f}, "
            f"dynamic_unknown_threshold={model.dynamic_unknown_distance_threshold:.3f}"
        )
        return 0

    return run_live_dual_classification(
        model=model,
        config=config,
        project_root=PROJECT_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
