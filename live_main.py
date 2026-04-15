from __future__ import annotations

import queue
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio import MicrophoneStream, describe_default_input, read_wav
from parrot_feedback import AudioFeedbackPlayer
from collect import (
    RecordingBatch,
    build_endpointer,
    drain_actions,
    save_batch_utterance,
    start_hotkey_listener,
    start_recording_batch,
)
from config import load_app_config
from data_augmentation import AugmentationConfig, build_augmented_feature_parts
from dual_knn import DynamicKeywordClassifier, KeywordClassifier
from base_svm import BaseSVMClassifier
from features.feature_core import extract_feature_parts
from features.feature_loading import load_sample_feature_parts_from_root
from features.feature_spaces import compute_delta_mfcc_mean
from storage import SampleRecord, ensure_storage

try:
    from servo.servo_snd import run_dance_sequence as _run_dance_sequence
except Exception as exc:
    _run_dance_sequence = None
    _DANCE_MOVEMENT_IMPORT_ERROR: Exception | None = exc
else:
    _DANCE_MOVEMENT_IMPORT_ERROR = None

from servo.servo_snd import open_serial

BAUDRATE = 115200
CONFIG_PATH = None
BASE_KEYWORD_SOURCE_ROOT = PROJECT_ROOT / "data" / "base_keywords"
DYNAMIC_KEYWORD_SOURCE_ROOT = PROJECT_ROOT / "data" / "dynamic_keywords"
BASE_MODEL_TYPE = "knn"
BASE_K = 5
BASE_SVM_KERNEL = "rbf"
BASE_SVM_C = 10.0
BASE_SVM_GAMMA = "scale"
BASE_SVM_MARGIN_THRESHOLD = 0.45
DYNAMIC_K = 3
BASE_UNKNOWN_DISTANCE_THRESHOLD = None
BASE_UNKNOWN_DISTANCE_PERCENTILE = 96.0
BASE_UNKNOWN_DISTANCE_MARGIN = 1.05
DYNAMIC_UNKNOWN_DISTANCE_THRESHOLD = None
DYNAMIC_UNKNOWN_DISTANCE_PERCENTILE = 97.0
DYNAMIC_UNKNOWN_DISTANCE_MARGIN = 1.10
BASE_MIN_LABEL_VOTE_RATIO = 0.80
LEARN_DANCE_KEY = "d"
LEARN_SING_KEY = "s"
DELETE_DYNAMIC_KEYWORDS_KEY = "x"
QUIT_KEY = None
DRY_RUN = False
USE_DELTA_FOR_DYNAMIC_KNN = True
DANCE_ACTION_LABEL = "dance"
SING_ACTION_LABEL = "sing"
DANCE_PREFIX = "dance"
SING_PREFIX = "sing"
PARROT_FEEDBACK_ROOT = PROJECT_ROOT / "data" / "parrot_voice"
DANCE_FEEDBACK_PROBABILITY = 0.45
MAX_DYNAMIC_KEYWORDS = 20
REQUIRE_DELETE_CONFIRMATION = True
FEEDBACK_RECOGNIZED_PAUSE_SECONDS = 4.80
FEEDBACK_NOT_RECOGNIZED_PAUSE_SECONDS = 2.20
FEEDBACK_TRAINING_PAUSE_SECONDS = 2.20

_DANCE_MOVEMENT_LOCK = threading.Lock()
_dance_movement_thread: threading.Thread | None = None


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
    base_classifier: KeywordClassifier | BaseSVMClassifier
    base_model_type: str
    dynamic_classifier: DynamicKeywordClassifier | None
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
        distances = np.linalg.norm(
            train_vectors_scaled - train_vectors_scaled[index],
            axis=1,
        )
        distances[index] = np.inf
        nearest_distances = np.sort(distances)[:effective_k]
        mean_neighbor_distances.append(
            float(np.mean(nearest_distances, dtype=np.float64))
        )

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


def _build_dynamic_delta_map(sample_parts, project_root: Path) -> dict[str, np.ndarray]:
    if not USE_DELTA_FOR_DYNAMIC_KNN:
        return {}

    delta_map: dict[str, np.ndarray] = {}
    for item in sample_parts:
        sample_path = project_root / item.record.path
        if not sample_path.exists():
            continue
        samples, sample_rate = read_wav(sample_path)
        delta_map[item.record.path] = compute_delta_mfcc_mean(samples, sample_rate)
    return delta_map


def _count_dynamic_keyword_folders(source_root: Path) -> int:
    source_root = Path(source_root).resolve()
    if not source_root.exists():
        return 0
    return sum(1 for child in source_root.iterdir() if child.is_dir())


def _can_start_new_dynamic_keyword(source_root: Path) -> bool:
    return _count_dynamic_keyword_folders(source_root) < int(MAX_DYNAMIC_KEYWORDS)


def _delete_dynamic_keyword_dataset(source_root: Path) -> None:
    source_root = Path(source_root).resolve()
    if not source_root.exists():
        source_root.mkdir(parents=True, exist_ok=True)
        return

    for child in source_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        elif child.is_file():
            child.unlink()


def _pause_listening_during_feedback(
    microphone: MicrophoneStream,
    endpointer,
    *,
    seconds: float,
) -> None:
    pause_seconds = max(0.0, float(seconds))
    if pause_seconds <= 0.0:
        endpointer.arm()
        return

    deadline = time.monotonic() + pause_seconds
    while time.monotonic() < deadline:
        _ = microphone.read_block(timeout=0.05)

    endpointer.arm()


def _play_feedback_and_pause(
    microphone: MicrophoneStream,
    endpointer,
    playback_started: bool,
    *,
    pause_seconds: float,
) -> None:
    if playback_started:
        _pause_listening_during_feedback(
            microphone,
            endpointer,
            seconds=pause_seconds,
        )
    else:
        endpointer.arm()


def prepare_dynamic_classifier(project_root: Path, dynamic_source_root: Path):
    dynamic_parts = []
    if Path(dynamic_source_root).exists():
        dynamic_parts = load_sample_feature_parts_from_root(project_root, dynamic_source_root)

    dynamic_classifier: DynamicKeywordClassifier | None = None
    dynamic_unknown_distance_threshold = float("inf")

    if dynamic_parts:
        delta_map = _build_dynamic_delta_map(dynamic_parts, project_root)
        dynamic_classifier = DynamicKeywordClassifier(k=DYNAMIC_K).fit(
            dynamic_parts,
            delta_map=delta_map if delta_map else None,
        )
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

    return dynamic_classifier, dynamic_unknown_distance_threshold


def prepare_base_classifier(base_parts, *, model_type: str):
    normalized_type = str(model_type).strip().lower()

    if normalized_type == "knn":
        base_parts_aug = build_augmented_feature_parts(
            base_parts,
            project_root=PROJECT_ROOT,
            config=AugmentationConfig(copies_per_sample=3),
            seed=42,
        )
        classifier = KeywordClassifier(k=BASE_K).fit(base_parts_aug)
        classifier = classifier.fit(base_parts)
        threshold = (
            float(BASE_UNKNOWN_DISTANCE_THRESHOLD)
            if BASE_UNKNOWN_DISTANCE_THRESHOLD is not None
            else _auto_unknown_distance_threshold(
                classifier._train_vectors,
                k=classifier.k,
                percentile=BASE_UNKNOWN_DISTANCE_PERCENTILE,
                margin=BASE_UNKNOWN_DISTANCE_MARGIN,
            )
        )
        return classifier, float(threshold)

    if normalized_type == "svm":
        classifier = BaseSVMClassifier(
            kernel=BASE_SVM_KERNEL,
            C=BASE_SVM_C,
            gamma=BASE_SVM_GAMMA,
        ).fit(base_parts)
        return classifier, float(BASE_SVM_MARGIN_THRESHOLD)

    raise ValueError(f"Unsupported base model type: {model_type}")


def prepare_dual_live_model(
    *,
    project_root: Path,
    base_source_root: Path,
    dynamic_source_root: Path,
    base_model_type: str = BASE_MODEL_TYPE,
) -> DualLiveModel:
    base_parts = load_sample_feature_parts_from_root(project_root, base_source_root)
    if not base_parts:
        raise ValueError(f"No base keyword samples found under {base_source_root}")

    base_classifier, base_unknown_distance_threshold = prepare_base_classifier(
        base_parts,
        model_type=base_model_type,
    )
    dynamic_classifier, dynamic_unknown_distance_threshold = prepare_dynamic_classifier(
        project_root,
        dynamic_source_root,
    )

    return DualLiveModel(
        base_source_root=Path(base_source_root).resolve(),
        dynamic_source_root=Path(dynamic_source_root).resolve(),
        base_classifier=base_classifier,
        base_model_type=str(base_model_type).strip().lower(),
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
    closest_label = str(prediction["predicted_label"])

    if model.base_model_type == "svm":
        margin = float(model.base_classifier.decision_margin(parts))
        is_known = margin >= model.base_unknown_distance_threshold
        return BasePrediction(
            bool(is_known),
            closest_label if is_known else None,
            closest_label,
            margin,
            float(model.base_unknown_distance_threshold),
            margin,
            float(model.base_unknown_distance_threshold),
            [],
            [],
        )

    mean_neighbor_distance = model.base_classifier.mean_neighbor_distance(parts)
    neighbor_labels = list(prediction["neighbor_labels"])
    neighbor_distances = list(prediction["neighbor_distances"])
    label_vote_count = sum(1 for label in neighbor_labels if label == closest_label)
    label_vote_ratio = (
        float(label_vote_count) / float(len(neighbor_labels)) if neighbor_labels else 0.0
    )
    is_known = (
        mean_neighbor_distance <= model.base_unknown_distance_threshold
        and label_vote_ratio >= BASE_MIN_LABEL_VOTE_RATIO
    )

    return BasePrediction(
        bool(is_known),
        closest_label if is_known else None,
        closest_label,
        float(mean_neighbor_distance),
        float(model.base_unknown_distance_threshold),
        float(label_vote_ratio),
        float(BASE_MIN_LABEL_VOTE_RATIO),
        neighbor_labels,
        neighbor_distances,
    )


def predict_dynamic_command(
    model: DualLiveModel,
    samples: np.ndarray,
    sample_rate: int,
) -> DynamicPrediction:
    if model.dynamic_classifier is None:
        return DynamicPrediction(
            False,
            None,
            "none",
            float("inf"),
            float(model.dynamic_unknown_distance_threshold),
            [],
            [],
        )

    parts = extract_feature_parts(_build_live_record("live_dynamic"), samples, sample_rate)
    delta = compute_delta_mfcc_mean(samples, sample_rate) if model.dynamic_uses_delta else None
    prediction = model.dynamic_classifier.predict(parts, delta_mfcc_mean=delta)
    mean_neighbor_distance = model.dynamic_classifier.mean_neighbor_distance(
        parts,
        delta_mfcc_mean=delta,
    )
    closest_label = str(prediction["predicted_label"])
    action_label = _infer_action_label(closest_label)
    is_known = mean_neighbor_distance <= model.dynamic_unknown_distance_threshold

    return DynamicPrediction(
        bool(is_known),
        action_label if is_known else None,
        closest_label,
        float(mean_neighbor_distance),
        float(model.dynamic_unknown_distance_threshold),
        list(prediction["neighbor_labels"]),
        list(prediction["neighbor_distances"]),
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


def _begin_learning_batch(
    *,
    source_root: Path,
    action_label: str,
    batch_size: int,
) -> RecordingBatch:
    prefix = DANCE_PREFIX if action_label == DANCE_ACTION_LABEL else SING_PREFIX
    return start_recording_batch(
        _next_anonymous_keyword(source_root, prefix),
        batch_size=batch_size,
    )


def _reload_dynamic_model(model: DualLiveModel) -> None:
    print("Rebuilding dynamic classifier with the updated dataset...")
    model.dynamic_classifier, model.dynamic_unknown_distance_threshold = (
        prepare_dynamic_classifier(
            project_root=PROJECT_ROOT,
            dynamic_source_root=model.dynamic_source_root,
        )
    )


def _run_dance_sequence_worker() -> None:
    global _dance_movement_thread

    try:
        if _run_dance_sequence is None:
            if _DANCE_MOVEMENT_IMPORT_ERROR is not None:
                print(f"Dance movement is unavailable: {_DANCE_MOVEMENT_IMPORT_ERROR}")
            return
        _run_dance_sequence()
    except Exception as exc:
        print(f"Dance movement failed: {exc}")
    finally:
        with _DANCE_MOVEMENT_LOCK:
            _dance_movement_thread = None


def _trigger_dance_movement() -> bool:
    global _dance_movement_thread

    if _run_dance_sequence is None:
        if _DANCE_MOVEMENT_IMPORT_ERROR is not None:
            print(f"Dance movement is unavailable: {_DANCE_MOVEMENT_IMPORT_ERROR}")
        return False

    with _DANCE_MOVEMENT_LOCK:
        if _dance_movement_thread is not None and _dance_movement_thread.is_alive():
            print("Dance movement is already running. Skipping duplicate trigger.")
            return False

        _dance_movement_thread = threading.Thread(
            target=_run_dance_sequence_worker,
            name="dance-movement",
            daemon=True,
        )
        _dance_movement_thread.start()
        return True


def _perform_action(action_label: str) -> None:
    if action_label == DANCE_ACTION_LABEL:
        print("Parrot is dancing")
        if _trigger_dance_movement():
            print("Dance movement started.")
    elif action_label == SING_ACTION_LABEL:
        print("Parrot is singing")


def _print_base_prediction(prediction: BasePrediction, *, model_type: str, k: int) -> None:
    print("")
    model_name = str(model_type).upper()

    if model_type == "svm":
        if prediction.is_known and prediction.predicted_label is not None:
            print(f"Base {model_name} prediction: {prediction.predicted_label}")
        else:
            print(f"Base {model_name} rejected the utterance.")
        print(
            f"Closest base label was {prediction.closest_label} with decision margin "
            f"{prediction.mean_neighbor_distance:.3f} "
            f"(threshold {prediction.unknown_distance_threshold:.3f})"
        )
        return

    neighbors = ", ".join(
        f"{label} ({distance:.3f})"
        for label, distance in zip(
            prediction.neighbor_labels,
            prediction.neighbor_distances,
        )
    )
    if prediction.is_known and prediction.predicted_label is not None:
        print(f"Base {model_name} prediction: {prediction.predicted_label}")
    else:
        print(f"Base {model_name} rejected the utterance.")
    print(
        f"Closest base label was {prediction.closest_label} at mean {k}-NN distance "
        f"{prediction.mean_neighbor_distance:.3f} "
        f"(threshold {prediction.unknown_distance_threshold:.3f})"
    )
    print(f"Nearest base neighbors: {neighbors}")


def _print_dynamic_prediction(prediction: DynamicPrediction, *, k: int) -> None:
    neighbors = ", ".join(
        f"{label} ({distance:.3f})"
        for label, distance in zip(
            prediction.neighbor_labels,
            prediction.neighbor_distances,
        )
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


def run_live_dual_classification(*, model: DualLiveModel, config, project_root: Path) -> int:
    feedback = AudioFeedbackPlayer(
        PARROT_FEEDBACK_ROOT,
        rng_seed=42,
        dance_trigger_probability=DANCE_FEEDBACK_PROBABILITY,
    )
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
            DELETE_DYNAMIC_KEYWORDS_KEY: "delete_dynamic_keywords",
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
        f"Live classification is ready. base_model_type={model.base_model_type}, "
        f"dynamic_k={DYNAMIC_K}, base_training_samples={len(model.base_classifier._train_labels)}, "
        f"dynamic_training_samples={dynamic_count}."
    )
    print(f"Parrot feedback clips: {feedback.available_summary()}")
    print(f"Dynamic keyword folders: {_count_dynamic_keyword_folders(dynamic_root)}/{MAX_DYNAMIC_KEYWORDS}")

    if model.base_model_type == "knn":
        print(
            f"Base unknown threshold: mean {model.base_classifier.k}-NN distance <= "
            f"{model.base_unknown_distance_threshold:.3f}"
        )
    else:
        print(
            f"Base classifier uses SVM with kernel={BASE_SVM_KERNEL}, C={BASE_SVM_C}, "
            f"gamma={BASE_SVM_GAMMA}, margin_threshold={model.base_unknown_distance_threshold:.3f}"
        )

    print(
        f"Dynamic unknown threshold: mean {DYNAMIC_K}-NN distance <= "
        f"{model.dynamic_unknown_distance_threshold:.3f}"
    )
    print("System is waiting for either a spoken word or a key press.")
    print(
        f"Press {LEARN_DANCE_KEY} to teach a new anonymous keyword mapped to dance, "
        f"{LEARN_SING_KEY} to teach one mapped to sing, "
        f"{DELETE_DYNAMIC_KEYWORDS_KEY} to delete all dynamic keywords, "
        f"and {config.collection.quit_key} to quit."
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
                            f"Ignoring '{action}' while learning batch "
                            f"{current_learning.batch.keyword} is in progress."
                        )
                        continue

                    if action in {"learn_dance", "learn_sing"}:
                        if not _can_start_new_dynamic_keyword(dynamic_root):
                            print(
                                f"Dynamic keyword limit reached ({MAX_DYNAMIC_KEYWORDS}). "
                                "Delete the current dynamic dataset before recording a new keyword."
                            )
                            continue

                        if action == "learn_dance":
                            batch = _begin_learning_batch(
                                source_root=dynamic_root,
                                action_label=DANCE_ACTION_LABEL,
                                batch_size=config.collection.batch_size,
                            )
                            current_learning = LearningContext(
                                action_label=DANCE_ACTION_LABEL,
                                batch=batch,
                            )
                            print(
                                f"\nLearning new anonymous keyword {batch.keyword} mapped to "
                                f"{DANCE_ACTION_LABEL}. Speak {batch.total} examples."
                            )
                        else:
                            batch = _begin_learning_batch(
                                source_root=dynamic_root,
                                action_label=SING_ACTION_LABEL,
                                batch_size=config.collection.batch_size,
                            )
                            current_learning = LearningContext(
                                action_label=SING_ACTION_LABEL,
                                batch=batch,
                            )
                            print(
                                f"\nLearning new anonymous keyword {batch.keyword} mapped to "
                                f"{SING_ACTION_LABEL}. Speak {batch.total} examples."
                            )

                        endpointer.arm()
                        continue

                    if action == "delete_dynamic_keywords":
                        if REQUIRE_DELETE_CONFIRMATION:
                            confirmation = input(
                                "Delete all dynamic keyword datasets? Type DELETE to confirm: "
                            ).strip()
                            if confirmation != "DELETE":
                                print("Dynamic dataset deletion cancelled.")
                                continue

                        _delete_dynamic_keyword_dataset(dynamic_root)
                        _reload_dynamic_model(model)
                        print("All dynamic keyword datasets were deleted. Dynamic classifier reset.")
                        print(
                            f"Dynamic keyword folders: "
                            f"{_count_dynamic_keyword_folders(dynamic_root)}/{MAX_DYNAMIC_KEYWORDS}"
                        )
                        endpointer.arm()
                        continue

                chunk = microphone.read_block(timeout=0.1)
                if chunk is None:
                    continue

                for event in endpointer.process_chunk(chunk):
                    if event.kind == "discarded":
                        if current_learning is not None:
                            print(
                                f"Discarded short noise while learning {current_learning.batch.keyword} "
                                f"({event.speech_ms} ms speech, "
                                f"{event.trailing_silence_ms} ms trailing silence)."
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
                            f"Saved {record.sample_id} to {record.path} for "
                            f"{current_learning.batch.keyword} mapped to "
                            f"{current_learning.action_label} "
                            f"({completed}/{current_learning.batch.total})"
                        )

                        if not is_complete:
                            playback_started = feedback.maybe_play_training()
                            _play_feedback_and_pause(
                                microphone,
                                endpointer,
                                playback_started,
                                pause_seconds=FEEDBACK_TRAINING_PAUSE_SECONDS,
                            )
                            continue

                        print(
                            f"Finished learning batch {current_learning.batch.keyword} for action "
                            f"{current_learning.action_label}. Updating dynamic classifier."
                        )
                        _reload_dynamic_model(model)
                        current_learning = None
                        endpointer.arm()
                        print(
                            f"Continuous listening resumed. Dynamic keyword folders: "
                            f"{_count_dynamic_keyword_folders(dynamic_root)}/{MAX_DYNAMIC_KEYWORDS}\n"
                        )
                        continue

                    base_prediction = predict_base_command(model, event.audio, sample_rate)
                    _print_base_prediction(
                        base_prediction,
                        model_type=model.base_model_type,
                        k=getattr(model.base_classifier, "k", 0),
                    )

                    if base_prediction.is_known and base_prediction.predicted_label is not None:
                        playback_started = feedback.maybe_play_recognized(
                            base_prediction.predicted_label
                        )
                        _perform_action(base_prediction.predicted_label)
                        _play_feedback_and_pause(
                            microphone,
                            endpointer,
                            playback_started,
                            pause_seconds=FEEDBACK_RECOGNIZED_PAUSE_SECONDS,
                        )
                        print("Ready for the next word.\n")
                        continue

                    dynamic_prediction = predict_dynamic_command(
                        model,
                        event.audio,
                        sample_rate,
                    )
                    _print_dynamic_prediction(dynamic_prediction, k=DYNAMIC_K)

                    if dynamic_prediction.is_known and dynamic_prediction.action_label is not None:
                        playback_started = feedback.maybe_play_recognized(
                            dynamic_prediction.action_label
                        )
                        _perform_action(dynamic_prediction.action_label)
                        _play_feedback_and_pause(
                            microphone,
                            endpointer,
                            playback_started,
                            pause_seconds=FEEDBACK_RECOGNIZED_PAUSE_SECONDS,
                        )
                    else:
                        playback_started = feedback.play_not_recognized()
                        _play_feedback_and_pause(
                            microphone,
                            endpointer,
                            playback_started,
                            pause_seconds=FEEDBACK_NOT_RECOGNIZED_PAUSE_SECONDS,
                        )
    finally:
        listener.stop()


def main() -> int:
    config = load_app_config(config_path=CONFIG_PATH)
    if QUIT_KEY:
        config.collection.quit_key = QUIT_KEY

    chosen_base_model = BASE_MODEL_TYPE
    open_serial(baudrate=BAUDRATE)
    model = prepare_dual_live_model(
        project_root=PROJECT_ROOT,
        base_source_root=Path(BASE_KEYWORD_SOURCE_ROOT).resolve(),
        dynamic_source_root=Path(DYNAMIC_KEYWORD_SOURCE_ROOT).resolve(),
        base_model_type=chosen_base_model,
    )

    if DRY_RUN:
        dynamic_count = 0 if model.dynamic_classifier is None else len(model.dynamic_classifier._train_labels)
        dynamic_dim = (
            0
            if model.dynamic_classifier is None or model.dynamic_classifier._train_vectors.size == 0
            else model.dynamic_classifier._train_vectors.shape[1]
        )
        print(
            f"Dry run complete. base_model_type={model.base_model_type}, "
            f"base_k={getattr(model.base_classifier, 'k', 'n/a')}, dynamic_k={DYNAMIC_K}, "
            f"base_source_root={model.base_source_root}, dynamic_source_root={model.dynamic_source_root}, "
            f"base_training_samples={len(model.base_classifier._train_labels)}, "
            f"dynamic_training_samples={dynamic_count}, base_dim={model.base_classifier._train_vectors.shape[1]}, "
            f"dynamic_dim={dynamic_dim}, base_unknown_threshold={model.base_unknown_distance_threshold:.3f}, "
            f"dynamic_unknown_threshold={model.dynamic_unknown_distance_threshold:.3f}, "
            f"dynamic_keyword_limit={MAX_DYNAMIC_KEYWORDS}"
        )
        return 0

    return run_live_dual_classification(
        model=model,
        config=config,
        project_root=PROJECT_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
