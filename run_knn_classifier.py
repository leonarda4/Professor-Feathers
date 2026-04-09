from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_core import (  # noqa: E402
    build_feature_matrix,
    split_keyword_label,
)
from feature_loading import load_sample_feature_parts_from_root  # noqa: E402
from knn_utils import (  # noqa: E402
    choose_knn_label,
    compute_accuracy,
    knn_predict,
    standardize_feature_matrices,
)


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "knn_split"


def collect_wav_files_by_label(source_root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for wav_path in sorted(source_root.rglob("*.wav")):
        relative_path = wav_path.relative_to(source_root)
        if len(relative_path.parts) < 3:
            continue
        label = relative_path.parts[0].strip()
        grouped.setdefault(label, []).append(wav_path)
    return grouped


def split_paths_by_label(
    grouped_paths: dict[str, list[Path]],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], dict[str, dict[str, int]]]:
    rng = random.Random(seed)
    train_paths: list[Path] = []
    test_paths: list[Path] = []
    counts: dict[str, dict[str, int]] = {}

    for label in sorted(grouped_paths):
        paths = list(grouped_paths[label])
        rng.shuffle(paths)
        sample_count = len(paths)
        if sample_count <= 1:
            label_test_count = 0
        else:
            label_test_count = int(round(sample_count * float(test_ratio)))
            if test_ratio > 0.0:
                label_test_count = max(1, label_test_count)
            label_test_count = min(sample_count - 1, label_test_count)

        label_test = paths[:label_test_count]
        label_train = paths[label_test_count:]
        train_paths.extend(label_train)
        test_paths.extend(label_test)
        counts[label] = {
            "total": sample_count,
            "train": len(label_train),
            "test": len(label_test),
        }
    return train_paths, test_paths, counts


def reset_output_root(output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Re-run with --force to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def copy_split_files(paths: list[Path], *, source_root: Path, split_root: Path) -> None:
    for source_path in paths:
        relative_path = source_path.relative_to(source_root)
        destination_path = split_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def write_predictions_csv(
    predictions_path: Path,
    test_features,
    predictions: list[dict[str, Any]],
) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "path",
                "session_id",
                "true_label",
                "predicted_label",
                "true_base_keyword",
                "predicted_base_keyword",
                "neighbor_labels",
                "neighbor_distances",
            ],
        )
        writer.writeheader()
        for sample_feature, prediction in zip(test_features, predictions):
            true_base_keyword, _ = split_keyword_label(sample_feature.record.keyword)
            predicted_base_keyword, _ = split_keyword_label(prediction["predicted_label"])
            writer.writerow(
                {
                    "sample_id": sample_feature.record.sample_id,
                    "path": sample_feature.record.path,
                    "session_id": sample_feature.record.session_id,
                    "true_label": sample_feature.record.keyword,
                    "predicted_label": prediction["predicted_label"],
                    "true_base_keyword": true_base_keyword,
                    "predicted_base_keyword": predicted_base_keyword,
                    "neighbor_labels": "|".join(prediction["neighbor_labels"]),
                    "neighbor_distances": "|".join(
                        f"{distance:.6f}" for distance in prediction["neighbor_distances"]
                    ),
                }
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_confusion_matrix(
    true_labels: list[str],
    predicted_labels: list[str],
    labels: list[str],
) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        true_index = label_to_index[true_label]
        predicted_index = label_to_index[predicted_label]
        matrix[true_index, predicted_index] += 1
    return matrix


def plot_confusion_matrix(
    axis,
    matrix: np.ndarray,
    labels: list[str],
    *,
    title: str,
    annotate: bool,
):
    row_sums = matrix.sum(axis=1, keepdims=True).astype(np.float64)
    normalized = np.divide(
        matrix.astype(np.float64),
        np.maximum(row_sums, 1.0),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=np.maximum(row_sums, 1.0) > 0,
    )
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)

    if annotate:
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                count = int(matrix[row_index, column_index])
                if count == 0:
                    continue
                value = normalized[row_index, column_index]
                text_color = "white" if value >= 0.6 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{count}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
    return image


def save_confusion_matrix_figure(
    output_path: Path,
    *,
    true_labels: list[str],
    predicted_labels: list[str],
) -> None:
    exact_labels = sorted(set(true_labels) | set(predicted_labels))
    base_true_labels = [split_keyword_label(label)[0] for label in true_labels]
    base_predicted_labels = [split_keyword_label(label)[0] for label in predicted_labels]
    base_labels = sorted(set(base_true_labels) | set(base_predicted_labels))

    exact_matrix = build_confusion_matrix(true_labels, predicted_labels, exact_labels)
    base_matrix = build_confusion_matrix(base_true_labels, base_predicted_labels, base_labels)
    exact_accuracy = compute_accuracy(true_labels, predicted_labels)
    base_accuracy = compute_accuracy(base_true_labels, base_predicted_labels)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(22, 10),
        gridspec_kw={"width_ratios": [2.5, 1.0]},
    )
    exact_image = plot_confusion_matrix(
        axes[0],
        exact_matrix,
        exact_labels,
        title=f"Exact Labels ({exact_accuracy:.1%} accuracy)",
        annotate=len(exact_labels) <= 12,
    )
    base_image = plot_confusion_matrix(
        axes[1],
        base_matrix,
        base_labels,
        title=f"Base Keywords ({base_accuracy:.1%} accuracy)",
        annotate=True,
    )
    figure.colorbar(exact_image, ax=axes[0], fraction=0.046, pad=0.04, label="Row-normalized")
    figure.colorbar(base_image, ax=axes[1], fraction=0.046, pad=0.04, label="Row-normalized")
    figure.suptitle("KNN Confusion Matrices", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a random train/test split from data/raw, copy the WAV files into "
            "train and test folders, and run a simple KNN classifier that writes "
            "predicted labels for the test set."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Folder containing raw keyword/session/sample.wav files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Folder where the train/test copy and KNN outputs will be written.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of each label assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the split.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to use.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    train_root = output_root / "train"
    test_root = output_root / "test"

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if not 0.0 <= float(args.test_ratio) < 1.0:
        raise ValueError("--test-ratio must be in the range [0.0, 1.0).")
    if int(args.k) < 1:
        raise ValueError("--k must be at least 1.")

    grouped_paths = collect_wav_files_by_label(source_root)
    if not grouped_paths:
        raise ValueError(f"No WAV files found under {source_root}")

    train_paths, test_paths, split_counts = split_paths_by_label(
        grouped_paths,
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    if not train_paths:
        raise ValueError("The split produced no training samples.")
    if not test_paths:
        raise ValueError("The split produced no test samples. Increase the dataset or test ratio.")

    reset_output_root(output_root, force=bool(args.force))
    copy_split_files(train_paths, source_root=source_root, split_root=train_root)
    copy_split_files(test_paths, source_root=source_root, split_root=test_root)

    train_parts = load_sample_feature_parts_from_root(PROJECT_ROOT, train_root)
    test_parts = load_sample_feature_parts_from_root(PROJECT_ROOT, test_root)
    train_vectors, train_labels = build_feature_matrix(
        train_parts,
        variant="f0_mean_std",
    )
    test_vectors, test_labels = build_feature_matrix(
        test_parts,
        variant="f0_mean_std",
    )

    train_vectors_scaled, test_vectors_scaled, scaler = standardize_feature_matrices(
        train_vectors,
        test_vectors,
    )
    predictions = knn_predict(
        train_vectors_scaled,
        train_labels,
        test_vectors_scaled,
        k=int(args.k),
    )

    predicted_labels = [item["predicted_label"] for item in predictions]
    true_base_labels = [split_keyword_label(label)[0] for label in test_labels]
    predicted_base_labels = [split_keyword_label(label)[0] for label in predicted_labels]
    exact_accuracy = compute_accuracy(test_labels, predicted_labels)
    base_accuracy = compute_accuracy(true_base_labels, predicted_base_labels)

    predictions_path = output_root / "predictions.csv"
    write_predictions_csv(predictions_path, test_parts, predictions)
    confusion_matrix_path = output_root / "confusion_matrices.png"
    save_confusion_matrix_figure(
        confusion_matrix_path,
        true_labels=test_labels,
        predicted_labels=predicted_labels,
    )

    write_json(output_root / "feature_scaler.json", scaler)
    write_json(
        output_root / "run_summary.json",
        {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "train_root": str(train_root),
            "test_root": str(test_root),
            "seed": int(args.seed),
            "test_ratio": float(args.test_ratio),
            "k": int(args.k),
            "feature_dimension": int(train_vectors.shape[1]),
            "train_sample_count": int(len(train_parts)),
            "test_sample_count": int(len(test_parts)),
            "exact_label_accuracy": exact_accuracy,
            "base_keyword_accuracy": base_accuracy,
            "split_counts": split_counts,
        },
    )

    print(f"Created train split at {train_root}")
    print(f"Created test split at {test_root}")
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote confusion matrices to {confusion_matrix_path}")
    print(f"Exact-label accuracy: {exact_accuracy:.3f}")
    print(f"Base-keyword accuracy: {base_accuracy:.3f}")
    print("")
    print("Example predictions:")
    for sample_feature, prediction in list(zip(test_parts, predictions))[:10]:
        print(
            f"{sample_feature.record.sample_id} | true={sample_feature.record.keyword} | "
            f"pred={prediction['predicted_label']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
