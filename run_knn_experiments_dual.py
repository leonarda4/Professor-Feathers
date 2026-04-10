from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio import read_wav  # noqa: E402
from features.feature_core import SampleFeatureParts  # noqa: E402
from features.feature_loading import load_sample_feature_parts_from_root  # noqa: E402
from features.feature_spaces import build_keyword_matrix, build_person_keyword_matrix, compute_delta_mfcc_mean  # noqa: E402
from knn_utils import compute_accuracy, knn_predict, standardize_feature_matrices  # noqa: E402

DEFAULT_BASE_SOURCE_ROOT = PROJECT_ROOT / "data" / "base_keywords"
DEFAULT_DYNAMIC_SOURCE_ROOT = PROJECT_ROOT / "data" / "dynamic_keywords"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "knn_experiments_dual"
DEFAULT_BASE_K_VALUES = [1, 3, 5, 7, 9, 11]
DEFAULT_DYNAMIC_K_VALUES = [1, 3, 5]
DANCE_PREFIX = "dance"
SING_PREFIX = "sing"


def infer_dynamic_action_label(keyword: str) -> str:
    cleaned = keyword.strip().lower()
    if cleaned.startswith(f"{DANCE_PREFIX}_"):
        return "dance"
    if cleaned.startswith(f"{SING_PREFIX}_"):
        return "sing"
    return cleaned


def reset_output_root(output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_root}. Re-run with --force to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def build_dynamic_delta_map(sample_parts: list[SampleFeatureParts], project_root: Path) -> dict[str, np.ndarray]:
    delta_map: dict[str, np.ndarray] = {}
    for item in sample_parts:
        sample_path = project_root / item.record.path
        if not sample_path.exists():
            continue
        samples, sample_rate = read_wav(sample_path)
        delta_map[item.record.sample_id] = compute_delta_mfcc_mean(samples, sample_rate)
    return delta_map


def evaluate_knn_grid(train_vectors, train_labels, test_vectors, test_labels, *, k_values, test_action_labels=None):
    train_vectors_scaled, test_vectors_scaled, _ = standardize_feature_matrices(train_vectors, test_vectors)
    rows = []
    for k in k_values:
        preds = knn_predict(train_vectors_scaled, train_labels, test_vectors_scaled, k=k)
        predicted_labels = [item["predicted_label"] for item in preds]
        row = {
            "k": int(k),
            "keyword_accuracy": float(compute_accuracy(test_labels, predicted_labels)),
        }
        if test_action_labels is not None:
            predicted_action_labels = [infer_dynamic_action_label(label) for label in predicted_labels]
            row["action_accuracy"] = float(compute_accuracy(test_action_labels, predicted_action_labels))
        rows.append(row)
    return rows


def leave_one_keyword_out_indices(labels: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    all_indices = np.arange(len(labels), dtype=int)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for label, indices in sorted(grouped.items()):
        test_indices = np.asarray(sorted(indices), dtype=int)
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]
        if train_indices.size == 0:
            continue
        folds.append((train_indices, test_indices))
    return folds


def evaluate_dynamic_keyword_generalization(vectors, labels, *, k_values):
    folds = leave_one_keyword_out_indices(labels)
    rows = []
    for k in k_values:
        action_fold_accuracies = []
        for train_indices, test_indices in folds:
            train_vectors = vectors[train_indices]
            test_vectors = vectors[test_indices]
            train_labels = [infer_dynamic_action_label(labels[i]) for i in train_indices]
            test_action_labels = [infer_dynamic_action_label(labels[i]) for i in test_indices]
            fold_rows = evaluate_knn_grid(
                train_vectors,
                train_labels,
                test_vectors,
                test_action_labels,
                k_values=[k],
                test_action_labels=test_action_labels,
            )
            action_fold_accuracies.append(float(fold_rows[0]["action_accuracy"]))
        rows.append(
            {
                "k": int(k),
                "mean_action_accuracy": float(np.mean(action_fold_accuracies, dtype=np.float64)),
                "std_action_accuracy": float(np.std(action_fold_accuracies, dtype=np.float64)),
                "fold_count": int(len(action_fold_accuracies)),
            }
        )
    return rows


def save_results_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_plot(path: Path, holdout_rows: list[dict[str, Any]], generalization_rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    base_rows = sorted([row for row in holdout_rows if row["model"] == "base"], key=lambda row: row["k"])
    dyn_rows = sorted([row for row in holdout_rows if row["model"] == "dynamic"], key=lambda row: row["k"])
    gen_rows = sorted(generalization_rows, key=lambda row: row["k"])

    axes[0].plot([r["k"] for r in base_rows], [r["keyword_accuracy"] * 100 for r in base_rows], marker="o")
    axes[0].set_title("Base holdout keyword accuracy")
    axes[1].plot([r["k"] for r in dyn_rows], [r["keyword_accuracy"] * 100 for r in dyn_rows], marker="o", label="keyword")
    axes[1].plot([r["k"] for r in dyn_rows], [r["action_accuracy"] * 100 for r in dyn_rows], marker="s", linestyle="--", label="action")
    axes[1].set_title("Dynamic holdout accuracy")
    axes[1].legend()
    axes[2].errorbar(
        [r["k"] for r in gen_rows],
        [r["mean_action_accuracy"] * 100 for r in gen_rows],
        yerr=[r["std_action_accuracy"] * 100 for r in gen_rows],
        marker="o",
        capsize=4,
    )
    axes[2].set_title("Dynamic leave-one-keyword-out")

    for axis in axes:
        axis.set_xlabel("k")
        axis.set_ylabel("Accuracy (%)")
        axis.grid(alpha=0.3)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate dual KNN with harder dynamic validation.")
    parser.add_argument("--base-source-root", type=Path, default=DEFAULT_BASE_SOURCE_ROOT)
    parser.add_argument("--dynamic-source-root", type=Path, default=DEFAULT_DYNAMIC_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-k-values", type=int, nargs="+", default=DEFAULT_BASE_K_VALUES)
    parser.add_argument("--dynamic-k-values", type=int, nargs="+", default=DEFAULT_DYNAMIC_K_VALUES)
    parser.add_argument("--force", action="store_true")
    return parser


def split_indices_by_label(labels: list[str], test_ratio: float, seed: int):
    grouped = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    split_counts = {}
    for label, indices in grouped.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        test_count = int(math.ceil(len(shuffled) * test_ratio)) if len(shuffled) > 1 else 0
        if test_count >= len(shuffled):
            test_count = len(shuffled) - 1
        label_test = sorted(shuffled[:test_count])
        label_train = sorted(shuffled[test_count:])
        if not label_train:
            label_train = [label_test.pop()]
        train_indices.extend(label_train)
        test_indices.extend(label_test)
        split_counts[label] = {"train": len(label_train), "test": len(label_test)}
    return np.asarray(sorted(train_indices), dtype=int), np.asarray(sorted(test_indices), dtype=int), split_counts


def main() -> int:
    args = build_arg_parser().parse_args()
    base_source_root = Path(args.base_source_root).resolve()
    dynamic_source_root = Path(args.dynamic_source_root).resolve()
    output_root = Path(args.output_root).resolve()
    reset_output_root(output_root, force=bool(args.force))

    base_parts = load_sample_feature_parts_from_root(PROJECT_ROOT, base_source_root)
    dynamic_parts = load_sample_feature_parts_from_root(PROJECT_ROOT, dynamic_source_root)
    base_vectors, base_labels = build_keyword_matrix(base_parts)
    dynamic_label_map = {item.record.sample_id: item.record.keyword for item in dynamic_parts}
    dynamic_delta_map = build_dynamic_delta_map(dynamic_parts, PROJECT_ROOT)
    dynamic_vectors, dynamic_labels = build_person_keyword_matrix(dynamic_parts, label_map=dynamic_label_map, delta_map=dynamic_delta_map)
    dynamic_action_labels = [infer_dynamic_action_label(label) for label in dynamic_labels]

    holdout_rows = []

    base_train_idx, base_test_idx, _ = split_indices_by_label(base_labels, float(args.test_ratio), int(args.seed))
    for row in evaluate_knn_grid(
        base_vectors[base_train_idx],
        [base_labels[i] for i in base_train_idx],
        base_vectors[base_test_idx],
        [base_labels[i] for i in base_test_idx],
        k_values=sorted({int(v) for v in args.base_k_values if int(v) >= 1}),
    ):
        holdout_rows.append({"model": "base", **row})

    dyn_train_idx, dyn_test_idx, _ = split_indices_by_label(dynamic_labels, float(args.test_ratio), int(args.seed))
    for row in evaluate_knn_grid(
        dynamic_vectors[dyn_train_idx],
        [dynamic_labels[i] for i in dyn_train_idx],
        dynamic_vectors[dyn_test_idx],
        [dynamic_labels[i] for i in dyn_test_idx],
        k_values=sorted({int(v) for v in args.dynamic_k_values if int(v) >= 1}),
        test_action_labels=[dynamic_action_labels[i] for i in dyn_test_idx],
    ):
        holdout_rows.append({"model": "dynamic", **row})

    generalization_rows = evaluate_dynamic_keyword_generalization(
        dynamic_vectors,
        dynamic_labels,
        k_values=sorted({int(v) for v in args.dynamic_k_values if int(v) >= 1}),
    )

    save_results_csv(
        output_root / "holdout_results.csv",
        holdout_rows,
        ["model", "k", "keyword_accuracy", "action_accuracy"],
    )
    save_results_csv(
        output_root / "dynamic_keyword_generalization.csv",
        generalization_rows,
        ["k", "mean_action_accuracy", "std_action_accuracy", "fold_count"],
    )
    save_plot(output_root / "accuracy.png", holdout_rows, generalization_rows)

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"holdout": holdout_rows, "dynamic_keyword_generalization": generalization_rows}, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
