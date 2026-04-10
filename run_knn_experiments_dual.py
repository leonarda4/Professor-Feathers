from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
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
DEFAULT_DYNAMIC_K_VALUES = [1, 3, 5, 7]
DEFAULT_CV_FOLDS = [3, 5]
DANCE_PREFIX = "dance"
SING_PREFIX = "sing"


def infer_dynamic_action_label(keyword: str) -> str:
    cleaned = keyword.strip().lower()
    if cleaned.startswith(f"{DANCE_PREFIX}_"):
        return "dance"
    if cleaned.startswith(f"{SING_PREFIX}_"):
        return "sing"
    return cleaned


def dynamic_key(item: SampleFeatureParts) -> str:
    return str(item.record.path)


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
        delta_map[dynamic_key(item)] = compute_delta_mfcc_mean(samples, sample_rate)
    return delta_map


def confusion_matrix_counts(true_labels: list[str], predicted_labels: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label in index and predicted_label in index:
            matrix[index[true_label]][index[predicted_label]] += 1
    return matrix


def save_results_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
        if not label_train and label_test:
            label_train = [label_test.pop()]
        train_indices.extend(label_train)
        test_indices.extend(label_test)
        split_counts[label] = {"train": len(label_train), "test": len(label_test)}
    return np.asarray(sorted(train_indices), dtype=int), np.asarray(sorted(test_indices), dtype=int), split_counts


def make_stratified_folds(labels: list[str], n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[label].append(idx)
    if len(labels) == 0:
        return []
    min_count = min(len(v) for v in grouped.values())
    effective_splits = max(2, min(int(n_splits), int(min_count))) if min_count >= 2 else 1
    if effective_splits < 2:
        return []
    rng = np.random.default_rng(seed)
    fold_buckets = [[] for _ in range(effective_splits)]
    for label in sorted(grouped):
        idxs = list(grouped[label])
        rng.shuffle(idxs)
        for pos, idx in enumerate(idxs):
            fold_buckets[pos % effective_splits].append(idx)
    all_indices = np.arange(len(labels), dtype=int)
    folds = []
    for bucket in fold_buckets:
        test_idx = np.asarray(sorted(bucket), dtype=int)
        mask = np.ones(len(labels), dtype=bool)
        mask[test_idx] = False
        train_idx = all_indices[mask]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def evaluate_single_split(train_vectors, train_labels, test_vectors, test_labels, k: int) -> tuple[float, list[str]]:
    train_scaled, test_scaled, _ = standardize_feature_matrices(train_vectors, test_vectors)
    preds = knn_predict(train_scaled, train_labels, test_scaled, k=k)
    predicted_labels = [item["predicted_label"] for item in preds]
    return float(compute_accuracy(test_labels, predicted_labels)), predicted_labels


def run_grid_search(vectors: np.ndarray, labels: list[str], *, k_values: list[int], fold_values: list[int], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for folds in sorted({int(v) for v in fold_values if int(v) >= 2}):
        split_folds = make_stratified_folds(labels, folds, seed)
        if not split_folds:
            continue
        for k in sorted({int(v) for v in k_values if int(v) >= 1}):
            fold_scores = []
            for train_idx, val_idx in split_folds:
                acc, _ = evaluate_single_split(
                    vectors[train_idx],
                    [labels[i] for i in train_idx],
                    vectors[val_idx],
                    [labels[i] for i in val_idx],
                    k,
                )
                fold_scores.append(acc)
            row = {
                "folds": int(len(split_folds)),
                "requested_folds": int(folds),
                "k": int(k),
                "mean_validation_accuracy": float(np.mean(fold_scores, dtype=np.float64)),
                "std_validation_accuracy": float(np.std(fold_scores, dtype=np.float64)),
                "fold_scores": [float(x) for x in fold_scores],
            }
            rows.append(row)
            if best_row is None or (row["mean_validation_accuracy"], -row["k"], row["folds"]) > (
                best_row["mean_validation_accuracy"],
                -best_row["k"],
                best_row["folds"],
            ):
                best_row = row
    if best_row is None:
        raise ValueError("Grid search could not build any validation folds.")
    return rows, best_row


def plot_best_model(path: Path, validation_grid_rows: list[dict[str, Any]], test_accuracy_rows: list[dict[str, Any]], *, best_k: int, best_folds: int, title_prefix: str) -> None:
    val_rows = [row for row in validation_grid_rows if row["folds"] == best_folds]
    val_rows = sorted(val_rows, key=lambda row: row["k"])
    test_rows = sorted(test_accuracy_rows, key=lambda row: row["k"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot([r["k"] for r in val_rows], [r["mean_validation_accuracy"] * 100 for r in val_rows], marker="o", label="Validation")
    axes[0].axvline(best_k, color="tab:red", linestyle="--", alpha=0.7)
    axes[0].set_title(f"{title_prefix} validation accuracy")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.3)

    axes[1].plot([r["k"] for r in test_rows], [r["test_accuracy"] * 100 for r in test_rows], marker="s", color="tab:green", label="Test")
    axes[1].axvline(best_k, color="tab:red", linestyle="--", alpha=0.7)
    axes[1].set_title(f"{title_prefix} test accuracy")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(path: Path, matrix: list[list[int]], labels: list[str], title: str) -> None:
    arr = np.asarray(matrix, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_grid_rows(name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{name} grid search results:")
    for row in sorted(rows, key=lambda r: (r["folds"], r["k"])):
        scores = ", ".join(f"{x:.3f}" for x in row["fold_scores"])
        print(
            f"  folds={row['folds']}, k={row['k']}: "
            f"val_mean={row['mean_validation_accuracy']:.3f}, "
            f"val_std={row['std_validation_accuracy']:.3f}, "
            f"fold_scores=[{scores}]"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid search and cross-validation for dual KNN experiments.")
    parser.add_argument("--base-source-root", type=Path, default=DEFAULT_BASE_SOURCE_ROOT)
    parser.add_argument("--dynamic-source-root", type=Path, default=DEFAULT_DYNAMIC_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-k-values", type=int, nargs="+", default=DEFAULT_BASE_K_VALUES)
    parser.add_argument("--dynamic-k-values", type=int, nargs="+", default=DEFAULT_DYNAMIC_K_VALUES)
    parser.add_argument("--cv-fold-values", type=int, nargs="+", default=DEFAULT_CV_FOLDS)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    base_source_root = Path(args.base_source_root).resolve()
    dynamic_source_root = Path(args.dynamic_source_root).resolve()
    output_root = Path(args.output_root).resolve()
    reset_output_root(output_root, force=bool(args.force))

    base_parts = load_sample_feature_parts_from_root(PROJECT_ROOT, base_source_root)
    dynamic_parts = load_sample_feature_parts_from_root(PROJECT_ROOT, dynamic_source_root)

    base_vectors, base_labels = build_keyword_matrix(base_parts)

    dynamic_label_map = {dynamic_key(item): item.record.keyword for item in dynamic_parts}
    dynamic_delta_map = build_dynamic_delta_map(dynamic_parts, PROJECT_ROOT)
    dynamic_vectors, dynamic_labels = build_person_keyword_matrix(
        dynamic_parts,
        label_map=dynamic_label_map,
        delta_map=dynamic_delta_map,
    )

    base_train_idx, base_test_idx, base_split_counts = split_indices_by_label(base_labels, float(args.test_ratio), int(args.seed))
    dyn_train_idx, dyn_test_idx, dyn_split_counts = split_indices_by_label(dynamic_labels, float(args.test_ratio), int(args.seed))

    base_k_values = sorted({int(v) for v in args.base_k_values if int(v) >= 1})
    dynamic_k_values = sorted({int(v) for v in args.dynamic_k_values if int(v) >= 1})
    cv_fold_values = sorted({int(v) for v in args.cv_fold_values if int(v) >= 2})

    base_grid_rows, base_best = run_grid_search(
        base_vectors[base_train_idx],
        [base_labels[i] for i in base_train_idx],
        k_values=base_k_values,
        fold_values=cv_fold_values,
        seed=int(args.seed),
    )
    dynamic_grid_rows, dynamic_best = run_grid_search(
        dynamic_vectors[dyn_train_idx],
        [dynamic_labels[i] for i in dyn_train_idx],
        k_values=dynamic_k_values,
        fold_values=cv_fold_values,
        seed=int(args.seed),
    )

    print_grid_rows("Base", base_grid_rows)
    print_grid_rows("Dynamic", dynamic_grid_rows)

    base_test_rows = []
    for k in base_k_values:
        test_acc, preds = evaluate_single_split(
            base_vectors[base_train_idx],
            [base_labels[i] for i in base_train_idx],
            base_vectors[base_test_idx],
            [base_labels[i] for i in base_test_idx],
            k,
        )
        base_test_rows.append({"k": int(k), "test_accuracy": float(test_acc)})
        if k == base_best["k"]:
            base_best_preds = preds

    dynamic_test_rows = []
    for k in dynamic_k_values:
        test_acc, preds = evaluate_single_split(
            dynamic_vectors[dyn_train_idx],
            [dynamic_labels[i] for i in dyn_train_idx],
            dynamic_vectors[dyn_test_idx],
            [dynamic_labels[i] for i in dyn_test_idx],
            k,
        )
        dynamic_test_rows.append({"k": int(k), "test_accuracy": float(test_acc)})
        if k == dynamic_best["k"]:
            dynamic_best_preds = preds

    base_test_labels = [base_labels[i] for i in base_test_idx]
    dynamic_test_labels = [dynamic_labels[i] for i in dyn_test_idx]

    base_conf_labels = sorted(set(base_test_labels) | set(base_best_preds))
    dynamic_conf_labels = sorted(set(dynamic_test_labels) | set(dynamic_best_preds))

    base_conf = confusion_matrix_counts(base_test_labels, base_best_preds, base_conf_labels)
    dynamic_conf = confusion_matrix_counts(dynamic_test_labels, dynamic_best_preds, dynamic_conf_labels)

    save_results_csv(
        output_root / "base_grid_search.csv",
        [{**row, "fold_scores": json.dumps(row["fold_scores"])} for row in base_grid_rows],
        ["requested_folds", "folds", "k", "mean_validation_accuracy", "std_validation_accuracy", "fold_scores"],
    )
    save_results_csv(
        output_root / "dynamic_grid_search.csv",
        [{**row, "fold_scores": json.dumps(row["fold_scores"])} for row in dynamic_grid_rows],
        ["requested_folds", "folds", "k", "mean_validation_accuracy", "std_validation_accuracy", "fold_scores"],
    )
    save_results_csv(output_root / "base_test_accuracy.csv", base_test_rows, ["k", "test_accuracy"])
    save_results_csv(output_root / "dynamic_test_accuracy.csv", dynamic_test_rows, ["k", "test_accuracy"])
    save_results_csv(
        output_root / "base_confusion_matrix.csv",
        [{"true_label": label, **{pred: count for pred, count in zip(base_conf_labels, row)}} for label, row in zip(base_conf_labels, base_conf)],
        ["true_label", *base_conf_labels],
    )
    save_results_csv(
        output_root / "dynamic_confusion_matrix.csv",
        [{"true_label": label, **{pred: count for pred, count in zip(dynamic_conf_labels, row)}} for label, row in zip(dynamic_conf_labels, dynamic_conf)],
        ["true_label", *dynamic_conf_labels],
    )

    plot_best_model(
        output_root / "base_best_model_accuracy.png",
        base_grid_rows,
        base_test_rows,
        best_k=base_best["k"],
        best_folds=base_best["folds"],
        title_prefix="Base",
    )
    plot_best_model(
        output_root / "dynamic_best_model_accuracy.png",
        dynamic_grid_rows,
        dynamic_test_rows,
        best_k=dynamic_best["k"],
        best_folds=dynamic_best["folds"],
        title_prefix="Dynamic",
    )
    plot_confusion_matrix(output_root / "base_confusion_matrix.png", base_conf, base_conf_labels, "Base confusion matrix")
    plot_confusion_matrix(output_root / "dynamic_confusion_matrix.png", dynamic_conf, dynamic_conf_labels, "Dynamic confusion matrix")

    summary = {
        "base": {
            "split_counts": base_split_counts,
            "best_model": base_best,
            "test_accuracy_at_best_k": next(row["test_accuracy"] for row in base_test_rows if row["k"] == base_best["k"]),
            "confusion_labels": base_conf_labels,
            "confusion_matrix": base_conf,
        },
        "dynamic": {
            "split_counts": dyn_split_counts,
            "best_model": dynamic_best,
            "test_accuracy_at_best_k": next(row["test_accuracy"] for row in dynamic_test_rows if row["k"] == dynamic_best["k"]),
            "confusion_labels": dynamic_conf_labels,
            "confusion_matrix": dynamic_conf,
        },
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("\nBest base model:", json.dumps(base_best, indent=2))
    print("Best dynamic model:", json.dumps(dynamic_best, indent=2))
    print(f"Outputs written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
