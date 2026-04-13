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
from sklearn.svm import SVC

from data_augmentation import AugmentationConfig, build_augmented_feature_parts

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features.feature_loading import load_sample_feature_parts_from_root  # noqa: E402
from features.feature_spaces import build_keyword_matrix  # noqa: E402
from knn_utils import compute_accuracy, knn_predict, standardize_feature_matrices  # noqa: E402

DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "base_keywords"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "base_model_compare"
DEFAULT_K_VALUES = [1, 3, 5, 7, 9, 11]
DEFAULT_C_VALUES = [0.1, 1.0, 10.0, 100.0]
DEFAULT_GAMMA_VALUES = ["scale", "auto"]
DEFAULT_CV_FOLDS = [3, 5]


def reset_output_root(output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_root}. Re-run with --force to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


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


def confusion_matrix_counts(true_labels: list[str], predicted_labels: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label in index and predicted_label in index:
            matrix[index[true_label]][index[predicted_label]] += 1
    return matrix


def per_class_recall(true_labels: list[str], predicted_labels: list[str], labels: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for label in labels:
        positives = sum(1 for x in true_labels if x == label)
        hits = sum(1 for t, p in zip(true_labels, predicted_labels) if t == label and p == label)
        metrics[f"recall_{label}"] = float(hits / positives) if positives else 0.0
    return metrics


def evaluate_knn(train_vectors, train_labels, test_vectors, test_labels, *, k: int) -> tuple[float, list[str]]:
    train_scaled, test_scaled, _ = standardize_feature_matrices(train_vectors, test_vectors)
    preds = knn_predict(train_scaled, train_labels, test_scaled, k=k)
    predicted_labels = [item["predicted_label"] for item in preds]
    return float(compute_accuracy(test_labels, predicted_labels)), predicted_labels


def evaluate_svm(train_vectors, train_labels, test_vectors, test_labels, *, kernel: str, C: float, gamma: str | float) -> tuple[float, list[str]]:
    train_scaled, test_scaled, _ = standardize_feature_matrices(train_vectors, test_vectors)
    clf = SVC(kernel=kernel, C=float(C), gamma=gamma)
    clf.fit(train_scaled, np.asarray(train_labels))
    predicted_labels = [str(x) for x in clf.predict(test_scaled)]
    return float(compute_accuracy(test_labels, predicted_labels)), predicted_labels


def run_knn_grid_search(vectors: np.ndarray, labels: list[str], *, k_values: list[int], fold_values: list[int], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for folds in sorted({int(v) for v in fold_values if int(v) >= 2}):
        split_folds = make_stratified_folds(labels, folds, seed)
        if not split_folds:
            continue
        for k in sorted({int(v) for v in k_values if int(v) >= 1}):
            fold_scores = []
            for train_idx, val_idx in split_folds:
                acc, _ = evaluate_knn(vectors[train_idx], [labels[i] for i in train_idx], vectors[val_idx], [labels[i] for i in val_idx], k=k)
                fold_scores.append(acc)
            row = {
                "model": "knn",
                "folds": int(len(split_folds)),
                "requested_folds": int(folds),
                "k": int(k),
                "mean_validation_accuracy": float(np.mean(fold_scores, dtype=np.float64)),
                "std_validation_accuracy": float(np.std(fold_scores, dtype=np.float64)),
                "fold_scores": [float(x) for x in fold_scores],
            }
            rows.append(row)
            if best_row is None or (row["mean_validation_accuracy"], -row["k"], row["folds"]) > (best_row["mean_validation_accuracy"], -best_row.get("k", 10**9), best_row["folds"]):
                best_row = row
    if best_row is None:
        raise ValueError("KNN grid search could not build any validation folds.")
    return rows, best_row


def run_svm_grid_search(vectors: np.ndarray, labels: list[str], *, kernels: list[str], c_values: list[float], gamma_values: list[str | float], fold_values: list[int], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for folds in sorted({int(v) for v in fold_values if int(v) >= 2}):
        split_folds = make_stratified_folds(labels, folds, seed)
        if not split_folds:
            continue
        for kernel in kernels:
            gamma_candidates = ["scale"] if kernel == "linear" else gamma_values
            for c_value in c_values:
                for gamma in gamma_candidates:
                    fold_scores = []
                    for train_idx, val_idx in split_folds:
                        acc, _ = evaluate_svm(vectors[train_idx], [labels[i] for i in train_idx], vectors[val_idx], [labels[i] for i in val_idx], kernel=kernel, C=float(c_value), gamma=gamma)
                        fold_scores.append(acc)
                    row = {
                        "model": f"svm_{kernel}",
                        "kernel": kernel,
                        "C": float(c_value),
                        "gamma": gamma,
                        "folds": int(len(split_folds)),
                        "requested_folds": int(folds),
                        "mean_validation_accuracy": float(np.mean(fold_scores, dtype=np.float64)),
                        "std_validation_accuracy": float(np.std(fold_scores, dtype=np.float64)),
                        "fold_scores": [float(x) for x in fold_scores],
                    }
                    rows.append(row)
                    score_key = (row["mean_validation_accuracy"], -row["C"], row["folds"])
                    best_key = (best_row["mean_validation_accuracy"], -best_row["C"], best_row["folds"]) if best_row is not None else None
                    if best_row is None or score_key > best_key:
                        best_row = row
    if best_row is None:
        raise ValueError("SVM grid search could not build any validation folds.")
    return rows, best_row


def plot_model_compare(path: Path, *, knn_rows: list[dict[str, Any]], knn_best: dict[str, Any], svm_best: dict[str, Any], knn_test_acc: float, svm_test_acc: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    knn_plot_rows = sorted([row for row in knn_rows if row["folds"] == knn_best["folds"]], key=lambda row: row["k"])
    axes[0].plot([r["k"] for r in knn_plot_rows], [r["mean_validation_accuracy"] * 100 for r in knn_plot_rows], marker="o")
    axes[0].axvline(knn_best["k"], color="tab:red", linestyle="--", alpha=0.7)
    axes[0].set_title("KNN validation accuracy")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.3)
    axes[1].bar(["KNN", f"SVM\n({svm_best['kernel']})"], [knn_test_acc * 100, svm_test_acc * 100], color=["tab:blue", "tab:orange"])
    axes[1].set_title("Held-out test accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(axis="y", alpha=0.3)
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline comparison between base-classifier KNN and SVM.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copies-per-sample", type=int, default=3)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--svm-kernels", nargs="+", default=["linear", "rbf"])
    parser.add_argument("--svm-c-values", type=float, nargs="+", default=DEFAULT_C_VALUES)
    parser.add_argument("--svm-gamma-values", nargs="+", default=DEFAULT_GAMMA_VALUES)
    parser.add_argument("--cv-fold-values", type=int, nargs="+", default=DEFAULT_CV_FOLDS)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    reset_output_root(output_root, force=bool(args.force))
    base_parts = load_sample_feature_parts_from_root(PROJECT_ROOT, source_root)
    base_parts_aug = build_augmented_feature_parts(base_parts, project_root=PROJECT_ROOT, config=AugmentationConfig(copies_per_sample=int(args.copies_per_sample)), seed=int(args.seed))
    vectors, labels = build_keyword_matrix(base_parts_aug)
    train_idx, test_idx, split_counts = split_indices_by_label(labels, float(args.test_ratio), int(args.seed))
    knn_rows, knn_best = run_knn_grid_search(vectors[train_idx], [labels[i] for i in train_idx], k_values=sorted({int(v) for v in args.k_values if int(v) >= 1}), fold_values=sorted({int(v) for v in args.cv_fold_values if int(v) >= 2}), seed=int(args.seed))
    svm_rows, svm_best = run_svm_grid_search(vectors[train_idx], [labels[i] for i in train_idx], kernels=[str(v) for v in args.svm_kernels], c_values=sorted({float(v) for v in args.svm_c_values if float(v) > 0.0}), gamma_values=[str(v) for v in args.svm_gamma_values], fold_values=sorted({int(v) for v in args.cv_fold_values if int(v) >= 2}), seed=int(args.seed))

    test_labels = [labels[i] for i in test_idx]
    knn_test_acc, knn_test_preds = evaluate_knn(vectors[train_idx], [labels[i] for i in train_idx], vectors[test_idx], test_labels, k=int(knn_best["k"]))
    svm_test_acc, svm_test_preds = evaluate_svm(vectors[train_idx], [labels[i] for i in train_idx], vectors[test_idx], test_labels, kernel=str(svm_best["kernel"]), C=float(svm_best["C"]), gamma=svm_best["gamma"])

    eval_labels = sorted(set(test_labels) | set(knn_test_preds) | set(svm_test_preds))
    knn_conf = confusion_matrix_counts(test_labels, knn_test_preds, eval_labels)
    svm_conf = confusion_matrix_counts(test_labels, svm_test_preds, eval_labels)
    knn_metrics = {"test_accuracy": float(knn_test_acc), **per_class_recall(test_labels, knn_test_preds, eval_labels)}
    svm_metrics = {"test_accuracy": float(svm_test_acc), **per_class_recall(test_labels, svm_test_preds, eval_labels)}

    save_results_csv(output_root / "knn_grid_search.csv", [{**row, "fold_scores": json.dumps(row["fold_scores"])} for row in knn_rows], ["model", "requested_folds", "folds", "k", "mean_validation_accuracy", "std_validation_accuracy", "fold_scores"])
    save_results_csv(output_root / "svm_grid_search.csv", [{**row, "fold_scores": json.dumps(row["fold_scores"])} for row in svm_rows], ["model", "kernel", "C", "gamma", "requested_folds", "folds", "mean_validation_accuracy", "std_validation_accuracy", "fold_scores"])
    save_results_csv(output_root / "held_out_test_results.csv", [
        {"model": "knn", "test_accuracy": float(knn_test_acc), **{k: v for k, v in knn_metrics.items() if k != 'test_accuracy'}, "config": json.dumps({"k": int(knn_best['k'])})},
        {"model": "svm", "test_accuracy": float(svm_test_acc), **{k: v for k, v in svm_metrics.items() if k != 'test_accuracy'}, "config": json.dumps({"kernel": svm_best['kernel'], "C": svm_best['C'], "gamma": svm_best['gamma']})},
    ], ["model", "test_accuracy", *[k for k in knn_metrics.keys() if k != 'test_accuracy'], "config"])
    save_results_csv(output_root / "knn_confusion_matrix.csv", [{"true_label": label, **{pred: count for pred, count in zip(eval_labels, row)}} for label, row in zip(eval_labels, knn_conf)], ["true_label", *eval_labels])
    save_results_csv(output_root / "svm_confusion_matrix.csv", [{"true_label": label, **{pred: count for pred, count in zip(eval_labels, row)}} for label, row in zip(eval_labels, svm_conf)], ["true_label", *eval_labels])

    plot_model_compare(output_root / "base_model_compare.png", knn_rows=knn_rows, knn_best=knn_best, svm_best=svm_best, knn_test_acc=float(knn_test_acc), svm_test_acc=float(svm_test_acc))
    plot_confusion_matrix(output_root / "knn_confusion_matrix.png", knn_conf, eval_labels, "KNN confusion matrix")
    plot_confusion_matrix(output_root / "svm_confusion_matrix.png", svm_conf, eval_labels, "SVM confusion matrix")

    summary = {
        "split_counts": split_counts,
        "labels": eval_labels,
        "knn_best": knn_best,
        "svm_best": svm_best,
        "knn_metrics": knn_metrics,
        "svm_metrics": svm_metrics,
        "knn_confusion_matrix": knn_conf,
        "svm_confusion_matrix": svm_conf,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Outputs written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
