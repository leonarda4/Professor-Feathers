from __future__ import annotations

import argparse
import csv
import json
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

from feature_core import build_feature_matrix  # noqa: E402
from feature_loading import load_sample_feature_parts_from_root  # noqa: E402
from knn_utils import (  # noqa: E402
    compute_accuracy,
    knn_predict,
    standardize_feature_matrices,
)
from run_knn_classifier import (  # noqa: E402
    collect_wav_files_by_label,
    copy_split_files,
    split_paths_by_label,
)


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "knn_experiments"
DEFAULT_K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]


def reset_output_root(output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Re-run with --force to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def evaluate_knn_grid(
    train_vectors: np.ndarray,
    train_labels: list[str],
    test_vectors: np.ndarray,
    test_labels: list[str],
    *,
    k_values: list[int],
) -> list[dict[str, Any]]:
    train_vectors_scaled, test_vectors_scaled, _ = standardize_feature_matrices(
        train_vectors,
        test_vectors,
    )
    results: list[dict[str, Any]] = []
    true_base_labels = [strip_speaker_suffix(label) for label in test_labels]
    for k in k_values:
        predictions = knn_predict(
            train_vectors_scaled,
            train_labels,
            test_vectors_scaled,
            k=k,
        )
        predicted_labels = [item["predicted_label"] for item in predictions]
        predicted_base_labels = [strip_speaker_suffix(label) for label in predicted_labels]
        exact_accuracy = compute_accuracy(test_labels, predicted_labels)
        base_accuracy = compute_accuracy(true_base_labels, predicted_base_labels)
        results.append(
            {
                "k": int(k),
                "exact_accuracy": float(exact_accuracy),
                "base_accuracy": float(base_accuracy),
            }
        )
    return results


def strip_speaker_suffix(label: str) -> str:
    digits_start = len(label)
    while digits_start > 0 and label[digits_start - 1].isdigit():
        digits_start -= 1
    return label[:digits_start] if digits_start < len(label) else label


def choose_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("No grid-search results to rank.")
    return sorted(
        results,
        key=lambda item: (
            -item["exact_accuracy"],
            -item["base_accuracy"],
            item["k"],
        ),
    )[0]


def save_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "feature_dimension",
                "k",
                "exact_accuracy",
                "base_accuracy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    variants = sorted({row["variant"] for row in rows})
    colors = {
        "f0_mean_std": "#F58518",
        "no_f0": "#54A24B",
    }
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for variant in variants:
        variant_rows = sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda row: row["k"],
        )
        k_values = [row["k"] for row in variant_rows]
        exact_values = [row["exact_accuracy"] * 100.0 for row in variant_rows]
        base_values = [row["base_accuracy"] * 100.0 for row in variant_rows]
        color = colors.get(variant, None)
        axes[0].plot(k_values, exact_values, marker="o", linewidth=2, label=variant, color=color)
        axes[1].plot(k_values, base_values, marker="o", linewidth=2, label=variant, color=color)

    axes[0].set_title("Exact Label Accuracy")
    axes[1].set_title("Base Keyword Accuracy")
    for axis in axes:
        axis.set_xlabel("k")
        axis.set_ylabel("Accuracy (%)")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("KNN Grid Search Across Feature Variants", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("KNN experiment summary\n")
        handle.write("======================\n\n")
        handle.write(f"source_root: {summary['source_root']}\n")
        handle.write(f"seed: {summary['seed']}\n")
        handle.write(f"test_ratio: {summary['test_ratio']}\n")
        handle.write(f"train_samples: {summary['train_sample_count']}\n")
        handle.write(f"test_samples: {summary['test_sample_count']}\n")
        handle.write(f"k_values: {summary['k_values']}\n\n")
        handle.write("Best results by variant\n")
        handle.write("-----------------------\n")
        for item in summary["best_by_variant"]:
            handle.write(
                f"{item['variant']}: k={item['k']}, "
                f"feature_dim={item['feature_dimension']}, "
                f"exact_accuracy={item['exact_accuracy']:.4f}, "
                f"base_accuracy={item['base_accuracy']:.4f}\n"
            )
        handle.write("\nOverall winner\n")
        handle.write("--------------\n")
        winner = summary["overall_winner"]
        handle.write(
            f"{winner['variant']}: k={winner['k']}, "
            f"feature_dim={winner['feature_dimension']}, "
            f"exact_accuracy={winner['exact_accuracy']:.4f}, "
            f"base_accuracy={winner['base_accuracy']:.4f}\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare KNN feature variants on one fixed train/test split and grid-search "
            "over multiple k values."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="List of k values to evaluate.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    split_root = output_root / "split"
    train_root = split_root / "train"
    test_root = split_root / "test"
    k_values = sorted({int(value) for value in args.k_values if int(value) >= 1})

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if not 0.0 <= float(args.test_ratio) < 1.0:
        raise ValueError("--test-ratio must be in the range [0.0, 1.0).")
    if not k_values:
        raise ValueError("At least one valid k value is required.")

    grouped_paths = collect_wav_files_by_label(source_root)
    train_paths, test_paths, split_counts = split_paths_by_label(
        grouped_paths,
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    if not train_paths or not test_paths:
        raise ValueError("Split did not produce both train and test samples.")

    reset_output_root(output_root, force=bool(args.force))
    copy_split_files(train_paths, source_root=source_root, split_root=train_root)
    copy_split_files(test_paths, source_root=source_root, split_root=test_root)

    train_parts = load_sample_feature_parts_from_root(PROJECT_ROOT, train_root)
    test_parts = load_sample_feature_parts_from_root(PROJECT_ROOT, test_root)

    mean_std_train_vectors, mean_std_train_labels = build_feature_matrix(
        train_parts,
        variant="f0_mean_std",
    )
    mean_std_test_vectors, mean_std_test_labels = build_feature_matrix(
        test_parts,
        variant="f0_mean_std",
    )
    no_f0_train_vectors, no_f0_train_labels = build_feature_matrix(
        train_parts,
        variant="no_f0",
    )
    no_f0_test_vectors, no_f0_test_labels = build_feature_matrix(
        test_parts,
        variant="no_f0",
    )

    variant_payloads = [
        {
            "variant": "f0_mean_std",
            "feature_dimension": int(mean_std_train_vectors.shape[1]),
            "train_vectors": mean_std_train_vectors,
            "train_labels": mean_std_train_labels,
            "test_vectors": mean_std_test_vectors,
            "test_labels": mean_std_test_labels,
        },
        {
            "variant": "no_f0",
            "feature_dimension": int(no_f0_train_vectors.shape[1]),
            "train_vectors": no_f0_train_vectors,
            "train_labels": no_f0_train_labels,
            "test_vectors": no_f0_test_vectors,
            "test_labels": no_f0_test_labels,
        },
    ]

    all_rows: list[dict[str, Any]] = []
    best_by_variant: list[dict[str, Any]] = []

    for payload in variant_payloads:
        results = evaluate_knn_grid(
            payload["train_vectors"],
            payload["train_labels"],
            payload["test_vectors"],
            payload["test_labels"],
            k_values=k_values,
        )
        best = choose_best_result(results) | {
            "variant": payload["variant"],
            "feature_dimension": payload["feature_dimension"],
        }
        best_by_variant.append(best)
        for item in results:
            all_rows.append(
                {
                    "variant": payload["variant"],
                    "feature_dimension": payload["feature_dimension"],
                    "k": item["k"],
                    "exact_accuracy": item["exact_accuracy"],
                    "base_accuracy": item["base_accuracy"],
                }
            )

    overall_winner = sorted(
        best_by_variant,
        key=lambda item: (
            -item["exact_accuracy"],
            -item["base_accuracy"],
            item["k"],
        ),
    )[0]

    save_results_csv(output_root / "grid_search_results.csv", all_rows)
    save_plot(output_root / "grid_search_accuracy.png", all_rows)
    save_summary(
        output_root / "summary.txt",
        {
            "source_root": str(source_root),
            "seed": int(args.seed),
            "test_ratio": float(args.test_ratio),
            "train_sample_count": int(len(train_parts)),
            "test_sample_count": int(len(test_parts)),
            "k_values": k_values,
            "best_by_variant": best_by_variant,
            "overall_winner": overall_winner,
        },
    )
    with (output_root / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_root": str(source_root),
                "split_root": str(split_root),
                "train_root": str(train_root),
                "test_root": str(test_root),
                "seed": int(args.seed),
                "test_ratio": float(args.test_ratio),
                "k_values": k_values,
                "split_counts": split_counts,
                "best_by_variant": best_by_variant,
                "overall_winner": overall_winner,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(f"Saved grid-search CSV to {output_root / 'grid_search_results.csv'}")
    print(f"Saved accuracy plot to {output_root / 'grid_search_accuracy.png'}")
    print(f"Saved summary to {output_root / 'summary.txt'}")
    print("")
    print("Best results by variant:")
    for item in best_by_variant:
        print(
            f"{item['variant']}: k={item['k']}, "
            f"feature_dim={item['feature_dimension']}, "
            f"exact={item['exact_accuracy']:.3f}, "
            f"base={item['base_accuracy']:.3f}"
        )
    print("")
    print(
        f"Overall winner: {overall_winner['variant']} at k={overall_winner['k']} "
        f"(exact={overall_winner['exact_accuracy']:.3f}, base={overall_winner['base_accuracy']:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
