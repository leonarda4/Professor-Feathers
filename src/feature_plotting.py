from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config import load_app_config
from feature_core import PcaResult, SampleFeatures, split_keyword_label
from feature_loading import load_sample_features, load_sample_features_from_root


def compute_pca(feature_vectors: np.ndarray, n_dims: int = 3) -> PcaResult:
    vectors = np.asarray(feature_vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("feature_vectors must be a 2D array.")
    if vectors.shape[0] == 0:
        return PcaResult(
            coordinates=np.zeros((0, n_dims), dtype=np.float32),
            explained_variance_ratio=np.zeros(0, dtype=np.float32),
        )
    centered = vectors - np.mean(vectors, axis=0, keepdims=True, dtype=np.float64)
    scale = np.std(centered, axis=0, keepdims=True, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    normalized = centered / scale
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


def _pc_label(index: int, variance_ratio: np.ndarray) -> str:
    if index < variance_ratio.size:
        return f"PC{index + 1} ({variance_ratio[index] * 100.0:.1f}%)"
    return f"PC{index + 1}"


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
    axis.set_title("PC1-PC6 Score Heatmap")
    return image


def plot_feature_dashboard(
    sample_features: list[SampleFeatures],
    pca: PcaResult,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
    annotate: bool = False,
) -> Path | None:
    if not sample_features:
        raise ValueError("No samples available to plot.")
    coordinates = pca.coordinates
    variance_ratio = pca.explained_variance_ratio
    heatmap_pc_count = min(6, variance_ratio.size)
    all_pc_coordinates = coordinates[:, :heatmap_pc_count] if heatmap_pc_count else coordinates[:, :0]
    figure = plt.figure(figsize=(20, 15))
    grid = figure.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.15])
    axes = np.empty((2, 3), dtype=object)
    for row in range(2):
        for column in range(3):
            axes[row, column] = figure.add_subplot(grid[row, column])
    heatmap_axis = figure.add_subplot(grid[2, :])

    _scatter_by_exact_keyword(axes[0, 0], sample_features, coordinates)
    _scatter_by_base_keyword_and_speaker(axes[0, 1], sample_features, coordinates, 0, 1)
    axes[0, 1].set_title("Base Keyword + Speaker: PC1 vs PC2")
    _scatter_by_base_keyword_and_speaker(axes[0, 2], sample_features, coordinates, 0, 2)
    axes[0, 2].set_title("Base Keyword + Speaker: PC1 vs PC3")
    _scatter_by_base_keyword_and_speaker(axes[1, 0], sample_features, coordinates, 1, 2)
    axes[1, 0].set_title("Base Keyword + Speaker: PC2 vs PC3")
    _plot_label_centroids(axes[1, 1], sample_features, coordinates)

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

    if annotate:
        for point, item in zip(coordinates, sample_features):
            axes[0, 0].annotate(item.record.sample_id, (point[0], point[1]), fontsize=7, alpha=0.65)

    figure.suptitle("MFCC + F0 Stats PCA Dashboard with All-PC Overview", fontsize=18)
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
    variant: str = "f0_mean_std",
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
            variant=variant,
        )
    else:
        manifest_path = (project_root / config.storage.manifest_path).resolve()
        sample_features = load_sample_features(
            project_root,
            manifest_path,
            keyword=keyword,
            variant=variant,
        )
    if not sample_features:
        raise RuntimeError("No saved samples found for feature extraction.")
    vectors = np.vstack([item.vector for item in sample_features]).astype(np.float32)
    pca = compute_pca(vectors, n_dims=max(3, min(vectors.shape[0], vectors.shape[1])))
    return plot_feature_dashboard(
        sample_features,
        pca,
        output_path=output_path,
        show=show,
        annotate=annotate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract MFCC + F0 summary features and plot a PCA dashboard."
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
