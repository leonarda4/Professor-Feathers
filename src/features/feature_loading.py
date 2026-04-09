from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from audio import read_wav
from features.feature_core import SampleFeatureParts, SampleFeatures, build_sample_features, extract_feature_parts
from storage import SampleRecord, read_manifest_records, relative_to_root


def load_sample_feature_parts(
    project_root: Path,
    manifest_path: Path,
    keyword: Optional[str] = None,
) -> list[SampleFeatureParts]:
    project_root = Path(project_root).resolve()
    records = read_manifest_records(manifest_path, keyword=keyword)
    sample_parts: list[SampleFeatureParts] = []
    for record in records:
        sample_path = project_root / record.path
        if not sample_path.exists():
            continue
        samples, sample_rate = read_wav(sample_path)
        sample_parts.append(extract_feature_parts(record, samples, sample_rate=sample_rate))
    return sample_parts


def load_sample_features(
    project_root: Path,
    manifest_path: Path,
    keyword: Optional[str] = None,
    *,
    variant: str = "f0_mean_std",
) -> list[SampleFeatures]:
    sample_parts = load_sample_feature_parts(project_root, manifest_path, keyword=keyword)
    return build_sample_features(sample_parts, variant=variant)


def _build_scanned_record(
    *,
    project_root: Path,
    samples_root: Path,
    sample_path: Path,
    sample_rate: int,
    num_samples: int,
) -> SampleRecord:
    relative_path = sample_path.resolve().relative_to(samples_root.resolve())
    parts = relative_path.parts
    if len(parts) >= 2:
        keyword = parts[0].strip()
    else:
        keyword = sample_path.parent.name.strip() or "keyword"
    if len(parts) >= 3:
        session_id = parts[1].strip() or "manual"
    else:
        session_id = "manual"
    try:
        record_path = relative_to_root(project_root, sample_path)
    except ValueError:
        record_path = str(sample_path.resolve())
    return SampleRecord(
        sample_id=sample_path.stem,
        keyword=keyword,
        path=record_path,
        sample_rate=int(sample_rate),
        duration_ms=int(round((num_samples / float(sample_rate)) * 1000.0)) if sample_rate > 0 else 0,
        timestamp="folder-scan",
        session_id=session_id,
        num_samples=int(num_samples),
        status="accepted",
    )


def load_sample_feature_parts_from_root(
    project_root: Path,
    samples_root: Path,
    keyword: Optional[str] = None,
) -> list[SampleFeatureParts]:
    project_root = Path(project_root).resolve()
    samples_root = Path(samples_root).resolve()
    if not samples_root.exists():
        raise FileNotFoundError(f"Sample root does not exist: {samples_root}")
    wav_paths = sorted(
        path
        for path in samples_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    sample_parts: list[SampleFeatureParts] = []
    for sample_path in wav_paths:
        samples, sample_rate = read_wav(sample_path)
        record = _build_scanned_record(
            project_root=project_root,
            samples_root=samples_root,
            sample_path=sample_path,
            sample_rate=sample_rate,
            num_samples=int(np.asarray(samples).size),
        )
        if keyword and record.keyword != keyword:
            continue
        sample_parts.append(extract_feature_parts(record, samples, sample_rate=sample_rate))
    return sample_parts


def load_sample_features_from_root(
    project_root: Path,
    samples_root: Path,
    keyword: Optional[str] = None,
    *,
    variant: str = "f0_mean_std",
) -> list[SampleFeatures]:
    sample_parts = load_sample_feature_parts_from_root(project_root, samples_root, keyword=keyword)
    return build_sample_features(sample_parts, variant=variant)
