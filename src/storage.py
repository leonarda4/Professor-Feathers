from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def slugify_keyword(keyword: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", keyword.strip().lower()).strip("_")
    return slug or "keyword"


def ensure_storage(
    project_root: Path,
    data_dir: str = "data",
    manifest_path: str = "data/manifests/samples.jsonl",
) -> tuple[Path, Path]:
    project_root = Path(project_root).resolve()
    data_dir_path = (project_root / data_dir).resolve()
    raw_dir = (data_dir_path / "raw").resolve()
    manifest_file = (project_root / manifest_path).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    return raw_dir, manifest_file


def session_dir(raw_dir: Path, keyword: str, session_id: str) -> Path:
    return Path(raw_dir).resolve() / slugify_keyword(keyword) / session_id


def next_sample_path(raw_dir: Path, keyword: str, session_id: str) -> tuple[Path, str]:
    sample_dir = session_dir(raw_dir, keyword, session_id)
    existing = sorted(sample_dir.glob("sample_*.wav"))
    sample_id = f"sample_{len(existing) + 1:04d}"
    return sample_dir / f"{sample_id}.wav", sample_id


def relative_to_root(project_root: Path, path: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(project_root).resolve()))


@dataclass
class SampleRecord:
    sample_id: str
    keyword: str
    path: str
    sample_rate: int
    duration_ms: int
    timestamp: str
    session_id: str
    num_samples: int
    status: str = "accepted"

    @classmethod
    def from_dict(cls, payload: dict) -> "SampleRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            keyword=str(payload["keyword"]),
            path=str(payload["path"]),
            sample_rate=int(payload["sample_rate"]),
            duration_ms=int(payload["duration_ms"]),
            timestamp=str(payload["timestamp"]),
            session_id=str(payload["session_id"]),
            num_samples=int(payload["num_samples"]),
            status=str(payload.get("status", "accepted")),
        )


def append_manifest_record(manifest_path: Path, record: SampleRecord) -> SampleRecord:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True))
        handle.write("\n")
    return record


def read_manifest_records(manifest_path: Path, keyword: Optional[str] = None) -> list[SampleRecord]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []
    records: list[SampleRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = SampleRecord.from_dict(json.loads(line))
            if keyword and record.keyword != keyword:
                continue
            records.append(record)
    return records
