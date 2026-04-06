from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 30
    device: str | None = None


@dataclass
class EndpointConfig:
    frame_ms: int = 30
    pre_roll_ms: int = 250
    min_speech_ms: int = 200
    end_silence_ms: int = 500
    max_clip_ms: int = 2000
    start_threshold_db: float = -42.0
    stop_threshold_db: float = -48.0
    start_margin_db: float = 12.0
    stop_margin_db: float = 6.0
    min_peak: float = 0.015
    noise_floor_alpha: float = 0.92


@dataclass
class StorageConfig:
    data_dir: str = "data"
    manifest_path: str = "data/manifests/samples.jsonl"


@dataclass
class CollectionConfig:
    keyword: str = "keyword"
    arm_key: str = "space"
    quit_key: str = "esc"
    batch_size: int = 20


@dataclass
class AppConfig:
    audio: AudioConfig
    endpoint: EndpointConfig
    storage: StorageConfig
    collection: CollectionConfig

    @classmethod
    def defaults(cls) -> "AppConfig":
        return cls(
            audio=AudioConfig(),
            endpoint=EndpointConfig(),
            storage=StorageConfig(),
            collection=CollectionConfig(),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        defaults = cls.defaults()
        return cls(
            audio=AudioConfig(**_merge_dataclass(defaults.audio.__dict__, payload.get("audio", {}))),
            endpoint=EndpointConfig(
                **_merge_dataclass(defaults.endpoint.__dict__, payload.get("endpoint", {}))
            ),
            storage=StorageConfig(
                **_merge_dataclass(defaults.storage.__dict__, payload.get("storage", {}))
            ),
            collection=CollectionConfig(
                **_merge_dataclass(defaults.collection.__dict__, payload.get("collection", {}))
            ),
        )


def _merge_dataclass(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(overrides or {})
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return payload


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_app_config(config_path: str | None = None) -> AppConfig:
    defaults_payload = asdict(AppConfig.defaults())
    if config_path:
        defaults_payload = _deep_merge(defaults_payload, _load_yaml(Path(config_path)))
    return AppConfig.from_dict(defaults_payload)
