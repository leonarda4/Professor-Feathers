from __future__ import annotations

import argparse
import queue
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from audio import MicrophoneStream, describe_default_input, duration_ms, list_input_devices, write_wav
from config import AppConfig, load_app_config
from endpointer import UtteranceEndpointer
from storage import SampleRecord, append_manifest_record, ensure_storage, next_sample_path, relative_to_root


def _require_pynput():
    try:
        from pynput import keyboard
    except ImportError as exc:  # pragma: no cover - exercised through CLI only
        raise RuntimeError(
            "pynput is required for keyboard hotkeys. Install project dependencies first."
        ) from exc
    return keyboard


def _normalize_key(key) -> str:
    char = getattr(key, "char", None)
    if char:
        return str(char).lower()
    name = getattr(key, "name", None)
    if name:
        return str(name).lower()
    key_str = str(key)
    if "." in key_str:
        return key_str.split(".")[-1].lower()
    return key_str.lower()


def start_hotkey_listener(events: "queue.Queue[str]", arm_key: Optional[str] = "space", quit_key: Optional[str] = "esc"):
    arm_name = arm_key.lower() if arm_key else None
    quit_name = quit_key.lower() if quit_key else None
    keyboard = _require_pynput()

    def on_press(key) -> None:  # pragma: no cover - hardware path
        name = _normalize_key(key)
        if arm_name and name == arm_name:
            events.put("arm")
        elif quit_name and name == quit_name:
            events.put("quit")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def drain_actions(events: "queue.Queue[str]") -> list[str]:
    actions: list[str] = []
    while True:
        try:
            actions.append(events.get_nowait())
        except queue.Empty:
            return actions


def build_endpointer(config: AppConfig) -> UtteranceEndpointer:
    return UtteranceEndpointer(
        sample_rate=config.audio.sample_rate,
        frame_ms=config.endpoint.frame_ms,
        pre_roll_ms=config.endpoint.pre_roll_ms,
        min_speech_ms=config.endpoint.min_speech_ms,
        end_silence_ms=config.endpoint.end_silence_ms,
        max_clip_ms=config.endpoint.max_clip_ms,
        start_threshold_db=config.endpoint.start_threshold_db,
        stop_threshold_db=config.endpoint.stop_threshold_db,
        start_margin_db=config.endpoint.start_margin_db,
        stop_margin_db=config.endpoint.stop_margin_db,
        min_peak=config.endpoint.min_peak,
        noise_floor_alpha=config.endpoint.noise_floor_alpha,
    )


def save_utterance(
    audio: np.ndarray,
    *,
    project_root: Path,
    raw_dir: Path,
    manifest_path: Path,
    keyword: str,
    session_id: str,
    sample_rate: int,
) -> SampleRecord:
    sample_path, sample_id = next_sample_path(raw_dir, keyword, session_id)
    write_wav(sample_path, audio, sample_rate=sample_rate)
    record = SampleRecord(
        sample_id=sample_id,
        keyword=keyword,
        path=relative_to_root(project_root, sample_path),
        sample_rate=sample_rate,
        duration_ms=duration_ms(audio, sample_rate),
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=session_id,
        num_samples=int(np.asarray(audio).size),
    )
    append_manifest_record(manifest_path, record)
    return record


def apply_cli_overrides(
    config: AppConfig,
    *,
    keyword: str,
    arm_key: Optional[str] = None,
    quit_key: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> AppConfig:
    config.collection.keyword = keyword
    if arm_key:
        config.collection.arm_key = arm_key
    if quit_key:
        config.collection.quit_key = quit_key
    if batch_size is not None:
        config.collection.batch_size = max(1, int(batch_size))
    return config


def run_collection(config: AppConfig, project_root: Path, keyword: str, session_id: str) -> int:
    project_root = Path(project_root).resolve()
    raw_dir, manifest_path = ensure_storage(
        project_root=project_root,
        data_dir=config.storage.data_dir,
        manifest_path=config.storage.manifest_path,
    )
    sample_rate = int(config.audio.sample_rate)
    batch_size = max(1, int(config.collection.batch_size))
    remaining_batch_items = 0
    endpointer = build_endpointer(config)
    hotkey_events: "queue.Queue[str]" = queue.Queue()
    listener = start_hotkey_listener(
        hotkey_events,
        arm_key=config.collection.arm_key,
        quit_key=config.collection.quit_key,
    )

    default_device = describe_default_input()
    if default_device:
        print(
            f"Using input device {default_device.index}: {default_device.name} "
            f"({default_device.default_samplerate:.0f} Hz default)"
        )
    print(
        f"Collecting keyword '{keyword}'. Press {config.collection.arm_key} to start "
        f"a batch of {batch_size} recordings, speak once per sample, and press "
        f"{config.collection.quit_key} to quit."
    )

    try:
        with MicrophoneStream(
            sample_rate=sample_rate,
            channels=config.audio.channels,
            block_ms=config.audio.block_ms,
            device=config.audio.device,
        ) as microphone:
            while True:
                for status_message in microphone.pop_status_messages():
                    print(f"audio-status: {status_message}")
                for action in drain_actions(hotkey_events):
                    if action == "quit":
                        print("Stopping collection.")
                        return 0
                    if action == "arm":
                        remaining_batch_items = batch_size
                        endpointer.arm()
                        print(f"Batch armed for {batch_size} recordings. Waiting for word 1.")
                chunk = microphone.read_block(timeout=0.1)
                if chunk is None:
                    continue
                for event in endpointer.process_chunk(chunk):
                    if event.kind != "utterance" or event.audio is None or event.audio.size == 0:
                        continue
                    record = save_utterance(
                        event.audio,
                        project_root=project_root,
                        raw_dir=raw_dir,
                        manifest_path=manifest_path,
                        keyword=keyword,
                        session_id=session_id,
                        sample_rate=sample_rate,
                    )
                    print(f"Saved {record.sample_id} to {record.path} ({record.duration_ms} ms)")
                    if remaining_batch_items > 0:
                        remaining_batch_items -= 1
                    completed = batch_size - remaining_batch_items
                    if remaining_batch_items > 0:
                        endpointer.arm()
                        print(
                            f"Ready for next word ({completed}/{batch_size}). "
                            f"{remaining_batch_items} recordings remaining."
                        )
                    else:
                        print(
                            f"Batch complete ({completed}/{batch_size}). Press "
                            f"{config.collection.arm_key} to start the next batch."
                        )
    finally:
        listener.stop()


def build_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect keyword audio samples with hotkey-triggered recording.")
    parser.add_argument("--project-root", default=".", help="Project root containing src/ and data/.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    parser.add_argument("--keyword", default=None, help="Keyword label to save samples under.")
    parser.add_argument("--session-id", default=None, help="Optional session identifier.")
    parser.add_argument("--arm-key", default=None, help="Override the keyboard key used to arm recording.")
    parser.add_argument("--quit-key", default=None, help="Override the keyboard key used to stop recording.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of recordings to capture automatically after each arm key press.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List available microphone devices and exit.")
    args = parser.parse_args(argv)

    if args.list_devices:
        for device in list_input_devices():
            print(
                f"{device.index}: {device.name} "
                f"(inputs={device.max_input_channels}, default_sr={device.default_samplerate:.0f})"
            )
        return 0

    if not args.keyword:
        parser.error("--keyword is required unless --list-devices is used.")

    config = apply_cli_overrides(
        load_app_config(config_path=args.config),
        keyword=args.keyword,
        arm_key=args.arm_key,
        quit_key=args.quit_key,
        batch_size=args.batch_size,
    )
    return run_collection(
        config=config,
        project_root=Path(args.project_root),
        keyword=args.keyword,
        session_id=args.session_id or build_session_id(),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
