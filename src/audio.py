from __future__ import annotations

import queue
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np


@dataclass(frozen=True)
class InputDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float


def _require_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - exercised through CLI only
        raise RuntimeError(
            "sounddevice is required for microphone capture. Install project dependencies first."
        ) from exc
    return sd


def list_input_devices() -> list[InputDeviceInfo]:
    sd = _require_sounddevice()
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] <= 0:
            continue
        devices.append(
            InputDeviceInfo(
                index=index,
                name=str(device["name"]),
                max_input_channels=int(device["max_input_channels"]),
                default_samplerate=float(device["default_samplerate"]),
            )
        )
    return devices


def describe_default_input() -> Optional[InputDeviceInfo]:
    sd = _require_sounddevice()
    default_devices = sd.default.device
    if not default_devices:
        return None
    input_index = default_devices[0]
    if input_index is None or input_index < 0:
        return None
    device = sd.query_devices(input_index)
    return InputDeviceInfo(
        index=int(input_index),
        name=str(device["name"]),
        max_input_channels=int(device["max_input_channels"]),
        default_samplerate=float(device["default_samplerate"]),
    )


class RollingAudioBuffer:
    def __init__(self, max_samples: int) -> None:
        self.max_samples = max(0, int(max_samples))
        self._chunks: Deque[np.ndarray] = deque()
        self._length = 0

    def clear(self) -> None:
        self._chunks.clear()
        self._length = 0

    def extend(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self.max_samples <= 0 or samples.size == 0:
            return
        if samples.size >= self.max_samples:
            self._chunks = deque([samples[-self.max_samples :].copy()])
            self._length = self.max_samples
            return
        self._chunks.append(samples.copy())
        self._length += int(samples.size)
        while self._length > self.max_samples and self._chunks:
            overflow = self._length - self.max_samples
            head = self._chunks[0]
            if overflow >= head.size:
                self._chunks.popleft()
                self._length -= int(head.size)
                continue
            self._chunks[0] = head[overflow:].copy()
            self._length -= overflow
            break

    def snapshot(self) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(list(self._chunks)).astype(np.float32, copy=False)


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = float_to_pcm16(samples)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return samples.astype(np.float32, copy=False), int(sample_rate)


def duration_ms(samples: np.ndarray, sample_rate: int) -> int:
    total_samples = np.asarray(samples).reshape(-1).size
    if sample_rate <= 0:
        return 0
    return int(round((total_samples / float(sample_rate)) * 1000.0))


class MicrophoneStream:
    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        block_ms: int = 30,
        device: Optional[str] = None,
        dtype: str = "float32",
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.block_ms = int(block_ms)
        self.device = device
        self.dtype = dtype
        self.blocksize = max(1, int(self.sample_rate * self.block_ms / 1000))
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=64)
        self._stream = None
        self._status_messages: list[str] = []

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - hardware path
        del frames, time_info
        if status:
            self._status_messages.append(str(status))
        samples = np.asarray(indata, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1, dtype=np.float32)
        else:
            samples = samples.reshape(-1).astype(np.float32, copy=False)
        try:
            self._queue.put_nowait(samples.copy())
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(samples.copy())

    def start(self) -> "MicrophoneStream":
        sd = _require_sounddevice()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def __enter__(self) -> "MicrophoneStream":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def read_block(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def pop_status_messages(self) -> list[str]:
        messages = list(self._status_messages)
        self._status_messages.clear()
        return messages
