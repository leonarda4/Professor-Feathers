from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audio import RollingAudioBuffer


def rms_level_db(samples: np.ndarray, floor_db: float = -96.0) -> float:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return floor_db
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    if rms <= 1e-12:
        return floor_db
    return max(20.0 * math.log10(rms), floor_db)


@dataclass
class EndpointEvent:
    kind: str
    audio: np.ndarray | None = None
    speech_ms: int = 0
    trailing_silence_ms: int = 0
    reason: str = ""


class EnergyVad:
    def __init__(
        self,
        start_threshold_db: float = -42.0,
        stop_threshold_db: float = -48.0,
        start_margin_db: float = 12.0,
        stop_margin_db: float = 6.0,
        min_peak: float = 0.015,
        noise_floor_alpha: float = 0.92,
    ) -> None:
        self.start_threshold_db = float(start_threshold_db)
        self.stop_threshold_db = float(stop_threshold_db)
        self.start_margin_db = float(start_margin_db)
        self.stop_margin_db = float(stop_margin_db)
        self.min_peak = float(min_peak)
        self.noise_floor_alpha = float(noise_floor_alpha)
        self.noise_floor_db = -60.0

    def _update_noise_floor(self, level_db: float) -> None:
        self.noise_floor_db = (
            self.noise_floor_alpha * self.noise_floor_db
            + (1.0 - self.noise_floor_alpha) * level_db
        )

    def classify(self, frame: np.ndarray, in_speech: bool) -> bool:
        level_db = rms_level_db(frame)
        peak = float(np.max(np.abs(frame))) if frame.size else 0.0
        if not in_speech:
            self._update_noise_floor(level_db)
            threshold = max(self.start_threshold_db, self.noise_floor_db + self.start_margin_db)
        else:
            threshold = max(self.stop_threshold_db, self.noise_floor_db + self.stop_margin_db)
        if peak < self.min_peak:
            return False
        return level_db >= threshold


class UtteranceEndpointer:
    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 30,
        pre_roll_ms: int = 250,
        min_speech_ms: int = 200,
        end_silence_ms: int = 500,
        max_clip_ms: int = 2000,
        start_threshold_db: float = -42.0,
        stop_threshold_db: float = -48.0,
        start_margin_db: float = 12.0,
        stop_margin_db: float = 6.0,
        min_peak: float = 0.015,
        noise_floor_alpha: float = 0.92,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_samples = max(1, int(self.sample_rate * frame_ms / 1000))
        self.pre_roll_samples = max(0, int(self.sample_rate * pre_roll_ms / 1000))
        self.min_speech_samples = max(1, int(self.sample_rate * min_speech_ms / 1000))
        self.end_silence_samples = max(1, int(self.sample_rate * end_silence_ms / 1000))
        self.max_clip_samples = max(1, int(self.sample_rate * max_clip_ms / 1000))
        self.pre_buffer = RollingAudioBuffer(self.pre_roll_samples)
        self.vad = EnergyVad(
            start_threshold_db=start_threshold_db,
            stop_threshold_db=stop_threshold_db,
            start_margin_db=start_margin_db,
            stop_margin_db=stop_margin_db,
            min_peak=min_peak,
            noise_floor_alpha=noise_floor_alpha,
        )
        self._pending = np.empty(0, dtype=np.float32)
        self.armed = False
        self._reset_active_state(clear_pre_buffer=True)

    def _reset_active_state(self, clear_pre_buffer: bool = False) -> None:
        self._captured_chunks: list[np.ndarray] = []
        self._speech_samples = 0
        self._total_samples = 0
        self._trailing_silence_samples = 0
        self._in_speech = False
        if clear_pre_buffer:
            self.pre_buffer.clear()

    def arm(self) -> EndpointEvent:
        self.armed = True
        self._reset_active_state(clear_pre_buffer=False)
        return EndpointEvent(kind="armed")

    def disarm(self) -> None:
        self.armed = False
        self._reset_active_state(clear_pre_buffer=True)

    def _speech_duration_ms(self) -> int:
        return int(round((self._speech_samples / float(self.sample_rate)) * 1000.0))

    def _silence_duration_ms(self) -> int:
        return int(round((self._trailing_silence_samples / float(self.sample_rate)) * 1000.0))

    def _finalize_audio(self) -> np.ndarray:
        if not self._captured_chunks:
            return np.empty(0, dtype=np.float32)
        audio = np.concatenate(self._captured_chunks).astype(np.float32, copy=False)
        trim = min(self._trailing_silence_samples, max(0, audio.size - self.min_speech_samples))
        if trim > 0:
            audio = audio[:-trim]
        return audio

    def process_chunk(self, chunk: np.ndarray) -> list[EndpointEvent]:
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return []
        buffer = np.concatenate([self._pending, samples])
        events: list[EndpointEvent] = []
        while buffer.size >= self.frame_samples:
            frame = buffer[: self.frame_samples]
            buffer = buffer[self.frame_samples :]
            events.extend(self._process_frame(frame))
        self._pending = buffer.astype(np.float32, copy=False)
        return events

    def _process_frame(self, frame: np.ndarray) -> list[EndpointEvent]:
        if not self.armed:
            self.vad.classify(frame, in_speech=False)
            self.pre_buffer.extend(frame)
            return []

        prefix = self.pre_buffer.snapshot()
        is_speech = self.vad.classify(frame, in_speech=self._in_speech)

        if not self._in_speech:
            if not is_speech:
                self.pre_buffer.extend(frame)
                return []
            self._in_speech = True
            self._captured_chunks = [prefix, frame] if prefix.size else [frame]
            self._speech_samples = int(frame.size)
            self._total_samples = sum(chunk.size for chunk in self._captured_chunks)
            self._trailing_silence_samples = 0
            self.pre_buffer.clear()
            return [EndpointEvent(kind="speech_start", speech_ms=self._speech_duration_ms())]

        self._captured_chunks.append(frame)
        self._total_samples += int(frame.size)
        if is_speech:
            self._speech_samples += int(frame.size)
            self._trailing_silence_samples = 0
        else:
            self._trailing_silence_samples += int(frame.size)

        if self._total_samples >= self.max_clip_samples:
            audio = self._finalize_audio()
            event = EndpointEvent(
                kind="utterance",
                audio=audio,
                speech_ms=self._speech_duration_ms(),
                trailing_silence_ms=self._silence_duration_ms(),
                reason="max_clip",
            )
            self.armed = False
            self._reset_active_state(clear_pre_buffer=True)
            return [event]

        if self._trailing_silence_samples < self.end_silence_samples:
            return []

        if self._speech_samples >= self.min_speech_samples:
            audio = self._finalize_audio()
            event = EndpointEvent(
                kind="utterance",
                audio=audio,
                speech_ms=self._speech_duration_ms(),
                trailing_silence_ms=self._silence_duration_ms(),
                reason="end_silence",
            )
            self.armed = False
            self._reset_active_state(clear_pre_buffer=True)
            return [event]

        self._reset_active_state(clear_pre_buffer=True)
        return [
            EndpointEvent(
                kind="discarded",
                speech_ms=self._speech_duration_ms(),
                trailing_silence_ms=self._silence_duration_ms(),
                reason="short_noise",
            )
        ]
