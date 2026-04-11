from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from audio import read_wav

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None


@dataclass
class FeedbackClips:
    recognized_dance: list[Path] = field(default_factory=list)
    recognized_sing: list[Path] = field(default_factory=list)
    not_recognized: list[Path] = field(default_factory=list)
    training: list[Path] = field(default_factory=list)


class AudioFeedbackPlayer:
    def __init__(self, clips_root: Path, *, rng_seed: int | None = None, dance_trigger_probability: float = 0.45) -> None:
        self.clips_root = Path(clips_root).resolve()
        self.rng = random.Random(rng_seed)
        self.dance_trigger_probability = max(0.0, min(1.0, float(dance_trigger_probability)))
        self.clips = self._discover_clips(self.clips_root)
        self.training_trigger_probability = 0.1

    @staticmethod
    def _sorted_existing(paths: Iterable[Path]) -> list[Path]:
        return sorted([path.resolve() for path in paths if path.exists() and path.is_file()])

    def _discover_clips(self, clips_root: Path) -> FeedbackClips:
        return FeedbackClips(
            recognized_dance=self._sorted_existing([clips_root / 'dance_recognized.wav']),
            recognized_sing=self._sorted_existing([clips_root / 'sing1.wav', clips_root / 'sing2.wav']),
            not_recognized=self._sorted_existing([
                clips_root / 'not_recognized1.wav',
                clips_root / 'not_recognized2.wav',
                clips_root / 'not_recognized3.wav',
            ]),
            training=self._sorted_existing([clips_root / 'training1.wav', clips_root / 'training2.wav']),
        )

    def available_summary(self) -> dict[str, int]:
        return {
            'recognized_dance': len(self.clips.recognized_dance),
            'recognized_sing': len(self.clips.recognized_sing),
            'not_recognized': len(self.clips.not_recognized),
            'training': len(self.clips.training),
        }

    def _play_file(self, path: Path) -> bool:
        if sd is None:
            return False
        try:
            samples, sample_rate = read_wav(path)
            sd.play(samples, samplerate=int(sample_rate), blocking=False)
            return True
        except Exception:
            return False

    def play_random(self, candidates: list[Path]) -> bool:
        if not candidates:
            return False
        return self._play_file(self.rng.choice(candidates))

    def maybe_play_recognized(self, action_label: str) -> bool:
        cleaned = str(action_label).strip().lower()
        if cleaned == 'dance':
            if self.rng.random() > self.dance_trigger_probability:
                return False
            return self.play_random(self.clips.recognized_dance)
        if cleaned == 'sing':
            return self.play_random(self.clips.recognized_sing)
        return False

    def play_not_recognized(self) -> bool:
        return self.play_random(self.clips.not_recognized)

    def play_training_now(self) -> bool:
        return self.play_random(self.clips.training)

    def maybe_play_training(self, *, probability: float | None = None) -> bool:
        trigger_probability = self.training_trigger_probability if probability is None else float(probability)
        trigger_probability = max(0.0, min(1.0, trigger_probability))
        if self.rng.random() > trigger_probability:
            return False
        return self.play_training_now()
