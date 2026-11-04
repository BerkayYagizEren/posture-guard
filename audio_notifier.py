"""Gentle, non-blocking audio alerts for sustained bad posture (macOS)."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

AFPLAY = "afplay"
SYSTEM_SOUNDS = {
    "tink": Path("/System/Library/Sounds/Tink.aiff"),
    "purr": Path("/System/Library/Sounds/Purr.aiff"),
}
SYNTH_SOUND = "synth"
DEFAULT_SYNTH_PATH = Path(__file__).resolve().parent / "sound.wav"

_STOP_WORKER = object()


def generate_chime(
    path: Path | str = DEFAULT_SYNTH_PATH,
    *,
    sample_rate: int = 44_100,
    duration_sec: float = 1.6,
) -> Path:
    """Write a soft two-note 440 Hz -> 528 Hz ambient ding to a 16-bit mono WAV."""
    path = Path(path)
    t = np.arange(int(sample_rate * duration_sec)) / sample_rate

    def note(freq: float, delay_sec: float, gain: float) -> np.ndarray:
        local_t = np.clip(t - delay_sec, 0.0, None)
        # The quiet octave adds a little shimmer without any harshness.
        tone = np.sin(2 * np.pi * freq * local_t) + 0.12 * np.sin(4 * np.pi * freq * local_t)
        attack = np.clip(local_t / 0.015, 0.0, 1.0)
        return gain * tone * (0.5 - 0.5 * np.cos(np.pi * attack)) * np.exp(-local_t / 0.45)

    signal = note(440.0, 0.0, 0.8) + note(528.0, 0.12, 1.0)

    # The fade guarantees the file ends on silence, so playback can't click.
    fade_samples = int(sample_rate * 0.2)
    signal[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    signal *= 0.9 / np.max(np.abs(signal))
    pcm = (signal * np.iinfo(np.int16).max).astype("<i2")

    # Written atomically so a crash can't leave a truncated file to be reused.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".wav")
    os.close(fd)
    try:
        with wave.open(tmp_name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


class AlertState(Enum):
    GOOD = "good"
    PENDING = "pending"  # Slouching, still inside the grace period.
    ALERTING = "alerting"  # Slouching past the threshold; alerts are repeating.


@dataclass(frozen=True)
class NotifierConfig:
    slouch_threshold_sec: float = 2.5
    cooldown_sec: float = 3.5
    # Good posture must hold this long to end an episode, so a single noisy
    # pose frame doesn't restart the slouch timer.
    recovery_sec: float = 0.5
    sound: str = "tink"  # "tink", "purr" or "synth".
    volume: float = 0.15
    synth_path: Path | str = DEFAULT_SYNTH_PATH

    def __post_init__(self) -> None:
        if self.slouch_threshold_sec <= 0:
            raise ValueError("slouch_threshold_sec must be positive")
        if self.cooldown_sec <= 0:
            raise ValueError("cooldown_sec must be positive")
        if self.recovery_sec < 0:
            raise ValueError("recovery_sec must not be negative")
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")
        if self.sound != SYNTH_SOUND and self.sound not in SYSTEM_SOUNDS:
            raise ValueError(f"unknown sound: {self.sound!r}")


class AudioNotifier:
    """Turns per-frame posture readings into debounced, non-blocking alerts.

    Call update() and the mute methods from a single thread (the camera loop).
    Playback happens on a worker thread, so update() never waits on audio.
    """

    def __init__(self, config: NotifierConfig | None = None) -> None:
        self.config = config or NotifierConfig()
        self._sound_path = self._resolve_sound_path()
        self._muted = False

        self._slouch_start: float | None = None
        self._good_since: float | None = None
        self._last_alert: float | None = None

        self._closed = False
        self._process: subprocess.Popen | None = None
        self._process_lock = threading.Lock()
        self._requests: queue.Queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._run_worker, name="audio-notifier", daemon=True)
        self._worker.start()

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        with self._process_lock:
            self._muted = muted
            if muted:
                self._drain_requests()
                self._stop_process()

    def toggle_mute(self) -> bool:
        self.set_muted(not self._muted)
        return self._muted

    def update(self, is_slouching: bool, now: float | None = None) -> AlertState:
        """Feed one posture reading; plays an alert when one is due."""
        now = time.monotonic() if now is None else now

        if is_slouching:
            self._good_since = None
            if self._slouch_start is None:
                self._slouch_start = now
        elif self._slouch_start is None:
            return AlertState.GOOD
        else:
            if self._good_since is None:
                self._good_since = now
            if now - self._good_since >= self.config.recovery_sec:
                self.reset()
                return AlertState.GOOD

        if now - self._slouch_start < self.config.slouch_threshold_sec:
            return AlertState.PENDING

        # While muted _last_alert stays put, so unmuting mid-slouch alerts at once.
        if is_slouching and not self._muted and self._cooldown_elapsed(now):
            self._last_alert = now
            self.play()
        return AlertState.ALERTING

    def slouch_progress(self, now: float | None = None) -> float:
        """How far the current slouch is through the grace period, from 0.0 to 1.0."""
        if self._slouch_start is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return min((now - self._slouch_start) / self.config.slouch_threshold_sec, 1.0)

    def reset(self) -> None:
        """End the current episode, e.g. when no person is in frame."""
        self._slouch_start = None
        self._good_since = None
        self._last_alert = None

    def play(self) -> None:
        """Request playback without blocking; dropped if one is already queued."""
        if self._closed or self._muted:
            return
        try:
            self._requests.put_nowait(self._sound_path)
        except queue.Full:
            pass

    def close(self) -> None:
        """Stop any playing sound and shut down the worker. Safe to call twice."""
        with self._process_lock:
            if self._closed:
                return
            self._closed = True
            self._drain_requests()
            self._stop_process()
            self._requests.put_nowait(_STOP_WORKER)
        self._worker.join(timeout=1.0)

    def __enter__(self) -> AudioNotifier:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _cooldown_elapsed(self, now: float) -> bool:
        return self._last_alert is None or now - self._last_alert >= self.config.cooldown_sec

    def _resolve_sound_path(self) -> Path:
        system_path = SYSTEM_SOUNDS.get(self.config.sound)
        if system_path is not None and system_path.exists():
            return system_path
        if system_path is not None:
            logger.warning("%s not found, falling back to the synthesized chime", system_path)
        synth_path = Path(self.config.synth_path)
        return synth_path if synth_path.exists() else generate_chime(synth_path)

    def _drain_requests(self) -> None:
        try:
            while True:
                self._requests.get_nowait()
        except queue.Empty:
            pass

    def _stop_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def _run_worker(self) -> None:
        while True:
            request = self._requests.get()
            if request is _STOP_WORKER:
                return
            # Spawning under the lock means a concurrent mute or close either
            # prevents this sound or sees the process and terminates it.
            with self._process_lock:
                if self._muted or self._closed:
                    continue
                try:
                    process = subprocess.Popen(
                        [AFPLAY, "-v", str(self.config.volume), str(request)],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError as exc:
                    logger.warning("Could not start %s: %s", AFPLAY, exc)
                    continue
                self._process = process
            process.wait()
            with self._process_lock:
                self._process = None
            if process.returncode > 0:
                logger.warning("%s exited with status %d", AFPLAY, process.returncode)


if __name__ == "__main__":
    # Audition each alert sound: python audio_notifier.py
    for sound_name in (*SYSTEM_SOUNDS, SYNTH_SOUND):
        print(f"Playing {sound_name}...")
        with AudioNotifier(NotifierConfig(sound=sound_name)) as notifier:
            notifier.play()
            time.sleep(2.0)
