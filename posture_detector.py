"""Webcam posture analysis on top of the MediaPipe Pose Landmarker Tasks API."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = PROJECT_DIR / "pose_landmarker_lite.task"
DEFAULT_CALIBRATION_PATH = PROJECT_DIR / "calibration.json"
DOWNLOAD_TIMEOUT_SEC = 30.0

CALIBRATION_FRAMES = 30
MIN_VISIBILITY = 0.5
# Below this (as a fraction of frame height) the person is too far away or side
# on, and dividing by the shoulder width would only amplify landmark noise.
MIN_SHOULDER_WIDTH = 0.05

NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 7, 8, 11, 12
DRAWN_LANDMARKS = {
    "nose": NOSE,
    "left_ear": LEFT_EAR,
    "right_ear": RIGHT_EAR,
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
}

STATUS_GOOD = "GOOD"
STATUS_HEAD_DROP = "SLOUCH_HEAD_DROP"
STATUS_FORWARD_NECK = "SLOUCH_FORWARD_NECK"
STATUS_SHOULDER_DROP = "SLOUCH_SHOULDER_DROP"
STATUS_SHOULDER_TILT = "SHOULDER_TILT"
STATUS_NO_PERSON = "NO_PERSON"
STATUS_CALIBRATING = "CALIBRATING"

MESSAGES = {
    STATUS_GOOD: "Duruş Harika (Dik)",
    STATUS_HEAD_DROP: "Başınızı ve boynunuzu dikleştirin",
    STATUS_FORWARD_NECK: "Başınız öne eğildi, boynunuzu geriye alın",
    STATUS_SHOULDER_DROP: "Omuzlarınız çöktü, dikleşin",
    STATUS_SHOULDER_TILT: "Omuzlarınızı dengeleyin, bir yana eğiliyorsunuz",
    STATUS_NO_PERSON: "Kişi algılanamadı",
}

# Share of the 100-point score each check can take away. Dict order also breaks
# severity ties when choosing the status.
SCORE_WEIGHTS = {
    STATUS_HEAD_DROP: 35.0,
    STATUS_FORWARD_NECK: 35.0,
    STATUS_SHOULDER_DROP: 15.0,
    STATUS_SHOULDER_TILT: 15.0,
}
# A check has lost all of its points at this multiple of its tolerance.
FULL_PENALTY_SEVERITY = 2.0


def ensure_model(path: Path | str = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Return the path to a usable model bundle, downloading it if needed."""
    path = Path(path)
    # Task bundles are zip files, which also catches a truncated earlier copy.
    if zipfile.is_zipfile(path):
        return path
    if path.exists():
        logger.warning("%s is not a valid model bundle, downloading it again", path)

    logger.info("Downloading pose model from %s", url)
    # Written atomically so a crash can't leave a truncated file to be reused.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".task")
    try:
        with os.fdopen(fd, "wb") as tmp, urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
            while chunk := response.read(1 << 16):
                tmp.write(chunk)
        if not zipfile.is_zipfile(tmp_name):
            raise RuntimeError(f"Downloaded pose model from {url} is not a valid bundle")
        os.replace(tmp_name, path)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Could not download pose model from {url}: {exc}") from exc
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class PostureMetrics:
    """Per-frame measurements; lengths never depend on the frame resolution."""

    head_shoulder_ratio: float  # Head height above the shoulders / shoulder width.
    neck_angle_deg: float  # Shoulders -> head vector vs. vertical, unsigned, 3D.
    shoulder_y: float  # Shoulder midpoint height, fraction of frame height.
    shoulder_width: float  # Fraction of frame height.
    shoulder_tilt_deg: float  # Shoulder line vs. horizontal, signed.


@dataclass(frozen=True)
class CalibrationData:
    """The user's upright baseline, averaged over the calibration frames."""

    head_shoulder_ratio: float
    neck_angle_deg: float
    shoulder_y: float
    shoulder_width: float
    shoulder_tilt_deg: float

    @classmethod
    def from_samples(cls, samples: Sequence[PostureMetrics]) -> CalibrationData:
        if not samples:
            raise ValueError("calibration needs at least one sample")
        return cls(
            **{
                field.name: float(np.mean([getattr(sample, field.name) for sample in samples]))
                for field in fields(cls)
            }
        )

    def save(self, path: Path | str = DEFAULT_CALIBRATION_PATH) -> None:
        path = Path(path)
        # Written atomically so a crash can't leave a corrupt baseline behind.
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                json.dump(asdict(self), tmp, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CALIBRATION_PATH) -> CalibrationData | None:
        """Return the saved baseline, or None if it is missing or unusable."""
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            values = {field.name: raw[field.name] for field in fields(cls)}
            if not all(type(v) in (int, float) and math.isfinite(v) for v in values.values()):
                raise ValueError("non-numeric calibration value")
            if values["shoulder_width"] <= 0:
                raise ValueError("shoulder_width must be positive")
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unusable calibration file %s: %s", path, exc)
            return None
        return cls(**{name: float(value) for name, value in values.items()})


@dataclass(frozen=True)
class Tolerances:
    """How far each metric may drift from its reference before it is a violation."""

    head_drop: float
    neck_angle_deg: float
    shoulder_drop: float
    shoulder_tilt_deg: float

    def __post_init__(self) -> None:
        for field in fields(self):
            if getattr(self, field.name) <= 0:
                raise ValueError(f"{field.name} must be positive")


# Population averages stand in for a baseline until the user calibrates, which
# is why their tolerances are wider. shoulder_y and shoulder_width are unused
# here: shoulder drop can't be judged without a personal reference.
DEFAULT_REFERENCE = CalibrationData(
    head_shoulder_ratio=0.60,
    neck_angle_deg=30.0,  # The head sits ahead of the shoulders even when upright.
    shoulder_y=0.0,
    shoulder_width=0.0,
    shoulder_tilt_deg=0.0,
)
DEFAULT_TOLERANCES = Tolerances(head_drop=0.20, neck_angle_deg=20.0, shoulder_drop=0.15, shoulder_tilt_deg=8.0)
CALIBRATED_TOLERANCES = Tolerances(head_drop=0.12, neck_angle_deg=12.0, shoulder_drop=0.12, shoulder_tilt_deg=5.0)
# Past this relative change in shoulder width the user has moved toward or away
# from the camera, so comparing shoulder_y to the baseline would be meaningless.
SHOULDER_DROP_MAX_DISTANCE_CHANGE = 0.25


@dataclass(frozen=True)
class PostureResult:
    has_person: bool
    is_slouching: bool
    status: str
    score: float
    message: str
    landmarks: dict[str, tuple[int, int]]  # Pixel (x, y) by name, for drawing.
    metrics: PostureMetrics | None = None


def compute_metrics(
    landmarks: Sequence,
    world_landmarks: Sequence,
    image_width: int,
    image_height: int,
    min_visibility: float = MIN_VISIBILITY,
) -> PostureMetrics | None:
    """Measure one pose, or return None if its head or shoulders aren't reliable."""

    # Normalized x and y scale with different frame sides, so every geometric
    # measure has to happen in pixels.
    def to_pixels(index: int) -> np.ndarray:
        landmark = landmarks[index]
        return np.array([landmark.x * image_width, landmark.y * image_height])

    def to_world(index: int) -> np.ndarray:
        landmark = world_landmarks[index]
        return np.array([landmark.x, landmark.y, landmark.z])

    def is_visible(index: int) -> bool:
        return (landmarks[index].visibility or 0.0) >= min_visibility

    head_indices = [i for i in (NOSE, LEFT_EAR, RIGHT_EAR) if is_visible(i)]
    if not head_indices or not (is_visible(LEFT_SHOULDER) and is_visible(RIGHT_SHOULDER)):
        return None

    left_shoulder, right_shoulder = to_pixels(LEFT_SHOULDER), to_pixels(RIGHT_SHOULDER)
    # 2D on purpose: the Lite model's z is too noisy to divide every metric by.
    across = left_shoulder - right_shoulder
    shoulder_width = float(np.hypot(*across))
    if shoulder_width < MIN_SHOULDER_WIDTH * image_height:
        return None

    shoulder_mid = (left_shoulder + right_shoulder) / 2
    head_y = np.mean([to_pixels(i)[1] for i in head_indices]) - shoulder_mid[1]

    # Depth is what makes forward head posture visible to a front-facing camera.
    # World landmarks are used because the image landmarks' z overstates it and
    # shifts with framing. Both y axes grow downward.
    neck_x, neck_y, neck_z = np.mean([to_world(i) for i in head_indices], axis=0) - (
        to_world(LEFT_SHOULDER) + to_world(RIGHT_SHOULDER)
    ) / 2

    return PostureMetrics(
        head_shoulder_ratio=float(-head_y / shoulder_width),
        neck_angle_deg=math.degrees(math.atan2(math.hypot(neck_x, neck_z), -neck_y)),
        shoulder_y=float(shoulder_mid[1] / image_height),
        shoulder_width=shoulder_width / image_height,
        shoulder_tilt_deg=math.degrees(math.atan2(across[1], abs(across[0]))),
    )


def _severities(metrics: PostureMetrics, calibration: CalibrationData | None) -> dict[str, float]:
    """Each check's deviation in units of its tolerance; 1.0 and up is a violation."""
    reference = calibration or DEFAULT_REFERENCE
    tolerances = CALIBRATED_TOLERANCES if calibration else DEFAULT_TOLERANCES

    shoulder_drop = 0.0
    if calibration:
        distance_change = abs(metrics.shoulder_width / calibration.shoulder_width - 1.0)
        if distance_change <= SHOULDER_DROP_MAX_DISTANCE_CHANGE:
            shoulder_drop = (metrics.shoulder_y - calibration.shoulder_y) / metrics.shoulder_width

    return {
        STATUS_HEAD_DROP: (reference.head_shoulder_ratio - metrics.head_shoulder_ratio) / tolerances.head_drop,
        STATUS_FORWARD_NECK: (metrics.neck_angle_deg - reference.neck_angle_deg) / tolerances.neck_angle_deg,
        STATUS_SHOULDER_DROP: shoulder_drop / tolerances.shoulder_drop,
        STATUS_SHOULDER_TILT: abs(metrics.shoulder_tilt_deg - reference.shoulder_tilt_deg)
        / tolerances.shoulder_tilt_deg,
    }


def evaluate_posture(metrics: PostureMetrics, calibration: CalibrationData | None = None) -> tuple[str, float]:
    """Return (status, score); the status is the worst violated check, if any."""
    severities = _severities(metrics, calibration)
    penalty = sum(
        SCORE_WEIGHTS[status] * min(max(severity, 0.0) / FULL_PENALTY_SEVERITY, 1.0)
        for status, severity in severities.items()
    )
    worst = max(severities, key=severities.__getitem__)
    return (worst if severities[worst] >= 1.0 else STATUS_GOOD), round(100.0 - penalty, 1)


class PostureDetector:
    """Turns BGR camera frames into PostureResults.

    Call detect() and the calibration methods from a single thread (the camera
    loop). Works without calibration, using wider default tolerances.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        calibration_path: Path | str = DEFAULT_CALIBRATION_PATH,
    ) -> None:
        self._calibration_path = Path(calibration_path)
        self._calibration = CalibrationData.load(self._calibration_path)
        self._calibration_target = 0
        self._calibration_samples: list[PostureMetrics] = []

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    @property
    def calibration(self) -> CalibrationData | None:
        return self._calibration

    @property
    def is_calibrated(self) -> bool:
        return self._calibration is not None

    @property
    def is_calibrating(self) -> bool:
        return self._calibration_target > 0

    @property
    def calibration_progress(self) -> tuple[int, int] | None:
        """(collected, target) frames while calibrating, otherwise None."""
        if not self.is_calibrating:
            return None
        return len(self._calibration_samples), self._calibration_target

    def start_calibration(self, frames: int = CALIBRATION_FRAMES) -> None:
        """Collect a new baseline from the next valid frames, e.g. on the 'C' key.

        The current baseline stays in place until collection completes.
        """
        if frames <= 0:
            raise ValueError("frames must be positive")
        self._calibration_target = frames
        self._calibration_samples = []

    def cancel_calibration(self) -> None:
        self._calibration_target = 0
        self._calibration_samples = []

    def detect(self, frame: np.ndarray | None) -> PostureResult:
        """Analyze one BGR frame. A missing frame counts as nobody being there."""
        if frame is None or frame.size == 0:
            return self._no_person()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection = self._landmarker.detect(image)
        if not detection.pose_landmarks:
            return self._no_person()
        height, width = frame.shape[:2]
        return self._analyze(detection.pose_landmarks[0], detection.pose_world_landmarks[0], width, height)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PostureDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _no_person(self) -> PostureResult:
        return PostureResult(
            has_person=False,
            is_slouching=False,
            status=STATUS_NO_PERSON,
            score=0.0,
            message=MESSAGES[STATUS_NO_PERSON],
            landmarks={},
        )

    def _analyze(self, landmarks: Sequence, world_landmarks: Sequence, width: int, height: int) -> PostureResult:
        metrics = compute_metrics(landmarks, world_landmarks, width, height)
        if metrics is None:
            return self._no_person()

        points = {
            name: (round(landmarks[index].x * width), round(landmarks[index].y * height))
            for name, index in DRAWN_LANDMARKS.items()
        }

        if self.is_calibrating:
            self._calibration_samples.append(metrics)
            collected = len(self._calibration_samples)
            if collected < self._calibration_target:
                # Reported as upright so the user isn't alerted while calibrating.
                return PostureResult(
                    has_person=True,
                    is_slouching=False,
                    status=STATUS_CALIBRATING,
                    score=100.0,
                    message=f"Kalibrasyon: {collected}/{self._calibration_target} - dik oturun",
                    landmarks=points,
                    metrics=metrics,
                )
            self._finish_calibration()

        status, score = evaluate_posture(metrics, self._calibration)
        return PostureResult(
            has_person=True,
            is_slouching=status != STATUS_GOOD,
            status=status,
            score=score,
            message=MESSAGES[status],
            landmarks=points,
            metrics=metrics,
        )

    def _finish_calibration(self) -> None:
        self._calibration = CalibrationData.from_samples(self._calibration_samples)
        self.cancel_calibration()
        try:
            self._calibration.save(self._calibration_path)
        except OSError as exc:
            # The baseline still applies for this session; it just won't persist.
            logger.warning("Could not save calibration to %s: %s", self._calibration_path, exc)
