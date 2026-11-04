"""Unit tests for posture_detector: python -m unittest test_posture_detector"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import posture_detector as pd
from posture_detector import CalibrationData, PostureDetector, PostureMetrics, compute_metrics, evaluate_posture

WIDTH, HEIGHT = 1280, 720


def make_pose(
    *,
    shoulder_y: float = 500.0,
    half_width: float = 150.0,
    head_rise: float = 100.0,
    head_shift_x: float = 0.0,
    head_z: float = 0.0,
    tilt_px: float = 0.0,
    center_x: float = 640.0,
    width: int = WIDTH,
    height: int = HEIGHT,
    visibility: dict[int, float] | None = None,
) -> list[SimpleNamespace]:
    """Build a landmark list from pixel-space geometry. Positive tilt_px lowers the left shoulder."""
    visibility = visibility or {}

    def landmark(index: int, x: float, y: float, z: float = 0.0) -> SimpleNamespace:
        return SimpleNamespace(x=x / width, y=y / height, z=z / width, visibility=visibility.get(index, 1.0))

    pose = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]
    head_x, head_y = center_x + head_shift_x, shoulder_y - head_rise
    pose[pd.NOSE] = landmark(pd.NOSE, head_x, head_y, head_z)
    pose[pd.LEFT_EAR] = landmark(pd.LEFT_EAR, head_x + 40, head_y, head_z)
    pose[pd.RIGHT_EAR] = landmark(pd.RIGHT_EAR, head_x - 40, head_y, head_z)
    pose[pd.LEFT_SHOULDER] = landmark(pd.LEFT_SHOULDER, center_x + half_width, shoulder_y + tilt_px / 2)
    pose[pd.RIGHT_SHOULDER] = landmark(pd.RIGHT_SHOULDER, center_x - half_width, shoulder_y - tilt_px / 2)
    return pose


def to_world(pose: list[SimpleNamespace], width: int = WIDTH, height: int = HEIGHT) -> list[SimpleNamespace]:
    """World landmarks for a pose: same geometry, metres instead of normalized coordinates."""
    return [
        SimpleNamespace(x=lm.x * width / 1000, y=lm.y * height / 1000, z=lm.z * width / 1000, visibility=lm.visibility)
        for lm in pose
    ]


def measure(pose: list[SimpleNamespace], width: int = WIDTH, height: int = HEIGHT) -> PostureMetrics | None:
    return compute_metrics(pose, to_world(pose, width, height), width, height)


def metrics_of(**kwargs) -> PostureMetrics:
    metrics = measure(make_pose(**kwargs), kwargs.get("width", WIDTH), kwargs.get("height", HEIGHT))
    assert metrics is not None
    return metrics


BASELINE = CalibrationData(
    head_shoulder_ratio=0.60,
    neck_angle_deg=5.0,
    shoulder_y=0.7,
    shoulder_width=0.4,
    shoulder_tilt_deg=1.0,
)
UPRIGHT = PostureMetrics(
    head_shoulder_ratio=0.60,
    neck_angle_deg=5.0,
    shoulder_y=0.7,
    shoulder_width=0.4,
    shoulder_tilt_deg=1.0,
)


class ComputeMetricsTest(unittest.TestCase):
    def test_upright_geometry(self) -> None:
        metrics = metrics_of(head_rise=180.0)
        self.assertAlmostEqual(metrics.head_shoulder_ratio, 0.6)
        self.assertAlmostEqual(metrics.neck_angle_deg, 0.0)
        self.assertAlmostEqual(metrics.shoulder_tilt_deg, 0.0)
        self.assertAlmostEqual(metrics.shoulder_width, 300.0 / HEIGHT)
        self.assertAlmostEqual(metrics.shoulder_y, 500.0 / HEIGHT)

    def test_head_ratio_is_independent_of_camera_distance(self) -> None:
        far = metrics_of(half_width=100.0, head_rise=60.0)
        near = metrics_of(half_width=200.0, head_rise=120.0)
        self.assertAlmostEqual(far.head_shoulder_ratio, near.head_shoulder_ratio)
        self.assertAlmostEqual(far.neck_angle_deg, near.neck_angle_deg)

    def test_angles_are_correct_on_non_square_frames(self) -> None:
        # Would come out wrong if angles were taken from normalized coordinates.
        metrics = metrics_of(head_rise=100.0, head_shift_x=100.0, tilt_px=300.0)
        self.assertAlmostEqual(metrics.neck_angle_deg, 45.0)
        self.assertAlmostEqual(metrics.shoulder_tilt_deg, 45.0)

    def test_metrics_are_independent_of_resolution(self) -> None:
        small = metrics_of(shoulder_y=500 / 2, half_width=75, head_rise=50, center_x=320, width=640, height=360)
        large = metrics_of(shoulder_y=500, half_width=150, head_rise=100, center_x=640)
        self.assertAlmostEqual(small.head_shoulder_ratio, large.head_shoulder_ratio)
        self.assertAlmostEqual(small.shoulder_y, large.shoulder_y)
        self.assertAlmostEqual(small.shoulder_width, large.shoulder_width)

    def test_forward_head_in_depth_raises_neck_angle(self) -> None:
        self.assertAlmostEqual(metrics_of(head_rise=100.0, head_z=-100.0).neck_angle_deg, 45.0)

    def test_neck_angle_ignores_image_z_in_favor_of_world_landmarks(self) -> None:
        pose = make_pose(head_rise=100.0)
        world = to_world(make_pose(head_rise=100.0, head_z=-100.0))
        metrics = compute_metrics(pose, world, WIDTH, HEIGHT)
        assert metrics is not None
        self.assertAlmostEqual(metrics.neck_angle_deg, 45.0)

        pose = make_pose(head_rise=100.0, head_z=-500.0)
        metrics = compute_metrics(pose, to_world(make_pose(head_rise=100.0)), WIDTH, HEIGHT)
        assert metrics is not None
        self.assertAlmostEqual(metrics.neck_angle_deg, 0.0)

    def test_tilt_sign_follows_the_lower_shoulder(self) -> None:
        self.assertGreater(metrics_of(tilt_px=40.0).shoulder_tilt_deg, 0.0)
        self.assertLess(metrics_of(tilt_px=-40.0).shoulder_tilt_deg, 0.0)

    def test_head_below_shoulders_gives_negative_ratio(self) -> None:
        metrics = metrics_of(head_rise=-30.0)
        self.assertLess(metrics.head_shoulder_ratio, 0.0)
        self.assertGreater(metrics.neck_angle_deg, 90.0)

    def test_hidden_shoulder_is_unreliable(self) -> None:
        pose = make_pose(visibility={pd.LEFT_SHOULDER: 0.2})
        self.assertIsNone(measure(pose))

    def test_fully_hidden_head_is_unreliable(self) -> None:
        pose = make_pose(visibility={pd.NOSE: 0.1, pd.LEFT_EAR: 0.1, pd.RIGHT_EAR: 0.1})
        self.assertIsNone(measure(pose))

    def test_missing_visibility_counts_as_hidden(self) -> None:
        pose = make_pose()
        pose[pd.LEFT_SHOULDER].visibility = None
        self.assertIsNone(measure(pose))

    def test_hidden_ears_fall_back_to_the_nose(self) -> None:
        pose = make_pose(head_rise=120.0, visibility={pd.LEFT_EAR: 0.1, pd.RIGHT_EAR: 0.1})
        metrics = measure(pose)
        assert metrics is not None
        self.assertAlmostEqual(metrics.head_shoulder_ratio, 0.4)

    def test_side_on_person_is_unreliable(self) -> None:
        self.assertIsNone(measure(make_pose(half_width=5.0)))


class EvaluatePostureTest(unittest.TestCase):
    def assert_status(self, metrics: PostureMetrics, status: str) -> None:
        self.assertEqual(evaluate_posture(metrics, BASELINE)[0], status)

    def test_matching_baseline_is_a_perfect_score(self) -> None:
        self.assertEqual(evaluate_posture(UPRIGHT, BASELINE), (pd.STATUS_GOOD, 100.0))

    def test_head_drop(self) -> None:
        self.assert_status(replace(UPRIGHT, head_shoulder_ratio=0.45), pd.STATUS_HEAD_DROP)

    def test_forward_neck(self) -> None:
        self.assert_status(replace(UPRIGHT, neck_angle_deg=20.0), pd.STATUS_FORWARD_NECK)

    def test_shoulder_drop(self) -> None:
        # 0.2 of a shoulder width below the baseline.
        self.assert_status(replace(UPRIGHT, shoulder_y=0.78), pd.STATUS_SHOULDER_DROP)

    def test_shoulder_tilt_in_either_direction(self) -> None:
        self.assert_status(replace(UPRIGHT, shoulder_tilt_deg=8.0), pd.STATUS_SHOULDER_TILT)
        self.assert_status(replace(UPRIGHT, shoulder_tilt_deg=-6.0), pd.STATUS_SHOULDER_TILT)

    def test_small_deviations_stay_good(self) -> None:
        metrics = replace(UPRIGHT, head_shoulder_ratio=0.55, neck_angle_deg=9.0, shoulder_y=0.72, shoulder_tilt_deg=3.0)
        status, score = evaluate_posture(metrics, BASELINE)
        self.assertEqual(status, pd.STATUS_GOOD)
        self.assertLess(score, 100.0)

    def test_improving_beyond_the_baseline_is_not_penalized(self) -> None:
        metrics = replace(UPRIGHT, head_shoulder_ratio=0.9, neck_angle_deg=0.0, shoulder_y=0.6)
        self.assertEqual(evaluate_posture(metrics, BASELINE), (pd.STATUS_GOOD, 100.0))

    def test_moving_closer_does_not_trigger_shoulder_drop(self) -> None:
        # Shoulders sit lower in frame, but they are also 50% wider: a distance change.
        self.assert_status(replace(UPRIGHT, shoulder_y=0.85, shoulder_width=0.6), pd.STATUS_GOOD)

    def test_worst_violation_wins(self) -> None:
        metrics = replace(UPRIGHT, head_shoulder_ratio=0.45, neck_angle_deg=50.0)
        self.assertEqual(evaluate_posture(metrics, BASELINE)[0], pd.STATUS_FORWARD_NECK)

    def test_score_is_bounded_and_monotonic(self) -> None:
        scores = [
            evaluate_posture(replace(UPRIGHT, head_shoulder_ratio=ratio), BASELINE)[1]
            for ratio in (0.6, 0.5, 0.4, 0.2, -1.0)
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(scores[-1], 65.0)

        worst = PostureMetrics(-5.0, 170.0, 5.0, 0.4, 80.0)
        self.assertEqual(evaluate_posture(worst, BASELINE)[1], 0.0)

    def test_uncalibrated_defaults(self) -> None:
        self.assertEqual(evaluate_posture(metrics_of(head_rise=180.0))[0], pd.STATUS_GOOD)
        self.assertEqual(evaluate_posture(metrics_of(head_rise=90.0))[0], pd.STATUS_HEAD_DROP)
        self.assertEqual(evaluate_posture(metrics_of(head_rise=180.0, head_z=-300.0))[0], pd.STATUS_FORWARD_NECK)
        self.assertEqual(evaluate_posture(metrics_of(head_rise=180.0, tilt_px=100.0))[0], pd.STATUS_SHOULDER_TILT)

    def test_uncalibrated_never_reports_shoulder_drop(self) -> None:
        self.assertEqual(evaluate_posture(replace(metrics_of(head_rise=180.0), shoulder_y=0.99))[0], pd.STATUS_GOOD)


class CalibrationDataTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "calibration.json"

    def test_from_samples_averages_every_field(self) -> None:
        low = PostureMetrics(0.4, 2.0, 0.5, 0.2, -1.0)
        high = PostureMetrics(0.6, 4.0, 0.7, 0.4, 3.0)
        baseline = CalibrationData.from_samples([low, high])
        self.assertAlmostEqual(baseline.head_shoulder_ratio, 0.5)
        self.assertAlmostEqual(baseline.neck_angle_deg, 3.0)
        self.assertAlmostEqual(baseline.shoulder_y, 0.6)
        self.assertAlmostEqual(baseline.shoulder_width, 0.3)
        self.assertAlmostEqual(baseline.shoulder_tilt_deg, 1.0)

    def test_from_samples_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            CalibrationData.from_samples([])

    def test_save_load_round_trip(self) -> None:
        BASELINE.save(self.path)
        self.assertEqual(CalibrationData.load(self.path), BASELINE)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], [self.path.name])

    def test_missing_file_is_silent(self) -> None:
        with self.assertNoLogs("posture_detector", level="WARNING"):
            self.assertIsNone(CalibrationData.load(self.path))

    def test_unusable_files_are_ignored_with_a_warning(self) -> None:
        valid = asdict(BASELINE)
        for content in (
            "{not json",
            "[1, 2]",
            json.dumps({"head_shoulder_ratio": 0.5}),
            json.dumps({**valid, "neck_angle_deg": "5"}),
            json.dumps({**valid, "neck_angle_deg": True}),
            json.dumps({**valid, "shoulder_width": 0}),
            json.dumps({**valid, "shoulder_tilt_deg": float("nan")}),
        ):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                with self.assertLogs("posture_detector", level="WARNING"):
                    self.assertIsNone(CalibrationData.load(self.path))

    def test_failed_save_leaves_previous_file_and_no_temp(self) -> None:
        BASELINE.save(self.path)
        with mock.patch("posture_detector.json.dump", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                replace(BASELINE, neck_angle_deg=99.0).save(self.path)
        self.assertEqual(CalibrationData.load(self.path), BASELINE)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], [self.path.name])


def zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("model.tflite", b"weights")
    return buffer.getvalue()


class EnsureModelTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "model.task"

    def test_existing_model_is_not_downloaded(self) -> None:
        self.path.write_bytes(zip_bytes())
        with mock.patch("posture_detector.urllib.request.urlopen") as urlopen:
            self.assertEqual(pd.ensure_model(self.path), self.path)
        urlopen.assert_not_called()

    def test_missing_model_is_downloaded(self) -> None:
        with mock.patch("posture_detector.urllib.request.urlopen", return_value=io.BytesIO(zip_bytes())):
            self.assertEqual(pd.ensure_model(self.path), self.path)
        self.assertTrue(zipfile.is_zipfile(self.path))

    def test_corrupt_model_is_replaced(self) -> None:
        self.path.write_bytes(b"truncated")
        with mock.patch("posture_detector.urllib.request.urlopen", return_value=io.BytesIO(zip_bytes())):
            with self.assertLogs("posture_detector", level="WARNING"):
                pd.ensure_model(self.path)
        self.assertTrue(zipfile.is_zipfile(self.path))

    def test_network_failure_raises_and_leaves_nothing_behind(self) -> None:
        error = urllib.error.URLError("offline")
        with mock.patch("posture_detector.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Could not download"):
                pd.ensure_model(self.path)
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_invalid_download_raises_and_leaves_nothing_behind(self) -> None:
        with mock.patch("posture_detector.urllib.request.urlopen", return_value=io.BytesIO(b"<html>error</html>")):
            with self.assertRaisesRegex(RuntimeError, "not a valid bundle"):
                pd.ensure_model(self.path)
        self.assertEqual(list(self.path.parent.iterdir()), [])


class DetectorFlowTest(unittest.TestCase):
    """Drives PostureDetector end to end with a fake landmarker."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.calibration_path = Path(tmp.name) / "calibration.json"
        self.landmarker = mock.Mock()
        self.show_pose(make_pose(head_rise=180.0))
        self.frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    def show_pose(self, pose: list | None) -> None:
        self.landmarker.detect.return_value = SimpleNamespace(
            pose_landmarks=[pose] if pose else [],
            pose_world_landmarks=[to_world(pose)] if pose else [],
        )

    def make_detector(self) -> PostureDetector:
        with (
            mock.patch("posture_detector.ensure_model", return_value=Path("model.task")),
            mock.patch("posture_detector.vision.PoseLandmarker.create_from_options", return_value=self.landmarker),
        ):
            return PostureDetector(calibration_path=self.calibration_path)

    def test_runs_uncalibrated(self) -> None:
        detector = self.make_detector()
        self.assertFalse(detector.is_calibrated)
        result = detector.detect(self.frame)
        self.assertTrue(result.has_person)
        self.assertFalse(result.is_slouching)
        self.assertEqual(result.status, pd.STATUS_GOOD)
        self.assertEqual(result.message, "Duruş Harika (Dik)")

    def test_landmarks_are_pixel_points_for_drawing(self) -> None:
        points = self.make_detector().detect(self.frame).landmarks
        self.assertEqual(set(points), {"nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder"})
        self.assertEqual(points["left_shoulder"], (790, 500))
        self.assertEqual(points["nose"], (640, 320))

    def test_calibration_progress_counts_collected_frames(self) -> None:
        detector = self.make_detector()
        self.assertIsNone(detector.calibration_progress)
        detector.start_calibration(frames=3)
        self.assertEqual(detector.calibration_progress, (0, 3))
        detector.detect(self.frame)
        self.assertEqual(detector.calibration_progress, (1, 3))
        detector.detect(self.frame)
        detector.detect(self.frame)
        self.assertIsNone(detector.calibration_progress)

    def test_no_pose(self) -> None:
        self.show_pose(None)
        result = self.make_detector().detect(self.frame)
        self.assertEqual(result.status, pd.STATUS_NO_PERSON)
        self.assertFalse(result.has_person)
        self.assertFalse(result.is_slouching)
        self.assertEqual(result.landmarks, {})

    def test_failed_camera_read_is_no_person(self) -> None:
        detector = self.make_detector()
        self.assertEqual(detector.detect(None).status, pd.STATUS_NO_PERSON)
        self.assertEqual(detector.detect(np.empty((0, 0, 3), dtype=np.uint8)).status, pd.STATUS_NO_PERSON)
        self.landmarker.detect.assert_not_called()

    def test_frame_is_converted_to_rgb(self) -> None:
        self.frame[..., 0] = 255  # Pure blue in BGR.
        self.make_detector().detect(self.frame)
        image = self.landmarker.detect.call_args.args[0]
        self.assertEqual(tuple(image.numpy_view()[0, 0]), (0, 0, 255))

    def test_calibration_flow_persists_and_personalizes(self) -> None:
        # This person's upright head sits low enough that the defaults flag it.
        personal_upright = make_pose(head_rise=100.0)
        self.show_pose(personal_upright)
        detector = self.make_detector()
        self.assertEqual(detector.detect(self.frame).status, pd.STATUS_HEAD_DROP)

        detector.start_calibration()
        for frame_number in range(1, pd.CALIBRATION_FRAMES):
            result = detector.detect(self.frame)
            self.assertEqual(result.status, pd.STATUS_CALIBRATING)
            self.assertFalse(result.is_slouching)
            self.assertIn(f"{frame_number}/30", result.message)
        self.assertFalse(detector.is_calibrated)

        final = detector.detect(self.frame)
        self.assertEqual(final.status, pd.STATUS_GOOD)
        self.assertEqual(final.score, 100.0)
        self.assertFalse(detector.is_calibrating)
        self.assertTrue(detector.is_calibrated)
        self.assertTrue(self.calibration_path.exists())

        self.show_pose(make_pose(head_rise=40.0))
        self.assertEqual(detector.detect(self.frame).status, pd.STATUS_HEAD_DROP)

        # A new session picks the saved baseline up.
        self.show_pose(personal_upright)
        restarted = self.make_detector()
        self.assertTrue(restarted.is_calibrated)
        self.assertEqual(restarted.detect(self.frame).status, pd.STATUS_GOOD)

    def test_calibration_counts_only_valid_frames(self) -> None:
        detector = self.make_detector()
        detector.start_calibration(frames=2)
        detector.detect(self.frame)
        self.show_pose(None)
        for _ in range(5):
            self.assertEqual(detector.detect(self.frame).status, pd.STATUS_NO_PERSON)
        self.show_pose(make_pose(visibility={pd.LEFT_SHOULDER: 0.0}))
        self.assertEqual(detector.detect(self.frame).status, pd.STATUS_NO_PERSON)
        self.assertTrue(detector.is_calibrating)
        self.show_pose(make_pose(head_rise=180.0))
        detector.detect(self.frame)
        self.assertTrue(detector.is_calibrated)

    def test_recalibration_keeps_old_baseline_until_done_and_restart_clears_samples(self) -> None:
        BASELINE.save(self.calibration_path)
        detector = self.make_detector()
        detector.start_calibration(frames=2)
        detector.detect(self.frame)
        self.assertEqual(detector.calibration, BASELINE)
        detector.start_calibration(frames=2)
        self.assertIn("1/2", detector.detect(self.frame).message)

    def test_cancel_keeps_baseline(self) -> None:
        BASELINE.save(self.calibration_path)
        detector = self.make_detector()
        detector.start_calibration()
        detector.detect(self.frame)
        detector.cancel_calibration()
        self.assertFalse(detector.is_calibrating)
        self.assertEqual(detector.calibration, BASELINE)
        self.assertNotEqual(detector.detect(self.frame).status, pd.STATUS_CALIBRATING)

    def test_save_failure_keeps_baseline_for_the_session(self) -> None:
        detector = self.make_detector()
        detector.start_calibration(frames=1)
        with mock.patch.object(CalibrationData, "save", side_effect=OSError("read-only")):
            with self.assertLogs("posture_detector", level="WARNING"):
                detector.detect(self.frame)
        self.assertTrue(detector.is_calibrated)

    def test_invalid_frame_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make_detector().start_calibration(frames=0)

    def test_context_manager_closes_landmarker(self) -> None:
        with self.make_detector():
            pass
        self.landmarker.close.assert_called_once()


@unittest.skipUnless(zipfile.is_zipfile(pd.DEFAULT_MODEL_PATH), "pose model not downloaded")
class RealModelTest(unittest.TestCase):
    def test_blank_frame_has_no_person(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, PostureDetector(calibration_path=Path(tmp) / "c.json") as detector:
            result = detector.detect(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        self.assertEqual(result.status, pd.STATUS_NO_PERSON)


if __name__ == "__main__":
    unittest.main()
