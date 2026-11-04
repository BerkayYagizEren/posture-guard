"""Unit tests for app: python -m unittest test_app"""

from __future__ import annotations

import contextlib
import io
import itertools
import unittest
from unittest import mock

import numpy as np

import app as app_module
from app import NO_KEY, CameraError, PostureApp, open_camera, parse_args, run
from audio_notifier import AlertState, AudioNotifier, NotifierConfig
from posture_detector import MESSAGES, STATUS_FORWARD_NECK, STATUS_GOOD, STATUS_NO_PERSON, PostureResult

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
GOOD = PostureResult(True, False, STATUS_GOOD, 95.0, MESSAGES[STATUS_GOOD], {})
SLOUCH = PostureResult(True, True, STATUS_FORWARD_NECK, 40.0, MESSAGES[STATUS_FORWARD_NECK], {})
NOBODY = PostureResult(False, False, STATUS_NO_PERSON, 0.0, MESSAGES[STATUS_NO_PERSON], {})


class FakeDetector:
    def __init__(self) -> None:
        self.result = GOOD
        self.calibration_progress = None
        self.is_calibrated = True
        self.start_calibration = mock.Mock()

    @property
    def is_calibrating(self) -> bool:
        return self.calibration_progress is not None

    def detect(self, frame: np.ndarray) -> PostureResult:
        return self.result


class ParseArgsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        args, config = parse_args([])
        self.assertEqual(args.camera, 0)
        self.assertFalse(args.headless)
        self.assertEqual(
            (config.sound, config.volume, config.slouch_threshold_sec, config.cooldown_sec),
            ("tink", 0.15, 2.5, 3.5),
        )

    def test_overrides(self) -> None:
        args, config = parse_args(
            "--camera 2 --sound synth --volume 0.5 --threshold 4 --cooldown 9 --headless".split()
        )
        self.assertEqual(args.camera, 2)
        self.assertTrue(args.headless)
        self.assertEqual(
            (config.sound, config.volume, config.slouch_threshold_sec, config.cooldown_sec),
            ("synth", 0.5, 4.0, 9.0),
        )

    def test_invalid_values_exit_with_a_usage_error(self) -> None:
        for argv in ("--volume 2", "--threshold 0", "--cooldown -1", "--sound beep"):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parse_args(argv.split())
                self.assertEqual(caught.exception.code, 2)


class PostureAppTest(unittest.TestCase):
    def setUp(self) -> None:
        logging_patch = mock.patch.object(app_module, "logger")
        self.logger = logging_patch.start()
        self.addCleanup(logging_patch.stop)
        self.detector = FakeDetector()
        self.notifier = AudioNotifier(NotifierConfig())
        self.addCleanup(self.notifier.close)
        self.notifier.play = mock.Mock()
        self.app = PostureApp(self.detector, self.notifier)

    def test_alert_fires_only_after_the_threshold(self) -> None:
        self.detector.result = SLOUCH
        state = self.app.process(FRAME, 0.0)
        self.assertIs(state.alert_state, AlertState.PENDING)
        state = self.app.process(FRAME, 1.25)
        self.assertAlmostEqual(state.slouch_progress, 0.5)
        self.notifier.play.assert_not_called()
        state = self.app.process(FRAME, 2.5)
        self.assertIs(state.alert_state, AlertState.ALERTING)
        self.assertEqual(state.slouch_progress, 1.0)
        self.notifier.play.assert_called_once()

    def test_one_dropped_detection_does_not_restart_the_grace_period(self) -> None:
        self.detector.result = SLOUCH
        self.app.process(FRAME, 0.0)
        self.detector.result = NOBODY
        self.app.process(FRAME, 1.0)
        self.detector.result = SLOUCH
        self.assertAlmostEqual(self.app.process(FRAME, 1.1).slouch_progress, 1.1 / 2.5)

    def test_person_leaving_ends_the_episode_silently(self) -> None:
        self.detector.result = SLOUCH
        self.app.process(FRAME, 0.0)
        self.detector.result = NOBODY
        self.app.process(FRAME, 1.0)
        self.app.process(FRAME, 1.6)
        state = self.app.process(FRAME, 5.0)
        self.assertEqual(state.slouch_progress, 0.0)
        self.notifier.play.assert_not_called()

    def test_quit_keys(self) -> None:
        for key in (ord("q"), ord("Q"), 27):
            self.assertFalse(self.app.handle_key(key, 0.0))

    def test_unknown_key_is_ignored(self) -> None:
        self.assertTrue(self.app.handle_key(ord("x"), 0.0))

    def test_calibrate_key(self) -> None:
        self.assertTrue(self.app.handle_key(ord("c"), 0.0))
        self.detector.start_calibration.assert_called_once()

    def test_completed_calibration_shows_a_toast(self) -> None:
        self.detector.calibration_progress = (5, 30)
        self.app.handle_key(ord("c"), 0.0)
        self.assertIsNone(self.app.process(FRAME, 0.1).toast)
        self.detector.calibration_progress = None
        self.assertEqual(self.app.process(FRAME, 0.2).toast, "Kalibrasyon tamamlandı")
        self.assertIsNone(self.app.process(FRAME, 10.0).toast)

    def test_mute_key_toggles_and_reaches_the_hud(self) -> None:
        self.app.handle_key(ord("m"), 0.0)
        self.assertTrue(self.app.process(FRAME, 0.1).muted)
        self.app.handle_key(ord("M"), 0.2)
        self.assertFalse(self.app.process(FRAME, 0.3).muted)

    def test_sound_test_key(self) -> None:
        self.app.handle_key(ord("t"), 0.0)
        self.notifier.play.assert_called_once()

    def test_sound_test_while_muted_explains_why_nothing_plays(self) -> None:
        self.app.handle_key(ord("m"), 0.0)
        self.app.handle_key(ord("t"), 0.1)
        self.notifier.play.assert_not_called()
        self.assertIn("Sessiz", self.app.process(FRAME, 0.2).toast)

    def test_simple_view_key_toggles(self) -> None:
        self.app.handle_key(ord("h"), 0.0)
        self.assertTrue(self.app.process(FRAME, 0.1).simple_view)
        self.app.handle_key(ord("h"), 0.2)
        self.assertFalse(self.app.process(FRAME, 0.3).simple_view)

    def test_fps_is_measured_from_frame_times(self) -> None:
        for i in range(5):
            state = self.app.process(FRAME, i / 30)
        self.assertAlmostEqual(state.fps, 30.0, places=3)

    def test_logs_only_on_changes(self) -> None:
        for now in (0.0, 0.1, 0.2):
            self.app.process(FRAME, now)
        self.assertEqual(self.logger.info.call_count, 1)

    def test_startup_hint_only_when_uncalibrated(self) -> None:
        self.app.show_startup_hint(0.0)
        self.assertIsNone(self.app.process(FRAME, 0.1).toast)
        self.detector.is_calibrated = False
        self.app.show_startup_hint(0.2)
        self.assertIsNotNone(self.app.process(FRAME, 0.3).toast)


class CameraTest(unittest.TestCase):
    def test_camera_that_will_not_open_raises_and_is_released(self) -> None:
        camera = mock.Mock()
        camera.isOpened.return_value = False
        with mock.patch.object(app_module.cv2, "VideoCapture", return_value=camera):
            with self.assertRaises(CameraError):
                open_camera(3, timeout_sec=0.0)
        camera.release.assert_called_once()

    def test_camera_is_retried_while_the_permission_prompt_is_up(self) -> None:
        camera = mock.Mock()
        camera.isOpened.side_effect = [False, False, True]
        with (
            mock.patch.object(app_module.cv2, "VideoCapture", return_value=camera) as capture,
            mock.patch.object(app_module.time, "sleep"),
            mock.patch.object(app_module, "logger"),
        ):
            self.assertIs(open_camera(0), camera)
        self.assertEqual(capture.call_count, 3)

    def test_camera_that_stops_delivering_ends_the_loop(self) -> None:
        camera = mock.Mock()
        camera.read.return_value = (False, None)
        clock = iter(range(0, 1000, 5))
        with (
            mock.patch.object(app_module.time, "monotonic", side_effect=lambda: float(next(clock))),
            mock.patch.object(app_module.time, "sleep"),
        ):
            with self.assertRaises(CameraError):
                run(camera, mock.Mock(), headless=True)

    def test_single_failed_read_is_tolerated(self) -> None:
        camera = mock.Mock()
        camera.read.side_effect = [(False, None), (True, FRAME), KeyboardInterrupt]
        app = mock.Mock()
        with mock.patch.object(app_module.time, "sleep"), self.assertRaises(KeyboardInterrupt):
            run(camera, app, headless=True)
        app.process.assert_called_once()


class WindowedLoopTest(unittest.TestCase):
    def run_loop(self, keys: list[int], visible: float | Exception = 1.0, app: mock.Mock | None = None) -> mock.Mock:
        camera = mock.Mock()
        camera.read.return_value = (True, FRAME)
        if app is None:
            app = mock.Mock()
            app.process.return_value = app_module.HudState(GOOD)
            app.handle_key.side_effect = lambda key, now: key != ord("q")
        window_property = {"side_effect": visible} if isinstance(visible, Exception) else {"return_value": visible}
        with (
            mock.patch.object(app_module.cv2, "namedWindow"),
            mock.patch.object(app_module.cv2, "imshow"),
            mock.patch.object(app_module.cv2, "waitKey", side_effect=keys) as self.wait_key,
            mock.patch.object(app_module.cv2, "getWindowProperty", **window_property),
        ):
            run(camera, app, headless=False)
        return app

    def test_q_key_exits(self) -> None:
        app = self.run_loop([NO_KEY, ord("q")])
        self.assertEqual(app.process.call_count, 2)
        app.handle_key.assert_called_once()

    def test_minimized_window_keeps_monitoring(self) -> None:
        app = self.run_loop([NO_KEY, NO_KEY, NO_KEY, ord("q")], visible=0.0)
        self.assertEqual(app.process.call_count, 4)
        self.assertEqual(self.wait_key.call_count, 4)

    def test_slouching_while_minimized_still_alerts(self) -> None:
        detector = FakeDetector()
        detector.result = SLOUCH
        with AudioNotifier(NotifierConfig()) as notifier:
            notifier.play = mock.Mock()
            clock = itertools.count()
            with (
                mock.patch.object(app_module.time, "monotonic", side_effect=lambda: float(next(clock))),
                mock.patch.object(app_module, "logger"),
            ):
                self.run_loop([NO_KEY] * 4 + [ord("q")], visible=0.0, app=PostureApp(detector, notifier))
            notifier.play.assert_called()

    def test_closing_the_window_exits(self) -> None:
        app = self.run_loop([NO_KEY], visible=-1.0)
        self.assertEqual(app.process.call_count, 1)

    def test_window_error_exits(self) -> None:
        app = self.run_loop([NO_KEY], visible=app_module.cv2.error("null window"))
        self.assertEqual(app.process.call_count, 1)


class MainTest(unittest.TestCase):
    def test_camera_failure_cleans_up_and_returns_an_error(self) -> None:
        detector = mock.MagicMock()
        notifier = mock.MagicMock()
        with (
            mock.patch.object(app_module, "PostureDetector", return_value=detector),
            mock.patch.object(app_module, "AudioNotifier", return_value=notifier),
            mock.patch.object(app_module, "open_camera", side_effect=CameraError("yok")),
            mock.patch.object(app_module, "logger"),
            mock.patch.object(app_module.signal, "signal"),
        ):
            self.assertEqual(app_module.main(["--headless"]), 1)
        detector.__exit__.assert_called_once()
        notifier.__exit__.assert_called_once()

    def test_interrupt_releases_everything_and_exits_cleanly(self) -> None:
        detector = mock.MagicMock()
        notifier = mock.MagicMock()
        camera = mock.Mock()
        camera.read.side_effect = KeyboardInterrupt
        with (
            mock.patch.object(app_module, "PostureDetector", return_value=detector),
            mock.patch.object(app_module, "AudioNotifier", return_value=notifier),
            mock.patch.object(app_module, "open_camera", return_value=camera),
            mock.patch.object(app_module, "logger"),
            mock.patch.object(app_module.signal, "signal"),
        ):
            self.assertEqual(app_module.main(["--headless"]), 0)
        camera.release.assert_called_once()
        detector.__exit__.assert_called_once()
        notifier.__exit__.assert_called_once()


class SlouchProgressTest(unittest.TestCase):
    def test_progress_tracks_the_grace_period(self) -> None:
        with AudioNotifier(NotifierConfig(slouch_threshold_sec=2.0)) as notifier:
            notifier.play = mock.Mock()
            self.assertEqual(notifier.slouch_progress(0.0), 0.0)
            notifier.update(True, 10.0)
            self.assertAlmostEqual(notifier.slouch_progress(11.0), 0.5)
            self.assertEqual(notifier.slouch_progress(50.0), 1.0)
            notifier.reset()
            self.assertEqual(notifier.slouch_progress(51.0), 0.0)


if __name__ == "__main__":
    unittest.main()
