"""Posture Guard: watches your posture through the webcam and alerts you when you slouch."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Sequence

import cv2
import numpy as np

from audio_notifier import SYNTH_SOUND, SYSTEM_SOUNDS, AlertState, AudioNotifier, NotifierConfig
from gui_overlay import HudState, render_hud
from posture_detector import PostureDetector

logger = logging.getLogger("posture_guard")

WINDOW_NAME = "Posture Guard"
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1280, 720
# Opening fails at once while the macOS permission prompt is up, so this has to
# leave the user time to answer it on first launch.
CAMERA_OPEN_TIMEOUT_SEC = 30.0
CAMERA_OPEN_RETRY_SEC = 1.0
CAMERA_TIMEOUT_SEC = 10.0
CAMERA_RETRY_SEC = 0.05
TOAST_SEC = 3.0
STARTUP_TOAST_SEC = 6.0
FPS_SMOOTHING = 0.1
KEY_ESC = 27
NO_KEY = 255


class CameraError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, NotifierConfig]:
    parser = argparse.ArgumentParser(description="Posture Guard: web kamerasıyla duruş takibi ve sesli uyarı.")
    parser.add_argument("--camera", type=int, default=0, help="kamera indeksi (varsayılan: %(default)s)")
    parser.add_argument(
        "--sound",
        choices=(*SYSTEM_SOUNDS, SYNTH_SOUND),
        default="tink",
        help="uyarı sesi (varsayılan: %(default)s)",
    )
    parser.add_argument("--volume", type=float, default=0.15, help="ses seviyesi, 0.0-1.0 (varsayılan: %(default)s)")
    parser.add_argument(
        "--threshold", type=float, default=2.5, help="uyarıdan önceki kambur duruş süresi, sn (varsayılan: %(default)s)"
    )
    parser.add_argument("--cooldown", type=float, default=3.5, help="uyarı tekrar aralığı, sn (varsayılan: %(default)s)")
    parser.add_argument("--headless", action="store_true", help="penceresiz çalış: yalnızca konsol kaydı ve sesli uyarı")
    args = parser.parse_args(argv)
    try:
        config = NotifierConfig(
            slouch_threshold_sec=args.threshold,
            cooldown_sec=args.cooldown,
            sound=args.sound,
            volume=args.volume,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args, config


class PostureApp:
    """Connects the detector and notifier; holds the state the keyboard controls."""

    def __init__(self, detector: PostureDetector, notifier: AudioNotifier) -> None:
        self._detector = detector
        self._notifier = notifier
        self._simple_view = False
        self._toast: str | None = None
        self._toast_until = 0.0
        self._fps = 0.0
        self._last_frame_at: float | None = None
        self._was_calibrating = False
        self._last_logged: tuple[str, str] | None = None

    def process(self, frame: np.ndarray, now: float) -> HudState:
        """Analyze one camera frame and update alerts, logging and the frame rate."""
        result = self._detector.detect(frame)
        # Losing the person goes through the same recovery debounce as good posture
        # (reset() would not), so one dropped detection can't restart the grace period.
        alert_state = self._notifier.update(result.is_slouching, now)

        calibrating = self._detector.is_calibrating
        if self._was_calibrating and not calibrating:
            self._show_toast("Kalibrasyon tamamlandı", now)
            logger.info("Kalibrasyon tamamlandı")
        self._was_calibrating = calibrating

        if self._last_frame_at is not None and now > self._last_frame_at:
            instant_fps = 1.0 / (now - self._last_frame_at)
            self._fps = instant_fps if self._fps == 0.0 else self._fps + FPS_SMOOTHING * (instant_fps - self._fps)
        self._last_frame_at = now

        self._log_transition(result, alert_state, calibrating)
        if self._toast is not None and now >= self._toast_until:
            self._toast = None

        return HudState(
            result=result,
            alert_state=alert_state,
            slouch_progress=self._notifier.slouch_progress(now),
            threshold_sec=self._notifier.config.slouch_threshold_sec,
            calibration_progress=self._detector.calibration_progress,
            muted=self._notifier.muted,
            sound_name=self._notifier.config.sound,
            fps=self._fps,
            simple_view=self._simple_view,
            toast=self._toast,
        )

    def handle_key(self, key: int, now: float) -> bool:
        """Apply one key press; returns False when the app should quit."""
        if key == KEY_ESC or key in (ord("q"), ord("Q")):
            return False
        if key in (ord("c"), ord("C")):
            self._detector.start_calibration()
            self._was_calibrating = True
            logger.info("Kalibrasyon başladı")
        elif key in (ord("m"), ord("M")):
            muted = self._notifier.toggle_mute()
            self._show_toast("Ses kapatıldı" if muted else "Ses açıldı", now)
        elif key in (ord("t"), ord("T")):
            if self._notifier.muted:
                self._show_toast("Sessiz mod açık: önce [M] ile sesi açın", now)
            else:
                self._notifier.play()
                self._show_toast(f"Ses testi: {self._notifier.config.sound.capitalize()}", now)
        elif key in (ord("h"), ord("H")):
            self._simple_view = not self._simple_view
        return True

    def show_startup_hint(self, now: float) -> None:
        if not self._detector.is_calibrated:
            self._show_toast("Daha iyi sonuç için dik oturup [C] ile kalibre edin", now, STARTUP_TOAST_SEC)

    def _show_toast(self, text: str, now: float, duration: float = TOAST_SEC) -> None:
        self._toast = text
        self._toast_until = now + duration

    def _log_transition(self, result, alert_state: AlertState, calibrating: bool) -> None:
        if calibrating:
            current = ("calibrating", "")
        elif not result.has_person:
            current = ("no_person", "")
        else:
            current = (alert_state.value, result.status if alert_state is AlertState.ALERTING else "")
        if current == self._last_logged:
            return
        self._last_logged = current

        if calibrating:
            logger.info("Kalibrasyon sürüyor: dik durup kameraya bakın")
        elif not result.has_person:
            logger.info("Kamera açısında kimse yok")
        elif alert_state is AlertState.ALERTING:
            logger.warning("KAMBUR TESPİT EDİLDİ (puan %%%.0f): %s", result.score, result.message)
        elif alert_state is AlertState.PENDING:
            logger.info("Dikkat: duruş bozuldu (puan %%%.0f)", result.score)
        else:
            logger.info("Duruş iyi (puan %%%.0f)", result.score)


def open_camera(index: int, timeout_sec: float = CAMERA_OPEN_TIMEOUT_SEC) -> cv2.VideoCapture:
    deadline = time.monotonic() + timeout_sec
    camera = cv2.VideoCapture(index)
    while not camera.isOpened():
        camera.release()
        if time.monotonic() >= deadline:
            raise CameraError(
                f"Kamera {index} açılamadı. Başka bir uygulama kullanıyor olabilir ya da terminalin kamera izni "
                "kapalı olabilir (Sistem Ayarları > Gizlilik ve Güvenlik > Kamera)."
            )
        logger.info("Kamera %d bekleniyor. macOS izin isterse onaylayın.", index)
        time.sleep(CAMERA_OPEN_RETRY_SEC)
        camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    return camera


def window_closed() -> bool:
    # macOS reports a minimized window as 0.0, and monitoring has to carry on then,
    # so only a negative value or an error means the window is gone.
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 0
    except cv2.error:
        return True


def run(camera: cv2.VideoCapture, app: PostureApp, headless: bool) -> None:
    """Run the camera loop until the user quits or the camera stops delivering."""
    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    last_frame_at = time.monotonic()
    while True:
        ok, frame = camera.read()
        now = time.monotonic()
        if not ok:
            if now - last_frame_at >= CAMERA_TIMEOUT_SEC:
                raise CameraError(f"Kameradan {CAMERA_TIMEOUT_SEC:.0f} saniyedir görüntü alınamıyor.")
            time.sleep(CAMERA_RETRY_SEC)
            continue
        last_frame_at = now

        state = app.process(frame, now)
        if headless:
            continue

        cv2.imshow(WINDOW_NAME, render_hud(frame, state))
        key = cv2.waitKey(1) & 0xFF
        if key != NO_KEY and not app.handle_key(key, now):
            return
        if window_closed():
            return


def _raise_interrupt(*_: object) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args, config = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    # Stopping a background --headless run should clean up like Ctrl+C does.
    signal.signal(signal.SIGTERM, _raise_interrupt)

    camera = None
    try:
        with PostureDetector() as detector, AudioNotifier(config) as notifier:
            app = PostureApp(detector, notifier)
            app.show_startup_hint(time.monotonic())
            if not detector.is_calibrated:
                logger.warning("Kalibrasyon yok, varsayılan eşikler kullanılıyor. Pencereli modda [C] ile kalibre edin.")
            camera = open_camera(args.camera)
            logger.info("Başladı (%s). Çıkış: %s", "arka plan" if args.headless else "pencere", "Ctrl+C" if args.headless else "Q / ESC")
            run(camera, app, args.headless)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    finally:
        if camera is not None:
            camera.release()
        if not args.headless:
            cv2.destroyAllWindows()
            # macOS only tears the window down while the event loop is pumped.
            cv2.waitKey(1)
    logger.info("Kapatıldı")
    return 0


if __name__ == "__main__":
    sys.exit(main())
