"""Unit tests for gui_overlay: python -m unittest test_gui_overlay"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

import gui_overlay as gui
from audio_notifier import AlertState
from gui_overlay import HudState, banner_for, mirror_points, render_hud, shortcuts_for, shortcuts_panel
from posture_detector import MESSAGES, STATUS_FORWARD_NECK, STATUS_GOOD, STATUS_NO_PERSON, PostureResult

POINTS = {
    "nose": (640, 200),
    "left_ear": (700, 190),
    "right_ear": (580, 190),
    "left_shoulder": (800, 400),
    "right_shoulder": (480, 400),
}


def good_result(score: float = 95.0) -> PostureResult:
    return PostureResult(True, False, STATUS_GOOD, score, MESSAGES[STATUS_GOOD], POINTS)


def slouch_result() -> PostureResult:
    return PostureResult(True, True, STATUS_FORWARD_NECK, 40.0, MESSAGES[STATUS_FORWARD_NECK], POINTS)


def no_person_result() -> PostureResult:
    return PostureResult(False, False, STATUS_NO_PERSON, 0.0, MESSAGES[STATUS_NO_PERSON], {})


class BannerTest(unittest.TestCase):
    def test_good_posture_is_green_with_score(self) -> None:
        banner = banner_for(HudState(good_result()))
        self.assertEqual(banner.color, gui.GREEN)
        self.assertEqual(banner.title, "DİK DURUŞ - HARİKA")
        self.assertTrue(banner.show_score)

    def test_grace_period_is_amber(self) -> None:
        banner = banner_for(HudState(slouch_result(), AlertState.PENDING))
        self.assertEqual(banner.color, gui.AMBER)
        self.assertEqual(banner.title, "DİKKAT")

    def test_low_score_without_violation_is_amber(self) -> None:
        self.assertEqual(banner_for(HudState(good_result(score=60.0))).color, gui.AMBER)

    def test_alerting_is_red_with_advice(self) -> None:
        banner = banner_for(HudState(slouch_result(), AlertState.ALERTING))
        self.assertEqual(banner.color, gui.RED)
        self.assertEqual(banner.title, "DİK DURUN! KAMBUR TESPİT EDİLDİ")
        self.assertEqual(banner.subtitle, MESSAGES[STATUS_FORWARD_NECK])

    def test_recovering_alert_does_not_claim_posture_is_great(self) -> None:
        banner = banner_for(HudState(good_result(), AlertState.ALERTING))
        self.assertEqual(banner.color, gui.RED)
        self.assertNotEqual(banner.subtitle, MESSAGES[STATUS_GOOD])

    def test_no_person_is_gray(self) -> None:
        banner = banner_for(HudState(no_person_result()))
        self.assertEqual(banner.color, gui.GRAY)
        self.assertEqual(banner.title, "Kamera açısında kimse yok")
        self.assertFalse(banner.show_score)

    def test_calibration_is_blue_and_counts(self) -> None:
        banner = banner_for(HudState(good_result(), calibration_progress=(15, 30)))
        self.assertEqual(banner.color, gui.BLUE)
        self.assertEqual(banner.title, "KALİBRASYON: 15/30 - Dik durup kameraya bakın")

    def test_calibration_stays_visible_when_person_leaves(self) -> None:
        banner = banner_for(HudState(no_person_result(), calibration_progress=(15, 30)))
        self.assertEqual(banner.color, gui.BLUE)


class RenderTest(unittest.TestCase):
    def test_mirror_points(self) -> None:
        self.assertEqual(mirror_points({"a": (0, 5), "b": (99, 7)}, 100), {"a": (99, 5), "b": (0, 7)})

    def test_frame_is_mirrored_and_input_untouched(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # A marker in the middle of the frame, away from every HUD element.
        frame[400:420, 100:120] = (0, 0, 255)
        original = frame.copy()
        output = render_hud(frame, HudState(no_person_result()))
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(tuple(output[410, 1169]), (0, 0, 255))
        self.assertEqual(tuple(output[410, 110]), (0, 0, 0))

    def test_skeleton_uses_mirrored_coordinates_and_posture_color(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        x, y = POINTS["left_shoulder"]
        output = render_hud(frame, HudState(good_result()))
        self.assertEqual(tuple(output[y, 1279 - x]), gui.GREEN)
        self.assertEqual(tuple(output[y, x + 60]), (0, 0, 0))
        output = render_hud(frame, HudState(slouch_result()))
        self.assertEqual(tuple(output[y, 1279 - x]), gui.RED)

    def test_grace_bar_fill_grows_with_progress(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        def painted(progress: float) -> int:
            state = HudState(slouch_result(), AlertState.PENDING, slouch_progress=progress)
            return int(np.count_nonzero(render_hud(frame, state)[:300].sum(axis=2) > 300))

        self.assertLess(painted(0.2), painted(0.9))

    def test_no_grace_bar_without_a_person(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = HudState(no_person_result(), slouch_progress=0.5)
        np.testing.assert_array_equal(render_hud(frame, state), render_hud(frame, HudState(no_person_result())))

    def test_simple_view_hides_skeleton_and_footer(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        x, y = POINTS["left_shoulder"]
        output = render_hud(frame, HudState(good_result(), simple_view=True))
        self.assertEqual(tuple(output[y, 1279 - x]), (0, 0, 0))
        self.assertEqual(tuple(output[710, 5]), (0, 0, 0))

    def test_renders_at_any_size_in_every_state(self) -> None:
        states = [
            HudState(good_result()),
            HudState(slouch_result(), AlertState.PENDING, slouch_progress=0.4),
            HudState(slouch_result(), AlertState.ALERTING, slouch_progress=1.0, muted=True),
            HudState(no_person_result(), toast="Kalibrasyon tamamlandı"),
            HudState(good_result(), calibration_progress=(1, 30), simple_view=True),
        ]
        for height, width in ((240, 320), (480, 640), (720, 1280), (1080, 1920), (2160, 3840)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for state in states:
                with self.subTest(size=(width, height), state=state.alert_state):
                    self.assertEqual(render_hud(frame, state).shape, frame.shape)

    def test_landmarks_outside_the_frame_are_harmless(self) -> None:
        points = {name: (x * 10, -y) for name, (x, y) in POINTS.items()}
        result = PostureResult(True, False, STATUS_GOOD, 90.0, "", points)
        render_hud(np.zeros((720, 1280, 3), dtype=np.uint8), HudState(result))


class ShortcutsPanelTest(unittest.TestCase):
    SIZES = ((640, 480), (1280, 720), (1920, 1080))

    def scale(self, width: int, height: int) -> float:
        return max(min(width / 1280, height / 720), 0.6)

    def panel(self, width: int = 1280, height: int = 720, muted: bool = False) -> gui.ShortcutsPanel:
        panel = shortcuts_panel(width, height, muted, self.scale(width, height))
        self.assertIsNotNone(panel)
        return panel

    def test_lists_every_key(self) -> None:
        self.assertEqual([s.key for s in shortcuts_for(False)], ["C", "T", "M", "H", "Q/ESC"])

    def test_mute_row_follows_mute_state(self) -> None:
        unmuted = {s.key: s for s in shortcuts_for(False)}["M"]
        muted = {s.key: s for s in shortcuts_for(True)}["M"]
        self.assertEqual((unmuted.title, unmuted.status), ("Sesi Kapat", ""))
        self.assertEqual((muted.title, muted.status), ("Sesi Aç", "SESSİZ"))

    def test_muted_state_is_drawn(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        panel = self.panel(muted=True)
        output = render_hud(frame, HudState(good_result(), muted=True))
        card = output[panel.y : panel.y + panel.height, panel.x : panel.x + panel.width]
        self.assertTrue((card == gui.AMBER).all(axis=2).any())
        card = render_hud(frame, HudState(good_result()))[panel.y : panel.y + panel.height, panel.x : panel.x + panel.width]
        self.assertFalse((card == gui.AMBER).all(axis=2).any())

    def test_simple_view_hides_the_card(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        panel = self.panel()
        output = render_hud(frame, HudState(good_result(), simple_view=True))
        self.assertFalse(output[panel.y : panel.y + panel.height, panel.x : panel.x + panel.width].any())

    def test_card_and_text_fit_at_every_size(self) -> None:
        for width, height in self.SIZES:
            for muted in (False, True):
                with self.subTest(size=(width, height), muted=muted):
                    scale = self.scale(width, height)
                    panel = self.panel(width, height, muted)
                    self.assertLess(panel.x + panel.width, width)
                    self.assertLess(panel.y + panel.height, height - round(36 * scale))
                    _, _, font = gui._shortcut_fonts(scale)
                    for row in panel.rows:
                        for line in row.lines:
                            self.assertLessEqual(font.getlength(line), panel.text_width)

    def test_rows_stay_inside_the_card_and_do_not_overlap(self) -> None:
        for width, height in self.SIZES:
            with self.subTest(size=(width, height)):
                scale = self.scale(width, height)
                panel = self.panel(width, height)
                _, title_font, font = gui._shortcut_fonts(scale)
                bottom = panel.y
                for row in panel.rows:
                    self.assertGreaterEqual(row.y, bottom)
                    bottom = row.y + sum(title_font.getmetrics()) + len(row.lines) * sum(font.getmetrics())
                self.assertLessEqual(bottom, panel.y + panel.height)

    def test_nothing_crosses_into_the_card_in_any_state(self) -> None:
        # Status card, grace bar and toast all live left of the card, so any overflow
        # toward it has to paint the gap between them, which stays untouched.
        states = [
            HudState(good_result()),
            HudState(slouch_result(), AlertState.ALERTING, slouch_progress=1.0, muted=True, toast="Ses kapatıldı"),
            HudState(slouch_result(), AlertState.PENDING, slouch_progress=0.4, toast="Daha iyi sonuç için dik oturup [C] ile kalibre edin"),
            HudState(no_person_result(), calibration_progress=(15, 30), toast="Sessiz mod açık: önce [M] ile sesi açın"),
        ]
        for width, height in self.SIZES:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for state in states:
                with self.subTest(size=(width, height), state=state.alert_state):
                    panel = self.panel(width, height, state.muted)
                    # Without landmarks only the HUD cards can paint here.
                    output = render_hud(frame, replace(state, result=replace(state.result, landmarks={})))
                    bottom = height - round(36 * self.scale(width, height))
                    self.assertFalse(output[:bottom, panel.x - 1].any())
                    self.assertFalse(output[:bottom, panel.x + panel.width :].any())
                    left = output[:bottom, : panel.x]
                    self.assertTrue(left.any())

    def test_too_small_a_frame_drops_the_card(self) -> None:
        self.assertIsNone(shortcuts_panel(320, 240, False, 0.6))
        self.assertIsNone(shortcuts_panel(640, 200, False, 0.6))


if __name__ == "__main__":
    unittest.main()
