"""Head-up display drawn over the mirrored camera image."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audio_notifier import AlertState
from posture_detector import MESSAGES, PostureResult, STATUS_GOOD

# All colors are BGR, like the frames they are drawn on.
GREEN = (120, 200, 90)
AMBER = (40, 165, 250)
RED = (70, 70, 235)
BLUE = (235, 150, 60)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
SOFT_WHITE = (225, 225, 225)
INK = (28, 24, 22)

# An upright pose scoring below this is drifting, though no check has failed yet.
WARNING_SCORE = 70.0

# Every text here has to fit Turkish characters, which rules out cv2.putText.
_FONT_FILE = Path("/System/Library/Fonts/HelveticaNeue.ttc")
_FONT_INDEX_REGULAR, _FONT_INDEX_BOLD = 0, 1
_BASE_WIDTH, _BASE_HEIGHT = 1280, 720
_MIN_SCALE = 0.6
_MIN_FONT_SIZE = 8
_FOOTER_HEIGHT = 36
_BANNER_WIDTH = 780
# The shortcuts card gives way entirely on frames too small for it next to a
# usable status card; the footer still lists the keys there.
_SHORTCUTS_WIDTH = 400
_MIN_BANNER_WIDTH = 300


@dataclass(frozen=True)
class HudState:
    """Everything one HUD frame shows, assembled by the app loop."""

    result: PostureResult
    alert_state: AlertState = AlertState.GOOD
    slouch_progress: float = 0.0
    threshold_sec: float = 2.5
    calibration_progress: tuple[int, int] | None = None
    muted: bool = False
    sound_name: str = "tink"
    fps: float = 0.0
    simple_view: bool = False
    toast: str | None = None


@dataclass(frozen=True)
class Banner:
    color: tuple[int, int, int]
    title: str
    subtitle: str = ""
    show_score: bool = False


@dataclass(frozen=True)
class Shortcut:
    key: str
    title: str
    description: str = ""
    status: str = ""


@dataclass(frozen=True)
class ShortcutRow:
    shortcut: Shortcut
    y: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class ShortcutsPanel:
    """Where the shortcuts card and its rows go; y coordinates are frame-absolute."""

    x: int
    y: int
    width: int
    height: int
    text_x: int
    text_width: int
    rows: tuple[ShortcutRow, ...]


def shortcuts_for(muted: bool) -> tuple[Shortcut, ...]:
    if muted:
        mute = Shortcut("M", "Sesi Aç", "Uyarı sesleri şu an kapalı", "SESSİZ")
    else:
        mute = Shortcut("M", "Sesi Kapat", "Uyarı seslerini sessize alır (Mute)")
    return (
        Shortcut("C", "Kalibre Et", "Kameraya bakıp dik oturun ve basın; 1 saniye içinde ideal dik pozisyonunuzu referans alır"),
        Shortcut("T", "Ses Testi", "Kulaklığındaki uyarı sesinin yüksekliğini test eder"),
        mute,
        Shortcut("H", "Sade Görünüm", "Gereksiz bilgileri gizler, sadece ince durum çubuğu kalır"),
        Shortcut("Q/ESC", "Çıkış"),
    )


def banner_for(state: HudState) -> Banner:
    """Pick the status card. Calibrating wins even with nobody in frame, because
    collection is only paused then and the user needs to know it is still running."""
    result = state.result
    if state.calibration_progress is not None:
        collected, target = state.calibration_progress
        return Banner(BLUE, f"KALİBRASYON: {collected}/{target} - Dik durup kameraya bakın")
    if not result.has_person:
        return Banner(GRAY, "Kamera açısında kimse yok")
    # While recovering the status is already GOOD but the alert episode is still open.
    advice = result.message if result.is_slouching else "Düzeliyor, böyle kalın"
    if state.alert_state is AlertState.ALERTING:
        return Banner(RED, "DİK DURUN! KAMBUR TESPİT EDİLDİ", advice, show_score=True)
    if state.alert_state is AlertState.PENDING:
        return Banner(AMBER, "DİKKAT", advice, show_score=True)
    if result.score < WARNING_SCORE:
        return Banner(AMBER, "DİKKAT", "Duruşunuz bozulmaya başladı", show_score=True)
    return Banner(GREEN, "DİK DURUŞ - HARİKA", MESSAGES[STATUS_GOOD], show_score=True)


def render_hud(frame: np.ndarray, state: HudState) -> np.ndarray:
    """Return a mirrored copy of the BGR frame with the HUD drawn on it."""
    canvas = cv2.flip(frame, 1)
    height, width = canvas.shape[:2]
    scale = max(min(width / _BASE_WIDTH, height / _BASE_HEIGHT), _MIN_SCALE)
    banner = banner_for(state)
    margin = round(16 * scale)
    # The grace bar and toast stay centered in this span, clear of the shortcuts card.
    left, right = margin, width - margin

    if state.simple_view:
        bottom = _draw_status_pill(canvas, banner, scale)
        _draw_text_right(canvas, "[H] Tam Görünüm", width - round(14 * scale), height - round(28 * scale), 13, scale)
    else:
        panel = shortcuts_panel(width, height, state.muted, scale)
        if panel is not None:
            right = panel.x - margin
        right = min(right, left + round(_BANNER_WIDTH * scale))
        _draw_skeleton(canvas, state, scale)
        bottom = _draw_banner(canvas, banner, state, scale, right - left)
        if panel is not None:
            _draw_shortcuts(canvas, panel, scale)
        _draw_footer(canvas, state, scale)
    bottom = _draw_grace_bar(canvas, state, bottom, scale, left, right)
    if state.toast:
        _draw_toast(canvas, state.toast, bottom, scale, left, right)
    return canvas


def mirror_points(points: dict[str, tuple[int, int]], width: int) -> dict[str, tuple[int, int]]:
    """Convert landmark pixels from camera coordinates to mirrored ones."""
    return {name: (width - 1 - x, y) for name, (x, y) in points.items()}


def _draw_skeleton(canvas: np.ndarray, state: HudState, scale: float) -> None:
    points = state.result.landmarks
    if not points:
        return
    points = mirror_points(points, canvas.shape[1])
    if state.calibration_progress is not None:
        color = BLUE
    else:
        color = RED if state.result.is_slouching else GREEN

    left, right = np.array(points["left_shoulder"]), np.array(points["right_shoulder"])
    left_ear, right_ear = np.array(points["left_ear"]), np.array(points["right_ear"])
    shoulder_mid = tuple(int(v) for v in (left + right) // 2)
    head = tuple(int(v) for v in (left_ear + right_ear) // 2)

    thickness = max(round(4 * scale), 2)
    # The dark underlay keeps the lines readable against any clothing or wall.
    for line_color, line_thickness in ((INK, thickness + 3), (color, thickness)):
        cv2.line(canvas, points["left_shoulder"], points["right_shoulder"], line_color, line_thickness, cv2.LINE_AA)
        cv2.line(canvas, shoulder_mid, head, line_color, line_thickness, cv2.LINE_AA)

    joint_radius = max(round(7 * scale), 3)
    face_radius = max(round(5 * scale), 3)
    for name, radius in (
        ("left_shoulder", joint_radius),
        ("right_shoulder", joint_radius),
        ("left_ear", face_radius),
        ("right_ear", face_radius),
        ("nose", face_radius),
    ):
        cv2.circle(canvas, points[name], radius + 2, INK, -1, cv2.LINE_AA)
        cv2.circle(canvas, points[name], radius, color, -1, cv2.LINE_AA)


def _draw_banner(canvas: np.ndarray, banner: Banner, state: HudState, scale: float, card_width: int) -> int:
    """Draw the top-left status card and return the y coordinate of its bottom edge."""
    margin = round(16 * scale)
    card_height = round(88 * scale)
    x, y = margin, margin
    pad = round(24 * scale)

    _paint_card(canvas, x, y, card_width, card_height, _tinted(banner.color), 0.82, round(16 * scale))
    _paint(canvas, _rounded_mask(round(7 * scale), card_height - 2 * round(12 * scale), round(3 * scale)), x + round(10 * scale), y + round(12 * scale), banner.color)

    text_x = x + pad + round(4 * scale)
    text_right = x + card_width - pad
    if banner.show_score:
        score = f"%{round(state.result.score)}"
        score_font = _font(round(38 * scale), bold=True)
        score_width = round(score_font.getlength(score))
        _draw_text(canvas, score, text_right - score_width, y + round(12 * scale), score_font, WHITE)
        caption_font = _font(round(12 * scale))
        caption = "Duruş Puanı"
        _draw_text(
            canvas,
            caption,
            text_right - round(caption_font.getlength(caption)),
            y + round(60 * scale),
            caption_font,
            SOFT_WHITE,
        )
        text_right -= max(score_width, round(caption_font.getlength(caption))) + pad

    available = text_right - text_x
    title_font = _fit_font(banner.title, round(26 * scale), available, bold=True)
    if banner.subtitle:
        _draw_text(canvas, banner.title, text_x, y + round(14 * scale), title_font, WHITE)
        subtitle_font = _fit_font(banner.subtitle, round(17 * scale), available)
        _draw_text(canvas, banner.subtitle, text_x, y + round(50 * scale), subtitle_font, SOFT_WHITE)
    else:
        line_height = sum(title_font.getmetrics())
        _draw_text(canvas, banner.title, text_x, y + (card_height - line_height) // 2, title_font, WHITE)
    return y + card_height


def _draw_status_pill(canvas: np.ndarray, banner: Banner, scale: float) -> int:
    """Simple view's compact status: a colored dot and the title."""
    margin = round(14 * scale)
    pad = round(14 * scale)
    dot = round(12 * scale)
    pill_height = round(38 * scale)
    available = canvas.shape[1] - 2 * margin - 3 * pad - dot
    font = _fit_font(banner.title, round(16 * scale), available, bold=True)
    pill_width = 3 * pad + dot + round(font.getlength(banner.title))

    _paint_card(canvas, margin, margin, pill_width, pill_height, INK, 0.72, pill_height // 2)
    _paint(canvas, _rounded_mask(dot, dot, dot // 2), margin + pad, margin + (pill_height - dot) // 2, banner.color)
    line_height = sum(font.getmetrics())
    _draw_text(canvas, banner.title, margin + 2 * pad + dot, margin + (pill_height - line_height) // 2, font, WHITE)
    return margin + pill_height


def _draw_grace_bar(canvas: np.ndarray, state: HudState, top: int, scale: float, left: int, right: int) -> int:
    """Draw the countdown to the sound alert below `top`; return the new bottom edge."""
    result = state.result
    if state.slouch_progress <= 0.0 or not result.has_person or state.calibration_progress is not None:
        return top

    card_width = min(round(420 * scale), right - left)
    card_height = round(52 * scale)
    x, y = left + (right - left - card_width) // 2, top + round(10 * scale)
    pad = round(14 * scale)

    progress = min(state.slouch_progress, 1.0)
    if state.alert_state is AlertState.ALERTING:
        label = "Sessiz mod: uyarı sesi kapalı" if state.muted else "Ses uyarısı devrede"
    else:
        remaining = state.threshold_sec * (1.0 - progress)
        label = f"Ses uyarısına {remaining:.1f} sn"

    _paint_card(canvas, x, y, card_width, card_height, INK, 0.76, round(12 * scale))
    font = _fit_font(label, round(14 * scale), card_width - 2 * pad)
    _draw_text(canvas, label, x + pad, y + round(7 * scale), font, WHITE)

    bar_x, bar_y = x + pad, y + round(32 * scale)
    bar_width, bar_height = card_width - 2 * pad, max(round(10 * scale), 4)
    radius = bar_height // 2
    _paint(canvas, _rounded_mask(bar_width, bar_height, radius), bar_x, bar_y, WHITE, 0.18)
    fill_width = max(round(bar_width * progress), bar_height)
    _paint(canvas, _rounded_mask(fill_width, bar_height, radius), bar_x, bar_y, _mix(AMBER, RED, progress))
    return y + card_height


def shortcuts_panel(frame_width: int, frame_height: int, muted: bool, scale: float) -> ShortcutsPanel | None:
    """Lay out the top-right shortcuts card, or None if the frame has no room for it."""
    margin = round(16 * scale)
    width = round(_SHORTCUTS_WIDTH * scale)
    if frame_width - 3 * margin - width < round(_MIN_BANNER_WIDTH * scale):
        return None

    pad = round(16 * scale)
    gap = round(12 * scale)
    key_font, title_font, description_font = _shortcut_fonts(scale)
    title_height = sum(title_font.getmetrics())
    description_height = sum(description_font.getmetrics())
    shortcuts = shortcuts_for(muted)
    key_column = max(_keycap_width(shortcut.key, key_font, scale) for shortcut in shortcuts)

    x = frame_width - margin - width
    text_x = x + pad + key_column + gap
    text_width = x + width - pad - text_x
    row_y = margin + _shortcuts_header_height(scale)
    rows = []
    for shortcut in shortcuts:
        lines = _wrap_text(shortcut.description, description_font, text_width)
        rows.append(ShortcutRow(shortcut, row_y, lines))
        row_height = max(title_height + len(lines) * description_height, _keycap_height(scale))
        row_y += row_height + gap
    height = row_y - gap + pad - margin

    if margin + height > frame_height - round(_FOOTER_HEIGHT * scale) - margin:
        return None
    return ShortcutsPanel(x, margin, width, height, text_x, text_width, tuple(rows))


def _draw_shortcuts(canvas: np.ndarray, panel: ShortcutsPanel, scale: float) -> None:
    x, y, width, height = panel.x, panel.y, panel.width, panel.height
    pad = round(16 * scale)
    radius = round(16 * scale)
    key_font, title_font, description_font = _shortcut_fonts(scale)
    title_height = sum(title_font.getmetrics())
    description_height = sum(description_font.getmetrics())

    # Glass: the blurred camera image shows through dark tint, with a faint lit edge.
    _blur_behind(canvas, x, y, width, height, radius, scale)
    _paint_card(canvas, x, y, width, height, WHITE, 0.14, radius)
    edge = max(round(scale), 1)
    _paint_card(canvas, x + edge, y + edge, width - 2 * edge, height - 2 * edge, INK, 0.72, radius - edge)

    header_font = _font(max(round(13 * scale), _MIN_FONT_SIZE), bold=True)
    _draw_text(canvas, "Kısayollar / Kontroller", x + pad, y + round(12 * scale), header_font, SOFT_WHITE)
    rule_y = y + _shortcuts_header_height(scale) - round(10 * scale)
    _paint(canvas, _rounded_mask(width - 2 * pad, edge, 0), x + pad, rule_y, WHITE, 0.16)

    for row in panel.rows:
        shortcut = row.shortcut
        _draw_keycap(canvas, shortcut.key, x + pad, row.y, key_font, scale)
        _draw_text(canvas, shortcut.title, panel.text_x, row.y, title_font, WHITE)
        if shortcut.status:
            _draw_status_chip(canvas, shortcut.status, x + width - pad, row.y, title_height, scale)
        for index, line in enumerate(row.lines):
            _draw_text(canvas, line, panel.text_x, row.y + title_height + index * description_height, description_font, SOFT_WHITE)


def _draw_keycap(canvas: np.ndarray, key: str, x: int, y: int, font: ImageFont.FreeTypeFont, scale: float) -> None:
    """A key drawn as a raised cap: the darker lip below reads as its side."""
    width, height = _keycap_width(key, font, scale), _keycap_height(scale)
    radius = round(5 * scale)
    lip = max(round(2 * scale), 1)
    _paint_card(canvas, x, y, width, height, WHITE, 0.14, radius)
    _paint_card(canvas, x, y, width, height - lip, WHITE, 0.2, radius)
    text_x = x + (width - round(font.getlength(key))) // 2
    _draw_text(canvas, key, text_x, y + (height - lip - sum(font.getmetrics())) // 2, font, WHITE)


def _draw_status_chip(canvas: np.ndarray, text: str, right: int, y: int, height: int, scale: float) -> None:
    font = _font(max(round(11 * scale), _MIN_FONT_SIZE), bold=True)
    width = round(font.getlength(text)) + 2 * round(8 * scale)
    _paint_card(canvas, right - width, y, width, height, AMBER, 1.0, height // 2)
    _draw_text(canvas, text, right - width + round(8 * scale), y + (height - sum(font.getmetrics())) // 2, font, INK)


def _shortcut_fonts(scale: float) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """Key, title and description fonts. The title's higher floor keeps the hierarchy
    visible on small frames, where both sizes would otherwise hit the minimum."""
    return _font(round(12 * scale), bold=True), _font(max(round(14 * scale), 9), bold=True), _font(round(12 * scale))


def _shortcuts_header_height(scale: float) -> int:
    return round(46 * scale)


def _keycap_height(scale: float) -> int:
    return round(24 * scale)


def _keycap_width(key: str, font: ImageFont.FreeTypeFont, scale: float) -> int:
    return max(round(font.getlength(key)) + 2 * round(7 * scale), _keycap_height(scale))


@lru_cache(maxsize=128)
def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> tuple[str, ...]:
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and font.getlength(candidate) <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return tuple(lines)


def _blur_behind(canvas: np.ndarray, x: int, y: int, width: int, height: int, radius: int, scale: float) -> None:
    region = canvas[y : y + height, x : x + width]
    alpha = _rounded_mask(width, height, radius)[..., None].astype(np.float32) / 255.0
    # Blurring a quarter-size copy looks the same and keeps this cheap at high resolutions.
    small = cv2.resize(region, (max(width // 4, 1), max(height // 4, 1)), interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3 * scale), (width, height), interpolation=cv2.INTER_LINEAR)
    region[:] = (region * (1.0 - alpha) + blurred * alpha).astype(np.uint8)


def _draw_footer(canvas: np.ndarray, state: HudState, scale: float) -> None:
    height, width = canvas.shape[:2]
    bar_height = round(_FOOTER_HEIGHT * scale)
    y = height - bar_height
    pad = round(14 * scale)
    _paint(canvas, _rounded_mask(width, bar_height, 0), 0, y, INK, 0.72)

    sound = "Sessiz" if state.muted else f"Açık: {state.sound_name.capitalize()}"
    status = f"FPS {state.fps:.0f}  |  Ses: {sound}"
    status_font = _font(round(14 * scale), bold=True)
    status_width = round(status_font.getlength(status))
    status_color = AMBER if state.muted else WHITE
    line_y = y + (bar_height - sum(status_font.getmetrics())) // 2
    _draw_text(canvas, status, width - pad - status_width, line_y, status_font, status_color)

    mute = "[M] Sesi Aç" if state.muted else "[M] Sessize Al"
    shortcuts = f"[C] Kalibre Et | {mute} | [T] Ses Testi | [H] Sade Görünüm | [Q] Çıkış"
    font = _fit_font(shortcuts, round(14 * scale), width - status_width - 3 * pad)
    _draw_text(canvas, shortcuts, pad, y + (bar_height - sum(font.getmetrics())) // 2, font, SOFT_WHITE)


def _draw_toast(canvas: np.ndarray, text: str, top: int, scale: float, left: int, right: int) -> None:
    pad = round(14 * scale)
    font = _fit_font(text, round(15 * scale), right - left - 2 * pad)
    width = round(font.getlength(text)) + 2 * pad
    toast_height = round(34 * scale)
    x, y = left + (right - left - width) // 2, top + round(10 * scale)
    _paint_card(canvas, x, y, width, toast_height, INK, 0.8, toast_height // 2)
    _draw_text(canvas, text, x + pad, y + (toast_height - sum(font.getmetrics())) // 2, font, WHITE)


def _draw_text_right(canvas: np.ndarray, text: str, right: int, y: int, size: int, scale: float) -> None:
    font = _font(round(size * scale))
    _draw_text(canvas, text, right - round(font.getlength(text)), y, font, SOFT_WHITE)


def _paint_card(
    canvas: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int], opacity: float, radius: int
) -> None:
    _paint(canvas, _rounded_mask(width, height, radius), x, y, color, opacity)


def _draw_text(canvas: np.ndarray, text: str, x: int, y: int, font: ImageFont.FreeTypeFont, color: tuple[int, int, int]) -> None:
    _paint(canvas, _text_mask(text, font), x, y, color)


def _paint(canvas: np.ndarray, mask: np.ndarray, x: int, y: int, color: tuple[int, int, int], opacity: float = 1.0) -> None:
    """Blend a flat color through an 8-bit mask, clipped to the canvas."""
    height, width = canvas.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + mask.shape[1], width), min(y + mask.shape[0], height)
    if x0 >= x1 or y0 >= y1:
        return
    alpha = mask[y0 - y : y1 - y, x0 - x : x1 - x, None].astype(np.float32) / 255.0 * opacity
    region = canvas[y0:y1, x0:x1]
    region[:] = (region * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)


# Masks are cached because PIL rendering is what would otherwise cost frame rate:
# compositing just these small regions avoids converting the whole frame each time.
@lru_cache(maxsize=512)
def _rounded_mask(width: int, height: int, radius: int) -> np.ndarray:
    supersample = 4
    image = Image.new("L", (width * supersample, height * supersample), 0)
    ImageDraw.Draw(image).rounded_rectangle(
        (0, 0, width * supersample - 1, height * supersample - 1), radius * supersample, fill=255
    )
    return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))


@lru_cache(maxsize=512)
def _text_mask(text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    ascent, descent = font.getmetrics()
    # Sized from the font's line metrics, not the text's own bounds, so a line's
    # baseline stays put whichever characters it happens to contain.
    image = Image.new("L", (max(round(font.getlength(text)) + 2, 1), ascent + descent), 0)
    ImageDraw.Draw(image).text((0, 0), text, font=font, fill=255, anchor="la")
    return np.asarray(image)


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    index = _FONT_INDEX_BOLD if bold else _FONT_INDEX_REGULAR
    try:
        return ImageFont.truetype(str(_FONT_FILE), max(size, _MIN_FONT_SIZE), index=index)
    except OSError:
        return ImageFont.load_default(max(size, _MIN_FONT_SIZE))


def _fit_font(text: str, size: int, max_width: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Return the font at `size`, shrunk as far as needed for the text to fit."""
    while size > _MIN_FONT_SIZE and _font(size, bold).getlength(text) > max_width:
        size -= 1
    return _font(size, bold)


def _mix(start: tuple[int, int, int], end: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * amount) for a, b in zip(start, end))  # type: ignore[return-value]


def _tinted(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Darken a status color enough for white text to stay readable on it."""
    return _mix(INK, color, 0.42)
