"""Render a caption-led HiggsVision explainer from the saved project results.

Run from the repository root with:
    python dashboard/video/render_explainer.py

Requires Pillow, NumPy, and FFmpeg. The output is a 1080p H.264 MP4 beside
this script. All displayed scores and SHAP values come from research.json.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "dashboard" / "data" / "research.json"
OUTPUT_PATH = Path(__file__).with_name("HiggsVision_Explainer.mp4")
WIDTH, HEIGHT, FPS = 1600, 900, 24
DURATION = 70

BG = (15, 19, 17)
SURFACE = (27, 34, 29)
BORDER = (49, 61, 52)
TEXT = (239, 243, 234)
MUTED = (170, 181, 166)
FAINT = (116, 130, 114)
GREEN = (199, 245, 154)
TEAL = (117, 188, 177)
PURPLE = (185, 163, 234)
GOLD = (222, 184, 119)

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]


def load_font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (["georgiab.ttf", "DejaVuSerif-Bold.ttf"] if bold else ["georgia.ttf", "DejaVuSerif.ttf"]) if serif else (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    )
    for folder in FONT_DIRS:
        for name in names:
            candidate = folder / name
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONTS = {
    "micro": load_font(15),
    "small": load_font(18),
    "body": load_font(24),
    "body_bold": load_font(24, bold=True),
    "title": load_font(61, bold=True),
    "hero": load_font(76, serif=True),
    "number": load_font(78, bold=True),
    "feature": load_font(20, bold=True),
}


def make_background() -> Image.Image:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    x = x / WIDTH
    y = y / HEIGHT
    glow = np.exp(-(((x - 0.76) / 0.38) ** 2 + ((y - 0.24) / 0.52) ** 2))
    side = np.exp(-(((x - 0.08) / 0.44) ** 2 + ((y - 0.78) / 0.60) ** 2))
    red = 13 + 8 * glow + 2 * side
    green = 17 + 17 * glow + 4 * side
    blue = 15 + 11 * glow + 5 * side
    pixels = np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    for gx in range(0, WIDTH, 100):
        draw.line((gx, 0, gx, HEIGHT), fill=(21, 27, 23), width=1)
    for gy in range(0, HEIGHT, 100):
        draw.line((0, gy, WIDTH, gy), fill=(21, 27, 23), width=1)
    return image


BACKGROUND = make_background()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease(value: float) -> float:
    value = clamp(value)
    return value * value * (3 - 2 * value)


def rgba(color: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    return color[0], color[1], color[2], alpha


def rounded(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], radius: int,
            fill: tuple[int, ...], outline: tuple[int, ...] | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(tuple(int(v) for v in box), radius=radius, fill=fill, outline=outline, width=width)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], label: str,
             font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    draw.text(xy, label, font=font, fill=fill, anchor="mm")


def wrap_words(draw: ImageDraw.ImageDraw, label: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in label.split():
        test = f"{current} {word}".strip()
        if current and draw.textlength(test, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def draw_caption(frame: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(frame)
    top, bottom = 757, 866
    rounded(draw, (70, top, WIDTH - 70, bottom), 18, (23, 30, 26), (57, 71, 60), 2)
    draw.rounded_rectangle((70, top, 78, bottom), radius=4, fill=GREEN)
    lines = wrap_words(draw, label, FONTS["body"], WIDTH - 190)[:2]
    line_height = 34
    first_y = top + (bottom - top - len(lines) * line_height) / 2 - 1
    for index, line in enumerate(lines):
        draw.text((108, first_y + index * line_height), line, font=FONTS["body"], fill=TEXT)


def draw_chrome(frame: Image.Image, second: float, chapter: str) -> None:
    draw = ImageDraw.Draw(frame)
    draw.text((70, 38), "HIGGS", font=FONTS["body_bold"], fill=TEXT)
    draw.text((138, 38), "VISION", font=FONTS["body"], fill=GREEN)
    draw.text((260, 43), "A COLLISION, EXPLAINED", font=FONTS["micro"], fill=FAINT)
    chapter_width = draw.textlength(chapter.upper(), font=FONTS["micro"])
    draw.text((WIDTH - 70 - chapter_width, 46), chapter.upper(), font=FONTS["micro"], fill=MUTED)
    bar_x0, bar_x1, bar_y = 70, WIDTH - 70, 887
    draw.rounded_rectangle((bar_x0, bar_y, bar_x1, bar_y + 3), radius=2, fill=(43, 52, 45))
    draw.rounded_rectangle((bar_x0, bar_y, bar_x0 + (bar_x1 - bar_x0) * clamp(second / DURATION), bar_y + 3),
                           radius=2, fill=GREEN)


def draw_pill(draw: ImageDraw.ImageDraw, x: float, y: float, label: str,
              color: tuple[int, int, int] = GREEN, font: ImageFont.ImageFont | None = None) -> None:
    font = font or FONTS["micro"]
    width = draw.textlength(label, font=font) + 34
    rounded(draw, (x, y, x + width, y + 36), 18, SURFACE, BORDER)
    draw.ellipse((x + 13, y + 14, x + 21, y + 22), fill=color)
    draw.text((x + 28, y + 7), label, font=font, fill=MUTED)


def bezier(points: tuple[tuple[float, float], ...], t: float) -> tuple[float, float]:
    u = 1 - t
    p0, p1, p2, p3 = points
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def draw_bezier(draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...], progress: float,
                color: tuple[int, int, int], width: int = 5) -> None:
    count = max(2, int(46 * clamp(progress)))
    coords = [bezier(points, i / 45) for i in range(count)]
    if len(coords) > 1:
        draw.line(coords, fill=color, width=width, joint="curve")
        x, y = coords[-1]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)


def draw_ring(draw: ImageDraw.ImageDraw, cx: float, cy: float, rx: float, ry: float,
              color: tuple[int, int, int], width: int = 2) -> None:
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=width)


def draw_intro(frame: Image.Image, p: float, data: dict) -> None:
    draw = ImageDraw.Draw(frame)
    cx, cy = 1110, 410
    for i, size in enumerate((340, 270, 205, 135)):
        draw_ring(draw, cx, cy, size, size * 0.53, (34 + i * 5, 65 + i * 7, 46 + i * 4), 2)
    for n in range(14):
        angle = n * math.tau / 14 + p * math.tau * 0.18
        x = cx + math.cos(angle) * (255 + 24 * math.sin(p * 5 + n))
        y = cy + math.sin(angle) * (135 + 18 * math.cos(p * 5 + n))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=TEAL if n % 3 == 0 else GREEN)
    draw.text((128, 242), "What can the", font=FONTS["title"], fill=TEXT)
    draw.text((128, 323), "collision debris", font=FONTS["hero"], fill=GREEN)
    draw.text((128, 410), "tell us?", font=FONTS["title"], fill=TEXT)
    draw.text((133, 514), "A visual journey from particle traces", font=FONTS["body"], fill=MUTED)
    draw.text((133, 550), "to a model’s SHAP explanation.", font=FONTS["body"], fill=MUTED)
    draw_pill(draw, 133, 617, "SIMULATED COLLISIONS", GREEN)
    draw_pill(draw, 390, 617, "MODEL EXPLANATION", TEAL)


def draw_collision(frame: Image.Image, p: float, data: dict) -> None:
    draw = ImageDraw.Draw(frame)
    cx, cy = 800, 418
    for i in range(5):
        draw_ring(draw, cx, cy, 145 + i * 79, 70 + i * 41, (26 + i * 3, 43 + i * 4, 34 + i * 2), 2)
    approach = ease(clamp(p / 0.55))
    lx, rx = 190 + 610 * approach, 1410 - 610 * approach
    for x, direction in ((lx + 30, 1), (rx - 30, -1)):
        draw.line((x - 140 * direction, cy, x + 20 * direction, cy), fill=(75, 100, 75), width=2)
    for x, label in ((lx, "PROTON"), (rx, "PROTON")):
        draw.ellipse((x - 29, cy - 29, x + 29, cy + 29), fill=(54, 74, 49), outline=GREEN, width=3)
        for dx, dy in ((-9, -7), (9, 0), (-8, 9)):
            draw.ellipse((x + dx - 4, cy + dy - 4, x + dx + 4, cy + dy + 4), fill=TEXT)
        centered(draw, (x, cy - 71), label, FONTS["micro"], MUTED)
    if p > 0.53:
        impact = ease((p - 0.53) / 0.20)
        radius = 40 + 135 * impact
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(67, 98, 54), width=3)
        draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=GREEN)
        track_progress = clamp((p - 0.56) / 0.40)
        tracks = [
            (((cx, cy), (cx + 90, cy - 40), (cx + 150, cy - 180), (cx + 320, cy - 225)), GREEN),
            (((cx, cy), (cx + 110, cy + 35), (cx + 185, cy + 190), (cx + 340, cy + 215)), TEAL),
            (((cx, cy), (cx - 95, cy - 40), (cx - 190, cy - 150), (cx - 350, cy - 210)), PURPLE),
            (((cx, cy), (cx - 100, cy + 35), (cx - 205, cy + 160), (cx - 355, cy + 208)), GOLD),
        ]
        for points, color in tracks:
            draw_bezier(draw, points, track_progress, color, 5)
        if p > 0.70:
            draw_pill(draw, 1110, 210, "HIGGS DECAYS INSTANTLY", GREEN)


def draw_detector(frame: Image.Image, p: float, data: dict) -> None:
    draw = ImageDraw.Draw(frame)
    cx, cy = 790, 410
    for i in range(7):
        draw_ring(draw, cx, cy, 95 + i * 72, 52 + i * 32, (24 + i * 3, 48 + i * 4, 41 + i * 3), 2)
    draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=GREEN)
    q = ease(p)
    tracks = [
        (((cx, cy), (cx + 90, cy - 45), (cx + 235, cy - 167), (cx + 485, cy - 195)), GREEN, "JET"),
        (((cx, cy), (cx + 80, cy + 50), (cx + 245, cy + 170), (cx + 500, cy + 192)), TEAL, "JET"),
        (((cx, cy), (cx - 100, cy - 52), (cx - 245, cy - 142), (cx - 490, cy - 200)), PURPLE, "LEPTON"),
        (((cx, cy), (cx - 90, cy + 46), (cx - 225, cy + 170), (cx - 425, cy + 214)), GOLD, ""),
    ]
    for points, color, _ in tracks:
        draw_bezier(draw, points, q, color, 5)
    if p > 0.5:
        draw_pill(draw, 1180, 192, "JET", GREEN)
        draw_pill(draw, 1100, 630, "JET", TEAL)
        draw_pill(draw, 185, 210, "LEPTON", PURPLE)
        # Dashed line indicates the momentum imbalance inferred from visible tracks.
        x0, y0, x1, y1 = cx, cy, 365, 630
        for step in np.linspace(0, 1, 15):
            if int(step * 30) % 2 == 0:
                x = x0 + (x1 - x0) * step
                y = y0 + (y1 - y0) * step
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=GOLD)
        draw_pill(draw, 230, 660, "MISSING MOMENTUM", GOLD)


def draw_model(frame: Image.Image, p: float, data: dict) -> None:
    draw = ImageDraw.Draw(frame)
    # Twenty-eight little inputs, shown as clues entering the network.
    rounded(draw, (95, 250, 370, 585), 18, (25, 32, 27), BORDER, 2)
    draw.text((128, 282), "28 CLUES", font=FONTS["body_bold"], fill=GREEN)
    draw.text((128, 320), "measured + reconstructed", font=FONTS["micro"], fill=MUTED)
    active = int(28 * ease(p))
    for i in range(28):
        row, col = divmod(i, 7)
        x, y = 135 + col * 31, 390 + row * 39
        colr = GREEN if i < active else (57, 70, 60)
        draw.rounded_rectangle((x, y, x + 20, y + 20), radius=5, fill=colr)
    # A compact, animated neural-network diagram.
    layers = [
        [(485, 330 + i * 48) for i in range(5)],
        [(650, 295 + i * 38) for i in range(7)],
        [(820, 320 + i * 42) for i in range(6)],
        [(970, 355 + i * 52) for i in range(4)],
    ]
    line_progress = ease(p)
    for left, right in zip(layers, layers[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                if line_progress > 0.12:
                    draw.line((x1 + 8, y1 + 8, x2 + 8, y2 + 8), fill=(48, 76, 59), width=2)
    for layer_index, layer in enumerate(layers):
        for x, y in layer:
            color = GREEN if line_progress > layer_index * 0.2 else (57, 72, 60)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    draw.text((523, 605), "NEURAL NETWORK", font=FONTS["micro"], fill=FAINT)
    # Score from one saved simulated example; percent is deliberately avoided.
    score = float(data["events"][0]["probability"])
    card = (1090, 264, 1505, 575)
    rounded(draw, card, 20, SURFACE, BORDER, 2)
    draw.text((1130, 299), "SIGNAL-LIKE SCORE", font=FONTS["micro"], fill=MUTED)
    cx, cy = 1297, 434
    draw.ellipse((cx - 91, cy - 91, cx + 91, cy + 91), outline=(54, 67, 57), width=13)
    draw.arc((cx - 91, cy - 91, cx + 91, cy + 91), -90, -90 + 360 * clamp(p) * score, fill=GREEN, width=13)
    centered(draw, (cx, cy - 4), f"{score * 100:.1f}", FONTS["number"], TEXT)
    centered(draw, (cx, cy + 54), "OUT OF 100", FONTS["micro"], MUTED)
    draw.text((1130, 542), "model score · not a discovery chance", font=FONTS["micro"], fill=FAINT)


def friendly_feature(name: str) -> str:
    labels = {
        "m_bb": "MASS · TWO BOTTOM JETS",
        "m_wbb": "MASS · W + TWO BOTTOM JETS",
        "lepton_pT": "LEPTON MOMENTUM",
        "m_wwbb": "MASS · TWO W + TWO BOTTOM JETS",
        "m_jlv": "MASS · JET + LEPTON + UNSEEN",
    }
    return labels.get(name, name.replace("_", " ").upper())


def draw_waterfall(frame: Image.Image, p: float, data: dict) -> None:
    draw = ImageDraw.Draw(frame)
    event = data["events"][0]
    base = float(data["baseValue"]) * 100
    pairs = [(feature["name"], float(value) * 100) for feature, value in zip(data["features"], event["shap"])]
    top = sorted(pairs, key=lambda item: abs(item[1]), reverse=True)[:5]
    other = sum(value for name, value in pairs if name not in {item[0] for item in top})
    rows = top + [("other", other)]
    score = base
    x_left, x_right = 490, 1460
    score_min, score_max = 48, 96

    def sx(value: float) -> float:
        return x_left + (value - score_min) / (score_max - score_min) * (x_right - x_left)

    draw.text((112, 206), f"STARTING SCORE  {base:.1f}", font=FONTS["small"], fill=MUTED)
    draw.line((sx(base), 255, sx(base), 661), fill=(84, 99, 81), width=2)
    draw.text((x_left, 222), "MODEL SCORE / 100", font=FONTS["micro"], fill=FAINT)
    completed = p * len(rows)
    for i, (name, value) in enumerate(rows):
        y = 294 + i * 61
        if completed <= i:
            continue
        reveal = ease(completed - i)
        before, after = score, score + value
        color = GREEN if value >= 0 else PURPLE
        draw.text((112, y + 1), friendly_feature(name) if name != "other" else "OTHER 23 CLUES",
                  font=FONTS["feature"], fill=TEXT)
        draw.text((112, y + 30), "score push", font=FONTS["micro"], fill=FAINT)
        left, right = sx(min(before, after)), sx(max(before, after))
        width = max(3, (right - left) * reveal)
        if value < 0:
            left = right - width
        rounded(draw, (left, y, left + width, y + 31), 7, color)
        draw.text((x_right + 4, y + 2), f"{value:+.1f}", font=FONTS["small"], fill=color)
        if reveal > 0.96:
            score = after
    target = float(event["probability"]) * 100
    if p > 0.91:
        rounded(draw, (1120, 678, 1472, 726), 24, (39, 51, 35), (81, 106, 64), 2)
        centered(draw, (1296, 702), f"FINAL SCORE  {target:.1f}", FONTS["small"], GREEN)
    draw.text((x_left, 684), "Each contribution nudges the same score.", font=FONTS["small"], fill=MUTED)


def draw_reliance(frame: Image.Image, p: float, data: dict) -> None:
    draw = ImageDraw.Draw(frame)
    rows = data["faithfulness"]
    five = next(row for row in rows if int(row["k"]) == 5)
    base = float(rows[0]["shap"])
    x0, x1 = 440, 1435
    y_top, y_bottom = 285, 640
    value_low, value_high = 0.5, 1.0

    def sy(value: float) -> float:
        return y_bottom - (value - value_low) / (value_high - value_low) * (y_bottom - y_top)

    def sx(value: float) -> float:
        return x0 + value / 28 * (x1 - x0)

    draw.text((125, 227), "HOW WELL THE NETWORK SORTS SIGNAL FROM BACKGROUND", font=FONTS["micro"], fill=FAINT)
    for tick, label in ((0.5, "0.5 · RANDOM"), (0.75, "0.75"), (1.0, "1.0 · PERFECT")):
        y = sy(tick)
        draw.line((x0, y, x1, y), fill=(47, 58, 50), width=2)
        draw.text((x0 - 180, y - 12), label, font=FONTS["micro"], fill=MUTED)
    progress = ease(p) * (len(rows) - 1)
    for key, color in (("shap", GREEN), ("random", TEAL)):
        for index in range(len(rows) - 1):
            if progress <= index:
                break
            partial = clamp(progress - index)
            first, second = rows[index], rows[index + 1]
            x_a, y_a = sx(float(first["k"])), sy(float(first[key]))
            x_b = x_a + (sx(float(second["k"])) - x_a) * partial
            value_b = float(first[key]) + (float(second[key]) - float(first[key])) * partial
            y_b = sy(value_b)
            draw.line((x_a, y_a, x_b, y_b), fill=color, width=5)
            if partial >= 1:
                draw.ellipse((x_b - 5, y_b - 5, x_b + 5, y_b + 5), fill=color)
    draw.line((x0, y_bottom, x1, y_bottom), fill=BORDER, width=2)
    draw.text((x0, 657), "0", font=FONTS["micro"], fill=FAINT)
    centered(draw, ((x0 + x1) / 2, 668), "NUMBER OF HIDDEN CLUES", FONTS["micro"], FAINT)
    draw.text((x1 - 26, 657), "28", font=FONTS["micro"], fill=FAINT)
    if p > 0.65:
        hx = sx(5)
        draw.ellipse((hx - 12, sy(float(five["shap"])) - 12, hx + 12, sy(float(five["shap"])) + 12), fill=GREEN)
        draw.ellipse((hx - 12, sy(float(five["random"])) - 12, hx + 12, sy(float(five["random"])) + 12), fill=TEAL)
        draw_pill(draw, 110, 688, f"HIDE TOP 5 SHAP CLUES  →  {five['shap']:.3f}", GREEN)
        draw_pill(draw, 600, 688, f"HIDE 5 RANDOM CLUES  →  {five['random']:.3f}", TEAL)
    else:
        draw_pill(draw, 110, 688, f"STARTING SORT SCORE  {base:.3f}", GREEN)


def draw_outro(frame: Image.Image, p: float, data: dict) -> None:
    draw = ImageDraw.Draw(frame)
    cx, cy = 1130, 405
    for i in range(6):
        draw_ring(draw, cx, cy, 95 + i * 70, 55 + i * 37, (24 + i * 3, 50 + i * 5, 37 + i * 3), 2)
    for index in range(9):
        angle = index * math.tau / 9 + p * 0.8
        x1 = cx + math.cos(angle) * 22
        y1 = cy + math.sin(angle) * 14
        x2 = cx + math.cos(angle) * 400
        y2 = cy + math.sin(angle) * 226
        draw.line((x1, y1, x2, y2), fill=GREEN if index % 3 else TEAL, width=3)
        draw.ellipse((x2 - 5, y2 - 5, x2 + 5, y2 + 5), fill=GREEN if index % 3 else TEAL)
    draw.text((135, 238), "SHAP makes", font=FONTS["title"], fill=TEXT)
    draw.text((135, 323), "the score", font=FONTS["hero"], fill=GREEN)
    draw.text((135, 416), "easier to understand.", font=FONTS["title"], fill=TEXT)
    rounded(draw, (140, 542, 785, 621), 16, (30, 36, 31), (62, 72, 62), 2)
    draw.text((169, 562), "USEFUL EXPLANATION", font=FONTS["body_bold"], fill=GREEN)
    draw.text((169, 593), "≠ Higgs discovery", font=FONTS["body"], fill=MUTED)
    draw.text((144, 663), "A model study on simulated collisions.", font=FONTS["body"], fill=MUTED)


SCENES = [
    (0, 6, "THE QUESTION", draw_intro,
     "A Higgs boson vanishes almost as soon as it appears. Can the debris explain what happened?"),
    (6, 17, "THE COLLISION", draw_collision,
     "Two protons collide. If a Higgs is produced, it decays almost immediately into other particles."),
    (17, 28, "THE DETECTOR", draw_detector,
     "The detector records the particles left behind—not the Higgs itself. Other processes can leave similar clues."),
    (28, 38, "THE MODEL", draw_model,
     "The network reads 28 clues. This simulated example scores 88.8 out of 100: signal-like to the model, not an 88.8% chance of discovery."),
    (38, 54, "THE SHAP EXPLANATION", draw_waterfall,
     "SHAP estimates how each clue nudged this score. Green raises it, purple lowers it; together the pushes add from 53.6 to 88.8."),
    (54, 63, "DO THE CLUES MATTER?", draw_reliance,
     "Hiding the five strongest SHAP clues hurts the model’s sorting more than hiding five random clues. That supports that these clues matter to the model."),
    (63, 70, "THE TAKEAWAY", draw_outro,
     "This explains a model trained on simulations. It does not analyze real LHC data or claim a new Higgs discovery."),
]


def render() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Project results were not found: {DATA_PATH}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to export the MP4. Install FFmpeg and try again.")
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not data.get("events") or len(data.get("features", [])) != 28:
        raise ValueError("The saved project results do not contain the expected event and 28 feature clues.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0",
        "-vf", "scale=1920:1080:flags=lanczos", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUTPUT_PATH),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    assert process.stderr is not None
    total_frames = DURATION * FPS
    try:
        for frame_number in range(total_frames):
            second = frame_number / FPS
            scene = next(scene for scene in SCENES if scene[0] <= second < scene[1])
            start, end, chapter, renderer, caption = scene
            local = clamp((second - start) / (end - start))
            frame = BACKGROUND.copy()
            renderer(frame, local, data)
            draw_chrome(frame, second, chapter)
            draw_caption(frame, caption)
            process.stdin.write(frame.tobytes())
            if frame_number % (FPS * 7) == 0:
                print(f"Rendering {int(second):02d}/{DURATION}s · {chapter}", flush=True)
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if code != 0:
        raise RuntimeError(f"FFmpeg could not finish the MP4 export:\n{stderr}")
    print(f"Video ready: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    try:
        render()
    except Exception as exc:
        print(f"Could not render the explainer: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
