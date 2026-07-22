from __future__ import annotations

import cv2
import numpy as np

import config

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _eye_lines(eye: dict) -> list[tuple[str, tuple[int, int, int]]]:
    if eye["status"] == "UNGRADABLE":
        return [("UNGRADABLE - retake", YELLOW)]
    label, prob = eye["cataract"]
    lines = [(f"{label}, {prob * 100:.1f}%",
              RED if label == "Cataract" else GREEN)]
    if "retinoblastoma" in eye:
        rb_label, rb_prob = eye["retinoblastoma"]
        lines.append((f"RB: {rb_label}, {rb_prob * 100:.1f}%",
                      RED if rb_label == "Refer" else GREEN))
    return lines


def draw_results(display_img: np.ndarray, results: dict) -> np.ndarray:
    img = display_img.copy()
    scale = max(0.5, img.shape[1] / 1300)
    thickness = max(1, int(scale * 2))
    line_h = int(34 * scale)

    for key, roi in (("left", config.EYE_ROI_LEFT),
                     ("right", config.EYE_ROI_RIGHT)):
        x0, y0, x1, y1 = roi
        cv2.rectangle(img, (x0, y0), (x1, y1), WHITE, 1)
        lines = _eye_lines(results[key])
        block_h = len(lines) * line_h
        if y0 - 10 - block_h >= 0:
            first_y = y0 - 10 - block_h + line_h
        else:
            first_y = min(y1 + line_h, img.shape[0] - block_h + line_h)
        for i, (text, color) in enumerate(lines):
            cv2.putText(img, text, (x0 + 4, first_y + i * line_h),
                        _FONT, scale, color, thickness)
    return img
