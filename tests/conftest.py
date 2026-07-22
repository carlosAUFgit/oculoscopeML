import cv2
import numpy as np
import pytest

import config

FRAME_W, FRAME_H = 640, 480
ROI_LEFT = (100, 150, 280, 330)
ROI_RIGHT = (360, 150, 540, 330)


def make_face(leuko_right: bool = False) -> np.ndarray:
    img = np.full((FRAME_H, FRAME_W, 3), (180, 190, 210), np.uint8)
    for (x0, y0, x1, y1), leuko in ((ROI_LEFT, False), (ROI_RIGHT, leuko_right)):
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.circle(img, (cx, cy), 60, (140, 120, 90), -1)
        pupil = (240, 240, 240) if leuko else (20, 15, 10)
        cv2.circle(img, (cx, cy), 25, pupil, -1)
    return img


@pytest.fixture
def small_frame_config(monkeypatch):
    monkeypatch.setattr(config, "FRAME_SIZE", (FRAME_W, FRAME_H))
    monkeypatch.setattr(config, "EYE_ROI_LEFT", ROI_LEFT)
    monkeypatch.setattr(config, "EYE_ROI_RIGHT", ROI_RIGHT)
