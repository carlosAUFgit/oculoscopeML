from __future__ import annotations

import logging
from typing import NamedTuple

import cv2
import numpy as np

import config
import hardware

log = logging.getLogger(__name__)


class EyePair(NamedTuple):
    left: np.ndarray
    right: np.ndarray


class DiffuseCapture(NamedTuple):
    eyes: EyePair
    full: np.ndarray


def sharpest(frames: list[np.ndarray]) -> np.ndarray:
    def score(f: np.ndarray) -> float:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    return max(frames, key=score)


def _white_balance(img: np.ndarray) -> np.ndarray:
    gains = np.array(config.WB_REFERENCE_GAINS, dtype=np.float32)
    return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def _brighten(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=config.BRIGHTNESS_GAIN, beta=0)


def post_process(img: np.ndarray) -> np.ndarray:
    return _brighten(_white_balance(img))


def crop_eyes(img: np.ndarray) -> EyePair:
    lx0, ly0, lx1, ly1 = config.EYE_ROI_LEFT
    rx0, ry0, rx1, ry1 = config.EYE_ROI_RIGHT
    return EyePair(img[ly0:ly1, lx0:lx1], img[ry0:ry1, rx0:rx1])


def capture_set(backend: hardware.Backend) -> tuple[list[EyePair], DiffuseCapture]:
    leds, cam = backend.leds, backend.camera
    coax: list[EyePair] = []

    leds.fixation_blink(config.CAPTURE_FIXATION_PIN)
    try:
        if config.ENABLE_RB:
            for pin in config.FIXATION_LED_PINS:
                leds.fixation(pin, True)
                leds.coaxial(True)
                burst = cam.capture_burst(config.BURST_N)
                leds.coaxial(False)
                leds.fixation(pin, False)
                frame = post_process(sharpest(burst))
                del burst
                coax.append(crop_eyes(frame))

        leds.surround(True)
        burst = cam.capture_burst(config.BURST_N)
        leds.surround(False)
        full = post_process(sharpest(burst))
        del burst
        return coax, DiffuseCapture(crop_eyes(full), full)
    finally:
        leds.fixation(config.CAPTURE_FIXATION_PIN, False)
