from __future__ import annotations

import logging

import cv2
import numpy as np

import config
import hardware
from capture import DiffuseCapture, EyePair

log = logging.getLogger(__name__)


def load_model(backend: hardware.Backend) -> hardware.Model:
    return backend.load_model()


def to_tensor(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, config.NN_INPUT, interpolation=cv2.INTER_AREA)


def infer(model: hardware.Model, img: np.ndarray) -> tuple[float, float, float]:
    return model.run(to_tensor(img))


def _analyze_eye(model: hardware.Model, diff_eye: np.ndarray,
                 coax_eyes: list[np.ndarray]) -> dict:
    quality, p_cat, _ = infer(model, diff_eye)
    if quality < config.QUALITY_THRESHOLD:
        return {"status": "UNGRADABLE"}

    result: dict = {
        "status": "OK",
        "cataract": ("Cataract" if p_cat >= config.CATARACT_THRESHOLD
                     else "Normal", p_cat),
    }
    if config.ENABLE_RB:
        p_rb = 0.0
        for frame in coax_eyes:
            q_c, _, p_leuk = infer(model, frame)
            if q_c >= config.QUALITY_THRESHOLD:
                p_rb = max(p_rb, p_leuk)
        result["retinoblastoma"] = (
            "Refer" if p_rb >= config.RETINOBLASTOMA_THRESHOLD else "Normal",
            p_rb)
    return result


def analyze(model: hardware.Model, coax_frames: list[EyePair],
            diffuse: DiffuseCapture) -> dict:
    left = _analyze_eye(model, diffuse.eyes.left,
                        [p.left for p in coax_frames])
    right = _analyze_eye(model, diffuse.eyes.right,
                         [p.right for p in coax_frames])
    status = "OK" if "OK" in (left["status"], right["status"]) else "UNGRADABLE"
    return {"status": status, "left": left, "right": right}
