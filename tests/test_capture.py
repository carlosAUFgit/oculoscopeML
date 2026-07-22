import cv2
import numpy as np
import pytest

import config
import capture
import hardware
from tests.conftest import make_face, ROI_LEFT, ROI_RIGHT


class ListCamera:

    def __init__(self, frames):
        self._frames = frames

    def capture_burst(self, n):
        return list(self._frames[:n])


class FailingCamera:

    def capture_burst(self, n):
        raise RuntimeError("sensor timeout")


class RecordingLeds:
    def __init__(self):
        self.events = []

    def coaxial(self, on):
        self.events.append(("coaxial", on))

    def surround(self, on):
        self.events.append(("surround", on))

    def fixation(self, pin, on):
        self.events.append((f"fixation{pin}", on))

    def fixation_blink(self, pin):
        self.events.append((f"fixation{pin}", "blink"))


def _backend(frames):
    leds = RecordingLeds()
    return hardware.Backend(
        camera=ListCamera(frames), button=None, leds=leds,
        load_model=hardware.MockModel), leds


def test_sharpest_picks_unblurred():
    sharp = make_face()
    blurry = cv2.GaussianBlur(sharp, (21, 21), 8)
    assert capture.sharpest([blurry, sharp, blurry]) is sharp


def test_crop_eyes_shapes(small_frame_config):
    left, right = capture.crop_eyes(make_face())
    assert left.shape == (ROI_LEFT[3] - ROI_LEFT[1], ROI_LEFT[2] - ROI_LEFT[0], 3)
    assert right.shape == (ROI_RIGHT[3] - ROI_RIGHT[1], ROI_RIGHT[2] - ROI_RIGHT[0], 3)


def test_post_process_preserves_shape_and_dtype():
    img = make_face()
    out = capture.post_process(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_post_process_lifts_brightness():
    img = np.full((60, 80, 3), 90, np.uint8)
    out = capture.post_process(img)
    assert out.mean() > img.mean()


def test_post_process_counters_green_cast():
    img = np.full((60, 80, 3), (90, 110, 90), np.uint8)
    out = capture.post_process(img)
    b, g, r = out.astype(np.float64).mean(axis=(0, 1))
    bi, gi, ri = img.astype(np.float64).mean(axis=(0, 1))
    assert g / gi < b / bi and g / gi < r / ri


def test_capture_set_rb_disabled(small_frame_config, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_RB", False)
    backend, leds = _backend([make_face()] * config.BURST_N)
    coax, diffuse = capture.capture_set(backend)
    assert coax == []
    assert isinstance(diffuse, capture.DiffuseCapture)
    assert diffuse.full.shape == (480, 640, 3)
    blink = f"fixation{config.CAPTURE_FIXATION_PIN}"
    assert leds.events == [(blink, "blink"),
                           ("surround", True), ("surround", False),
                           (blink, False)]


def test_capture_set_rb_enabled(small_frame_config, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_RB", True)
    backend, leds = _backend([make_face()] * config.BURST_N)
    coax, diffuse = capture.capture_set(backend)
    assert len(coax) == len(config.FIXATION_LED_PINS)
    for pair in coax:
        assert isinstance(pair, capture.EyePair)
    blink = f"fixation{config.CAPTURE_FIXATION_PIN}"
    expected = [(blink, "blink")]
    for pin in config.FIXATION_LED_PINS:
        expected += [(f"fixation{pin}", True), ("coaxial", True),
                     ("coaxial", False), (f"fixation{pin}", False)]
    expected += [("surround", True), ("surround", False)]
    expected += [(blink, False)]
    assert leds.events == expected


def test_capture_set_stops_blink_when_capture_fails(small_frame_config, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_RB", False)
    leds = RecordingLeds()
    backend = hardware.Backend(camera=FailingCamera(), button=None, leds=leds,
                               load_model=hardware.MockModel)
    with pytest.raises(RuntimeError, match="sensor timeout"):
        capture.capture_set(backend)
    assert leds.events[-1] == (f"fixation{config.CAPTURE_FIXATION_PIN}", False)
