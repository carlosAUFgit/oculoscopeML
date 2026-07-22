import threading

import cv2
import numpy as np
import pytest

import hardware
import main
from tests.conftest import make_face


class OneShotButton:

    def __init__(self):
        self._pressed = threading.Event()
        self._pressed.set()
        self._done = threading.Event()

    def wait_for_press(self):
        if self._pressed.is_set():
            self._pressed.clear()
            return
        self._done.wait()


class FaceCamera:
    def capture_burst(self, n):
        return [make_face() for _ in range(n)]


class FailingCamera:
    def capture_burst(self, n):
        raise RuntimeError("injected bad frame")


def _backend(camera):
    return hardware.Backend(camera=camera, button=OneShotButton(),
                            leds=hardware.MockLeds(),
                            load_model=lambda: hardware.MockModel(seed=1))


@pytest.fixture(autouse=True)
def reset_latest():
    main._set_latest_for_tests(None)
    yield
    main._set_latest_for_tests(None)


def test_run_capture_once_produces_jpeg(small_frame_config):
    backend = _backend(FaceCamera())
    model = backend.load_model()
    main.run_capture_once(backend, model)
    data = main.get_latest()
    assert data is not None
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (480, 640, 3)


def test_capture_loop_survives_bad_frame(small_frame_config):
    backend = _backend(FailingCamera())
    model = backend.load_model()
    t = threading.Thread(target=main.capture_loop, args=(backend, model),
                         daemon=True)
    t.start()
    t.join(timeout=1.0)
    assert t.is_alive()
    assert main.get_latest() is None
