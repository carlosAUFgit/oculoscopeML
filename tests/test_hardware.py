import threading

import numpy as np
import cv2
import pytest

import config
import hardware


@pytest.fixture
def images_dir(tmp_path):
    img = np.full((60, 80, 3), 128, np.uint8)
    cv2.imwrite(str(tmp_path / "sample.png"), img)
    return str(tmp_path)


def test_mock_camera_burst(images_dir):
    cam = hardware.MockCamera(images_dir)
    burst = cam.capture_burst(4)
    assert len(burst) == 4
    for f in burst:
        assert f.shape == (60, 80, 3) and f.dtype == np.uint8


def test_mock_camera_empty_dir_raises(tmp_path):
    cam = hardware.MockCamera(str(tmp_path))
    with pytest.raises(RuntimeError, match="no images"):
        cam.capture_burst(1)


def test_mock_button_http_trigger_unblocks():
    btn = hardware.MockButton(listen_stdin=False)
    released = threading.Event()

    def waiter():
        btn.wait_for_press()
        released.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    assert not released.wait(0.1)
    btn.simulate_press()
    assert released.wait(2.0)


def test_mock_model_returns_three_probs():
    m = hardware.MockModel(seed=42)
    out = m.run(np.zeros((160, 160, 3), np.uint8))
    assert len(out) == 3
    assert all(0.0 <= p <= 1.0 for p in out)


def test_get_backend_simulation(monkeypatch, images_dir):
    monkeypatch.setattr(config, "SIMULATION", True)
    monkeypatch.setattr(config, "TEST_IMAGES_DIR", images_dir)
    b = hardware.get_backend(listen_stdin=False)
    assert isinstance(b.camera, hardware.MockCamera)
    assert callable(b.load_model)
    assert b.simulate_press is not None
    model = b.load_model()
    assert len(model.run(np.zeros((160, 160, 3), np.uint8))) == 3
