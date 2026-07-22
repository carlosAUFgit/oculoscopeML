import pytest

import config
import hardware


def test_real_backend_unavailable_off_pi(monkeypatch):
    monkeypatch.setattr(config, "SIMULATION", False)
    with pytest.raises((ImportError, RuntimeError)):
        hardware.get_backend()


def test_model_file_missing_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "nope.tflite"))
    with pytest.raises(SystemExit, match="model"):
        hardware._load_tflite_model()
