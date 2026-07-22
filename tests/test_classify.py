import numpy as np
import pytest

import config
import classify
from capture import DiffuseCapture, EyePair


class ScriptedModel:

    def __init__(self, outputs):
        self._out = list(outputs)

    def run(self, tensor):
        assert tensor.shape[:2] == (config.NN_INPUT[1], config.NN_INPUT[0])
        return self._out.pop(0) if len(self._out) > 1 else self._out[0]


def _eye(v=100):
    return np.full((180, 180, 3), v, np.uint8)


def _diffuse():
    return DiffuseCapture(EyePair(_eye(90), _eye(110)), _eye(100))


def test_to_tensor_resizes():
    t = classify.to_tensor(_eye())
    assert t.shape == (config.NN_INPUT[1], config.NN_INPUT[0], 3)


def test_analyze_cataract_labels(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_RB", False)
    model = ScriptedModel([(0.9, 0.8, 0.0), (0.9, 0.1, 0.0)])
    r = classify.analyze(model, [], _diffuse())
    assert r["status"] == "OK"
    assert r["left"]["cataract"][0] == "Cataract"
    assert r["right"]["cataract"][0] == "Normal"
    assert "retinoblastoma" not in r["left"]


def test_analyze_ungradable_eye_and_overall(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_RB", False)
    model = ScriptedModel([(0.2, 0.8, 0.0), (0.9, 0.1, 0.0)])
    r = classify.analyze(model, [], _diffuse())
    assert r["left"] == {"status": "UNGRADABLE"}
    assert r["right"]["status"] == "OK"
    assert r["status"] == "OK"

    model = ScriptedModel([(0.2, 0.8, 0.0), (0.1, 0.1, 0.0)])
    r = classify.analyze(model, [], _diffuse())
    assert r["status"] == "UNGRADABLE"


def test_analyze_rb_max_over_gazes(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_RB", True)
    coax = [EyePair(_eye(), _eye()), EyePair(_eye(), _eye())]
    model = ScriptedModel([
        (0.9, 0.1, 0.0),
        (0.9, 0.0, 0.05),
        (0.9, 0.0, 0.30),
        (0.9, 0.1, 0.0),
        (0.4, 0.0, 0.99),
        (0.9, 0.0, 0.10),
    ])
    r = classify.analyze(model, coax, _diffuse())
    assert r["left"]["retinoblastoma"] == ("Refer", pytest.approx(0.30))
    assert r["right"]["retinoblastoma"] == ("Normal", pytest.approx(0.10))
