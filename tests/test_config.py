import importlib
import os


def _fresh_config(monkeypatch, sim_env: str | None):
    if sim_env is None:
        monkeypatch.delenv("SIMULATION", raising=False)
    else:
        monkeypatch.setenv("SIMULATION", sim_env)
    import config
    return importlib.reload(config)


def test_simulation_flag_from_env(monkeypatch):
    assert _fresh_config(monkeypatch, "1").SIMULATION is True
    assert _fresh_config(monkeypatch, "0").SIMULATION is False
    assert _fresh_config(monkeypatch, None).SIMULATION is False


def test_rois_are_valid_rects(monkeypatch):
    cfg = _fresh_config(monkeypatch, "1")
    for roi in (cfg.EYE_ROI_LEFT, cfg.EYE_ROI_RIGHT):
        x0, y0, x1, y1 = roi
        assert x0 < x1 and y0 < y1
        assert x1 <= cfg.FRAME_SIZE[0] and y1 <= cfg.FRAME_SIZE[1]


def test_rois_split_frame_at_midline(monkeypatch):
    cfg = _fresh_config(monkeypatch, "1")
    half = cfg.FRAME_SIZE[0] // 2
    assert cfg.EYE_ROI_LEFT[2] == half == cfg.EYE_ROI_RIGHT[0]
    assert cfg.EYE_ROI_LEFT[0] == cfg.FRAME_SIZE[0] - cfg.EYE_ROI_RIGHT[2]
    assert cfg.EYE_ROI_LEFT[0] > 0


def test_capture_fixation_blink_settings(monkeypatch):
    cfg = _fresh_config(monkeypatch, "1")
    assert cfg.CAPTURE_FIXATION_PIN in cfg.FIXATION_LED_PINS
    assert cfg.FIXATION_BLINK_INTERVAL > 0


def test_colour_and_exposure_calibration(monkeypatch):
    cfg = _fresh_config(monkeypatch, "1")
    b, g, r = cfg.WB_REFERENCE_GAINS
    assert g < b and g < r
    assert cfg.BRIGHTNESS_GAIN > 1.0


def test_thresholds_and_staging(monkeypatch):
    cfg = _fresh_config(monkeypatch, "1")
    for t in (cfg.CATARACT_THRESHOLD, cfg.RETINOBLASTOMA_THRESHOLD,
              cfg.QUALITY_THRESHOLD):
        assert 0.0 <= t <= 1.0
    assert cfg.RETINOBLASTOMA_THRESHOLD < cfg.CATARACT_THRESHOLD
    assert cfg.ENABLE_RB is False
    assert cfg.NN_INPUT == (160, 160)
    assert cfg.BURST_N == 5
    assert cfg.SERVER_PORT == 8000
