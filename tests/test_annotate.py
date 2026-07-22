import numpy as np

import annotate
from tests.conftest import make_face


def _results(status="OK"):
    if status == "UNGRADABLE":
        eye = {"status": "UNGRADABLE"}
        return {"status": "UNGRADABLE", "left": eye, "right": dict(eye)}
    return {
        "status": "OK",
        "left": {"status": "OK", "cataract": ("Cataract", 0.83),
                 "retinoblastoma": ("Refer", 0.35)},
        "right": {"status": "OK", "cataract": ("Normal", 0.07)},
    }


def test_draw_returns_modified_copy(small_frame_config):
    img = make_face()
    before = img.copy()
    out = annotate.draw_results(img, _results())
    assert out.shape == img.shape
    assert np.array_equal(img, before)
    assert not np.array_equal(out, img)


def test_draw_ungradable(small_frame_config):
    out = annotate.draw_results(make_face(), _results("UNGRADABLE"))
    assert not np.array_equal(out, make_face())


def test_refer_draws_red_and_normal_draws_green(small_frame_config):
    out = annotate.draw_results(make_face(), _results())
    red_pixels = np.all(out == (0, 0, 255), axis=-1)
    green_pixels = np.all(out == (0, 255, 0), axis=-1)
    assert red_pixels.any()
    assert green_pixels.any()


def test_labels_avoid_eye_region_near_top_edge(small_frame_config, monkeypatch):
    import config
    monkeypatch.setattr(config, "EYE_ROI_LEFT", (100, 5, 280, 185))
    monkeypatch.setattr(config, "EYE_ROI_RIGHT", (360, 5, 540, 185))
    img = make_face()
    out = annotate.draw_results(img, _results())
    for x0, y0, x1, y1 in (config.EYE_ROI_LEFT, config.EYE_ROI_RIGHT):
        interior = (slice(y0 + 2, y1 - 2), slice(x0 + 2, x1 - 2))
        assert np.array_equal(out[interior], img[interior])
