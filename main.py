from __future__ import annotations

import logging
import threading

import cv2

import annotate
import capture
import classify
import config
import hardware
import server

log = logging.getLogger(__name__)

_lock = threading.Lock()
_latest_image: bytes | None = None


def get_latest() -> bytes | None:
    with _lock:
        return _latest_image


def _set_latest(jpeg: bytes | None) -> None:
    global _latest_image
    with _lock:
        _latest_image = jpeg


def _set_latest_for_tests(value: bytes | None) -> None:
    _set_latest(value)


def run_capture_once(backend: hardware.Backend, model: hardware.Model) -> None:
    coax, diffuse = capture.capture_set(backend)
    results = classify.analyze(model, coax, diffuse)
    annotated = annotate.draw_results(diffuse.full, results)
    ok, buf = cv2.imencode(".jpg", annotated,
                           [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    _set_latest(buf.tobytes())
    log.info("capture complete: status=%s", results["status"])


def capture_loop(backend: hardware.Backend, model: hardware.Model) -> None:
    while True:
        backend.button.wait_for_press()
        try:
            run_capture_once(backend, model)
        except Exception:
            log.exception("capture failed - waiting for next press")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    backend = hardware.get_backend()
    model = classify.load_model(backend)
    threading.Thread(target=capture_loop, args=(backend, model),
                     daemon=True, name="capture-loop").start()
    log.info("oculoscope up (simulation=%s) - press the button",
             config.SIMULATION)
    server.start_server(get_latest, config.SERVER_PORT,
                        trigger_fn=backend.simulate_press)


if __name__ == "__main__":
    main()
