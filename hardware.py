from __future__ import annotations

import glob
import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

import config

log = logging.getLogger(__name__)


class Camera(Protocol):
    def capture_burst(self, n: int) -> list[np.ndarray]:
        ...


class Button(Protocol):
    def wait_for_press(self) -> None:
        ...


class Leds(Protocol):
    def coaxial(self, on: bool) -> None: ...
    def surround(self, on: bool) -> None: ...
    def fixation(self, pin: int, on: bool) -> None: ...

    def fixation_blink(self, pin: int) -> None:
        ...


class Model(Protocol):
    def run(self, tensor: np.ndarray) -> tuple[float, float, float]:
        ...


@dataclass
class Backend:
    camera: Camera
    button: Button
    leds: Leds
    load_model: Callable[[], Model]
    simulate_press: Callable[[], None] | None = None


class MockCamera:

    def __init__(self, images_dir: str | None = None) -> None:
        self._dir = images_dir if images_dir is not None else config.TEST_IMAGES_DIR
        self._paths = sorted(
            p for p in glob.glob(os.path.join(self._dir, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        self._i = 0
        self._rng = np.random.default_rng()
        if not self._paths:
            log.warning("MockCamera: no images in %s - captures will fail "
                        "until both-eyes sample photos are added", self._dir)

    def capture_burst(self, n: int) -> list[np.ndarray]:
        import cv2
        if not self._paths:
            raise RuntimeError(
                f"MockCamera: no images in '{self._dir}' (config.TEST_IMAGES_DIR). "
                "Add both-eyes sample photos to run the simulation.")
        path = self._paths[self._i % len(self._paths)]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(
                f"MockCamera: could not read '{path}' - corrupt or unsupported image file.")
        self._i += 1
        noise = self._rng.integers(-3, 4, img.shape, dtype=np.int16)
        return [np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                for _ in range(n)]


class MockButton:

    def __init__(self, listen_stdin: bool = True) -> None:
        self._event = threading.Event()
        if listen_stdin:
            threading.Thread(target=self._stdin_loop, daemon=True,
                             name="mock-button-stdin").start()

    def _stdin_loop(self) -> None:
        try:
            while True:
                input()
                self._event.set()
        except EOFError:
            log.info("MockButton: stdin closed - Enter trigger disabled "
                     "(HTTP POST /trigger still works)")

    def simulate_press(self) -> None:
        self._event.set()

    def wait_for_press(self) -> None:
        self._event.wait()
        self._event.clear()


class MockLeds:

    def coaxial(self, on: bool) -> None:
        log.info("LED coaxial -> %s", "ON" if on else "OFF")

    def surround(self, on: bool) -> None:
        log.info("LED surround -> %s", "ON" if on else "OFF")

    def fixation(self, pin: int, on: bool) -> None:
        log.info("LED fixation[%d] -> %s", pin, "ON" if on else "OFF")

    def fixation_blink(self, pin: int) -> None:
        log.info("LED fixation[%d] -> BLINK", pin)


class MockModel:

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def run(self, tensor: np.ndarray) -> tuple[float, float, float]:
        quality = float(self._rng.uniform(0.7, 1.0))
        cataract = float(self._rng.uniform(0.0, 1.0))
        leukocoria = float(self._rng.uniform(0.0, 0.4))
        return (quality, cataract, leukocoria)


def _mock_backend(listen_stdin: bool = True) -> Backend:
    button = MockButton(listen_stdin=listen_stdin)
    return Backend(
        camera=MockCamera(),
        button=button,
        leds=MockLeds(),
        load_model=MockModel,
        simulate_press=button.simulate_press,
    )


def _load_tflite_model() -> Model:
    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit(
            f"FATAL: model file '{config.MODEL_PATH}' not found. Place the "
            "multi-head model.tflite in the working directory "
            "(see README 'Swapping the model').")
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=config.MODEL_PATH, num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    if len(outs) != 3:
        raise SystemExit(
            f"FATAL: model has {len(outs)} outputs, expected 3 "
            "(quality, cataract, leukocoria - in that order).")

    class TfliteModel:
        def run(self, tensor: np.ndarray) -> tuple[float, float, float]:
            data = tensor.astype(np.float32)
            scale, zero = inp["quantization"]
            if scale:
                data = np.rint(data / scale + zero).astype(inp["dtype"])
            interp.set_tensor(inp["index"], data[np.newaxis, ...])
            interp.invoke()
            vals = []
            for out in outs:
                raw = interp.get_tensor(out["index"]).squeeze()
                o_scale, o_zero = out["quantization"]
                vals.append(float(raw) * 1.0 if not o_scale
                            else (float(raw) - o_zero) * o_scale)
            return (vals[0], vals[1], vals[2])

    return TfliteModel()


def _real_backend() -> Backend:
    from picamera2 import Picamera2
    from gpiozero import Button as GpioButton, LED

    class RealCamera:
        def __init__(self) -> None:
            self._cam = Picamera2()
            cam_config = self._cam.create_still_configuration(
                main={"size": config.FRAME_SIZE, "format": "RGB888"},
                buffer_count=1)
            self._cam.configure(cam_config)
            self._cam.start()

        def capture_burst(self, n: int) -> list[np.ndarray]:
            return [self._cam.capture_array("main") for _ in range(n)]

    class RealButton:
        def __init__(self) -> None:
            self._btn = GpioButton(config.BUTTON_PIN, pull_up=True,
                                   bounce_time=0.05)

        def wait_for_press(self) -> None:
            self._btn.wait_for_press()

    class RealLeds:
        def __init__(self) -> None:
            self._coax = LED(config.COAXIAL_LED_PIN)
            self._surround = LED(config.SURROUND_LED_PIN)
            self._fixation = {p: LED(p) for p in config.FIXATION_LED_PINS}

        def coaxial(self, on: bool) -> None:
            (self._coax.on if on else self._coax.off)()

        def surround(self, on: bool) -> None:
            (self._surround.on if on else self._surround.off)()

        def fixation(self, pin: int, on: bool) -> None:
            led = self._fixation[pin]
            (led.on if on else led.off)()

        def fixation_blink(self, pin: int) -> None:
            self._fixation[pin].blink(
                on_time=config.FIXATION_BLINK_INTERVAL,
                off_time=config.FIXATION_BLINK_INTERVAL)

    return Backend(camera=RealCamera(), button=RealButton(), leds=RealLeds(),
                   load_model=_load_tflite_model, simulate_press=None)


def get_backend(listen_stdin: bool = True) -> Backend:
    if config.SIMULATION:
        log.info("hardware: SIMULATION mode - mock backend")
        return _mock_backend(listen_stdin=listen_stdin)
    return _real_backend()
