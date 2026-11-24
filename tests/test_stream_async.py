"""The live stream must keep flowing while a slow detector runs (Pi-class
CPU): inference happens on a worker thread and the video never waits on it."""
import time
from threading import RLock
from types import SimpleNamespace

import numpy as np

import backend.app.main as m
from src.config import load_config
from src.inference import Detection


class SlowDet:
    algorithm = "slow"

    def __init__(self):
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        time.sleep(0.5)
        return [Detection((1, 1, 5, 5), 0.9, 0, "ecoli")]


class Cam:
    def read(self):
        time.sleep(0.01)                      # ~100 fps camera
        return True, np.zeros((72, 128, 3), np.uint8)

    def __exit__(self, *exc):
        pass


def test_stream_not_blocked_by_slow_detector():
    saved = m.service, m._open_source_with_retry
    m.service = SimpleNamespace(config=load_config("config.yaml"), lock=RLock())
    m._open_source_with_retry = lambda source: Cam()
    try:
        det = SlowDet()
        gen = m._mjpeg_generator(0, annotate=True, detector=det)
        t0 = time.perf_counter()
        frames = [next(gen) for _ in range(30)]
        dt = time.perf_counter() - t0
        gen.close()
    finally:
        m.service, m._open_source_with_retry = saved
    assert all(f.startswith(b"--frame") for f in frames)
    # 30 frames at the 15 fps cap ~ 2 s; waiting on the model would take 30 x 0.5 s.
    assert dt < 4.0, dt
    assert 1 <= det.calls <= 6, det.calls     # ran in the background, one at a time


if __name__ == "__main__":
    test_stream_not_blocked_by_slow_detector()
    print("test_stream_async ok")
