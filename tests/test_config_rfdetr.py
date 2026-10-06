"""config.yaml with and without the rfdetr block; validation of the new keys."""
import tempfile
from pathlib import Path

import yaml

from src.config import load_config

BASE = {
    "model": {"weights": "models/x.pt", "device": "cpu", "imgsz": 640,
              "conf_threshold": 0.25, "iou_threshold": 0.45},
    "capture": {"source": 0, "width": 640, "height": 480, "fps": 30},
    "preprocessing": {"apply_clahe": False, "clahe_clip_limit": 2.0, "clahe_tile_grid_size": 8},
    "classes": ["ecoli"],
    "logging": {"level": "INFO", "file": ""},
    "output": {"save_video": False, "output_dir": "outputs"},
}


def _load(raw):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_config(str(p))


def _raises(raw):
    try:
        _load(raw)
    except ValueError:
        return True
    return False


def test_old_yaml_still_loads_with_defaults():
    cfg = _load(BASE)
    assert cfg.model.algorithm == "yolov9"
    assert cfg.rfdetr.variant == "medium" and cfg.rfdetr.resolution == 640
    assert cfg.rfdetr.num_queries == 1200 and cfg.rfdetr.weights.endswith("best_rfdetr.pth")


def test_rfdetr_block_and_default_algorithm():
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    raw["model"]["algorithm"] = "rfdetr"
    raw["rfdetr"] = {"weights": "models/r.pth", "variant": "small", "resolution": 512, "num_queries": 300}
    cfg = _load(raw)
    assert cfg.model.algorithm == "rfdetr"
    assert (cfg.rfdetr.weights, cfg.rfdetr.variant, cfg.rfdetr.resolution, cfg.rfdetr.num_queries) \
        == ("models/r.pth", "small", 512, 300)


def test_validation():
    bad_algo = yaml.safe_load(yaml.safe_dump(BASE)); bad_algo["model"]["algorithm"] = "yolov8"
    bad_res = yaml.safe_load(yaml.safe_dump(BASE)); bad_res["rfdetr"] = {"resolution": 600}
    bad_var = yaml.safe_load(yaml.safe_dump(BASE)); bad_var["rfdetr"] = {"variant": "xl"}
    bad_key = yaml.safe_load(yaml.safe_dump(BASE)); bad_key["rfdetr"] = {"nope": 1}
    assert _raises(bad_algo) and _raises(bad_res) and _raises(bad_var) and _raises(bad_key)


if __name__ == "__main__":
    test_old_yaml_still_loads_with_defaults()
    test_rfdetr_block_and_default_algorithm()
    test_validation()
    print("test_config_rfdetr ok")
