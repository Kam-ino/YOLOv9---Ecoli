"""One-off: re-encode any dataset image that cv2.imread can't read.

Some uploaded PNG / WEBP variants (16-bit, palette, embedded ICC
profile, certain AI-upscaler outputs) decode fine in Pillow but return
None from cv2.imread. Ultralytics' trainer uses cv2.imread directly,
so those files raise ``FileNotFoundError: Image Not Found`` mid-epoch.

Run from the repo root:
    python scripts/repair_dataset_images.py
"""
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    fixed = ok_already = unfixable = 0
    root = Path("data/ecoli/images")

    for img_path in root.rglob("*"):
        if not img_path.is_file():
            continue
        raw = img_path.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            ok_already += 1
            continue

        # cv2 refused — try Pillow.
        try:
            with Image.open(io.BytesIO(raw)) as img:
                rgb = img.convert("RGB")
                arr_rgb = np.array(rgb)
            frame = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            print(f"FAIL decode {img_path}: {exc}")
            unfixable += 1
            continue

        ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            print(f"FAIL encode {img_path}")
            unfixable += 1
            continue

        img_path.write_bytes(buf.tobytes())
        print(f"FIXED {img_path}")
        fixed += 1

    print(f"--- summary: fixed={fixed} ok-already={ok_already} unfixable={unfixable}")

    # Drop Ultralytics' cache so it re-scans these on the next run.
    for c in Path("data/ecoli/labels").glob("*.cache"):
        c.unlink()
        print(f"deleted {c}")


if __name__ == "__main__":
    main()
