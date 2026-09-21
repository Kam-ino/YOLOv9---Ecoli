"""Tile layout and edge filtering for tiled inference.

Run from the repo root:  python -m tests.test_tiling
"""
from src.inference import keep_tile_box, tile_origins


def test_tile_origins() -> None:
    assert tile_origins(480, 640) == [0]                 # smaller than a tile
    assert tile_origins(640, 640) == [0]
    o = tile_origins(2000, 640)
    assert o[0] == 0 and o[-1] == 2000 - 640             # covers both borders
    # Neighbours overlap by at least the promised 20 % (128 px).
    assert all(b - a <= 640 - 128 for a, b in zip(o, o[1:])), o


def test_keep_tile_box() -> None:
    W = H = 2000
    inside = (100, 100, 140, 140)
    assert keep_tile_box(inside, 512, 512, 640, 640, W, H)
    # Cut by an interior edge: a neighbouring tile has it whole.
    assert not keep_tile_box((0, 100, 30, 140), 512, 512, 640, 640, W, H)
    assert not keep_tile_box((600, 100, 640, 140), 512, 512, 640, 640, W, H)
    # Same position, but that tile edge IS the frame border: keep it.
    assert keep_tile_box((0, 100, 30, 140), 0, 512, 640, 640, W, H)
    assert keep_tile_box((600, 100, 640, 140), W - 640, 512, 640, 640, W, H)


if __name__ == "__main__":
    test_tile_origins()
    test_keep_tile_box()
    print("ok")
