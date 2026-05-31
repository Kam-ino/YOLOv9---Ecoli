"""
backend/app/cluster_merge.py
=============================
Post-processing pass: collapse dense overlapping detections of one
source class into a single bounding box of a target class.

Use case: the model emits ~50 individual ``ecoli`` boxes packed into a
purple smear; the user wants to see that smear as one big
``ecoli_cluster`` box instead, while loose isolated bacteria stay as
individual ``ecoli`` detections.

Algorithm
---------
1. Pull out every detection whose ``class_id`` matches the source class
   (others are passed through untouched).
2. Inflate each source box by ``margin_frac × max(image_width, image_height)``
   pixels on all sides — a scale-invariant way to define "near".
3. Build connected components via union-find: A and B are connected if
   their inflated boxes intersect. Transitive closure handles A–B–C
   chains automatically.
4. For each component with at least ``min_size`` members, emit ONE box
   equal to the union of the original (non-inflated) corners, labelled
   with the target class. Confidence = mean of member confidences so
   the cluster's score reflects how confident the model was about its
   constituents.
5. Components smaller than ``min_size`` keep their individual
   source-class boxes.

Complexity is O(N²) on source-class boxes. Typical N is ≤ a few hundred
per frame, so this finishes in microseconds.
"""
from typing import List, Tuple

from src.inference import Detection


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _inflate(
    bbox: Tuple[float, float, float, float], margin: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def _intersects(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    """Axis-aligned rectangle overlap test (inclusive)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------

class _UnionFind:
    """Standard union-find with path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        # Iterative path compression — avoids recursion-depth issues on
        # pathological inputs.
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def merge_clusters(
    detections: List[Detection],
    image_size: Tuple[int, int],
    source_class_id: int,
    target_class_id: int,
    target_class_name: str,
    margin_frac: float = 0.01,
    min_size: int = 3,
) -> List[Detection]:
    """Collapse dense ``source_class_id`` detections into clusters.

    Args:
        detections:        raw detector output. Returned unmodified if
                           there's nothing to do.
        image_size:        (width, height) in pixels — needed to make
                           ``margin_frac`` scale-invariant.
        source_class_id:   only boxes of this class are candidates for
                           merging. Others pass through.
        target_class_id:   class_id stamped on emitted cluster boxes.
                           Used by the frontend palette for colour.
        target_class_name: class_name stamped on emitted cluster boxes.
                           Used by the frontend for the label text.
        margin_frac:       neighbour radius as a fraction of the longer
                           image edge. 0.01 = 1% (e.g. ~13 px on a
                           1280×720 frame).
        min_size:          smallest component that becomes a cluster.
                           Components below this stay as individual
                           source-class boxes.

    Returns:
        A new list of ``Detection`` objects, sorted by confidence
        descending so the sidebar reads naturally.
    """
    if not detections or min_size < 2:
        return list(detections)

    # Partition.
    src_dets = [d for d in detections if d.class_id == source_class_id]
    other_dets = [d for d in detections if d.class_id != source_class_id]

    if len(src_dets) < min_size:
        # Not enough source boxes for any cluster — short-circuit.
        return list(detections)

    w, h = image_size
    margin_px = margin_frac * max(w, h)

    inflated = [_inflate(d.bbox, margin_px) for d in src_dets]

    uf = _UnionFind(len(src_dets))
    for i in range(len(src_dets)):
        ai = inflated[i]
        for j in range(i + 1, len(src_dets)):
            if _intersects(ai, inflated[j]):
                uf.union(i, j)

    # Group members by component root.
    groups: dict = {}
    for i in range(len(src_dets)):
        groups.setdefault(uf.find(i), []).append(i)

    out: List[Detection] = []
    for members in groups.values():
        if len(members) >= min_size:
            xs1: List[float] = []
            ys1: List[float] = []
            xs2: List[float] = []
            ys2: List[float] = []
            confs: List[float] = []
            for i in members:
                x1, y1, x2, y2 = src_dets[i].bbox
                xs1.append(x1); ys1.append(y1)
                xs2.append(x2); ys2.append(y2)
                confs.append(src_dets[i].confidence)
            out.append(Detection(
                bbox=(min(xs1), min(ys1), max(xs2), max(ys2)),
                confidence=sum(confs) / len(confs),
                class_id=target_class_id,
                class_name=target_class_name,
            ))
        else:
            for i in members:
                out.append(src_dets[i])

    out.extend(other_dets)
    # Stable, intuitive ordering — highest confidence first.
    out.sort(key=lambda d: -d.confidence)
    return out
