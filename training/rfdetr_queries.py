"""
training/rfdetr_queries.py
==========================
Grow an RF-DETR checkpoint's object-query count.

RF-DETR packs its learned query embeddings as ``nn.Embedding(num_queries *
group_detr, …)`` with group ``g`` in slots ``[g*nq, (g+1)*nq)``. The library
can *shrink* that on load (it slices per group) but not grow it, and the
published COCO checkpoints carry 300 queries — fewer than the cells on a
dense stained slide (1,043 here). This module tiles each group's learned
queries up to the requested count, adds a little noise to the copies so they
separate during fine-tuning, updates the stored training args, and writes a
derived checkpoint that loads with ``num_queries=N``.

    python -m training.rfdetr_queries --variant medium --num-queries 1200
    → models/rf-detr-medium-q1200.pth

``training/train_rfdetr.py`` calls :func:`expand_if_needed` automatically.
"""
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

QUERY_KEYS = ("refpoint_embed.weight", "query_feat.weight")
NOISE = 0.02          # relative to each tensor's std; copies only


def _args_dict(args):
    return vars(args) if hasattr(args, "__dict__") else dict(args)


def _load(path: Path) -> dict:
    """Load a checkpoint without unpickling arbitrary objects.

    rfdetr checkpoints hold tensors plus an ``argparse.Namespace`` of training
    args, so that one class is allow-listed and ``weights_only=True`` stays on.
    """
    import torch
    torch.serialization.add_safe_globals([argparse.Namespace])
    return torch.load(str(path), map_location="cpu", weights_only=True)


def expand_checkpoint(src: Path, dst: Path, num_queries: int, noise: float = NOISE, seed: int = 0) -> Path:
    """Write ``dst`` = ``src`` with query embeddings tiled to ``num_queries`` per group."""
    import torch

    ck = _load(src)
    a = _args_dict(ck["args"])
    nq, groups = int(a["num_queries"]), int(a.get("group_detr", 1))
    if num_queries < nq:
        raise ValueError(f"{src} already has {nq} queries; the library can shrink to {num_queries} itself.")
    gen = torch.Generator().manual_seed(seed)
    for key in QUERY_KEYS:
        w = ck["model"][key]
        assert w.shape[0] == nq * groups, (key, tuple(w.shape), nq, groups)
        per_group = w.reshape(groups, nq, -1)
        reps = math.ceil(num_queries / nq)
        tiled = per_group.repeat(1, reps, 1)[:, :num_queries]
        copies = torch.zeros_like(tiled, dtype=torch.bool)
        copies[:, nq:] = True
        jitter = torch.randn(tiled.shape, generator=gen) * (w.std() * noise)
        ck["model"][key] = torch.where(copies, tiled + jitter, tiled).reshape(groups * num_queries, -1).contiguous()
    if hasattr(ck["args"], "__dict__"):
        ck["args"].num_queries = num_queries
        if hasattr(ck["args"], "num_select"):
            ck["args"].num_select = max(int(a.get("num_select", nq)), num_queries)
    else:
        ck["args"]["num_queries"] = num_queries
        ck["args"]["num_select"] = max(int(a.get("num_select", nq)), num_queries)
    for heavy in ("optimizer", "lr_scheduler", "ema"):      # not needed to initialise a fine-tune
        ck.pop(heavy, None)
    ck["query_expansion"] = {"source": str(src), "from": nq, "to": num_queries, "groups": groups,
                             "noise": noise, "seed": seed}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(dst))
    return dst


def checkpoint_queries(path: Path) -> Optional[int]:
    """``num_queries`` recorded in a checkpoint, or None if it has no args."""
    a = _load(path).get("args")
    return int(_args_dict(a)["num_queries"]) if a is not None else None


def coco_checkpoint(variant: str) -> Path:
    """Path of the library's COCO checkpoint for ``variant`` (downloaded if missing)."""
    import rfdetr.config as cfg
    from rfdetr.assets.model_weights import download_pretrain_weights, get_model_cache_dir
    name = getattr(cfg, f"RFDETR{variant.capitalize()}Config").model_fields["pretrain_weights"].default
    path = Path(get_model_cache_dir()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    download_pretrain_weights(str(path))
    return path


def expand_if_needed(weights: Optional[str], variant: str, num_queries: int,
                     models_dir: Path = Path("models")) -> Optional[str]:
    """Return a checkpoint path carrying ``num_queries`` slots, or None for the library default.

    ``weights`` is None (COCO base of ``variant``) or a ``.pth`` path. Nothing is
    written when the checkpoint already has at least ``num_queries`` queries.
    Derived files are cached as ``<models_dir>/<stem>-q<N>.pth``.
    """
    src = Path(weights) if weights else None
    if src is None:
        default_nq = checkpoint_queries(coco_checkpoint(variant))
        if default_nq is None or num_queries <= default_nq:
            return None
        src = coco_checkpoint(variant)
        dst = models_dir / f"rf-detr-{variant}-q{num_queries}.pth"
    else:
        have = checkpoint_queries(src)
        if have is None or num_queries <= have:
            return str(src)
        dst = models_dir / f"{src.stem}-q{num_queries}.pth"
    if not dst.exists():
        expand_checkpoint(src, dst, num_queries)
    return str(dst)


def _selftest() -> None:
    import tempfile
    import torch
    groups, nq, d = 2, 3, 4
    base = torch.arange(groups * nq * d, dtype=torch.float32).reshape(groups * nq, d)
    ck = {"model": {"refpoint_embed.weight": base.clone(), "query_feat.weight": base.clone() * 10},
          "args": argparse.Namespace(num_queries=nq, group_detr=groups, num_select=nq), "optimizer": {"x": 1}}
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "a.pth", Path(td) / "b.pth"
        torch.save(ck, src)
        expand_checkpoint(src, dst, 7)
        out = _load(dst)
        w = out["model"]["refpoint_embed.weight"]
        assert w.shape == (groups * 7, d), w.shape
        per = w.reshape(groups, 7, d)
        for g in range(groups):
            assert torch.equal(per[g, :nq], base.reshape(groups, nq, d)[g]), "originals must be untouched"
            assert torch.allclose(per[g, nq:2 * nq], base.reshape(groups, nq, d)[g], atol=base.std() * NOISE * 6)
        assert out["args"].num_queries == 7 and out["args"].num_select == 7 and "optimizer" not in out
        assert checkpoint_queries(dst) == 7
        assert expand_if_needed(str(dst), "medium", 5) == str(dst)        # no shrink, no rewrite
    print("rfdetr_queries selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", default="medium")
    ap.add_argument("--weights", default=None, help="Checkpoint to expand (default: the variant's COCO base).")
    ap.add_argument("--num-queries", type=int, default=1200)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    out = expand_if_needed(a.weights, a.variant, a.num_queries, a.models_dir)
    print(out or f"library default already has >= {a.num_queries} queries")


if __name__ == "__main__":
    main()
