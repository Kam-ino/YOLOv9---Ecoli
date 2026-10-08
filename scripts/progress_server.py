"""
scripts/progress_server.py
==========================
Live training monitor: one page, refreshes every 15 s, reads the logs that
the detached drivers and the app's Train tab write.

    python scripts/progress_server.py            # http://127.0.0.1:8766/
    python scripts/progress_server.py --port 8770

Shows, per run: state, epoch / total, iteration speed, best validation
score so far, last log line, and a wall-clock ETA from the measured epoch
rate. Also lists the deployable weights in models/ with timestamps and the
DONE/FAILED markers under runs/compare/.
"""
import argparse
import html
import json
import re
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMP = ROOT / "runs" / "compare"

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
YOLO_EPOCH = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G\s+.*?(\d+)%")
YOLO_VAL = re.compile(r"^\s+all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
RF_EPOCH = re.compile(r"^Epoch (\d+):\s+(\d+)%.*?(\d+)/(\d+).*?([\d.]+)it/s")
RF_BEST = re.compile(r"val/ema_mAP_50_95=([\d.\-]+)")


def tail_text(path: Path, n_bytes: int = 400_000) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(max(0, path.stat().st_size - n_bytes))
            return ANSI.sub("", fh.read().decode("utf-8", "replace").replace("\r", "\n"))
    except OSError:
        return ""


def marker_lines(path: Path, needle: str) -> bool:
    """True if any line of the whole file contains ``needle`` (small scan)."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return any(needle in ln for ln in fh)
    except OSError:
        return False


def parse_yolo(log: Path, total_default=30) -> dict:
    t = tail_text(log)
    lines = [ln for ln in t.splitlines() if ln.strip()]
    ep = tot = pct = None
    for ln in reversed(lines):
        m = YOLO_EPOCH.search(ln)
        if m:
            ep, tot, pct = int(m.group(1)), int(m.group(2)), int(m.group(3))
            break
    best = None
    for ln in lines:
        m = YOLO_VAL.search(ln)
        if m:
            v = float(m.group(4))
            best = v if best is None else max(best, v)
    return {"epoch": ep, "total": tot or total_default, "pct": pct, "best": best,
            "last": lines[-1][:140] if lines else ""}


def parse_rf(log: Path, total_default=25) -> dict:
    t = tail_text(log)
    lines = [ln for ln in t.splitlines() if ln.strip()]
    ep = pct = speed = None
    for ln in reversed(lines):
        m = RF_EPOCH.search(ln)
        if m:
            ep, pct, speed = int(m.group(1)) + 1, int(m.group(2)), float(m.group(5))
            break
    vals = [float(x) for x in RF_BEST.findall(t) if float(x) >= 0]
    return {"epoch": ep, "total": total_default, "pct": pct, "speed": speed,
            "best": max(vals) if vals else None, "last": lines[-1][:140] if lines else ""}


def run_rows() -> list:
    rows = []
    # Deployable runs (the detached driver's single log) and their outputs.
    ylog = ROOT / "runs" / "train" / "deploy_yolov9"
    dlog = CMP / "deploy_all.log"
    if dlog.exists():
        yolo_done = marker_lines(dlog, "activated runs/train/deploy_yolov9")
        rf_done = marker_lines(dlog, "activated runs/train/deploy_rfdetr")
        rf_started = (ROOT / "runs" / "train" / "deploy_rfdetr").exists()
        py = parse_yolo(dlog)
        if yolo_done:
            py.update(epoch=30, pct=100)
        rows.append({"name": "deployable YOLOv9-c (all 552)", "state": "done" if yolo_done else
                     ("failed" if (CMP / "FAILED_deploy_yolov9").exists() else "running"),
                     **py, "mtime": dlog.stat().st_mtime})
        if rf_started or yolo_done:
            pr = parse_rf(dlog)
            rows.append({"name": "deployable RF-DETR (all 552)",
                         "state": "done" if rf_done else
                         ("failed" if (CMP / "FAILED_deploy_rfdetr").exists() else
                          ("running" if rf_started else "queued")),
                         **pr, "mtime": dlog.stat().st_mtime})
    # Fold runs.
    for algo, parser in (("yolov9", parse_yolo), ("rfdetr", parse_rf)):
        for k in range(5):
            log = CMP / algo / f"fold{k}_x8_full.log"
            if not log.exists():
                continue
            best_file = CMP / algo / f"fold{k}_x8_full" / ("weights/best.pt" if algo == "yolov9" else "checkpoint_best_ema.pth")
            p = parser(log)
            done = best_file.exists() and (time.time() - log.stat().st_mtime > 600 or (p["epoch"] or 0) >= p["total"])
            if done:
                p.update(epoch=p["total"], pct=100)
            rows.append({"name": f"{algo} fold {k}", "state": "done" if done else "running", **p,
                         "mtime": log.stat().st_mtime})
    # App Train tab (if a run is going).
    return rows


def eta(row: dict, rate_cache: dict) -> str:
    if row["state"] != "running" or not row.get("epoch"):
        return ""
    key = row["name"]
    now = time.time()
    prog = (row["epoch"] - 1) + (row.get("pct") or 0) / 100
    hist = rate_cache.setdefault(key, [])
    hist.append((now, prog))
    hist[:] = hist[-40:]
    if len(hist) < 2 or hist[-1][1] <= hist[0][1]:
        return "measuring…"
    rate = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])   # epochs / s
    left = (row["total"] - prog) / rate if rate > 0 else None
    if left is None:
        return "measuring…"
    return f"{left / 60:.0f} min → {datetime.fromtimestamp(now + left):%H:%M}"


RATE: dict = {}


def page() -> str:
    rows = run_rows()
    markers = sorted(p.name for p in CMP.glob("DONE_*")) + sorted(p.name for p in CMP.glob("FAILED_*"))
    weights = []
    for w in sorted((ROOT / "models").glob("best_*")):
        if ".bak-" in w.name:
            continue
        weights.append(f"{w.name} — {datetime.fromtimestamp(w.stat().st_mtime):%Y-%m-%d %H:%M} — {w.stat().st_size / 1e6:.0f} MB")
    results = ROOT / "experiments" / "rfdetr_vs_yolov9" / "RESULTS.md"
    trs = []
    for r in rows:
        stale = time.time() - r["mtime"] > 900 and r["state"] == "running"
        state = r["state"] + (" (no log output for 15 min — machine asleep?)" if stale else "")
        ep = f"{r['epoch']}/{r['total']}" + (f" ({r['pct']}%)" if r.get("pct") is not None and r["state"] == "running" else "") if r.get("epoch") else "–"
        best = "–" if r.get("best") is None else f"{r['best']:.3f}"
        speed = f"{r['speed']:.2f} it/s" if r.get("speed") else ""
        cls = {"done": "ok", "running": "run", "failed": "bad", "queued": "muted"}[r["state"]]
        trs.append(f"<tr class='{cls}'><td>{html.escape(r['name'])}</td><td>{html.escape(state)}</td><td>{ep}</td>"
                   f"<td>{best}</td><td>{speed}</td><td>{html.escape(eta(r, RATE))}</td>"
                   f"<td class='muted'>{html.escape(r['last'])}</td></tr>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Training monitor</title>
<style>
body{{margin:0;background:#111;color:#eee;font:14px/1.45 system-ui,sans-serif;padding:16px}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#999;margin-bottom:14px}}
table{{border-collapse:collapse;width:100%;margin-bottom:18px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #2a2a2a;vertical-align:top}}
th{{color:#9a9a9a;font-weight:500}} tr.ok td:nth-child(2){{color:#5ad75a}} tr.run td:nth-child(2){{color:#ffb347}} tr.bad td:nth-child(2){{color:#ff6b6b}} .muted{{color:#888;font-size:12px;font-family:ui-monospace,monospace}}
h2{{font-size:14px;color:#9a9a9a;margin:14px 0 6px;text-transform:uppercase;letter-spacing:.04em}} ul{{margin:0;padding-left:18px}}
</style></head><body>
<h1>E. coli training monitor</h1>
<div class="sub">refreshes every 15 s · {datetime.now():%Y-%m-%d %H:%M:%S} · best = inner-val mAP50-95 (YOLO) / EMA mAP50-95 (RF-DETR)</div>
<table><tr><th>run</th><th>state</th><th>epoch</th><th>best val</th><th>speed</th><th>ETA</th><th>last log line</th></tr>{''.join(trs)}</table>
<h2>Deployable weights (models/)</h2><ul>{''.join(f'<li>{html.escape(w)}</li>' for w in weights) or '<li>none yet</li>'}</ul>
<h2>Markers (runs/compare/)</h2><ul>{''.join(f'<li>{html.escape(m)}</li>' for m in markers) or '<li>none</li>'}</ul>
<h2>Comparison report</h2><ul><li>{'written ' + datetime.fromtimestamp(results.stat().st_mtime).strftime('%Y-%m-%d %H:%M') + ' — experiments/rfdetr_vs_yolov9/RESULTS.md' if results.exists() else 'not yet'}</li></ul>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    print(f"monitor at http://127.0.0.1:{a.port}/")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
