"""
scripts/make_gallery.py
=======================
Contact-sheet gallery of an expanded dataset (data/ecoli_x8 by default):
every image as a thumbnail with its boxes drawn — green = human label,
orange = pseudo-label (from pseudo.json) — plus a filter bar.

    python scripts/make_gallery.py                      # -> runs/compare/gallery/index.html
    python -m http.server 8765 -d runs/compare/gallery  # then open http://127.0.0.1:8765
"""
import argparse
import html
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.dataset_view import IMG_EXTS, label_of, read_labels  # noqa: E402

HUMAN, PSEUDO = (60, 220, 60), (0, 140, 255)      # BGR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ecoli_x8")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "compare" / "gallery")
    ap.add_argument("--width", type=int, default=400, help="Thumbnail long side in px.")
    a = ap.parse_args()

    pseudo = json.loads((a.data / "pseudo.json").read_text(encoding="utf-8")) if (a.data / "pseudo.json").exists() else {"images": {}}
    (a.out / "thumbs").mkdir(parents=True, exist_ok=True)
    cards = []
    for split in ("train", "val", "test"):
        for img in sorted(p for p in (a.data / "images" / split).glob("*") if p.suffix.lower() in IMG_EXTS):
            frame = cv2.imread(str(img))
            h, w = frame.shape[:2]
            rows = read_labels(label_of(img))
            pseudo_rows = {b["row"] for b in pseudo["images"].get(f"{split}/{img.name}", [])}
            s = a.width / max(h, w)
            thumb = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            for i, (cls, cx, cy, bw, bh) in enumerate(rows):
                x1, y1 = int((cx - bw / 2) * w * s), int((cy - bh / 2) * h * s)
                x2, y2 = int((cx + bw / 2) * w * s), int((cy + bh / 2) * h * s)
                cv2.rectangle(thumb, (x1, y1), (x2, y2), PSEUDO if i in pseudo_rows else HUMAN, 1)
            name = f"{split}__{img.stem}.jpg"
            cv2.imwrite(str(a.out / "thumbs" / name), thumb, [cv2.IMWRITE_JPEG_QUALITY, 78])
            stem, _, k = img.stem.partition("__d4-")
            cards.append({"split": split, "file": img.name, "source": stem, "k": k or "0",
                          "thumb": f"thumbs/{name}", "human": len(rows) - len(pseudo_rows),
                          "pseudo": len(pseudo_rows), "w": w, "h": h})

    data_json = json.dumps(cards)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E. coli expanded dataset</title>
<style>
:root{{--bg:#111;--fg:#eee;--muted:#9a9a9a;--card:#1b1b1b;--line:#2c2c2c}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}}
header{{position:sticky;top:0;background:#000c;backdrop-filter:blur(6px);padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
header input,header select{{background:#222;color:var(--fg);border:1px solid #444;border-radius:6px;padding:6px 8px}}
.legend span{{display:inline-block;width:12px;height:12px;border:2px solid;margin:0 4px -2px 10px}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px;padding:12px 16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.card img{{width:100%;display:block;background:#000}}
.card .cap{{padding:6px 8px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:6px}}
.card .cap b{{color:var(--fg);font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.h{{color:#5ad75a}} .p{{color:#ff9a3c}}
#count{{color:var(--muted)}}
</style></head><body>
<header>
 <strong>E. coli expanded dataset</strong>
 <input id="q" placeholder="filter by name…" size="24">
 <select id="split"><option value="">all splits</option><option>train</option><option>val</option><option>test</option></select>
 <select id="k"><option value="">all orientations</option><option value="0">0 original</option><option value="1">1 rot 90</option><option value="2">2 rot 180</option><option value="3">3 rot 270</option><option value="4">4 mirror</option><option value="5">5 mirror+90</option><option value="6">6 mirror+180</option><option value="7">7 mirror+270</option></select>
 <label><input type="checkbox" id="onlyp"> only with pseudo boxes</label>
 <span class="legend"><span style="border-color:#5ad75a"></span>human <span style="border-color:#ff9a3c"></span>pseudo</span>
 <span id="count"></span>
</header>
<main id="grid"></main>
<script>
const cards={data_json};
const grid=document.getElementById('grid'),q=document.getElementById('q'),sp=document.getElementById('split'),kk=document.getElementById('k'),op=document.getElementById('onlyp');
function render(){{
  const t=q.value.toLowerCase();
  const rows=cards.filter(c=>(!sp.value||c.split===sp.value)&&(!kk.value||c.k===kk.value)&&(!op.checked||c.pseudo>0)&&(!t||c.file.toLowerCase().includes(t)));
  document.getElementById('count').textContent=rows.length+' / '+cards.length+' images · '+rows.reduce((s,c)=>s+c.human,0)+' human + '+rows.reduce((s,c)=>s+c.pseudo,0)+' pseudo boxes';
  const esc=s=>String(s).replace(/[&<>"']/g,ch=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[ch]);
  grid.innerHTML=rows.map(c=>`<div class="card"><img loading="lazy" src="${{esc(c.thumb)}}" alt="${{esc(c.file)}}" title="${{esc(c.file)}} (${{c.w}}×${{c.h}})"><div class="cap"><b>${{esc(c.split)}}/${{esc(c.file)}}</b><span><span class="h">${{c.human}}</span>+<span class="p">${{c.pseudo}}</span></span></div></div>`).join('');
}}
[q,sp,kk,op].forEach(e=>e.addEventListener('input',render));render();
</script></body></html>"""
    (a.out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(cards)} images -> {a.out / 'index.html'}  "
          f"(human {sum(c['human'] for c in cards)}, pseudo {sum(c['pseudo'] for c in cards)} boxes)")


if __name__ == "__main__":
    main()
