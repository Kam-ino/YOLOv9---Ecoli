// Client-side port of backend/app/cluster_merge.py.
//
// Used by the Upload tab so the merge sliders give instant feedback
// instead of round-tripping to /api/predict (which re-runs the model
// each call, ~100 ms+). The backend exposes /api/predict?merge=false
// to skip its own pass; the result here is identical to what the
// backend would have produced for the same parameters.
//
// Keep this in sync with the Python implementation. Behavior tests
// live next to that file.

import type { Detection } from './api'

export type MergeParams = {
  enabled: boolean
  marginFrac: number    // 0..~0.05, fraction of max(imageW, imageH)
  minSize: number       // smallest component that becomes a cluster (≥2)
  sourceClassName: string
  targetClassName: string
}

type BBox = [number, number, number, number]

function inflate([x1, y1, x2, y2]: BBox, m: number): BBox {
  return [x1 - m, y1 - m, x2 + m, y2 + m]
}

function intersects(a: BBox, b: BBox): boolean {
  return !(a[2] < b[0] || b[2] < a[0] || a[3] < b[1] || b[3] < a[1])
}

class UnionFind {
  private parent: number[]
  constructor(n: number) { this.parent = Array.from({ length: n }, (_, i) => i) }
  find(i: number): number {
    let r = i
    while (this.parent[r] !== r) r = this.parent[r]
    while (this.parent[i] !== r) { const next = this.parent[i]; this.parent[i] = r; i = next }
    return r
  }
  union(i: number, j: number): void {
    const ri = this.find(i), rj = this.find(j)
    if (ri !== rj) this.parent[ri] = rj
  }
}

export function mergeClusters(
  detections: Detection[],
  imageSize: [number, number],
  params: MergeParams,
): Detection[] {
  if (!params.enabled || params.minSize < 2 || detections.length === 0) {
    return detections.slice()
  }

  const src = detections.filter(d => d.class_name === params.sourceClassName)
  const other = detections.filter(d => d.class_name !== params.sourceClassName)
  if (src.length < params.minSize) return detections.slice()

  const [imgW, imgH] = imageSize
  const marginPx = params.marginFrac * Math.max(imgW, imgH)
  const inflated = src.map(d => inflate(d.bbox as BBox, marginPx))

  const uf = new UnionFind(src.length)
  for (let i = 0; i < src.length; i++) {
    const ai = inflated[i]
    for (let j = i + 1; j < src.length; j++) {
      if (intersects(ai, inflated[j])) uf.union(i, j)
    }
  }

  // Group source-class boxes by component root.
  const groups = new Map<number, number[]>()
  for (let i = 0; i < src.length; i++) {
    const r = uf.find(i)
    const arr = groups.get(r); if (arr) arr.push(i); else groups.set(r, [i])
  }

  // Use the first non-source class_id from the model's output we've
  // ever seen, if any, so the cluster colour stays distinct. Otherwise
  // synthesize one past the highest class_id in this response.
  let targetClassId = src[0].class_id + 1
  for (const d of detections) {
    if (d.class_id > targetClassId) targetClassId = d.class_id + 1
  }

  const out: Detection[] = []
  for (const members of groups.values()) {
    if (members.length >= params.minSize) {
      let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity, c = 0
      for (const i of members) {
        const [a, b, cR, dR] = src[i].bbox
        if (a < x1) x1 = a; if (b < y1) y1 = b
        if (cR > x2) x2 = cR; if (dR > y2) y2 = dR
        c += src[i].confidence
      }
      out.push({
        bbox: [x1, y1, x2, y2],
        confidence: c / members.length,
        class_id: targetClassId,
        class_name: params.targetClassName,
      })
    } else {
      for (const i of members) out.push(src[i])
    }
  }

  return out.concat(other).sort((a, b) => b.confidence - a.confidence)
}
