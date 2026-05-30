// Typed thin wrappers around the FastAPI backend on /api/*.
// The Vite dev server proxies /api → http://localhost:8000 (see vite.config.ts);
// in production the same FastAPI process serves the built bundle alongside the
// API, so the URLs are identical in both modes.

export type Detection = {
  bbox: [number, number, number, number]
  class_id: number
  class_name: string
  confidence: number
}

export type PredictResponse = {
  detections: Detection[]
  image_size: [number, number]
  inference_ms: number
}

export type HealthInfo = {
  status: string
  model_loaded: boolean
  device: string
  classes: string[]
}

export async function fetchHealth(): Promise<HealthInfo> {
  const r = await fetch('/api/health')
  if (!r.ok) throw new Error(`/api/health → ${r.status}`)
  return r.json()
}

export async function predict(file: File): Promise<PredictResponse> {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch('/api/predict', { method: 'POST', body: fd })
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`/api/predict → ${r.status}: ${text || r.statusText}`)
  }
  return r.json()
}

export function streamUrl(
  source: string,
  inferEvery: number,
  cacheBust: number,
): string {
  const params = new URLSearchParams({
    source,
    infer_every: String(inferEvery),
    _t: String(cacheBust),
  })
  return `/api/stream?${params.toString()}`
}

// BGR palette from the backend visualization, mirrored as web hex.
const PALETTE = [
  '#00ff00', '#0080ff', '#ff0040', '#ffff00',
  '#ff00ff', '#00ffff', '#ffffff', '#ff0080',
]
export function colorForClass(id: number): string {
  return PALETTE[id % PALETTE.length]
}

// ---- Labelling / dataset --------------------------------------------------

export type LabelBox = {
  class_id: number
  cx: number   // 0..1
  cy: number   // 0..1
  w: number    // 0..1
  h: number    // 0..1
}

export type DatasetEntry = {
  filename: string
  split: string
  image_url: string
  num_boxes: number
  created_at: number
}

export type DatasetSplitStats = { images: number; boxes: number }

export type DatasetStats = {
  classes: string[]
  splits: Record<string, DatasetSplitStats>
  totals: DatasetSplitStats
}

export type Split = 'train' | 'val' | 'test'
export const SPLITS: Split[] = ['train', 'val', 'test']

export async function fetchLabelClasses(): Promise<string[]> {
  const r = await fetch('/api/label-classes')
  if (!r.ok) throw new Error(`/api/label-classes → ${r.status}`)
  const data = (await r.json()) as { classes: string[] }
  return data.classes
}

export async function fetchDatasetStats(): Promise<DatasetStats> {
  const r = await fetch('/api/dataset/stats')
  if (!r.ok) throw new Error(`/api/dataset/stats → ${r.status}`)
  return r.json()
}

export async function fetchDatasetList(split?: Split): Promise<DatasetEntry[]> {
  const params = split ? `?split=${split}` : ''
  const r = await fetch(`/api/dataset/list${params}`)
  if (!r.ok) throw new Error(`/api/dataset/list → ${r.status}`)
  return r.json()
}

export async function saveDatasetEntry(
  image: Blob,
  boxes: LabelBox[],
  split: Split,
  filename?: string,
): Promise<DatasetEntry> {
  const fd = new FormData()
  // Pass a usable filename so the backend can derive an extension.
  fd.append('file', image, filename ?? 'snapshot.jpg')
  fd.append('boxes', JSON.stringify(boxes))
  fd.append('split', split)
  if (filename) fd.append('filename', filename)
  const r = await fetch('/api/dataset/save', { method: 'POST', body: fd })
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`/api/dataset/save → ${r.status}: ${text || r.statusText}`)
  }
  return r.json()
}

export type DatasetEntryWithBoxes = {
  entry: DatasetEntry
  boxes: LabelBox[]
}

export async function findDatasetEntry(
  filename: string,
): Promise<DatasetEntryWithBoxes | null> {
  const r = await fetch(
    `/api/dataset/find?filename=${encodeURIComponent(filename)}`,
  )
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`/api/dataset/find → ${r.status}`)
  return r.json()
}

export async function deleteDatasetEntry(split: Split, filename: string): Promise<void> {
  const r = await fetch(
    `/api/dataset/${split}/${encodeURIComponent(filename)}`,
    { method: 'DELETE' },
  )
  if (!r.ok) throw new Error(`delete → ${r.status}`)
}

export async function snapshotImage(source: string): Promise<Blob> {
  const r = await fetch(
    `/api/snapshot?source=${encodeURIComponent(source)}`,
    { method: 'POST' },
  )
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`/api/snapshot → ${r.status}: ${text || r.statusText}`)
  }
  return r.blob()
}
