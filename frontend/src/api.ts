// Typed thin wrappers around the FastAPI backend on /api/*.
//
// API_BASE resolves the API URL:
//   - empty (default): use relative `/api/*` — works for local dev (Vite
//     proxies to localhost:8003) and for local single-process serve
//     (uvicorn serves both UI and API on the same port).
//   - set via `VITE_API_BASE_URL` at build time: prepended to every
//     fetch. Required when the UI is hosted somewhere other than the
//     backend (e.g. UI on Vercel, backend reached through a Cloudflare
//     or ngrok tunnel).
//
// IMPORTANT: when the UI is HTTPS (Vercel) the API base MUST also be
// HTTPS, otherwise the browser blocks the request as mixed content.
const API_BASE: string = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

function api(path: string): string {
  return `${API_BASE}${path}`
}

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
  const r = await fetch(api('/api/health'))
  if (!r.ok) throw new Error(`/api/health → ${r.status}`)
  return r.json()
}

export async function predict(
  file: File,
  opts: { merge?: boolean } = {},
): Promise<PredictResponse> {
  const fd = new FormData()
  fd.append('file', file)
  // merge=false → backend returns raw detections so client sliders can
  // re-merge on the fly without re-running the model on each tick.
  const merge = opts.merge ?? true
  const r = await fetch(api(`/api/predict?merge=${merge}`), { method: 'POST', body: fd })
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
  return api(`/api/stream?${params.toString()}`)
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
  const r = await fetch(api('/api/label-classes'))
  if (!r.ok) throw new Error(`/api/label-classes → ${r.status}`)
  const data = (await r.json()) as { classes: string[] }
  return data.classes
}

export async function addLabelClass(name: string): Promise<string[]> {
  const r = await fetch(api('/api/label-classes'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`/api/label-classes → ${r.status}: ${text || r.statusText}`)
  }
  const data = (await r.json()) as { classes: string[] }
  return data.classes
}

export async function fetchDatasetStats(): Promise<DatasetStats> {
  const r = await fetch(api('/api/dataset/stats'))
  if (!r.ok) throw new Error(`/api/dataset/stats → ${r.status}`)
  return r.json()
}

export async function fetchDatasetList(split?: Split): Promise<DatasetEntry[]> {
  const params = split ? `?split=${split}` : ''
  const r = await fetch(api(`/api/dataset/list${params}`))
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
  const r = await fetch(api('/api/dataset/save'), { method: 'POST', body: fd })
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
    api(`/api/dataset/find?filename=${encodeURIComponent(filename)}`),
  )
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`/api/dataset/find → ${r.status}`)
  return r.json()
}

export async function deleteDatasetEntry(split: Split, filename: string): Promise<void> {
  const r = await fetch(
    api(`/api/dataset/${split}/${encodeURIComponent(filename)}`),
    { method: 'DELETE' },
  )
  if (!r.ok) throw new Error(`delete → ${r.status}`)
}

export async function snapshotImage(source: string): Promise<Blob> {
  const r = await fetch(
    api(`/api/snapshot?source=${encodeURIComponent(source)}`),
    { method: 'POST' },
  )
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`/api/snapshot → ${r.status}: ${text || r.statusText}`)
  }
  return r.blob()
}

// ---- Training -------------------------------------------------------------

export type TrainState = 'idle' | 'running' | 'completed' | 'failed' | 'killed'

export type TrainingStatus = {
  state: TrainState
  pid: number | null
  started_at: number | null
  finished_at: number | null
  return_code: number | null
  name: string | null
  command: string[] | null
  log_lines: string[]
}

export type TrainStartRequest = {
  weights?: string
  epochs?: number
  batch?: number
  imgsz?: number
  device?: string
  name?: string
}

export async function fetchTrainingStatus(): Promise<TrainingStatus> {
  const r = await fetch(api('/api/train/status'))
  if (!r.ok) throw new Error(`/api/train/status → ${r.status}`)
  return r.json()
}

export async function startTraining(req: TrainStartRequest): Promise<TrainingStatus> {
  const r = await fetch(api('/api/train'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`/api/train → ${r.status}: ${text || r.statusText}`)
  }
  return r.json()
}

export async function stopTraining(): Promise<TrainingStatus> {
  const r = await fetch(api('/api/train/stop'), { method: 'POST' })
  if (!r.ok) throw new Error(`/api/train/stop → ${r.status}`)
  return r.json()
}

// ---- Model reset ----------------------------------------------------------

export type ResetResult = {
  backup: string | null
  active_weights: string
  classes: string[]
  device: string
}

export async function resetModel(): Promise<ResetResult> {
  const r = await fetch(api('/api/model/reset'), { method: 'POST' })
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`/api/model/reset → ${r.status}: ${text || r.statusText}`)
  }
  return r.json()
}
