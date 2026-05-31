import { useCallback, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  predict,
  colorForClass,
  type Detection,
  type LabelBox,
  type PredictResponse,
} from './api'
import DetectionOverlay from './DetectionOverlay'


// Convert a model detection (pixel xyxy + class_id) into the YOLO-format
// LabelBox the labelling UI uses (normalized cx/cy/w/h). Detections that
// somehow have w or h ≤ 0 are dropped so they don't crash the canvas.
function detectionsToLabelBoxes(
  detections: Detection[],
  imageW: number,
  imageH: number,
): LabelBox[] {
  const boxes: LabelBox[] = []
  for (const d of detections) {
    const [x1, y1, x2, y2] = d.bbox
    const w = (x2 - x1) / imageW
    const h = (y2 - y1) / imageH
    if (w <= 0 || h <= 0) continue
    boxes.push({
      class_id: d.class_id,
      cx: (x1 + (x2 - x1) / 2) / imageW,
      cy: (y1 + (y2 - y1) / 2) / imageH,
      w,
      h,
    })
  }
  return boxes
}


// Shape of the data we hand off via React Router's navigate state when
// the user wants to edit predictions as training labels. LabelView
// consumes this on mount.
export type SeedFromPredictionState = {
  fromPrediction: true
  imageBlob: Blob
  imageName: string | null
  boxes: LabelBox[]
  sourceCount: number   // total detections before filtering, for UX
}

export default function UploadView() {
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confFilter, setConfFilter] = useState(0.0)

  const onPick = useCallback((f: File) => {
    setFile(f)
    setImageUrl((prev) => {
      // Revoke the previous object URL so we don't leak across uploads.
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(f)
    })
    setResult(null)
    setError(null)
  }, [])

  const onSubmit = useCallback(async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      setResult(await predict(file))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [file])

  const filtered = useMemo(
    () => (result ? result.detections.filter((d) => d.confidence >= confFilter) : []),
    [result, confFilter],
  )

  // Hand the current image + (filtered) predictions off to the Label
  // tab as a labelling seed. Only the filtered detections come over —
  // the user has just told us the rest are noise via the slider, so
  // there's no point dragging them into the manual review.
  const onSendToLabel = useCallback(() => {
    if (!file || !result) return
    const boxes = detectionsToLabelBoxes(
      filtered,
      result.image_size[0],
      result.image_size[1],
    )
    const seed: SeedFromPredictionState = {
      fromPrediction: true,
      imageBlob: file,
      imageName: file.name,
      boxes,
      sourceCount: result.detections.length,
    }
    navigate('/train', { state: seed })
  }, [file, result, filtered, navigate])

  return (
    <div className="upload-view">
      <div className="upload-controls">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && onPick(e.target.files[0])}
        />
        <button onClick={onSubmit} disabled={!file || loading}>
          {loading ? 'Running…' : 'Detect'}
        </button>
        {result && (
          <label className="conf-filter">
            <span>Min confidence: <strong>{confFilter.toFixed(2)}</strong></span>
            <input
              type="range" min={0} max={1} step={0.01}
              value={confFilter}
              onChange={(e) => setConfFilter(parseFloat(e.target.value))}
            />
          </label>
        )}
        {result && filtered.length > 0 && (
          <button
            onClick={onSendToLabel}
            style={{ background: 'var(--bg-elev-2)', color: 'var(--fg)' }}
            title="Open these detections in the Label tab so you can edit them and save as training data"
          >
            Edit &amp; save as training data ({filtered.length})
          </button>
        )}
        {error && <span className="error">{error}</span>}
      </div>

      <div className="result-pane">
        <div className="image-pane">
          {imageUrl ? (
            <DetectionOverlay imageUrl={imageUrl} detections={filtered} />
          ) : (
            <div className="placeholder">Choose a microscopy image to begin.</div>
          )}
        </div>
        <DetectionList result={result} filteredCount={filtered.length} confFilter={confFilter} />
      </div>
    </div>
  )
}

function DetectionList({
  result,
  filteredCount,
  confFilter,
}: {
  result: PredictResponse | null
  filteredCount: number
  confFilter: number
}) {
  if (!result) {
    return <aside className="det-list empty">No results yet — choose an image and click Detect.</aside>
  }
  const shown = result.detections.filter((d) => d.confidence >= confFilter)
  return (
    <aside className="det-list">
      <div className="det-summary">
        <div>
          <strong>{filteredCount}</strong>
          <span className="muted"> / {result.detections.length} detections</span>
        </div>
        <div className="muted">{result.inference_ms.toFixed(0)} ms inference</div>
        <div className="muted">{result.image_size[0]} × {result.image_size[1]} px</div>
      </div>
      {shown.length === 0 ? (
        <p className="muted small">Nothing above the current confidence threshold.</p>
      ) : (
        <ul>
          {shown.map((d, i) => (
            <li key={i}>
              <span className="cls-dot" style={{ background: colorForClass(d.class_id) }} />
              <span className="cls-name">{d.class_name}</span>
              <span className="conf">{(d.confidence * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ul>
      )}
    </aside>
  )
}
