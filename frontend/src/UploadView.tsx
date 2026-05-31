import { useCallback, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  predict,
  colorForClass,
  type Detection,
  type LabelBox,
  type PredictResponse,
} from './api'
import { mergeClusters, type MergeParams } from './clusterMerge'
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

  // Cluster-merge controls — applied client-side over the raw detections
  // returned by the backend so the sliders give instant feedback without
  // re-running the model. Defaults mirror the backend config defaults.
  const [mergeEnabled, setMergeEnabled] = useState(true)
  const [mergeMargin, setMergeMargin] = useState(0.01)
  const [mergeMinSize, setMergeMinSize] = useState(3)

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
      // merge=false → backend returns raw boxes. We re-merge client-side
      // so the sliders below are instant.
      setResult(await predict(file, { merge: false }))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [file])

  // Pipeline: raw detections → cluster-merge → confidence filter.
  // (Merging first so a cluster's mean-confidence can pass the filter
  // even when individual members were below it.)
  const merged = useMemo(() => {
    if (!result) return [] as Detection[]
    const params: MergeParams = {
      enabled: mergeEnabled,
      marginFrac: mergeMargin,
      minSize: mergeMinSize,
      sourceClassName: 'ecoli',
      targetClassName: 'ecoli_cluster',
    }
    return mergeClusters(result.detections, result.image_size, params)
  }, [result, mergeEnabled, mergeMargin, mergeMinSize])

  const filtered = useMemo(
    () => merged.filter((d) => d.confidence >= confFilter),
    [merged, confFilter],
  )

  const clusterCount = useMemo(
    () => filtered.filter((d) => d.class_name === 'ecoli_cluster').length,
    [filtered],
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

        {/* Merge settings are persistent — adjustable before Detect,
            then continue to live-tune the result after Detect. */}
        <label className="conf-filter">
          <input
            type="checkbox"
            checked={mergeEnabled}
            onChange={(e) => setMergeEnabled(e.target.checked)}
            style={{ marginRight: 6 }}
          />
          <span>Cluster merging</span>
        </label>
        <label className="conf-filter" title="How close two boxes have to be to count as neighbours. Fraction of the longer image edge.">
          <span>Closeness: <strong>{(mergeMargin * 100).toFixed(1)}%</strong></span>
          <input
            type="range" min={0} max={0.05} step={0.001}
            value={mergeMargin}
            onChange={(e) => setMergeMargin(parseFloat(e.target.value))}
            disabled={!mergeEnabled}
          />
        </label>
        <label className="conf-filter" title="Smallest neighbour group that becomes a cluster. Higher = only large clumps merge; loose pairs stay individual.">
          <span>Min cluster size: <strong>{mergeMinSize}</strong></span>
          <input
            type="range" min={10} max={100} step={1}
            value={mergeMinSize}
            onChange={(e) => setMergeMinSize(parseInt(e.target.value, 10))}
            disabled={!mergeEnabled}
          />
        </label>
        {result && mergeEnabled && clusterCount > 0 && (
          <span className="muted small">
            → {clusterCount} cluster{clusterCount === 1 ? '' : 's'}
          </span>
        )}

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
