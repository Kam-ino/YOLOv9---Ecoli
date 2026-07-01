import { useCallback, useEffect, useState } from 'react'
import {
  streamUrl,
  snapshotImage,
  predict,
  saveDatasetEntry,
  fetchLabelClasses,
  detectionsToLabelBoxes,
  colorForClass,
  SPLITS,
  type LabelBox,
  type Split,
} from './api'
import LabelCanvas from './LabelCanvas'

// After Stop / hiding the stream <img>, the MJPEG generator needs a
// moment to release the camera before /api/snapshot can reopen the same
// device. Retry a few times to ride out that handoff rather than failing
// on a transient "device busy".
async function snapshotWithRetry(
  source: string, tries = 6, delayMs = 300,
): Promise<Blob> {
  let lastErr: unknown
  for (let i = 0; i < tries; i++) {
    try {
      // clahe=true → the captured frame is contrast-enhanced like the
      // live view (and the domain the model infers on), so it doesn't
      // look washed-out next to the stream.
      return await snapshotImage(source, { clahe: true })
    } catch (e) {
      lastErr = e
      await new Promise((r) => setTimeout(r, delayMs))
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr))
}

// Live MJPEG stream pulled from /api/stream. The browser handles
// multipart/x-mixed-replace in <img> natively. Cache-bust forces a
// fresh connection on Start so the browser doesn't reuse an idle one.
//
// inferEvery: run inference once every N frames; reuse last boxes in
// between. On CPU this is the single biggest live-FPS knob.
//
// Min confidence and cluster-merge settings can be changed WHILE
// running: the boxes are drawn server-side (baked into each JPEG), so
// the only way to re-apply them is to reconnect the stream with new
// query params. `applied` holds the values currently in the stream URL;
// the individual states are the live slider values. A debounce copies
// the live values into `applied` so dragging doesn't reopen the camera
// on every tick.
//
// "Capture & label": freeze a clean frame, pre-fill it with the model's
// detections, then edit (add/remove boxes) and save to the dataset —
// the same flow as the Upload tab, but sourced from the live camera.
export default function LiveView() {
  const [source, setSource] = useState('0')
  const [inferEvery, setInferEvery] = useState(3)
  const [minConf, setMinConf] = useState(0.25)
  // The confidence currently baked into the live stream connection. A
  // debounce copies `minConf` here so dragging doesn't reopen the camera
  // on every tick; changing it forces a stream reconnect.
  const [appliedConf, setAppliedConf] = useState(0.25)
  const [running, setRunning] = useState(false)
  const [bust, setBust] = useState(0)
  const [error, setError] = useState<string | null>(null)

  // ---- capture-to-label mode ----
  const [editing, setEditing] = useState(false)
  const [busy, setBusy] = useState(false)
  const [editBlob, setEditBlob] = useState<Blob | null>(null)
  const [editUrl, setEditUrl] = useState<string | null>(null)
  const [editBoxes, setEditBoxes] = useState<LabelBox[]>([])
  const [classes, setClasses] = useState<string[]>([])
  const [classId, setClassId] = useState(0)
  const [split, setSplit] = useState<Split>('train')
  const [saveMsg, setSaveMsg] = useState<string | null>(null)

  // Labelling vocabulary for the class selector + canvas labels.
  useEffect(() => {
    fetchLabelClasses().then(setClasses).catch(() => {})
  }, [])

  // Revoke the captured-frame object URL when it changes or on unmount.
  useEffect(() => {
    return () => { if (editUrl) URL.revokeObjectURL(editUrl) }
  }, [editUrl])

  const start = () => {
    setError(null)
    setAppliedConf(minConf)
    setBust((b) => b + 1)
    setRunning(true)
  }
  const stop = () => setRunning(false)

  // True while the confidence slider differs from what the stream is
  // currently using — i.e. a reconnect is pending.
  const settingsDirty = running && minConf !== appliedConf

  // While running, debounce slider changes into the stream URL: 350ms
  // after the last adjustment, reconnect with the new threshold. The
  // changed query param is what triggers the <img> to reconnect.
  useEffect(() => {
    if (!settingsDirty) return
    const id = setTimeout(() => setAppliedConf(minConf), 350)
    return () => clearTimeout(id)
  }, [settingsDirty, minConf])

  // Freeze the current frame for labelling. Setting busy unmounts the
  // stream <img> (see the render gate), which releases the camera so the
  // snapshot can grab a clean, un-annotated frame. We then run detection
  // on it to pre-fill boxes; detection is best-effort, so a failure
  // there still lets you label by hand.
  const captureForLabel = useCallback(async () => {
    setError(null); setSaveMsg(null); setBusy(true)
    try {
      const blob = await snapshotWithRetry(source)
      const file = new File([blob], 'frame.jpg', {
        type: blob.type || 'image/jpeg',
      })
      let boxes: LabelBox[] = []
      try {
        // preprocess=false → the snapshot is already CLAHE-enhanced, so
        // don't let predict apply CLAHE a second time.
        const res = await predict(file, { preprocess: false })
        // Only pre-fill detections that pass the Min confidence slider —
        // same threshold the live overlay uses. Without this the canvas
        // floods with every weak box down to the model's 0.01 floor.
        const kept = res.detections.filter((d) => d.confidence >= minConf)
        boxes = detectionsToLabelBoxes(
          kept, res.image_size[0], res.image_size[1],
        )
      } catch {
        // Detection failed — fall back to a blank canvas.
      }
      if (editUrl) URL.revokeObjectURL(editUrl)
      setEditBlob(blob)
      setEditUrl(URL.createObjectURL(blob))
      setEditBoxes(boxes)
      setEditing(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [source, editUrl, minConf])

  const deleteBox = useCallback((idx: number) => {
    setEditBoxes((prev) => prev.filter((_, i) => i !== idx))
  }, [])

  const saveEdit = useCallback(async () => {
    if (!editBlob) return
    setBusy(true); setError(null); setSaveMsg(null)
    try {
      // No filename → backend names it MicroscopeCapture(N).png.
      const entry = await saveDatasetEntry(editBlob, editBoxes, split)
      setSaveMsg(
        `Saved ${entry.num_boxes} box${entry.num_boxes === 1 ? '' : 'es'} ` +
        `→ ${entry.split}/${entry.filename}. Capture another or go back to live.`,
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [editBlob, editBoxes, split])

  const backToLive = useCallback(() => {
    if (editUrl) URL.revokeObjectURL(editUrl)
    setEditing(false)
    setEditBlob(null); setEditUrl(null); setEditBoxes([])
    setSaveMsg(null); setError(null)
    // Reconnect the stream (it was unmounted during the edit session).
    setAppliedConf(minConf)
    setBust((b) => b + 1)
    setRunning(true)
  }, [editUrl, minConf])

  // ---- Edit mode: clean frozen frame + editable boxes + save ----------
  if (editing) {
    return (
      <div className="live-view">
        <div className="live-controls">
          <label className="conf-filter">
            <span>Class</span>
            <select
              value={classId}
              onChange={(e) => setClassId(parseInt(e.target.value, 10))}
            >
              {classes.length === 0 && <option value={0}>(loading…)</option>}
              {classes.map((c, i) => (
                <option key={i} value={i}>{i}: {c}</option>
              ))}
            </select>
          </label>
          <label className="conf-filter">
            <span>Split</span>
            <select value={split} onChange={(e) => setSplit(e.target.value as Split)}>
              {SPLITS.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <button onClick={saveEdit} disabled={busy || !editBlob}>
            {busy ? 'Saving…' : `Save${editBoxes.length ? ` (${editBoxes.length} box${editBoxes.length === 1 ? '' : 'es'})` : ''}`}
          </button>
          <button
            onClick={backToLive}
            disabled={busy}
            style={{ background: 'var(--bg-elev-2)', color: 'var(--fg-muted)' }}
            title="Discard this frame and resume the live stream"
          >Back to live</button>
          {saveMsg && <span className="badge badge-ok">{saveMsg}</span>}
          {error && <span className="error">{error}</span>}
        </div>

        <div className="result-pane">
          <div className="image-pane">
            {editUrl && (
              <LabelCanvas
                imageUrl={editUrl}
                boxes={editBoxes}
                classId={classId}
                classes={classes}
                onChange={setEditBoxes}
                onDelete={deleteBox}
              />
            )}
          </div>
          <aside className="det-list">
            <div className="det-summary">
              <div>
                <strong>{editBoxes.length}</strong>
                <span className="muted"> box{editBoxes.length === 1 ? '' : 'es'} on this frame</span>
              </div>
              <div className="muted">Class: {classes[classId] ?? `cls_${classId}`}</div>
              <div className="muted">Split: {split}</div>
              <div className="muted small">
                Pre-filled with detections ≥ {minConf.toFixed(2)} confidence.
              </div>
              <div className="muted small">
                Drag to add a box (in the selected class); click a box to remove it.
              </div>
              {editBoxes.length > 0 && (
                <button
                  onClick={() => setEditBoxes([])}
                  style={{ marginTop: 8, padding: '4px 10px', fontSize: '0.8rem' }}
                >Clear all</button>
              )}
            </div>
            {editBoxes.length === 0 ? (
              <p className="muted small">No boxes. Drag on the image to add one.</p>
            ) : (
              <ul>
                {editBoxes.map((b, i) => (
                  <li key={i}>
                    <span className="cls-dot" style={{ background: colorForClass(b.class_id) }} />
                    <span className="cls-name">
                      #{i + 1} {classes[b.class_id] ?? `cls_${b.class_id}`}
                    </span>
                    <button
                      onClick={() => deleteBox(i)}
                      style={{ background: 'transparent', color: 'var(--fg-muted)', padding: '2px 8px', fontSize: '0.8rem' }}
                      title="Remove this box"
                    >×</button>
                  </li>
                ))}
              </ul>
            )}
          </aside>
        </div>
      </div>
    )
  }

  // ---- Live mode ------------------------------------------------------
  return (
    <div className="live-view">
      <div className="live-controls">
        <label>
          <span>Source</span>
          <input
            type="text"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="0, 1, or path/to/file.mp4"
            disabled={running}
          />
        </label>
        <label className="conf-filter">
          <span>Infer every <strong>{inferEvery}</strong></span>
          <input
            type="range" min={1} max={10} step={1}
            value={inferEvery}
            onChange={(e) => setInferEvery(parseInt(e.target.value, 10))}
            disabled={running}
          />
        </label>
        <label className="conf-filter">
          <span>
            Min confidence <strong>{minConf.toFixed(2)}</strong>
            {settingsDirty && (
              <span className="muted small"> · applying…</span>
            )}
          </span>
          <input
            type="range" min={0} max={1} step={0.05}
            value={minConf}
            onChange={(e) => setMinConf(parseFloat(e.target.value))}
          />
        </label>
        {running ? (
          <button onClick={stop}>Stop</button>
        ) : (
          <button onClick={start} disabled={!source.trim() || busy}>Start</button>
        )}
        {running && (
          <button
            onClick={captureForLabel}
            disabled={busy}
            style={{ background: 'var(--bg-elev-2)', color: 'var(--fg)' }}
            title="Freeze the current frame, pre-fill the model's detections, then edit and save as training data"
          >
            {busy ? 'Capturing…' : 'Capture & label'}
          </button>
        )}
        {error && <span className="error">{error}</span>}
      </div>

      <div className="image-pane">
        {busy ? (
          <div className="placeholder"><p>Capturing a clean frame…</p></div>
        ) : running ? (
          <img
            className="stream-img"
            src={streamUrl(source, inferEvery, bust, { minConf: appliedConf })}
            alt="live stream"
            onError={() => {
              setError('Stream ended or failed (camera busy / source invalid).')
              setRunning(false)
            }}
          />
        ) : (
          <div className="placeholder">
            <p>For a USB microscope, try <code>0</code> or <code>1</code>.</p>
            <p>For file playback, use a path, e.g. <code>data/synthetic_microscopy.mp4</code>.</p>
            <p className="muted small">
              Higher "Infer every N" → smoother but boxes lag slightly behind motion.
              "Min confidence" can be adjusted live while streaming. Use
              "Capture &amp; label" to save a frame with its detections as training data.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
