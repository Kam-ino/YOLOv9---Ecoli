import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  SPLITS,
  addLabelClass,
  colorForClass,
  deleteDatasetEntry,
  fetchDatasetList,
  fetchDatasetStats,
  fetchLabelClasses,
  saveDatasetEntry,
  streamUrl,
  type DatasetEntry,
  type DatasetStats,
  type LabelBox,
  type Split,
} from './api'
import type { SeedFromPredictionState } from './UploadView'
import LabelCanvas from './LabelCanvas'
import TrainPanel from './TrainPanel'

type SourceMode = 'upload' | 'snapshot'

export default function LabelView() {
  // ---- vocabulary / dataset summary state ---------------------------------
  const [classes, setClasses] = useState<string[]>([])
  const [classId, setClassId] = useState(0)
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [entries, setEntries] = useState<DatasetEntry[]>([])

  // ---- current image being labelled ---------------------------------------
  const [sourceMode, setSourceMode] = useState<SourceMode>('upload')
  const [snapshotSource, setSnapshotSource] = useState('0')
  // Live MJPEG preview shown in Snapshot mode before a frame is grabbed.
  // Bumping liveBust forces a fresh stream connection (on mode switch,
  // source change, or "New image"). liveReady gates the Capture button
  // until the <img> has decoded at least one frame.
  const liveImgRef = useRef<HTMLImageElement>(null)
  const [liveBust, setLiveBust] = useState(0)
  const [liveReady, setLiveReady] = useState(false)
  const [liveError, setLiveError] = useState<string | null>(null)
  const [imageBlob, setImageBlob] = useState<Blob | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageName, setImageName] = useState<string | null>(null)

  // ---- working set of boxes for the current image -------------------------
  const [boxes, setBoxes] = useState<LabelBox[]>([])
  const [split, setSplit] = useState<Split>('train')
  // Sidebar/canvas class filter: null = show all boxes, else isolate a
  // single class id.
  const [classFilter, setClassFilter] = useState<number | null>(null)

  // ---- IO state -----------------------------------------------------------
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  // Native <input type="file"> is uncontrolled — clearing component
  // state doesn't reset the "ecoli.png" label. Hold a ref so we can
  // also reset .value when the user clicks "New image" (and as a bonus,
  // this lets them re-pick the same file, which the input otherwise
  // ignores because onChange doesn't fire for the same selection).
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ---- bootstrap ----------------------------------------------------------
  useEffect(() => {
    fetchLabelClasses().then(setClasses).catch((e) => setError(String(e)))
    refreshDataset()
  }, [])

  // Revoke the object URL when we swap images or unmount.
  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl)
    }
  }, [imageUrl])

  // If the user clicked "Edit & save as training data" on the Upload
  // tab, navigate state arrives carrying the original blob + the
  // detection boxes already converted to YOLO format. Seed the canvas
  // with that, then clear the history state so a refresh of /train
  // doesn't re-seed surprise the user.
  const location = useLocation()
  const navigate = useNavigate()
  useEffect(() => {
    const seed = location.state as SeedFromPredictionState | null | undefined
    if (!seed || !seed.fromPrediction || !seed.imageBlob) return

    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageBlob(seed.imageBlob)
    setImageUrl(URL.createObjectURL(seed.imageBlob))
    setImageName(seed.imageName)
    setBoxes(seed.boxes)
    setError(null)
    setMessage(
      `Loaded ${seed.boxes.length} box${seed.boxes.length === 1 ? '' : 'es'} ` +
      `from your last prediction (of ${seed.sourceCount} total). ` +
      `Add boxes the model missed, delete wrong ones, then Save.`,
    )
    // Clear the navigate state so re-renders / refreshes don't reload.
    navigate(location.pathname, { replace: true, state: null })
    // We deliberately depend on location.key — only run when the user
    // arrives via a fresh navigate, not on every imageUrl change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.key])

  const refreshDataset = useCallback(() => {
    fetchDatasetStats().then(setStats).catch(() => setStats(null))
    fetchDatasetList().then(setEntries).catch(() => setEntries([]))
  }, [])

  const setSource = useCallback(
    (blob: Blob, name: string | null) => {
      if (imageUrl) URL.revokeObjectURL(imageUrl)
      setImageBlob(blob)
      setImageUrl(URL.createObjectURL(blob))
      setImageName(name)
      setBoxes([])
      setClassFilter(null)
      setMessage(null)
      setError(null)
    },
    [imageUrl],
  )

  const onPickFile = useCallback(
    (f: File) => setSource(f, f.name),
    [setSource],
  )

  // Grab the frame currently showing in the live preview <img>. We draw
  // it to a canvas client-side rather than calling /api/snapshot: the
  // preview already holds the camera open, so a server-side snapshot of
  // the same device would fail with "device busy". The preview runs with
  // annotate=false, so the captured frame is raw (no boxes/HUD baked in).
  const onCaptureFromLive = useCallback(() => {
    setError(null); setMessage(null)
    const img = liveImgRef.current
    if (!img || !img.naturalWidth || !img.naturalHeight) {
      setError('Live feed not ready yet — give it a moment to connect.')
      return
    }
    const canvas = document.createElement('canvas')
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) { setError('Could not get a canvas context.'); return }
    ctx.drawImage(img, 0, 0)
    canvas.toBlob((blob) => {
      if (blob) setSource(blob, null)
      else setError('Could not capture the current frame.')
    }, 'image/png')
  }, [setSource])

  // (Re)connect the live preview whenever we enter snapshot mode or the
  // source changes, resetting the per-connection ready / error flags.
  useEffect(() => {
    if (sourceMode !== 'snapshot') return
    setLiveReady(false)
    setLiveError(null)
    setLiveBust((b) => b + 1)
  }, [sourceMode, snapshotSource])

  // Press "P" to capture the current live frame. Only fires while the
  // preview is actually showing (snapshot mode, no frame grabbed yet,
  // stream ready) and never while typing in a field or using a modifier
  // chord (so Ctrl/Cmd+P → print still works).
  useEffect(() => {
    const canCapture =
      sourceMode === 'snapshot' && !imageUrl && liveReady
    if (!canCapture) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'p' && e.key !== 'P') return
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const el = e.target as HTMLElement | null
      const tag = el?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' ||
          el?.isContentEditable) return
      e.preventDefault()
      onCaptureFromLive()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [sourceMode, imageUrl, liveReady, onCaptureFromLive])

  const onDeleteBox = useCallback((idx: number) => {
    setBoxes((prev) => prev.filter((_, i) => i !== idx))
  }, [])

  // ---- class filter (isolate boxes by class) ------------------------------
  // Box count per class on the current image.
  const classCounts = useMemo(() => {
    const m = new Map<number, number>()
    for (const b of boxes) m.set(b.class_id, (m.get(b.class_id) ?? 0) + 1)
    return m
  }, [boxes])
  // Distinct class ids present, for the filter tabs.
  const presentClasses = useMemo(
    () => [...classCounts.keys()].sort((a, b) => a - b),
    [classCounts],
  )
  // Auto-fall back to "All" if the filtered class no longer has any boxes
  // (e.g. you deleted the last one) — avoids a dead, empty view.
  const activeFilter =
    classFilter !== null && classCounts.has(classFilter) ? classFilter : null
  // Boxes shown in the list/canvas, carrying their original index so
  // deletes hit the right entry in the full array.
  const visible = useMemo(
    () => boxes
      .map((b, idx) => ({ b, idx }))
      .filter(({ b }) => activeFilter === null || b.class_id === activeFilter),
    [boxes, activeFilter],
  )
  const visibleBoxes = useMemo(() => visible.map((v) => v.b), [visible])

  // LabelCanvas only ever appends (it draws new boxes; deletes go through
  // the list). So any boxes beyond the filtered set it was given are new
  // — append them to the FULL list rather than replacing it, which keeps
  // the hidden (other-class) boxes intact while a filter is active.
  const onCanvasChange = useCallback((updated: LabelBox[]) => {
    const added = updated.slice(visibleBoxes.length)
    if (added.length) setBoxes((prev) => [...prev, ...added])
  }, [visibleBoxes.length])

  // "Clear all" clears everything; when a class is isolated it clears
  // just that class.
  const onClearBoxes = useCallback(() => {
    if (activeFilter === null) setBoxes([])
    else setBoxes((prev) => prev.filter((b) => b.class_id !== activeFilter))
  }, [activeFilter])

  const onSave = useCallback(async () => {
    if (!imageBlob) return
    setBusy(true); setError(null); setMessage(null)
    try {
      const entry = await saveDatasetEntry(imageBlob, boxes, split, imageName ?? undefined)
      // Keep the canvas loaded so the user can keep refining and re-save.
      // We deliberately do NOT adopt the backend-assigned filename: the
      // backend auto-renames on collision (foo.png → foo - 1.png → …),
      // so every Save click creates a new entry rather than overwriting.
      setMessage(
        `Saved ${entry.num_boxes} box${entry.num_boxes === 1 ? '' : 'es'} ` +
        `→ ${entry.split}/${entry.filename}. Each Save creates a new ` +
        `entry — click "New image" when you're done with this one.`,
      )
      refreshDataset()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [imageBlob, boxes, split, imageName, refreshDataset])

  const onClearImage = useCallback(() => {
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageBlob(null); setImageUrl(null); setImageName(null); setBoxes([])
    setMessage(null); setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
    // Force a fresh live-preview connection so we don't reuse a stale,
    // possibly-disconnected stream after labelling the captured frame.
    setLiveReady(false); setLiveError(null)
    setLiveBust((b) => b + 1)
  }, [imageUrl])

  const onDeleteEntry = useCallback(async (entry: DatasetEntry) => {
    if (!confirm(`Delete ${entry.split}/${entry.filename}?`)) return
    try {
      await deleteDatasetEntry(entry.split as Split, entry.filename)
      refreshDataset()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [refreshDataset])

  const onAddClass = useCallback(async () => {
    // window.prompt is plain but it's the right tool here — single
    // short input, no rich validation needed (backend enforces).
    const raw = window.prompt(
      "New class name?\n\n" +
      "Letters, digits, '.', '_' and '-' only. No spaces or slashes.\n" +
      "Example: ecoli_cluster",
    )
    if (raw === null) return
    const name = raw.trim()
    if (!name) return
    try {
      const updated = await addLabelClass(name)
      setClasses(updated)
      // Select the newly added class so the next box uses it.
      const newId = updated.indexOf(name)
      if (newId >= 0) setClassId(newId)
      setMessage(`Added class ${name} (id ${newId}).`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  return (
    <div className="label-view">
      <div className="label-controls">
        <div className="seg">
          <button
            className={sourceMode === 'upload' ? 'active' : ''}
            onClick={() => setSourceMode('upload')}
          >Upload</button>
          <button
            className={sourceMode === 'snapshot' ? 'active' : ''}
            onClick={() => setSourceMode('snapshot')}
          >Snapshot</button>
        </div>

        {sourceMode === 'upload' ? (
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && onPickFile(e.target.files[0])}
          />
        ) : (
          <>
            <input
              type="text"
              value={snapshotSource}
              onChange={(e) => setSnapshotSource(e.target.value)}
              placeholder="0 or path/to/file.mp4"
              style={{ width: 240 }}
            />
            <button
              onClick={onCaptureFromLive}
              disabled={busy || !snapshotSource.trim() || !liveReady}
            >
              Capture frame
            </button>
          </>
        )}

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
          <button
            type="button"
            onClick={onAddClass}
            title="Add a new class to the labelling vocabulary"
            style={{
              background: 'var(--bg-elev-2)',
              color: 'var(--fg)',
              padding: '4px 10px',
              fontSize: '1rem',
              lineHeight: 1,
              border: '1px solid var(--border)',
            }}
          >+</button>
        </label>

        <label className="conf-filter">
          <span>Split</span>
          <select value={split} onChange={(e) => setSplit(e.target.value as Split)}>
            {SPLITS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>

        <button onClick={onSave} disabled={!imageBlob || busy}>
          {busy ? 'Saving…' : `Save${boxes.length ? ` (${boxes.length} box${boxes.length === 1 ? '' : 'es'})` : ''}`}
        </button>

        {imageBlob && (
          <button
            onClick={onClearImage}
            disabled={busy}
            style={{ background: 'var(--bg-elev-2)', color: 'var(--fg-muted)' }}
            title="Discard the current image and start over"
          >New image</button>
        )}

        {message && <span className="badge badge-ok">{message}</span>}
        {error && <span className="error">{error}</span>}
      </div>

      <div className="result-pane">
        <div className="image-pane">
          {imageUrl ? (
            <LabelCanvas
              imageUrl={imageUrl}
              boxes={visibleBoxes}
              classId={classId}
              classes={classes}
              onChange={onCanvasChange}
              onDelete={(visIdx) => {
                const orig = visible[visIdx]
                if (orig) onDeleteBox(orig.idx)
              }}
            />
          ) : sourceMode === 'snapshot' && snapshotSource.trim() ? (
            <div className="live-preview">
              {liveError ? (
                <div className="placeholder">
                  <p className="error">{liveError}</p>
                  <p className="muted small">
                    Check the source (try <code>0</code> or <code>1</code>), make
                    sure no other tab is holding the camera, then adjust the
                    source field to reconnect.
                  </p>
                </div>
              ) : (
                <>
                  <img
                    ref={liveImgRef}
                    className="stream-img"
                    src={streamUrl(snapshotSource, 1, liveBust, { minConf: 0, annotate: false })}
                    alt="live preview"
                    onLoad={() => setLiveReady(true)}
                    onError={() => {
                      setLiveReady(false)
                      setLiveError('Live feed failed (camera busy / source invalid).')
                    }}
                  />
                  <p className="muted small">
                    {liveReady
                      ? 'Live feed — click "Capture frame" or press P to grab the current frame for labelling.'
                      : 'Connecting to the camera…'}
                  </p>
                </>
              )}
            </div>
          ) : (
            <div className="placeholder">
              <p>Pick an image to start labelling — upload one, or capture a frame from your microscope.</p>
              <p className="muted small">Click and drag on the image to draw a bounding box. Click an existing box to remove it (or use the × in the list on the right).</p>
            </div>
          )}
        </div>

        <aside className="det-list">
          {imageBlob ? (
            <>
              <div className="det-summary">
                <div>
                  <strong>{boxes.length}</strong>
                  <span className="muted"> box{boxes.length === 1 ? '' : 'es'} on this image</span>
                  {activeFilter !== null && (
                    <span className="muted"> · showing {visible.length} {classes[activeFilter] ?? `cls_${activeFilter}`}</span>
                  )}
                </div>
                <div className="muted">Class: {classes[classId] ?? `cls_${classId}`}</div>
                <div className="muted">Split: {split}</div>
                {boxes.length > 0 && (
                  <button
                    onClick={onClearBoxes}
                    style={{ marginTop: 8, padding: '4px 10px', fontSize: '0.8rem' }}
                  >
                    {activeFilter === null
                      ? 'Clear all'
                      : `Clear ${classes[activeFilter] ?? `cls_${activeFilter}`} (${classCounts.get(activeFilter) ?? 0})`}
                  </button>
                )}
              </div>
              {boxes.length > 0 && (
                <div className="seg" style={{ marginBottom: 8, flexWrap: 'wrap' }}>
                  <button
                    className={activeFilter === null ? 'active' : ''}
                    onClick={() => setClassFilter(null)}
                  >All ({boxes.length})</button>
                  {presentClasses.map((cid) => (
                    <button
                      key={cid}
                      className={activeFilter === cid ? 'active' : ''}
                      onClick={() => setClassFilter(cid)}
                    >
                      {classes[cid] ?? `cls_${cid}`} ({classCounts.get(cid)})
                    </button>
                  ))}
                </div>
              )}
              {visible.length === 0 ? (
                <p className="muted small">Drag to add a box; click a box to remove it.</p>
              ) : (
                <ul>
                  {visible.map(({ b, idx }) => (
                    <li key={idx}>
                      <span className="cls-dot" style={{ background: colorForClass(b.class_id) }} />
                      <span className="cls-name">
                        #{idx + 1} {classes[b.class_id] ?? `cls_${b.class_id}`}
                      </span>
                      <button
                        onClick={() => onDeleteBox(idx)}
                        style={{ background: 'transparent', color: 'var(--fg-muted)', padding: '2px 8px', fontSize: '0.8rem' }}
                        title="Remove this box"
                      >×</button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          ) : (
            <>
              <DatasetSummary stats={stats} entries={entries} onDelete={onDeleteEntry} />
              <TrainPanel />
            </>
          )}
        </aside>
      </div>
    </div>
  )
}

function DatasetSummary({
  stats, entries, onDelete,
}: {
  stats: DatasetStats | null
  entries: DatasetEntry[]
  onDelete: (e: DatasetEntry) => void
}) {
  if (!stats) return <p className="muted">Loading dataset…</p>
  return (
    <>
      <div className="det-summary">
        <div>
          <strong>{stats.totals.images}</strong>
          <span className="muted"> images · </span>
          <strong>{stats.totals.boxes}</strong>
          <span className="muted"> boxes</span>
        </div>
        {SPLITS.map((s) => (
          <div className="muted" key={s}>
            {s}: {stats.splits[s]?.images ?? 0} img / {stats.splits[s]?.boxes ?? 0} box
          </div>
        ))}
        <div className="muted" style={{ marginTop: 8 }}>Boxes by class</div>
        {stats.classes.map((c, i) => (
          <div className="muted" key={c}>
            <span className="cls-dot" style={{ background: colorForClass(i) }} />
            {c}: {stats.per_class[c] ?? 0}
          </div>
        ))}
        {Object.entries(stats.per_class)
          .filter(([name]) => !stats.classes.includes(name))
          .map(([name, count]) => (
            <div className="muted" key={name}>{name}: {count}</div>
          ))}
      </div>
      {entries.length === 0 ? (
        <p className="muted small">No saved entries yet.</p>
      ) : (
        <ul>
          {entries.slice(0, 15).map((e) => (
            <li key={`${e.split}/${e.filename}`}>
              <span className="cls-name" title={e.filename} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {e.split}/{e.filename}
              </span>
              <span className="conf">{e.num_boxes}</span>
              <button
                onClick={() => onDelete(e)}
                style={{ background: 'transparent', color: 'var(--fg-muted)', padding: '2px 8px', fontSize: '0.8rem' }}
                title="Delete entry"
              >×</button>
            </li>
          ))}
        </ul>
      )}
    </>
  )
}
