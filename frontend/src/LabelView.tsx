import { useCallback, useEffect, useState } from 'react'
import {
  SPLITS,
  colorForClass,
  deleteDatasetEntry,
  fetchDatasetList,
  fetchDatasetStats,
  fetchLabelClasses,
  findDatasetEntry,
  saveDatasetEntry,
  snapshotImage,
  type DatasetEntry,
  type DatasetStats,
  type LabelBox,
  type Split,
} from './api'
import LabelCanvas from './LabelCanvas'

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
  const [imageBlob, setImageBlob] = useState<Blob | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageName, setImageName] = useState<string | null>(null)

  // ---- working set of boxes for the current image -------------------------
  const [boxes, setBoxes] = useState<LabelBox[]>([])
  const [split, setSplit] = useState<Split>('train')

  // ---- IO state -----------------------------------------------------------
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

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
      setMessage(null)
      setError(null)
    },
    [imageUrl],
  )

  const onPickFile = useCallback(
    async (f: File) => {
      setSource(f, f.name)
      // Cross-session continuity: if we've already labelled an image
      // with this filename, pull back its boxes so the user can extend
      // them rather than start over.
      try {
        const existing = await findDatasetEntry(f.name)
        if (existing) {
          setBoxes(existing.boxes)
          setSplit(existing.entry.split as Split)
          setMessage(
            `Loaded ${existing.boxes.length} existing box${
              existing.boxes.length === 1 ? '' : 'es'
            } from ${existing.entry.split}/${existing.entry.filename}. ` +
            `Add more boxes and click Save again.`,
          )
        }
      } catch {
        // Lookup is best-effort — fall through to an empty canvas.
      }
    },
    [setSource],
  )

  const onSnapshot = useCallback(async () => {
    setBusy(true); setError(null); setMessage(null)
    try {
      const blob = await snapshotImage(snapshotSource)
      setSource(blob, null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [snapshotSource, setSource])

  const onClearBoxes = useCallback(() => setBoxes([]), [])

  const onDeleteBox = useCallback((idx: number) => {
    setBoxes((prev) => prev.filter((_, i) => i !== idx))
  }, [])

  const onSave = useCallback(async () => {
    if (!imageBlob) return
    setBusy(true); setError(null); setMessage(null)
    try {
      const entry = await saveDatasetEntry(imageBlob, boxes, split, imageName ?? undefined)
      // Keep the image + boxes loaded so the user can keep adding to the
      // same entry. We adopt the backend-assigned filename (matters for
      // snapshots, which arrive without one) so the next save overwrites
      // the same entry instead of creating a duplicate.
      setImageName(entry.filename)
      setMessage(
        `Saved ${entry.num_boxes} box${entry.num_boxes === 1 ? '' : 'es'} ` +
        `→ ${entry.split}/${entry.filename}. Keep labelling, or click ` +
        `"New image" to start fresh.`,
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
            <button onClick={onSnapshot} disabled={busy || !snapshotSource.trim()}>
              {busy ? 'Capturing…' : 'Capture frame'}
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
              boxes={boxes}
              classId={classId}
              classes={classes}
              onChange={setBoxes}
            />
          ) : (
            <div className="placeholder">
              <p>Pick an image to start labelling — upload one, or capture a frame from your microscope.</p>
              <p className="muted small">Click and drag on the image to draw a bounding box. Existing boxes are listed on the right and can be removed individually.</p>
            </div>
          )}
        </div>

        <aside className="det-list">
          {imageBlob ? (
            <>
              <div className="det-summary">
                <div><strong>{boxes.length}</strong><span className="muted"> box{boxes.length === 1 ? '' : 'es'} on this image</span></div>
                <div className="muted">Class: {classes[classId] ?? `cls_${classId}`}</div>
                <div className="muted">Split: {split}</div>
                {boxes.length > 0 && (
                  <button
                    onClick={onClearBoxes}
                    style={{ marginTop: 8, padding: '4px 10px', fontSize: '0.8rem' }}
                  >Clear all</button>
                )}
              </div>
              {boxes.length === 0 ? (
                <p className="muted small">Draw a box by clicking and dragging.</p>
              ) : (
                <ul>
                  {boxes.map((b, i) => (
                    <li key={i}>
                      <span className="cls-dot" style={{ background: colorForClass(b.class_id) }} />
                      <span className="cls-name">
                        #{i + 1} {classes[b.class_id] ?? `cls_${b.class_id}`}
                      </span>
                      <button
                        onClick={() => onDeleteBox(i)}
                        style={{ background: 'transparent', color: 'var(--fg-muted)', padding: '2px 8px', fontSize: '0.8rem' }}
                        title="Remove this box"
                      >×</button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          ) : (
            <DatasetSummary stats={stats} entries={entries} onDelete={onDeleteEntry} />
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
