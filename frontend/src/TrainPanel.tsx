import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchTrainingStatus,
  startTraining,
  stopTraining,
  type TrainingStatus,
} from './api'

// Polls /api/train/status. The polling interval adapts: fast while a
// run is active, slow otherwise — keeps idle CPU/network noise down
// without sacrificing log latency when something is actually running.
const POLL_RUNNING_MS = 2000
const POLL_IDLE_MS = 10_000


export default function TrainPanel() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [weights, setWeights] = useState('yolov9c.pt')
  const [epochs, setEpochs] = useState(100)
  const [batch, setBatch] = useState(16)
  const [imgsz, setImgsz] = useState(640)
  const [device, setDevice] = useState('auto')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    let handle: number | undefined

    const tick = async () => {
      try {
        const s = await fetchTrainingStatus()
        if (cancelled) return
        setStatus(s)
        const next = s.state === 'running' ? POLL_RUNNING_MS : POLL_IDLE_MS
        handle = window.setTimeout(tick, next)
      } catch {
        if (cancelled) return
        handle = window.setTimeout(tick, POLL_IDLE_MS)
      }
    }
    tick()
    return () => {
      cancelled = true
      if (handle !== undefined) window.clearTimeout(handle)
    }
  }, [])

  // Auto-scroll the log view to the bottom as new lines arrive — but
  // only while the run is active. After it finishes, leave the user
  // free to scroll up and read.
  useEffect(() => {
    if (logRef.current && status?.state === 'running') {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [status])

  const onStart = useCallback(async () => {
    setBusy(true); setError(null)
    try {
      setStatus(await startTraining({ weights, epochs, batch, imgsz, device }))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [weights, epochs, batch, imgsz, device])

  const onStop = useCallback(async () => {
    setBusy(true); setError(null)
    try {
      setStatus(await stopTraining())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [])

  const running = status?.state === 'running'
  const elapsed = status?.started_at
    ? Math.max(0, Math.floor(Date.now() / 1000 - status.started_at))
    : 0

  return (
    <div className="train-panel">
      <h3>Train model</h3>

      <div className="train-form">
        <label>
          <span>Weights</span>
          <input
            type="text" value={weights}
            onChange={(e) => setWeights(e.target.value)}
            disabled={running || busy}
          />
        </label>
        <label>
          <span>Epochs</span>
          <input
            type="number" min={1} max={2000} step={1} value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value || '0', 10) || 1)}
            disabled={running || busy}
          />
        </label>
        <label>
          <span>Batch</span>
          <input
            type="number" min={1} max={256} step={1} value={batch}
            onChange={(e) => setBatch(parseInt(e.target.value || '0', 10) || 1)}
            disabled={running || busy}
          />
        </label>
        <label>
          <span>Image size</span>
          <input
            type="number" min={64} max={2048} step={32} value={imgsz}
            onChange={(e) => setImgsz(parseInt(e.target.value || '0', 10) || 64)}
            disabled={running || busy}
          />
        </label>
        <label>
          <span>Device</span>
          <input
            type="text" value={device}
            onChange={(e) => setDevice(e.target.value)}
            placeholder="auto, 0, cpu"
            disabled={running || busy}
          />
        </label>
      </div>

      <div className="train-actions">
        {running ? (
          <button onClick={onStop} disabled={busy}>Stop training</button>
        ) : (
          <button onClick={onStart} disabled={busy}>Start training</button>
        )}
        {status && (
          <StatusPill
            state={status.state}
            elapsed={running ? elapsed : null}
            returnCode={status.return_code}
          />
        )}
        {status?.name && <span className="muted small">run: {status.name}</span>}
      </div>

      {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}

      {status && status.log_lines.length > 0 && (
        <div ref={logRef} className="train-log">
          {status.log_lines.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}

      {status?.state === 'completed' && status.name && (
        <p className="muted small" style={{ marginTop: 8 }}>
          Best weights:&nbsp;
          <code>runs/train/{status.name}/weights/best.pt</code>
          . Copy it to <code>models/ecoli_yolov9c.pt</code> and switch the
          backend to <code>config.yaml</code> to deploy.
        </p>
      )}
    </div>
  )
}


function StatusPill({
  state, elapsed, returnCode,
}: {
  state: TrainingStatus['state']
  elapsed: number | null
  returnCode: number | null
}) {
  const cls =
    state === 'running' ? 'badge-ok'
    : state === 'completed' ? 'badge-ok'
    : state === 'failed' || state === 'killed' ? 'badge-warn'
    : ''
  const label =
    state === 'running' && elapsed !== null
      ? `running · ${formatElapsed(elapsed)}`
      : state === 'completed'
        ? `completed (rc=${returnCode})`
        : state === 'failed'
          ? `failed (rc=${returnCode})`
          : state
  return <span className={`badge ${cls}`}>{label}</span>
}


function formatElapsed(s: number): string {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${sec}s`
  return `${sec}s`
}
