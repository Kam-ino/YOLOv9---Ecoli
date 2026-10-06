import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ALGORITHMS,
  ALGORITHM_LABEL,
  fetchTrainingStatus,
  resetModel,
  startTraining,
  stopTraining,
  type Algorithm,
  type TrainingStatus,
} from './api'
import TrainingConsole from './TrainingConsole'
import { useAlgorithm } from './useAlgorithm'

// Per-algorithm form defaults. RF-DETR: COCO-pretrained variant name as
// "weights", smaller batch (DINOv2 backbone at 640 px on an 8 GB GPU).
const DEFAULTS: Record<Algorithm, { weights: string; epochs: number; batch: number }> = {
  yolov9: { weights: 'yolov9c.pt', epochs: 250, batch: 16 },
  rfdetr: { weights: 'medium', epochs: 200, batch: 4 },
}
const ACTIVE_WEIGHTS: Record<Algorithm, string> = {
  yolov9: 'models/best_yolov9c.pt',
  rfdetr: 'models/best_rfdetr.pth',
}
const BEST_FILE: Record<Algorithm, string> = {
  yolov9: 'weights/best.pt',
  rfdetr: 'checkpoint_best_ema.pth',
}

// Polls /api/train/status. Adapts the interval: fast while a run is
// active, slow otherwise — keeps idle CPU/network noise down without
// sacrificing log latency when something is actually running.
const POLL_RUNNING_MS = 2000
const POLL_IDLE_MS = 10_000


export default function TrainPanel() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [algorithm, setAlgorithmStored] = useAlgorithm()
  const [weights, setWeights] = useState(DEFAULTS[algorithm].weights)
  const [epochs, setEpochs] = useState(DEFAULTS[algorithm].epochs)
  const [batch, setBatch] = useState(DEFAULTS[algorithm].batch)
  const [imgsz, setImgsz] = useState(640)
  const [device, setDevice] = useState('auto')
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [consoleOpen, setConsoleOpen] = useState(false)

  // Track the last seen state so we can auto-open the modal on the
  // transition into 'running' (covers both "user clicks Start here" and
  // "user refreshes while a run is in flight").
  const prevStateRef = useRef<TrainingStatus['state'] | null>(null)
  const userClosedRef = useRef(false)

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

  // Auto-open the modal the first time we observe a 'running' state
  // during this component's life, *unless* the user has explicitly
  // dismissed it. That way the modal pops as soon as Start is clicked,
  // or on a fresh page load if training is already in flight — but
  // closing it sticks until the next manual reopen.
  useEffect(() => {
    if (!status) return
    if (
      status.state === 'running' &&
      prevStateRef.current !== 'running' &&
      !userClosedRef.current
    ) {
      setConsoleOpen(true)
    }
    if (status.state !== 'running') {
      // Reset the "user closed" guard so the next run can auto-open again.
      userClosedRef.current = false
    }
    prevStateRef.current = status.state
  }, [status])

  const onAlgorithm = useCallback((a: Algorithm) => {
    setAlgorithmStored(a)
    setWeights(DEFAULTS[a].weights)
    setEpochs(DEFAULTS[a].epochs)
    setBatch(DEFAULTS[a].batch)
  }, [setAlgorithmStored])

  const onStart = useCallback(async () => {
    setBusy(true)
    setError(null)
    userClosedRef.current = false
    setConsoleOpen(true)
    try {
      setStatus(await startTraining({ algorithm, weights, epochs, batch, imgsz, device }))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [algorithm, weights, epochs, batch, imgsz, device])

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

  const onCloseConsole = useCallback(() => {
    setConsoleOpen(false)
    userClosedRef.current = true
  }, [])

  // These derived values are referenced by callbacks below — declare
  // them before any useCallback that lists them in its dependency array
  // to avoid a temporal-dead-zone error at render time.
  const running = status?.state === 'running'
  const elapsed = status?.started_at
    ? Math.max(0, Math.floor(Date.now() / 1000 - status.started_at))
    : 0
  const hasLogs = !!status && status.log_lines.length > 0

  const onReset = useCallback(async () => {
    // Belt and braces: the backend rejects reset during a run, but
    // catch it here too so the user doesn't even get a chance.
    if (running) {
      setError("Stop the training run before resetting the model.")
      return
    }
    const active = ACTIVE_WEIGHTS[algorithm]
    const ok = window.confirm(
      `Reset the ${ALGORITHM_LABEL[algorithm]} model to the pretrained base?\n\n` +
      `• Your current trained weights will be moved to ${active.replace(/(\.\w+)$/, '.bak-<timestamp>$1')} (recoverable).\n` +
      "• The detector reloads from the COCO base — it will report 80 classes.\n" +
      "• Your labels and dataset are NOT touched. Re-run training to fine-tune from scratch.",
    )
    if (!ok) return
    setResetting(true); setError(null); setInfo(null)
    try {
      const result = await resetModel(algorithm)
      const backed = result.backup ? ` (backup: ${result.backup})` : ''
      setInfo(`${ALGORITHM_LABEL[algorithm]} reset — now serving the base ${result.active_weights}${backed}.`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setResetting(false)
    }
  }, [running, algorithm])

  return (
    <div className="train-panel">
      <h3>Train model</h3>

      <div className="train-form">
        <label>
          <span>Algorithm</span>
          <select
            value={algorithm}
            onChange={(e) => onAlgorithm(e.target.value as Algorithm)}
            disabled={running || busy}
          >
            {ALGORITHMS.map((a) => <option key={a} value={a}>{ALGORITHM_LABEL[a]}</option>)}
          </select>
        </label>
        <label>
          <span>Weights</span>
          <input
            type="text" value={weights}
            onChange={(e) => setWeights(e.target.value)}
            placeholder={algorithm === 'rfdetr' ? 'nano | small | medium | large | file.pth' : 'yolov9c.pt'}
            title={algorithm === 'rfdetr'
              ? 'COCO-pretrained variant name, or a .pth checkpoint to continue from'
              : 'Pretrained .pt name (auto-downloaded) or a file under models/'}
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
      </div>

      <div className="train-actions">
        {hasLogs && (
          <button
            onClick={() => setConsoleOpen(true)}
            style={{ background: 'var(--bg-elev-2)', color: 'var(--fg)' }}
            title="Open the training console"
          >
            {running ? 'View live console' : 'View last console'}
          </button>
        )}
        {status?.name && <span className="muted small">run: {status.name}</span>}
      </div>

      {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
      {info && <div className="muted small" style={{ marginTop: 8 }}>{info}</div>}

      {status?.state === 'completed' && status.name && (
        <p className="muted small" style={{ marginTop: 8 }}>
          Best weights:&nbsp;
          <code>runs/train/{status.name}/{BEST_FILE[status.algorithm ?? 'yolov9']}</code>
          &nbsp;— auto-activated as <code>{ACTIVE_WEIGHTS[status.algorithm ?? 'yolov9']}</code>
          (previous weights kept as a <code>.bak-&lt;timestamp&gt;</code> file).
        </p>
      )}

      <div className="train-danger">
        <h4>Danger zone</h4>
        <button
          onClick={onReset}
          disabled={resetting || running}
          className="danger-button"
          title={
            running
              ? "Stop training first"
              : "Move trained weights to a .bak file and reload from pretrained base"
          }
        >
          {resetting ? 'Resetting…' : `Reset ${ALGORITHM_LABEL[algorithm]} to pretrained base`}
        </button>
      </div>

      {consoleOpen && (
        <TrainingConsole
          status={status}
          onClose={onCloseConsole}
          onStopped={(s) => setStatus(s)}
        />
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
