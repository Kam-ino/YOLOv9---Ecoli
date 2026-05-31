import { useCallback, useEffect, useRef } from 'react'
import type { TrainingStatus } from './api'
import { stopTraining } from './api'

// Big modal console for training logs. Lives at the top of the layout
// (fixed-position overlay) so it isn't constrained by the cramped
// sidebar — you can read 30+ lines at a time and the font is actually
// legible.
//
// Behavior:
//   - opens when training starts (TrainPanel sets visible=true)
//   - Esc and backdrop click both close it
//   - closing it does NOT stop training (the backend keeps going)
//   - auto-scrolls to the bottom only while state === 'running' so the
//     user can pause to read after a run finishes
type Props = {
  status: TrainingStatus | null
  onClose: () => void
  onStopped?: (status: TrainingStatus) => void
}

export default function TrainingConsole({ status, onClose, onStopped }: Props) {
  const logRef = useRef<HTMLDivElement>(null)
  const running = status?.state === 'running'

  // Esc to close. Listener is global so focus doesn't matter.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // Auto-scroll only while running — after it finishes the user usually
  // wants to scroll up to read the failure point or final metrics.
  useEffect(() => {
    if (logRef.current && running) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [status, running])

  const onStop = useCallback(async () => {
    try {
      const s = await stopTraining()
      onStopped?.(s)
    } catch {
      // surfaced by the polling status update in TrainPanel
    }
  }, [onStopped])

  if (!status) return null

  const elapsed =
    status.started_at
      ? Math.max(0, Math.floor(Date.now() / 1000 - status.started_at))
      : 0

  return (
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="train-console-title"
      onClick={onClose}
    >
      <div className="modal-window" onClick={(e) => e.stopPropagation()}>
        <header className="modal-header">
          <h2 id="train-console-title">Training console</h2>
          <ConsoleStatus status={status} elapsed={elapsed} />
          {status.name && <span className="muted small">run: {status.name}</span>}
          <button
            className="modal-close"
            onClick={onClose}
            title="Close (Esc) — does not stop training"
            aria-label="Close"
          >×</button>
        </header>

        <div ref={logRef} className="modal-log">
          {status.log_lines.length === 0 ? (
            <div className="muted">Waiting for output…</div>
          ) : (
            status.log_lines.map((line, i) => <div key={i}>{line}</div>)
          )}
        </div>

        <footer className="modal-footer">
          {running ? (
            <>
              <button onClick={onStop}>Stop training</button>
              <span className="muted small">Closing this window does not stop training.</span>
            </>
          ) : (
            <span className="muted small">
              {status.state === 'completed'
                ? `Run completed (rc=${status.return_code}). Best weights at runs/train/${status.name}/weights/best.pt.`
                : status.state === 'failed'
                  ? `Run failed (rc=${status.return_code}). Scroll up for the traceback.`
                  : status.state === 'killed'
                    ? 'Run killed.'
                    : 'No active run.'}
            </span>
          )}
        </footer>
      </div>
    </div>
  )
}


function ConsoleStatus({ status, elapsed }: { status: TrainingStatus; elapsed: number }) {
  const cls =
    status.state === 'running' || status.state === 'completed' ? 'badge-ok' :
    status.state === 'failed' || status.state === 'killed' ? 'badge-warn' :
    ''
  const label =
    status.state === 'running' ? `running · ${formatElapsed(elapsed)}` :
    status.state === 'completed' ? `completed (rc=${status.return_code})` :
    status.state === 'failed' ? `failed (rc=${status.return_code})` :
    status.state
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
