import { useState } from 'react'
import { streamUrl } from './api'

// Live MJPEG stream pulled from /api/stream. The browser handles
// multipart/x-mixed-replace in <img> natively. Cache-bust forces a
// fresh connection on Start so the browser doesn't reuse an idle one.
//
// inferEvery: run inference once every N frames; reuse last boxes in
// between. On CPU this is the single biggest live-FPS knob.
export default function LiveView() {
  const [source, setSource] = useState('0')
  const [inferEvery, setInferEvery] = useState(3)
  const [running, setRunning] = useState(false)
  const [bust, setBust] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const start = () => {
    setError(null)
    setBust((b) => b + 1)
    setRunning(true)
  }
  const stop = () => setRunning(false)

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
        {running ? (
          <button onClick={stop}>Stop</button>
        ) : (
          <button onClick={start} disabled={!source.trim()}>Start</button>
        )}
        {error && <span className="error">{error}</span>}
      </div>

      <div className="image-pane">
        {running ? (
          <img
            className="stream-img"
            src={streamUrl(source, inferEvery, bust)}
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
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
