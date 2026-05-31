import { useEffect, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes, Navigate } from 'react-router-dom'
import { fetchHealth, type HealthInfo } from './api'
import UploadView from './UploadView'
import LiveView from './LiveView'
import LabelView from './LabelView'

// Routes:
//   /         → Upload image  (also matches any unknown path → fallback)
//   /live     → Live stream
//   /train    → Label & train
//
// Direct loads of /train work both locally (Vite's history-API fallback)
// and in production on Vercel (vercel.json rewrites /(.*) → /index.html).
export default function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  )
}

function Shell() {
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const tick = () =>
      fetchHealth()
        .then((h) => { if (!cancelled) { setHealth(h); setHealthError(null) } })
        .catch((e) => { if (!cancelled) { setHealth(null); setHealthError(String(e)) } })
    tick()
    const id = window.setInterval(tick, 10_000)
    return () => { cancelled = true; window.clearInterval(id) }
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <h1>E. coli Detection</h1>
        </NavLink>
        <HealthBadge health={health} error={healthError} />
      </header>
      <nav className="tabs">
        <NavLink
          to="/"
          end
          className={({ isActive }) => (isActive ? 'tab active' : 'tab')}
        >Upload image</NavLink>
        <NavLink
          to="/live"
          className={({ isActive }) => (isActive ? 'tab active' : 'tab')}
        >Live stream</NavLink>
        <NavLink
          to="/train"
          className={({ isActive }) => (isActive ? 'tab active' : 'tab')}
        >Label & train</NavLink>
      </nav>
      <main className="content">
        <Routes>
          <Route path="/" element={<UploadView />} />
          <Route path="/live" element={<LiveView />} />
          <Route path="/train" element={<LabelView />} />
          {/* Anything else falls back to the home page. */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}

function HealthBadge({ health, error }: { health: HealthInfo | null; error: string | null }) {
  if (!health) {
    return (
      <span className="badge badge-warn" title={error ?? 'no response from backend'}>
        backend offline
      </span>
    )
  }
  const cls = health.model_loaded ? 'badge-ok' : 'badge-warn'
  return (
    <span className={`badge ${cls}`}>
      {health.status} · {health.device} · {health.classes.length} class{health.classes.length === 1 ? '' : 'es'}
    </span>
  )
}
