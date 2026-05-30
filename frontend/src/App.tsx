import { useEffect, useState } from 'react'
import { fetchHealth, type HealthInfo } from './api'
import UploadView from './UploadView'
import LiveView from './LiveView'
import LabelView from './LabelView'

type Tab = 'upload' | 'live' | 'label'

export default function App() {
  const [tab, setTab] = useState<Tab>('upload')
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
        <h1>E. coli Detection</h1>
        <HealthBadge health={health} error={healthError} />
      </header>
      <nav className="tabs">
        <button className={tab === 'upload' ? 'active' : ''} onClick={() => setTab('upload')}>
          Upload image
        </button>
        <button className={tab === 'live' ? 'active' : ''} onClick={() => setTab('live')}>
          Live stream
        </button>
        <button className={tab === 'label' ? 'active' : ''} onClick={() => setTab('label')}>
          Label & train
        </button>
      </nav>
      <main className="content">
        {tab === 'upload' && <UploadView />}
        {tab === 'live' && <LiveView />}
        {tab === 'label' && <LabelView />}
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
