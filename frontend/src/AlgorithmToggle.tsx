import { useEffect, useState } from 'react'
import { ALGORITHMS, ALGORITHM_LABEL, fetchHealth, type Algorithm, type HealthInfo } from './api'

// YOLOv9 / RF-DETR segmented switch (same .seg styling as the Label page's
// Upload / Snapshot toggle). A backend whose RF-DETR package is missing
// greys that option out with the reason in the tooltip.
export default function AlgorithmToggle({
  value, onChange, disabled = false,
}: {
  value: Algorithm
  onChange: (a: Algorithm) => void
  disabled?: boolean
}) {
  const [health, setHealth] = useState<HealthInfo | null>(null)
  useEffect(() => {
    let cancelled = false
    fetchHealth().then((h) => { if (!cancelled) setHealth(h) }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  return (
    <div className="seg" role="radiogroup" aria-label="Detector">
      {ALGORITHMS.map((a) => {
        const info = health?.algorithms?.[a]
        const unavailable = !!info && !info.available
        const title = unavailable
          ? `${ALGORITHM_LABEL[a]} is not installed on the backend`
          : info && !info.weights_exists
            ? `${ALGORITHM_LABEL[a]}: no fine-tuned weights yet (COCO base)`
            : ALGORITHM_LABEL[a]
        return (
          <button
            key={a}
            type="button"
            role="radio"
            aria-checked={value === a}
            className={value === a ? 'active' : ''}
            disabled={disabled || unavailable}
            title={title}
            onClick={() => onChange(a)}
          >{ALGORITHM_LABEL[a]}</button>
        )
      })}
    </div>
  )
}
