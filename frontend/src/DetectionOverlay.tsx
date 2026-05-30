import { useState } from 'react'
import type { Detection } from './api'
import { colorForClass } from './api'

// Renders the uploaded image with an absolutely-positioned SVG on top.
// The SVG viewBox is the natural image size, so detection coords (which
// are in original-frame pixels from the backend) map 1:1 onto the SVG
// regardless of how the browser scales the displayed <img>.
export default function DetectionOverlay({
  imageUrl,
  detections,
}: {
  imageUrl: string
  detections: Detection[]
}) {
  const [natural, setNatural] = useState<[number, number] | null>(null)

  return (
    <div className="overlay-wrapper">
      <img
        src={imageUrl}
        alt="upload"
        onLoad={(e) => {
          const img = e.currentTarget
          setNatural([img.naturalWidth, img.naturalHeight])
        }}
      />
      {natural && (
        <svg
          className="det-svg"
          viewBox={`0 0 ${natural[0]} ${natural[1]}`}
          preserveAspectRatio="none"
        >
          {detections.map((d, i) => {
            const [x1, y1, x2, y2] = d.bbox
            const color = colorForClass(d.class_id)
            // Stroke and font sized as a fraction of the longer edge so
            // boxes stay visible on any image resolution.
            const longer = Math.max(natural[0], natural[1])
            const strokeW = longer / 400
            const fontSize = longer / 55
            return (
              <g key={i}>
                <rect
                  x={x1} y={y1}
                  width={x2 - x1} height={y2 - y1}
                  fill="none" stroke={color} strokeWidth={strokeW}
                />
                <text
                  x={x1}
                  y={Math.max(y1 - strokeW * 2, fontSize)}
                  fill={color}
                  fontSize={fontSize}
                  fontFamily="system-ui, sans-serif"
                  style={{ paintOrder: 'stroke' }}
                  stroke="black"
                  strokeWidth={strokeW * 0.6}
                >
                  {d.class_name} {(d.confidence * 100).toFixed(0)}%
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}
