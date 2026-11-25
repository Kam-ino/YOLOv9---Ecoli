import { useState } from 'react'
import type { Detection } from './api'
import { colorForClass } from './api'

// Boxes as an SVG laid over an image or video. The viewBox is the size of
// the frame the backend saw, so detection coords (pixels in that frame) map
// 1:1 regardless of how the browser scales what's underneath.
export function DetectionSvg({
  size,
  detections,
}: {
  size: [number, number]
  detections: Detection[]
}) {
  // Stroke and font sized as a fraction of the longer edge so boxes stay
  // visible on any resolution.
  const longer = Math.max(size[0], size[1])
  const strokeW = longer / 400
  const fontSize = longer / 55
  return (
    <svg
      className="det-svg"
      viewBox={`0 0 ${size[0]} ${size[1]}`}
      preserveAspectRatio="none"
    >
      {detections.map((d, i) => {
        const [x1, y1, x2, y2] = d.bbox
        const color = colorForClass(d.class_id)
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
  )
}

// Renders the uploaded image with the detection boxes on top.
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
      {natural && <DetectionSvg size={natural} detections={detections} />}
    </div>
  )
}
