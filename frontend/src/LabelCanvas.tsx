import { useCallback, useRef, useState } from 'react'
import type { LabelBox } from './api'
import { colorForClass } from './api'

// Click-and-drag drawing surface for YOLO bounding boxes.
//
//   * The SVG overlay sits exactly on top of the image and uses
//     viewBox = natural image size, so user-drawn coords map 1:1 onto
//     image pixels regardless of how the browser scales the <img>.
//   * Existing boxes are rendered with pointer-events:none so the user
//     can draw new boxes that overlap previously-placed ones. Delete
//     happens from the sidebar list.
//   * All emitted box coords are normalized to [0, 1] (YOLO convention).
type Props = {
  imageUrl: string
  boxes: LabelBox[]
  classId: number
  classes: string[]
  onChange: (boxes: LabelBox[]) => void
}

export default function LabelCanvas({ imageUrl, boxes, classId, classes, onChange }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [natural, setNatural] = useState<[number, number] | null>(null)
  const [drag, setDrag] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)

  // Convert a pointer event to normalized [0, 1] coords against the
  // displayed image rectangle. We clip to the rect so drags that leave
  // the image bounds still produce sane boxes.
  const toNorm = useCallback((e: React.PointerEvent): { x: number; y: number } | null => {
    const img = imgRef.current
    if (!img) return null
    const r = img.getBoundingClientRect()
    if (r.width <= 0 || r.height <= 0) return null
    const x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))
    const y = Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))
    return { x, y }
  }, [])

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    const p = toNorm(e)
    if (!p) return
    setDrag({ x1: p.x, y1: p.y, x2: p.x, y2: p.y })
    ;(e.currentTarget as Element).setPointerCapture(e.pointerId)
    e.preventDefault()
  }
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag) return
    const p = toNorm(e)
    if (!p) return
    setDrag({ ...drag, x2: p.x, y2: p.y })
  }
  const onPointerUp = (_e: React.PointerEvent) => {
    if (!drag) return
    const x1 = Math.min(drag.x1, drag.x2)
    const x2 = Math.max(drag.x1, drag.x2)
    const y1 = Math.min(drag.y1, drag.y2)
    const y2 = Math.max(drag.y1, drag.y2)
    const w = x2 - x1
    const h = y2 - y1
    setDrag(null)
    // Reject sub-0.5% boxes — almost certainly a misclick.
    if (w < 0.005 || h < 0.005) return
    onChange([
      ...boxes,
      { class_id: classId, cx: x1 + w / 2, cy: y1 + h / 2, w, h },
    ])
  }

  return (
    <div className="label-canvas">
      <img
        ref={imgRef}
        src={imageUrl}
        alt="labelling"
        draggable={false}
        onLoad={(e) => setNatural([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
      />
      {natural && (
        <svg
          className="label-svg"
          viewBox={`0 0 ${natural[0]} ${natural[1]}`}
          preserveAspectRatio="none"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          {boxes.map((b, i) => {
            const color = colorForClass(b.class_id)
            const x = (b.cx - b.w / 2) * natural[0]
            const y = (b.cy - b.h / 2) * natural[1]
            const w = b.w * natural[0]
            const h = b.h * natural[1]
            const longer = Math.max(natural[0], natural[1])
            const strokeW = longer / 400
            const fontSize = longer / 70
            return (
              // pointer-events:none so the user can keep drawing through
              // existing boxes; deletion happens via the sidebar list.
              <g key={i} style={{ pointerEvents: 'none' }}>
                <rect
                  x={x} y={y} width={w} height={h}
                  fill="rgba(255,255,255,0.04)"
                  stroke={color} strokeWidth={strokeW}
                />
                <text
                  x={x} y={Math.max(y - strokeW * 2, fontSize)}
                  fill={color} fontSize={fontSize}
                  fontFamily="system-ui, sans-serif"
                  style={{ paintOrder: 'stroke' }}
                  stroke="black" strokeWidth={strokeW * 0.6}
                >
                  {classes[b.class_id] ?? `cls_${b.class_id}`} · {i + 1}
                </text>
              </g>
            )
          })}
          {drag && (() => {
            const x = Math.min(drag.x1, drag.x2) * natural[0]
            const y = Math.min(drag.y1, drag.y2) * natural[1]
            const w = Math.abs(drag.x2 - drag.x1) * natural[0]
            const h = Math.abs(drag.y2 - drag.y1) * natural[1]
            const longer = Math.max(natural[0], natural[1])
            return (
              <rect
                x={x} y={y} width={w} height={h}
                fill="none"
                stroke={colorForClass(classId)}
                strokeWidth={longer / 400}
                strokeDasharray={`${longer / 100},${longer / 200}`}
                style={{ pointerEvents: 'none' }}
              />
            )
          })()}
        </svg>
      )}
    </div>
  )
}
