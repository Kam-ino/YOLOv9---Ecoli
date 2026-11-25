import { useEffect, useRef, useState } from 'react'
import type { Detection } from './api'
import { DetectionSvg } from './DetectionOverlay'

// Microscope read by THIS browser (getUserMedia), for when the backend runs
// on a cloud server that has no camera. Needs HTTPS or localhost.

export type Camera = ReturnType<typeof useCamera>

// Opens the camera while `active`, closes it otherwise (frees the device).
export function useCamera(active: boolean) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!active) return
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This page needs HTTPS (or localhost) to use the camera.')
      return
    }
    let cancelled = false
    let opened: MediaStream | null = null
    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })
      .then(async (s) => {
        if (cancelled) { s.getTracks().forEach((t) => t.stop()); return }
        opened = s
        setStream(s)
        // Device names are only readable once permission is granted.
        const all = await navigator.mediaDevices.enumerateDevices()
        if (!cancelled) setDevices(all.filter((d) => d.kind === 'videoinput'))
      })
      .catch((e: unknown) => {
        if (cancelled) return
        const name = e instanceof DOMException ? e.name : ''
        setError(
          name === 'NotAllowedError' ? 'Camera permission was denied — allow it in the browser and retry.'
          : name === 'NotFoundError' || name === 'OverconstrainedError' ? 'No camera found — is the microscope plugged in?'
          : name === 'NotReadableError' ? 'The camera is busy (another app or tab is using it).'
          : `Could not open the camera: ${e instanceof Error ? e.message : String(e)}`,
        )
      })
    return () => {
      cancelled = true
      opened?.getTracks().forEach((t) => t.stop())
      setStream(null)
    }
  }, [active, deviceId])

  return { videoRef, stream, devices, deviceId, setDeviceId, error }
}

// Current video frame as an image, scaled down so its long side is at most
// `maxSide` (the backend tiles anything bigger, which costs ~7x the time).
export function grabFrame(
  video: HTMLVideoElement | null,
  maxSide = Infinity,
  type: 'image/jpeg' | 'image/png' = 'image/jpeg',
): Promise<Blob> {
  if (!video || !video.videoWidth) {
    return Promise.reject(new Error('Camera not ready yet — give it a moment.'))
  }
  const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(video.videoWidth * scale)
  canvas.height = Math.round(video.videoHeight * scale)
  const ctx = canvas.getContext('2d')
  if (!ctx) return Promise.reject(new Error('Could not get a canvas context.'))
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
  return new Promise((resolve, reject) =>
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('Could not capture a frame.'))),
      type, 0.9,
    ),
  )
}

export function CameraPicker({ cam, disabled }: { cam: Camera; disabled?: boolean }) {
  return (
    <label>
      <span>Camera</span>
      <select
        value={cam.deviceId}
        onChange={(e) => cam.setDeviceId(e.target.value)}
        disabled={disabled}
      >
        <option value="">Default camera</option>
        {cam.devices.map((d, i) => (
          <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>
        ))}
      </select>
    </label>
  )
}

// The live video, with detection boxes (in `frameSize` pixels) on top.
export function CameraView({
  cam,
  detections = [],
  frameSize = null,
}: {
  cam: Camera
  detections?: Detection[]
  frameSize?: [number, number] | null
}) {
  // Re-attach on every mount too: views unmount the <video> while busy.
  useEffect(() => {
    const v = cam.videoRef.current
    if (v && v.srcObject !== cam.stream) v.srcObject = cam.stream
  }, [cam.stream, cam.videoRef])

  return (
    <div className="overlay-wrapper">
      <video ref={cam.videoRef} className="stream-img" autoPlay muted playsInline />
      {frameSize && <DetectionSvg size={frameSize} detections={detections} />}
    </div>
  )
}
