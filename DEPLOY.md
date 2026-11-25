# Deployment — UI on Vercel, backend on your PC

The React UI is a static site on Vercel. The backend (PyTorch + YOLOv9 /
RF-DETR, training, the dataset) runs on a computer you control — ideally one
with an NVIDIA GPU — and is reached over HTTPS through a tunnel. Any device
(e.g. the Raspberry Pi with the microscope) just opens the Vercel link.

```
 Pi / laptop / phone                     Vercel                 your PC
┌──────────────────────┐   page   ┌────────────────────┐
│ browser + microscope │ ◄─────── │ static React UI    │
│ (getUserMedia)       │          └────────────────────┘
│                      │  HTTPS /api/*   ┌──────────────────┐   ┌──────────────┐
│                      │ ──────────────► │ Tailscale Funnel │ ─►│ uvicorn :8003│
└──────────────────────┘  frames, boxes  │ https://…ts.net  │   │ (GPU)        │
                                         └──────────────────┘   └──────────────┘
```

When the UI is built with `API_BASE_URL` (i.e. on Vercel), the Live
tab and the Label tab's Snapshot read the camera **of the device viewing the
page** and upload frames to the backend; boxes are drawn in the browser. The
backend's own camera is not used.

---

## One-time setup

### 1. Give the PC a fixed HTTPS address (Tailscale Funnel, free)

```powershell
winget install --id Tailscale.Tailscale     # then sign in from the tray icon
tailscale funnel --bg 8003
```

The first `funnel` run prints a link to enable HTTPS/Funnel for your
tailnet — open it, enable, run the command again. It then prints the
address, e.g. `https://my-pc.tail1234.ts.net`. `--bg` keeps the Funnel
across reboots, so the address never changes.

### 2. Point the Vercel project at it

The repo-root `vercel.json` builds `frontend/` into `frontend/dist`. In the
Vercel project → **Settings → Environment Variables**:

| Name | Value |
|---|---|
| `API_BASE_URL` | `https://my-pc.tail1234.ts.net` (no trailing slash) |

Then **Deployments → ⋯ → Redeploy**. Vite bakes the variable into the
bundle at build time, so changing it always needs a redeploy.

### 3. Stop the Pi's own backend (if you installed the service)

```bash
sudo systemctl disable --now ecoli-backend
```

The Pi only runs a browser now.

---

## Every time

On the PC, start the backend and allow your Vercel origin (CORS):

```powershell
deploy\serve-pc.bat https://YOUR-APP.vercel.app
```

(or once `setx CORS_ORIGINS https://YOUR-APP.vercel.app`, then just
`deploy\serve-pc.bat`). It listens on 127.0.0.1 only; the Funnel is the
public door. Keep the PC awake and plugged in.

On the Pi, open `https://YOUR-APP.vercel.app` in Chromium, allow the camera
when asked, go to **Live stream**, pick the microscope under **Camera**, Start.

---

## What works through the Vercel link

| Feature | Notes |
|---|---|
| Upload tab — image → predict | Image goes to the PC; boxes come back. |
| Live stream | Video plays locally from the viewer's camera; frames (≤960 px) go to the PC, boxes refresh as fast as the round trip allows. "Min confidence" applies instantly. |
| Capture & label / Snapshot | Grabs a full-resolution frame from the viewer's camera. |
| Label tab — save | Saved on the PC's disk (`data/ecoli/`). |
| Train tab | Runs on the PC's GPU. |

If the PC or the backend is off, the page loads but every API call fails.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Status badge shows an error / `Failed to fetch` | Backend or Funnel not running | `deploy\serve-pc.bat …`; `tailscale funnel status` |
| `CORS error` in the console | Vercel origin not allowed | Pass the exact `https://YOUR-APP.vercel.app` to `serve-pc.bat` |
| `Mixed content blocked` | `API_BASE_URL` is `http://` | Use the Funnel's `https://` address, redeploy |
| "Camera permission was denied" | Chromium blocked the camera | Click the camera icon in the address bar → Allow, reload |
| "No camera found" | Microscope not detected by the viewer's device | Replug it; check `ls /dev/video*` on the Pi |

**No Tailscale?** `cloudflared tunnel --url http://localhost:8003` gives an
HTTPS address without an account, but it changes on every start, so you'd
update `API_BASE_URL` and redeploy each time. Avoid free ngrok: its
browser-warning page breaks the API calls.

---

## Going back to fully local

Leaving `API_BASE_URL` empty makes every fetch a same-origin `/api/*`
call and the Live tab uses the backend's camera (`source` 0, 1, …):

```powershell
cd frontend; npm run build; cd ..    # writes to backend/app/static/
$env:ECOLI_CONFIG = "config.yaml"
.venv\Scripts\python -m uvicorn backend.app.main:app --port 8003
```

The vite build writes to `dist/` only when Vercel (`VERCEL=1`) runs it.
