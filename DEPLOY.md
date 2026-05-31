# Deployment — Static UI on Vercel + local backend

The frontend ships to Vercel as a static SPA. The backend (PyTorch +
YOLOv9 + OpenCV + camera access) stays on a machine you control and is
exposed over HTTPS via a tunnel. This is the only architecture that
both fits Vercel and preserves the parts of this app that need a real
process (live USB stream, training, dataset persistence).

```
┌────────────────────┐     HTTPS API calls         ┌─────────────────────┐
│  React UI          │ ─────────────────────────►  │  Cloudflare Tunnel  │
│  https://<x>.vercel│                             │  / ngrok            │
└────────────────────┘                             └──────────┬──────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────────┐
                                                    │  uvicorn :8003      │
                                                    │  (your laptop / Pi) │
                                                    └─────────────────────┘
```

---

## One-time setup

### 1. Deploy the frontend to Vercel

The repo root has a `vercel.json` that tells Vercel to build inside
`frontend/` and serve the `dist/` output. From a fresh Vercel project:

1. Import the repo on <https://vercel.com/new>.
2. Leave **Root Directory** at the repo root (the `vercel.json` handles
   the rest).
3. Under **Environment Variables**, set:

   | Name | Value | Notes |
   |---|---|---|
   | `VITE_API_BASE_URL` | `https://<your-tunnel>.trycloudflare.com` | Has to be HTTPS. See step 2 for getting one. |

4. Click **Deploy**. First deploy will fail to *function* until step 2
   is done, but the build itself should succeed.

> When you change `VITE_API_BASE_URL`, click **Redeploy** — Vite bakes
> env vars into the bundle at build time, not at runtime.

### 2. Expose your local backend over HTTPS

Browsers refuse to call plain HTTP from an HTTPS page. Two easy
options — both give you a public HTTPS URL pointing at
`http://localhost:8003`.

#### Cloudflare Tunnel (recommended — free, persistent URL possible)

```bash
# install once
brew install cloudflared           # macOS
# or:   sudo apt install cloudflared   # Debian/Ubuntu
# or:   winget install Cloudflare.cloudflared   # Windows

# quick start (ephemeral URL; changes every restart)
cloudflared tunnel --url http://localhost:8003
```

It prints something like
`https://abc-thing-xyz-123.trycloudflare.com` — that's your
`VITE_API_BASE_URL`. Paste it into Vercel, redeploy.

For a stable URL that survives reboots, follow
<https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/>.

#### ngrok (one-command, ephemeral)

```bash
ngrok http 8003
```

Use the `https://...ngrok-free.app` URL it prints. Free accounts get a
new URL every restart.

### 3. Allow the Vercel origin through CORS

On the machine running the backend:

```bash
# Linux / macOS
export CORS_ORIGINS="https://your-app.vercel.app"

# Windows PowerShell
$env:CORS_ORIGINS = "https://your-app.vercel.app"
```

Then start the backend as usual (`./start.sh`, `uvicorn ...`, `run.bat`,
etc.). The backend reads `CORS_ORIGINS` (comma-separated) at startup
and uses local-dev defaults if it's unset, so this is a no-op for
development.

---

## What works after this is set up

| Feature | Works on Vercel-hosted UI? | Notes |
|---|---|---|
| Upload tab — image → predict | ✓ | Image bytes travel through the tunnel to your backend. |
| Label tab — drawing + save | ✓ | Saves to the labelling host's filesystem (your machine, not Vercel). |
| Label tab — snapshot from microscope | ✓ * | Captures from the microscope plugged into your *backend host*, not the visitor's webcam. |
| Live Stream tab | ✓ * | MJPEG streams over the tunnel. Watch your upload bandwidth. |
| Train tab | ✓ * | Runs on the backend host's CPU/GPU. |

*"Works" assumes the backend is running while someone visits the UI.
If you close your laptop, the UI loads but every API call fails until
you bring the backend back up.

---

## What doesn't work and how to know

| Symptom in the browser | Likely cause | Fix |
|---|---|---|
| `Failed to fetch` on /api/health | Tunnel not running, or backend not running | Start `cloudflared tunnel --url http://localhost:8003` and the backend |
| `Mixed content blocked` warning | `VITE_API_BASE_URL` is `http://`, UI is `https://` | Use a tunnel URL (HTTPS) |
| `CORS error` in console | Vercel origin not in `CORS_ORIGINS` | Set `CORS_ORIGINS=https://your-app.vercel.app` on the backend host, restart |
| API works but very slow on /api/predict | Image uploads + tunnel = round-trip over your home upload bandwidth | Expected for now; not a Vercel issue |

---

## Going back to fully local

Nothing in this guide breaks the local single-process mode. Leaving
`VITE_API_BASE_URL` empty makes every fetch a same-origin `/api/*`
call. So:

```powershell
# Windows local (single process)
cd frontend; npm run build; cd ..    # writes to backend/app/static/
$env:ECOLI_CONFIG = "config.yaml"
uvicorn backend.app.main:app --port 8003
```

still works exactly as before. The vite build detects `VERCEL=1` only
when Vercel is the one running it.
