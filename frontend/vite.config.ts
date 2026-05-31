import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Build output dir depends on context:
//   - On Vercel (process.env.VERCEL === '1' at build time):
//     write to `dist/` so Vercel's static deploy picks it up.
//   - Locally (single-process serve via uvicorn):
//     write into the FastAPI app's static directory so one uvicorn
//     process serves both the UI and /api/* on the same port.
//
// The dev server proxy still forwards /api/* to localhost:8003 so the
// same React code works without thinking about the backend URL during
// local development.
//
// `process` is available at build time (vite.config.ts runs in Node)
// but we don't pull in @types/node — declare what we use locally.
declare const process: { env: Record<string, string | undefined> }
const isVercelBuild = process.env.VERCEL === '1'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 3003,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
        // MJPEG is a long-lived response; disable Vite's proxy timeout.
        ws: false,
      },
    },
  },
  build: {
    outDir: isVercelBuild ? 'dist' : '../backend/app/static',
    emptyOutDir: true,
  },
})
