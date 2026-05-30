import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During development, Vite proxies `/api/*` to the FastAPI backend on :8000
// so the React app talks to its API without CORS. The MJPEG stream goes
// through the same proxy.
//
// `build.outDir` writes the production bundle directly into the FastAPI
// app's static directory so a single `uvicorn` process can serve both
// the UI and the API in production.
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
    outDir: '../backend/app/static',
    emptyOutDir: true,
  },
})
