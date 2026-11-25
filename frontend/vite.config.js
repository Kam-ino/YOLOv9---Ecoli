import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
const isVercelBuild = process.env.VERCEL === '1';
export default defineConfig(({ mode }) => ({
    plugins: [react()],
    // API_BASE_URL (Vercel project setting, shell, or frontend/.env) is the
    // only variable baked into the browser bundle. Deliberately not
    // envPrefix: that would publish every API_* variable, secrets included.
    define: {
        'import.meta.env.API_BASE_URL': JSON.stringify(loadEnv(mode, process.cwd(), '').API_BASE_URL ?? ''),
    },
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
}));
