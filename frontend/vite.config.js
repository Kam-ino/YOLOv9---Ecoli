import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
const isVercelBuild = process.env.VERCEL === '1';
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
});
