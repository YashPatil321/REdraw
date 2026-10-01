import { defineConfig } from 'vitest/config';

// Dev server on 5173; `/api/*` is proxied to the FastAPI app with the `/api`
// prefix stripped (so `/api/assets/...` reaches the API's `/assets/...`).
export default defineConfig({
  server: {
    port: 5173,
    // The mock mode imports data/config/tools.yaml as a dev fixture.
    fs: { allow: ['..'] },
    proxy: {
      '/api': {
        target: process.env.REDRAW_API_URL ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      '/api': {
        target: process.env.REDRAW_API_URL ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  build: {
    target: 'es2022',
    // public/assets holds pipeline output served by the API; never bundle it
    copyPublicDir: false,
    chunkSizeWarningLimit: 1500,
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
