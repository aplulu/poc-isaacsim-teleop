import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// Dev server: `npm run dev` serves the app over HTTPS (WebXR needs a secure context) and proxies
// /api to the Isaac Sim server. Production: `npm run build` -> dist/ is served by the Python server.
const ISAAC_SERVER = process.env.ISAAC_SERVER ?? 'https://127.0.0.1:8443'

export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': { target: ISAAC_SERVER, changeOrigin: true, secure: false },
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 2000,
  },
})
