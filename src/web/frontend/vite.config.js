import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5050',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 500,
    rollupOptions: {
      output: {
        manualChunks: {
          // React 核心
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          // MUI 组件库（大！）
          'mui-vendor': [
            '@mui/material',
            '@mui/icons-material',
            '@emotion/react',
            '@emotion/styled',
          ],
          // Recharts 图表库
          'recharts-vendor': ['recharts'],
          // D3 可视化库
          'd3-vendor': [
            'd3',
            'd3-array',
            'd3-color',
            'd3-format',
            'd3-interpolate',
            'd3-scale',
            'd3-selection',
            'd3-shape',
            'd3-transition',
            'd3-zoom',
          ],
        },
      },
    },
  },
})
