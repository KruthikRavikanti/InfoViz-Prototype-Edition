import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, '.', '');
  const cortexCoreProxyTarget = environment.CORTEX_CORE_PROXY_TARGET || 'http://127.0.0.1:5556';

  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5174,
      strictPort: true,
      proxy: {
        // Keep session and temporary-image requests same-origin when InfoViz is
        // opened through VS Code Remote SSH or another single-port tunnel.
        '/cortex-core': {
          target: cortexCoreProxyTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/cortex-core/, ''),
        },
      },
    },
    preview: {
      port: 4174,
      strictPort: true,
    },
  };
});
