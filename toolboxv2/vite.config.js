import { defineConfig  } from 'vite';
export default defineConfig({
  build: {
    outDir: '../web/', // Verzeichnis, in das Vite baut
  },
    preview: {
        port: 8080,
    },
    server: {
        port: 8080,
        proxy: {
            '/api': 'http://127.0.0.1:5000',
            '/web/signup': 'http://127.0.0.1:5000',
            '/web/login': 'http://127.0.0.1:5000',
            '/web/dashboard': 'http://127.0.0.1:5000',
                // changeOrigin: true,
                // rewrite: (path) => path.replace(/^\/api/, ''), // Entfernen des "/api"-Präfixes
        },
    },
});
