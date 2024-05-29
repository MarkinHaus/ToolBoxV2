import { defineConfig  } from 'vite';
import { resolve } from 'path'

export default defineConfig({
    build: {
        outDir: './vweb/', // Verzeichnis, in das Vite baut
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                widgetbord: resolve(__dirname, 'web/dashboards/widgetbord.html'),
            },
        },
    },
    preview: {
        port: 8080,
        proxy: {
            '/api': 'http://127.0.0.1:5000',
            '/web': 'http://127.0.0.1:5000',
        }
    },
    server: {
        port: 8080,
        proxy: {
            '/api': 'http://127.0.0.1:5000',
            '/validateSession': 'http://127.0.0.1:5000',
            '/web/signup': 'http://127.0.0.1:5000',
            '/web/login': 'http://127.0.0.1:5000',
            '/web/logout': 'http://127.0.0.1:5000',
            '/web/logoutS': 'http://127.0.0.1:5000',
            '/web/dashboard': 'http://127.0.0.1:5000',
                // changeOrigin: true,
                // rewrite: (path) => path.replace(/^\/api/, ''), // Entfernen des "/api"-Präfixes
        },
    },
});
