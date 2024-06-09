import { defineConfig  } from 'vite';
import { resolve } from 'path'

export default defineConfig({
    preview: {
        port: 8080,
        proxy: {
            '/api': 'http://127.0.0.1:5000',
            '/validateSession': 'http://127.0.0.1:5000',
            '/IsValiSession': 'http://127.0.0.1:5000',
            '/web/signup': 'http://127.0.0.1:5000',
            '/web/login': 'http://127.0.0.1:5000',
            '/web/logout': 'http://127.0.0.1:5000',
            '/web/logoutS': 'http://127.0.0.1:5000',
            '/web/dashboard': 'http://127.0.0.1:5000',
                // changeOrigin: true,
                // rewrite: (path) => path.replace(/^\/api/, ''), // Entfernen des "/api"-Präfixes
        },
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
