// tbjs/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './src/**/*.{html,js,css}', // Durchsucht alle JS-Dateien in tbjs/src nach Klassen
      "./tbjs/**/*.js",
  ],
  darkMode: 'class', // Oder 'media', je nach Präferenz (stimmt mit ui/theme.js überein)
  theme: {
    extend: {
      // Hier kannst du tbjs-spezifische Design-Tokens (Farben, Schriftarten etc.) definieren,
      // die von den tbjs-Komponenten verwendet werden.
      // Diese könnten von der Hauptanwendung überschrieben oder erweitert werden.
      colors: {
        'primary-50': 'var(--tb-color-primary-50, #eff6ff)', // CSS-Variablen für Überschreibbarkeit
        'primary-100': 'var(--tb-color-primary-100, #dbeafe)',
        'primary-500': 'var(--tb-color-primary-500, #3b82f6)',
        'primary-600': 'var(--tb-color-primary-600, #2563eb)',
        'primary-700': 'var(--tb-color-primary-700, #1d4ed8)',
        // Definiere Text-, Hintergrund- und Akzentfarben, die deine Komponenten nutzen
        'background-color': 'var(--tb-color-background, #ffffff)', // Hellmodus Standard
        'text-color': 'var(--tb-color-text, #1f2937)',           // Hellmodus Standard
        'border-color': 'var(--tb-color-border, #e5e7eb)',
        'accent-color': 'var(--tb-color-accent, #2563eb)',
      },
      // Beispiel für Dark Mode Farben (werden angewendet, wenn .dark auf <html> ist)
      // Diese werden über CSS-Variablen im Dark-Modus gesetzt.
    },
  },
  plugins: [
    require('@tailwindcss/forms'), // Wenn du Formular-Styling-Reset brauchst
  ],
  prefix: 'tb-', // SEHR EMPFOHLEN! Verhindert Klassenkonflikte mit der Hauptanwendung.
                 // Deine Komponenten-Klassen wären dann z.B. `tb-bg-primary-500`
};
