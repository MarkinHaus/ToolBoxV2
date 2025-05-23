// tbjs/jest.config.js
export default {
  testEnvironment: 'jest-environment-jsdom',
  transform: {
    '^.+\\.js$': 'babel-jest',
  },
  // Mocken von CSS-Modulen oder direkten CSS-Importen
  moduleNameMapper: {
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    // Wenn du Aliase in Webpack hast, hier eintragen
    // '^@core/(.*)$': '<rootDir>/src/core/$1',
  },
  // Globale Mocks und Setup-Dateien
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'], // Optional, für globale Mocks
  // Coverage
  collectCoverage: true,
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/index.js', // Haupt-Exportdatei oft ausgeschlossen
    '!src/core/graphics.js', // THREE.js ist schwer zu unit-testen
    '!src/ui/components/MarkdownRenderer/MarkdownRenderer.js', // Externe Libs
    '!src/styles/**', // CSS-Dateien
    '!**/node_modules/**',
    '!**/dist/**',
  ],
  coverageReporters: ['text', 'lcov', 'clover'],
  // Verbessert die Lesbarkeit von Testfehlern
  verbose: true,
};
