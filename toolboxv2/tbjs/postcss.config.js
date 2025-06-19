export default {
  plugins: {
    tailwindcss: { config: './tbjs/tailwind.config.js' }, // Pfad zur tbjs-Tailwind-Konfig
    autoprefixer: {}, // Oder 'postcss-preset-env' für breitere Kompatibilität
    // 'postcss-preset-env': {
    //   stage: 1, // Oder eine andere Stufe
    // },
  },
};
