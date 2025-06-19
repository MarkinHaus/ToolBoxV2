// tbjs/webpack.config.js
import path from 'path';
import { fileURLToPath } from 'url';
import MiniCssExtractPlugin from 'mini-css-extract-plugin';
import { CleanWebpackPlugin } from 'clean-webpack-plugin';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isProduction = process.env.NODE_ENV === 'production';

export default {
  mode: isProduction ? 'production' : 'development',
  entry: {
    tbjs: './src/index.js', // Haupt-JS-Einstiegspunkt für tbjs
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js', // Wird zu tbjs.js
    library: {
        name: 'TB', // Globaler Name, wenn als <script> Tag geladen (optional)
        type: 'umd', // Universal Module Definition, macht es kompatibel
        export: 'default'
    },
    globalObject: 'this', // Wichtig für UMD
    clean: true, // Ersetzt CleanWebpackPlugin in Webpack 5+ für output.path
  },
  devtool: isProduction ? 'source-map' : 'eval-source-map',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env'],
          },
        },
      },
      {
        test: /\.css$/,
        exclude: /node_modules/,
        use: [
          MiniCssExtractPlugin.loader,
          'css-loader',
          {
            loader: 'postcss-loader',
            options: {
              postcssOptions: {
                config: path.resolve(__dirname, 'postcss.config.js'),
              },
            },
          },
        ],
      },
      // Wenn deine Komponenten direkt CSS importieren (nicht empfohlen mit Tailwind),
      // müsstest du eine weitere Regel hinzufügen oder die obige anpassen.
    ],
  },
  plugins: [
    new CleanWebpackPlugin(), // Stellt sicher, dass dist/ sauber ist vor dem Build
    new MiniCssExtractPlugin({
      filename: '[name].css', // Wird zu tbjs.css
    }),
  ],
  // Externe Abhängigkeiten, die nicht gebündelt werden sollen
  // (wenn sie als peerDependencies deklariert sind und von der App bereitgestellt werden)
  externals: {
    'htmx.org': 'htmx', // Wenn die App HTMX global als 'htmx' bereitstellt
    'three': 'THREE',   // Wenn die App Three.js global als 'THREE' bereitstellt
  },
  optimization: {
    minimize: isProduction,
    // ... (weitere Optimierungen wie TerserPlugin, CssMinimizerPlugin, wenn nötig)
  },
  resolve: {
    extensions: ['.js'],
    // Aliase, wenn nötig
  },
};
