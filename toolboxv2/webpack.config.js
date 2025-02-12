import path from 'path';
import { CleanWebpackPlugin } from 'clean-webpack-plugin';
import HtmlWebpackPlugin from 'html-webpack-plugin';
import MiniCssExtractPlugin from 'mini-css-extract-plugin';
import CopyWebpackPlugin from 'copy-webpack-plugin';
import TerserPlugin from 'terser-webpack-plugin';
import CssMinimizerPlugin from 'css-minimizer-webpack-plugin';
import CompressionPlugin from 'compression-webpack-plugin';
import webpack from 'webpack';

const isProduction = process.env.NODE_ENV === 'production';

export default {
  mode: isProduction ? 'production' : 'development',
  entry: {
    main: './index.js',
  },
  output: {
    filename: '[name].js',
    path: path.resolve(process.cwd(), 'dist'),
    publicPath: '/',
  },
  devtool: isProduction ? 'eval-source-map' : 'eval-source-map',
  devServer: {
    static: {
      directory: path.join(process.cwd(), 'web'),
    },
    hot: true,
    historyApiFallback: true,
    open: true,
    proxy: [
      {
        context: ['/web/', '/index.js', '/api', '/validateSession', '/IsValiSession'],
        target:  'http://0.0.0.0:5000', // isProduction ?  process.env.TOOLBOXV2_REMOTE_BASE : 'http://0.0.0.0:5000',
      },{
        context: ['/talk'], // Proxy für index.js
        target: 'http://0.0.0.0:5000',
      },
    ],
  },
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
        use: [
          isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
          'css-loader',
        ],
      },
      {
        test: /\.(png|jpg|jpeg|gif|svg|woff|woff2|eot|ttf|otf)$/,
        type: 'asset/resource',
      },
    ],
  },
  plugins: [
    new CleanWebpackPlugin(),
    new HtmlWebpackPlugin({
      template: './index.html',
      minify: isProduction && {
        removeComments: true,
      },
    }),
    // For index.html
    new HtmlWebpackPlugin({
      template: './web/core0/index.html',
      filename: './web/core0/index.html',
      chunks: ['main']
    }),
    // For index.html
    new HtmlWebpackPlugin({
      template: './web/core0/Installer.html',
      filename: './web/core0/Installer.html',
      chunks: ['main']
    }),
    // For MainIdea.html
    new HtmlWebpackPlugin({
      template: './web/core0/MainIdea.html',
      filename: './web/core0/MainIdea.html',
      chunks: ['main']
    }),
    // For roadmap.html
    new HtmlWebpackPlugin({
      template: './web/core0/roadmap.html',
      filename: './web/core0/roadmap.html',
      chunks: ['main']
    }),
    // For /web/assets/404.html
    new HtmlWebpackPlugin({
      template: './web/assets/404.html',
      filename: './web/assets/404.html',
      chunks: ['main']
    }),
    // For /web/assets/401.html
    new HtmlWebpackPlugin({
      template: './web/assets/401.html',
      filename: './web/assets/401.html',
      chunks: ['main']
    }),
    // For /web/assets/m_log_in.html
    new HtmlWebpackPlugin({
      template: './web/assets/m_log_in.html',
      filename: './web/assets/m_log_in.html',
      chunks: ['main']
    }),
    // For /web/assets/logout.html
    new HtmlWebpackPlugin({
      template: './web/assets/logout.html',
      filename: './web/assets/logout.html',
      chunks: ['main']
    }),
    // For /web/assets/login.html
    new HtmlWebpackPlugin({
      template: './web/assets/login.html',
      filename: './web/assets/login.html',
      chunks: ['main']
    }),
    // For /web/assets/signup.html
    new HtmlWebpackPlugin({
      template: './web/assets/signup.html',
      filename: './web/assets/signup.html',
      chunks: ['main']
    }),
    // For /web/assets/terms.html
    new HtmlWebpackPlugin({
      template: './web/assets/terms.html',
      filename: './web/assets/terms.html',
      chunks: ['main']
    }),
    // For dashboard
    new HtmlWebpackPlugin({
      template: './web/dashboards/dashboard.html',
      filename: './web/dashboards/dashboard.html',
      chunks: ['main']
    }),
      // For Apps
    new HtmlWebpackPlugin({
      template: './web/mainContent.html',
      filename: './web/mainContent.html',
      chunks: ['main']
    }),
    // For helper.js
    new HtmlWebpackPlugin({
      template: './helper.html',
      filename: './helper.html',
      chunks: ['main']
    }),
    // For user_dashboard.html
    new HtmlWebpackPlugin({
      template: './web/dashboards/user_dashboard.html',
      filename: './web/dashboards/user_dashboard.html',
      chunks: ['main']
    }),
    new MiniCssExtractPlugin({
      filename:  '[name].css',
    }),
    new CopyWebpackPlugin({
      patterns: [
        {
          from: 'web',
          to: 'web',
          globOptions: {
            ignore: ['**/node_modules/**',
             '**/web/core0/index.html',
             '**/web/core0/Installer.html',
             '**/web/core0/MainIdea.html',
             '**/web/core0/roadmap.html',
             '**/web/assets/404.html',
             '**/web/assets/401.html',
             '**/web/assets/m_log_in.html',
             '**/web/assets/logout.html',
             '**/web/mainContent.html',
             '**/web/dashboards/dashboard.html',
             '**/web/dashboards/user_dashboard.html',
             '**/web/assets/login.html',
             '**/web/assets/signup.html',
             '**/web/assets/terms.html' ], // Exclude node_modules
          },
        },
      ],
    }),
    new CompressionPlugin({
      test: /\.(js|css|html|svg)$/,
      algorithm: 'gzip',
      compressionOptions: { level: 9 },
    }),
    new webpack.ProgressPlugin({
      percentBy: 'entries',
    }),
  ],
  optimization: {
    splitChunks: {
      chunks: 'all',
      maxSize: 512000,
      cacheGroups: {
      vendor: {
        test: /[\\/]node_modules[\\/]/,
        name: 'vendors',
        chunks: 'all',
      },
    },
    },
    minimize: isProduction,
    minimizer: [
      new TerserPlugin({
        parallel: true,
        terserOptions: {
          compress: {
            drop_console: true,
          },
        },
      }),
      new CssMinimizerPlugin(),
    ],
  },
  resolve: {
    extensions: ['.js', '.css'],
    alias: {
            htmx: path.join(process.cwd(), '/web/node_modules/htmx.org/dist/htmx.js'),
            Three: path.join(process.cwd(), '/web/node_modules/three/src/Three.js'),
        }
  },
  stats: {
    colors: true,
    modules: true,
    reasons: true,
    errorDetails: true,
  },
  performance: {
    hints: isProduction ? 'warning' : false,
    maxEntrypointSize: 812000,
    maxAssetSize: 812000,
  },
};
