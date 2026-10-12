// Standalone SPA build: same src/ tree, no Grafana host, no externals.
// Build-time env:
//   INVOICEX_BASE_URL  origin/prefix of the standalone server routes (/pdf, /whoami). default ''
//   INVOICEX_BASENAME  router basename + public path. default '/'
import CopyWebpackPlugin from 'copy-webpack-plugin';
import ForkTsCheckerWebpackPlugin from 'fork-ts-checker-webpack-plugin';
// @ts-ignore — no type declarations available
import ReplaceInFileWebpackPlugin from 'replace-in-file-webpack-plugin';
import path from 'path';
import { fileURLToPath } from 'url';
import webpack, { Configuration } from 'webpack';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const appRoot = path.resolve(__dirname, '../..');
const outDir = path.resolve(appRoot, 'dist-standalone');
const hostDir = path.resolve(appRoot, 'src/host');

const baseUrl = process.env.INVOICEX_BASE_URL ?? '';
const basename = process.env.INVOICEX_BASENAME ?? '/';
const publicPath = basename.endsWith('/') ? basename : `${basename}/`;

const config = (): Configuration => ({
  mode: 'production',
  entry: { standalone: './src/standalone.tsx' },
  devtool: false,

  output: {
    path: outDir,
    filename: '[name].js',
    chunkFilename: '[name].[contenthash:8].js',
    publicPath,
    clean: true,
  },

  // No externals: @grafana/ui, @grafana/data, react, ... are bundled.
  // @grafana/runtime must never be reachable; src/host/index swaps to ./standalone below.
  performance: { hints: false },

  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.jsx'],
    alias: { '@': path.resolve(appRoot, 'src') },
  },

  module: {
    rules: [
      // @grafana/ui ships .mjs that imports extensionless subpaths (react-use/lib/*, rc-picker/lib/*)
      // and default-imports CJS with __esModule (react-use/lib/useMeasure). webpack types .mjs as strict
      // javascript/esm (default = whole module.exports); javascript/auto restores __esModule interop.
      { test: /\.m?js$/, type: 'javascript/auto', resolve: { fullySpecified: false } },
      {
        test: /\.[tj]sx?$/,
        exclude: /node_modules/,
        use: {
          loader: 'swc-loader',
          options: {
            jsc: {
              parser: { syntax: 'typescript', tsx: true },
              transform: { react: { runtime: 'automatic' } },
            },
          },
        },
      },
      { test: /\.css$/, use: ['style-loader', 'css-loader'] },
      { test: /\.s[ac]ss$/, use: ['style-loader', 'css-loader', 'sass-loader'] },
    ],
  },

  plugins: [
    // Host selection: ./grafana -> ./standalone, only for requests issued from src/host/.
    new webpack.NormalModuleReplacementPlugin(/^\.\/grafana$/, (resource: { context?: string; request: string }) => {
      if (path.resolve(resource.context ?? '') === hostDir) {
        resource.request = './standalone';
      }
    }),
    new webpack.DefinePlugin({
      INVOICEX_BASE_URL: JSON.stringify(baseUrl),
      INVOICEX_BASENAME: JSON.stringify(basename),
    }),
    new CopyWebpackPlugin({
      patterns: [
        {
          from: 'src/standalone.html',
          to: 'index.html',
          transform: (content: Buffer) => content.toString().replace(/%PUBLIC_PATH%/g, publicPath),
        },
        { from: 'node_modules/pdfjs-dist/build/pdf.worker.min.mjs', to: 'pdf.worker.min.js' },
      ],
    }),
    new ForkTsCheckerWebpackPlugin({
      async: false,
      typescript: { configFile: path.resolve(appRoot, 'tsconfig.json') },
    }),
    // Strip absolute local paths that pdfjs-dist embeds via createRequire()
    new ReplaceInFileWebpackPlugin([
      {
        dir: outDir,
        test: /\.js$/,
        rules: [{ search: /file:\/\/\/[^"]*?\/node_modules\//g, replace: './node_modules/' }],
      },
    ]),
  ],
});

export default config;
