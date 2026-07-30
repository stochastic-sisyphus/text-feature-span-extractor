import CopyWebpackPlugin from 'copy-webpack-plugin';
import ForkTsCheckerWebpackPlugin from 'fork-ts-checker-webpack-plugin';
// @ts-ignore — no type declarations available
import ReplaceInFileWebpackPlugin from 'replace-in-file-webpack-plugin';
import path from 'path';
import { fileURLToPath } from 'url';
import { Configuration, ExternalItemFunctionData } from 'webpack';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const config = (env: { production?: boolean }): Configuration => {
  const isProduction = env.production === true;

  return {
    mode: isProduction ? 'production' : 'development',
    entry: './src/module.ts',
    devtool: isProduction ? 'source-map' : 'eval-source-map',

    performance: {
      hints: isProduction ? 'warning' : false,
      // pdfjs-dist ships a ~1MB worker and a ~400KB parser — both are
      // irreducible vendor assets.  The worker is a static copy loaded
      // out-of-band; the parser chunk is already code-split.  Exclude
      // them so genuine regressions in *our* code still trigger warnings.
      assetFilter(assetFilename: string) {
        if (assetFilename.endsWith('.map')) return false;
        if (assetFilename === 'pdf.worker.min.js') return false;
        if (/^\d+\.module\.js$/.test(assetFilename)) return false;
        return true;
      },
    },

    output: {
      path: path.resolve(__dirname, '../../dist'),
      filename: 'module.js',
      libraryTarget: 'amd',
      publicPath: '/public/plugins/invoicex-labeling-app/',
      devtoolModuleFilenameTemplate: isProduction
        ? 'webpack:///[namespace]/[resource-path]'
        : undefined,
    },

    externals: [
      'lodash',
      'react',
      'react-dom',

      '@grafana/data',
      '@grafana/ui',
      '@grafana/runtime',
      ({ request }: ExternalItemFunctionData, callback: (err?: Error | null, result?: string) => void) => {
        const prefix = 'grafana/';
        if (request?.startsWith(prefix)) {
          return callback(null, request.substring(prefix.length));
        }
        callback();
      },
    ],

    resolve: {
      extensions: ['.ts', '.tsx', '.js', '.jsx'],
      alias: {
        '@': path.resolve(__dirname, '../../src'),
      },
    },

    module: {
      rules: [
        {
          test: /\.[tj]sx?$/,
          exclude: /node_modules/,
          use: {
            loader: 'swc-loader',
            options: {
              jsc: {
                parser: {
                  syntax: 'typescript',
                  tsx: true,
                },
                transform: {
                  react: {
                    runtime: 'automatic',
                  },
                },
              },
            },
          },
        },
        {
          test: /\.css$/,
          use: ['style-loader', 'css-loader'],
        },
        {
          test: /\.s[ac]ss$/,
          use: ['style-loader', 'css-loader', 'sass-loader'],
        },
      ],
    },

    plugins: [
      new CopyWebpackPlugin({
        patterns: [
          { from: 'plugin.json', to: '.' },
          { from: 'README.md', to: '.', noErrorOnMissing: true },
          { from: 'CHANGELOG.md', to: '.', noErrorOnMissing: true },
          { from: 'LICENSE', to: '.', noErrorOnMissing: true },
          { from: 'img/', to: 'img/', noErrorOnMissing: true },
          { from: 'node_modules/pdfjs-dist/build/pdf.worker.min.mjs', to: 'pdf.worker.min.js' },
        ],
      }),
      new ForkTsCheckerWebpackPlugin({
        async: !isProduction,
        typescript: {
          configFile: path.resolve(__dirname, '../../tsconfig.json'),
        },
      }),
      // Strip absolute local paths that pdfjs-dist embeds via createRequire()
      ...(isProduction
        ? [
            new ReplaceInFileWebpackPlugin([
              {
                dir: path.resolve(__dirname, '../../dist'),
                test: /\.js$/,
                rules: [
                  {
                    search: /file:\/\/\/[^"]*?\/node_modules\//g,
                    replace: './node_modules/',
                  },
                ],
              },
            ]),
          ]
        : []),
    ],
  };
};

export default config;
