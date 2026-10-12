// Default (plugin build, tsc, jest) resolves to the Grafana host.
// webpack.standalone.config.ts swaps './grafana' -> './standalone' at build time.
export { host } from './grafana';
export type { Host } from './types';
