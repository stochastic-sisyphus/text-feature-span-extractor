// Minimal window polyfill for node test environment
// api.ts reads window.__INVOICEX_API_BASE__ at module load time
if (typeof window === 'undefined') {
  global.window = { __INVOICEX_API_BASE__: undefined };
}
