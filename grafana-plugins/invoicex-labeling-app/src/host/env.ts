// Build-time constants injected by webpack.standalone.config.ts (DefinePlugin).
// Absent in the plugin build / jest / tsc, hence the typeof guards.
declare const INVOICEX_BASE_URL: string | undefined;
declare const INVOICEX_BASENAME: string | undefined;

/** Origin/prefix of the standalone server routes (/pdf, /whoami). Default '' = same origin. */
export const BASE_URL: string = (typeof INVOICEX_BASE_URL !== 'undefined' ? INVOICEX_BASE_URL : '').replace(/\/$/, '');

/** Router basename / public path of the SPA. Default '/'. */
export const BASENAME: string = typeof INVOICEX_BASENAME !== 'undefined' ? INVOICEX_BASENAME : '/';
