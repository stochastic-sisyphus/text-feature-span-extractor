import type { Host } from './types';
import { BASE_URL, BASENAME } from './env';

/** Standalone host: plain fetch against the standalone Go server. No Grafana, no auth scheme imposed. */
export const host: Host = {
  async currentUser() {
    try {
      const res = await fetch(`${BASE_URL}/whoami`);
      if (!res.ok) {
        return 'unknown';
      }
      const body = (await res.json()) as { user?: unknown };
      return typeof body.user === 'string' && body.user ? body.user : 'unknown';
    } catch {
      return 'unknown';
    }
  },

  async pdfBytes(sha256) {
    const res = await fetch(`${BASE_URL}/pdf/${sha256}`);
    if (!res.ok) {
      throw new Error(`PDF fetch failed (${res.status})`);
    }
    return res.arrayBuffer();
  },

  pdfWorkerSrc: `${BASENAME.replace(/\/$/, '')}/pdf.worker.min.js`,
};
