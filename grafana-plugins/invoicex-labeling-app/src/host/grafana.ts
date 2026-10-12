import { config, getBackendSrv } from '@grafana/runtime';
import { lastValueFrom } from 'rxjs';
import type { Host } from './types';

/** Grafana app-plugin host. The only file (besides module.ts) allowed to import @grafana/runtime. */
export const host: Host = {
  async currentUser() {
    const user = config.bootData.user;
    return user?.login || user?.email || 'unknown';
  },

  async pdfBytes(sha256) {
    // Same-origin resource handler of the Go backend; getBackendSrv carries Grafana auth.
    const obs = getBackendSrv().fetch<ArrayBuffer>({
      url: `/api/plugins/invoicex-labeling-app/resources/pdf/${sha256}`,
      responseType: 'arraybuffer',
    });
    const result = await lastValueFrom(obs);
    return result.data;
  },

  pdfWorkerSrc: '/public/plugins/invoicex-labeling-app/pdf.worker.min.js',
};
