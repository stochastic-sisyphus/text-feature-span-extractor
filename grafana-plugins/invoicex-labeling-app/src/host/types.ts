/**
 * Narrow waist between the labeling UI and whatever hosts it.
 * The UI never imports a host SDK directly; it imports `host` from './index'.
 */
export interface Host {
  /** Identity recorded as `submitted_by`. Must resolve (never reject); 'unknown' when not known. */
  currentUser(): Promise<string>;
  /** Raw PDF bytes for a document sha256. Rejects on failure. */
  pdfBytes(sha256: string): Promise<ArrayBuffer>;
  /** URL pdf.js loads its worker from. */
  pdfWorkerSrc: string;
}
