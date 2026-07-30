// Shared TypeScript types used across pages.

export type ComputedFn = 'infer_currency' | 'concat_strip' | 'coalesce';

/** Legacy server-side mode param — kept for back-compat with deep links. */
export type QueueViewMode = 'needs_review' | 'all_processed' | 'all';

const QUEUE_VIEW_MODES: ReadonlySet<string> = new Set<QueueViewMode>([
  'needs_review',
  'all_processed',
  'all',
]);

/** Type-narrowing guard — use instead of `as QueueViewMode` casts. */
export function isQueueViewMode(v: unknown): v is QueueViewMode {
  return typeof v === 'string' && QUEUE_VIEW_MODES.has(v);
}

/** Filter chips for the queue page. Maps to ?filter= URL param. */
export type QueueFilter = 'ready' | 'pending' | 'failed' | 'labeled' | 'all';

const QUEUE_FILTERS: ReadonlySet<string> = new Set<QueueFilter>([
  'ready',
  'pending',
  'failed',
  'labeled',
  'all',
]);

export function isQueueFilter(v: unknown): v is QueueFilter {
  return typeof v === 'string' && QUEUE_FILTERS.has(v);
}

/** Map a QueueFilter to the backend mode param (legacy until backend adds ?filter= support). */
export function filterToMode(f: QueueFilter): QueueViewMode {
  if (f === 'ready') { return 'needs_review'; }
  if (f === 'labeled') { return 'all_processed'; }
  if (f === 'all') { return 'all'; }
  // 'pending' and 'failed' — backend will handle via ?filter= when available
  return 'all_processed';
}

export const BASE_TYPES = ['str', 'decimal', 'date', 'int'] as const;
export type BaseType = typeof BASE_TYPES[number];

export const NORMALIZERS = ['amount', 'date', 'id', 'name', 'passthrough', 'currency', 'email', 'phone'] as const;
export type Normalizer = typeof NORMALIZERS[number];

export const ANCHOR_FAMILIES = ['total', 'date', 'id', 'name', 'tax'] as const;
export type AnchorFamily = typeof ANCHOR_FAMILIES[number];

export interface FieldDetail {
  name: string;
  base_type: BaseType;
  normalizer: Normalizer;
  anchor_family: AnchorFamily | null;
  required: boolean;
  description: string;
  dataverse_column: string | null;
  computed: boolean;
  computed_fn: ComputedFn | null;
  computed_from: string[] | null;
  // Fallback emitted when extraction and computed_fn both yield null
  default_value?: string | null;
  // Vendor-corpus resolver flag — exactly one field in the schema should be true
  is_vendor?: boolean;
  // Tunables — keyword_proximal is user-editable; others managed by ML calibration
  confidence_threshold: number;
  importance: number;
  priority_bonus: number;
  keyword_proximal: boolean;
  // Lifecycle
  status?: 'active' | 'planned' | 'deprecated';
  deprecation_reason?: string | null;
}

// Discriminated union for prediction responses (v2+ envelope vs legacy bare dict).
// The prediction map's value type is intentionally `unknown` here — callers
// (LabelPage.tsx) cast to their local FieldPrediction after unwrapping the envelope.
export type PredictionEnvelope =
  | { status: 'pending'; reason: string }
  | { status: 'ready'; prediction: Record<string, unknown> };

/**
 * A single corpus-wide candidate span from docs.payload->'all_candidates'.
 * Serialized by queue.py from candidates_df rows (token_ids/token_indices stripped).
 * Consumed by the LabelPage candidate overlay — filtered per page_idx at render time.
 */
export interface CandidateSpan {
  raw_text: string;
  page_idx: number;
  bbox_norm_x0: number;
  bbox_norm_y0: number;
  bbox_norm_x1: number;
  bbox_norm_y1: number;
  cohesion_score?: number;
  candidate_id?: string;
}
