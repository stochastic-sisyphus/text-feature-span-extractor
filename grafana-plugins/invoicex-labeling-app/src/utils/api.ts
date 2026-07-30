// PostgREST is the sole HTTP face. All writes go through /api/rpc/<fn>;
// all reads go through /api/<table> with query params.
// /api/ strips the prefix → PostgREST sees its native route roots.

export const API_BASE = '/api';

export type SortAxis = 'recent' | 'priority' | 'confidence' | 'alphabetical';

// ── Enqueue helpers ──────────────────────────────────────────────────────────
// PostgREST void RPCs return 204 No Content. These helpers return a synthetic
// { queued: true } so callers get a uniform acknowledgement shape.

export interface EnqueueResponse {
  queued: true;
}

async function rpcPost(path: string, body?: Record<string, unknown>): Promise<void> {
  const init: RequestInit = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  };
  const response = await fetch(`${API_BASE}/rpc/${path}`, init);
  if (!response.ok) {
    let message = `RPC ${path} failed (${response.status})`;
    const ct = response.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      const data = await response.json();
      message = data?.message || data?.detail || message;
    }
    throw new Error(message);
  }
}

// ── Pipeline enqueue ─────────────────────────────────────────────────────────

export async function wakePipeline(): Promise<EnqueueResponse> {
  await rpcPost('fn_pgqueuer_enqueue', { entrypoint: 'sharepoint_wake' });
  return { queued: true };
}

export async function reextractDoc(sha256: string): Promise<EnqueueResponse> {
  await rpcPost('fn_pgqueuer_enqueue', {
    entrypoint: 'reextract',
    payload: btoa(JSON.stringify({ sha: sha256 })),
    dedupe_key: sha256,
  });
  return { queued: true };
}

export async function triggerRetrain(): Promise<EnqueueResponse> {
  await rpcPost('fn_pgqueuer_enqueue', { entrypoint: 'retrain' });
  return { queued: true };
}

// ── Labeling actions ─────────────────────────────────────────────────────────
// Four explicit user actions; all POST to /api/rpc/submit_label.
// action ∈ { 'approve', 'correct', 'not_in_document', 'reject' }

export interface SubmitLabelParams {
  doc_id: string;
  field: string;
  action: 'approve' | 'correct' | 'not_in_document' | 'reject';
  payload?: Record<string, unknown>;
  submitted_by?: string;
}

export interface LabelRow {
  id: number;
  doc_id: string;
  field: string;
  action: string;
  submitted_by: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}

export async function submitLabel(params: SubmitLabelParams): Promise<LabelRow> {
  const body = {
    p_doc_id: params.doc_id,
    p_field: params.field,
    p_action: params.action,
    p_payload: params.payload ?? {},
    ...(params.submitted_by !== undefined ? { p_submitted_by: params.submitted_by } : {}),
  };
  const response = await fetch(`${API_BASE}/rpc/submit_label`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `submit_label failed (${response.status})`;
    const ct = response.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      const data = await response.json();
      message = data?.message || data?.detail || message;
    }
    throw new Error(message);
  }
  return response.json();
}

// ── Stale count ───────────────────────────────────────────────────────────────

export interface StaleCountResponse {
  count: number;
}

export async function getStaleCount(): Promise<StaleCountResponse> {
  // PostgREST GET on a void-returning RPC that returns json.
  const response = await fetch(`${API_BASE}/rpc/stale_count`);
  if (!response.ok) {
    throw new Error(`stale_count failed (${response.status})`);
  }
  return response.json();
}

// ── Config (contract schema) ─────────────────────────────────────────────────

// Config keys live in the contract_schema payload, written there by a future
// global-threshold UI. No writer exists yet — confidence_auto_approve /
// confidence_thresholds are currently only in Python Settings (env vars).
// Fields are optional so the hook falls through to its hardcoded defaults until
// the backend exposes them in the payload.
export interface Config {
  confidence_auto_approve?: number;
  confidence_thresholds?: {
    high: number;
    medium: number;
    low: number;
  };
}

let cachedConfig: Config | null = null;

export async function fetchConfig(): Promise<Config> {
  if (cachedConfig) {
    return cachedConfig;
  }
  const response = await fetch(`${API_BASE}/rpc/latest_contract_schema`);
  if (!response.ok) {
    throw new Error(`latest_contract_schema failed (${response.status})`);
  }
  // latest_contract_schema returns the full DB row {sequence_id, version, payload, created_at}.
  // Config fields live inside payload — mirror loadSchemaPayload() in SchemaEditorPage.
  const row = await response.json();
  cachedConfig = (typeof row?.payload === 'object' && row.payload !== null ? row.payload : {}) as Config;
  return cachedConfig;
}

/**
 * Returns the run_id of the most recent RETRAIN append in contract_schema,
 * or null if no retrain has ever completed. Schema-edit appends carry no
 * run_id and are ignored by the backing SQL — this is the correct discriminator
 * for retrain completion vs. schema edits (both bump version).
 * Never cached: each call is a fresh read.
 */
export async function getLatestModelRunId(): Promise<string | null> {
  const response = await fetch(`${API_BASE}/rpc/latest_model_run_id`);
  if (!response.ok) {
    throw new Error(`latest_model_run_id failed (${response.status})`);
  }
  const val = await response.json();
  return typeof val === 'string' ? val : null;
}

// ── Document filename ────────────────────────────────────────────────────────
// Derived from get_document RPC — PostgREST table routes are not reachable
// from the plugin (nginx routes /api/* to Grafana, only /api/rpc/* to PostgREST).

export async function getDocumentFilename(sha256: string): Promise<string | null> {
  try {
    const doc = await getDocument(sha256);
    return doc?.filename ?? null;
  } catch {
    return null;
  }
}

// ── Queue list ───────────────────────────────────────────────────────────────

export interface QueueItem {
  doc_id: string;
  sha256: string;
  imported_at: string | null;
  filename: string | null;
  source: string | null;
  source_id: string | null;
  field_count: number;
}

export interface QueueListResponse {
  items: QueueItem[];
  total_count: number;
  next_cursor: string | null;
}

export async function queueList(params: {
  sort?: SortAxis;
  filter?: string;
  page_size?: number;
  cursor?: string | null;
  searchQuery?: string;
}): Promise<QueueListResponse> {
  const body: Record<string, unknown> = {
    p_sort: params.sort ?? 'recent',
    p_filter: params.filter ?? 'ready',
    p_page_size: params.page_size ?? 50,
  };
  if (params.cursor) {
    body['p_cursor'] = params.cursor;
  }
  if (params.searchQuery) {
    body['p_search'] = params.searchQuery;
  }
  const response = await fetch(`${API_BASE}/rpc/queue_list`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `queue_list failed (${response.status})`;
    const ct = response.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      const data = await response.json();
      message = data?.message || data?.detail || message;
    }
    throw new Error(message);
  }
  return response.json();
}

// ── Get document detail ──────────────────────────────────────────────────────

export interface DocToken {
  token_id: string;
  page_idx: number;
  token_idx: number;
  text: string;
  bbox_norm_x0: number;
  bbox_norm_y0: number;
  bbox_norm_x1: number;
  bbox_norm_y1: number;
}

export interface DocPage {
  page_idx: number;
  tokens: DocToken[];
}

export interface DocData {
  sha256: string;
  pages: DocPage[];
}

export interface DocumentDetail {
  doc_id: string;
  sha256: string;
  filename: string | null;
  source_id: string | null;
  drive_id: string | null;
  doc: DocData | null;
  pages: number;
  predictions: Record<string, unknown> | null;
  /** Corpus-wide candidate spans from docs.payload->'candidates' (init.sql:396).
   * Null for docs that have not been extracted yet. */
  all_candidates?: Array<import('./types').CandidateSpan> | null;
}

export async function getDocument(sha256: string): Promise<DocumentDetail | null> {
  const response = await fetch(`${API_BASE}/rpc/get_document`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify({ p_sha: sha256 }),
  });
  if (!response.ok) {
    if (response.status === 404) { return null; }
    let message = `get_document failed (${response.status})`;
    const ct = response.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      const data = await response.json();
      message = data?.message || data?.detail || message;
    }
    throw new Error(message);
  }
  return response.json();
}

// ── Next document for auto-advance ───────────────────────────────────────────
// get_next_doc walks docs by created_at DESC from the current doc.
// No status or schema filter — every doc is a candidate.
// Returns null when current doc is oldest or table is empty.

export interface NextDocResponse {
  sha256: string;
  doc_id: string;
  filename: string | null;
}

export async function getNextDoc(currentSha: string): Promise<NextDocResponse | null> {
  const response = await fetch(`${API_BASE}/rpc/get_next_doc`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify({ p_current_sha: currentSha }),
  });
  if (!response.ok) {
    throw new Error(`get_next_doc failed (${response.status})`);
  }
  // PostgREST returns null body when LIMIT 1 finds no row
  const text = await response.text();
  if (!text || text === 'null') { return null; }
  return JSON.parse(text);
}

// ── Model status ─────────────────────────────────────────────────────────────

export interface ModelCurrent {
  version: number;
  run_id: string;
  trained_at: string;
  /** n_docs: document groups used in training (sourced from n_groups in payload) */
  n_docs: number;
  n_samples: number;
  n_positive: number;
  n_features: number;
  mlflow_model_prefix: string;
}

export interface ModelHistoryEntry {
  version: number;
  run_id: string;
  trained_at: string;
  /** n_docs: document groups used in training (sourced from n_groups in payload) */
  n_docs: number;
  n_positive: number;
}

export interface ModelRecentLabels {
  total: number;
  approve_count: number;
  correct_count: number;
  reject_count: number;
}

export interface ModelStatusResponse {
  current: ModelCurrent | null;
  history: ModelHistoryEntry[];
  recent_labels: ModelRecentLabels;
}

/**
 * Returns the current model state projection from contract_schema + labels.
 * current === null means no retrain has ever completed (first-run / empty state).
 * Never cached — each call is a fresh read.
 */
export async function modelStatus(): Promise<ModelStatusResponse | null> {
  const response = await fetch(`${API_BASE}/rpc/model_status`);
  if (!response.ok) {
    throw new Error(`model_status failed (${response.status})`);
  }
  const text = await response.text();
  if (!text || text === 'null') { return null; }
  return JSON.parse(text) as ModelStatusResponse;
}

// ── Fetch with retry ─────────────────────────────────────────────────────────

export async function fetchWithRetry(
  input: RequestInfo | URL,
  init?: RequestInit,
  opts?: { retries?: number; baseDelay?: number }
): Promise<Response> {
  const retries = opts?.retries ?? 1;
  const baseDelay = opts?.baseDelay ?? 1000;

  if (init?.method?.toUpperCase() === 'POST') {
    return fetch(input, init);
  }

  for (let attempt = 0; attempt <= retries; attempt++) {
    let response: Response;
    try {
      response = await fetch(input, init);
    } catch (err) {
      if (attempt === retries) {
        throw err;
      }
      await new Promise<void>((resolve) =>
        setTimeout(resolve, baseDelay * Math.pow(2, attempt))
      );
      continue;
    }
    const shouldRetry = [408, 429, 503, 504].includes(response.status);
    if (!shouldRetry || attempt === retries) {
      return response;
    }
    await new Promise<void>((resolve) =>
      setTimeout(resolve, baseDelay * Math.pow(2, attempt))
    );
  }
  throw new Error('fetchWithRetry: exhausted retries');
}

// ── Extraction results ────────────────────────────────────────────────────────

export interface ExtractionResultRow {
  filename: string | null;
  field: string;
  extracted_value: string | null;
  confidence: number | null;
  evaluator_version: string;
  evaluated_at: string;
  doc_id: string;
}

export async function fetchExtractionResults(): Promise<ExtractionResultRow[]> {
  const response = await fetch(`${API_BASE}/rpc/extraction_results_view`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: '{}',
  });
  if (!response.ok) {
    throw new Error(`extraction_results_view failed (${response.status})`);
  }
  return response.json();
}
