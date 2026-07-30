import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { css } from '@emotion/css';
import { GrafanaTheme2 } from '@grafana/data';
import { useStyles2, Button, Input, Badge, Spinner, Alert, Checkbox, Icon } from '@grafana/ui';
import { config, getBackendSrv } from '@grafana/runtime';
import { lastValueFrom } from 'rxjs';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import * as pdfjsLib from 'pdfjs-dist';
import { API_BASE, fetchWithRetry, getDocument, getNextDoc, queueList, reextractDoc, submitLabel } from '../utils/api';
import { getStatusColor, getConfidenceColor, formatConfidencePercent, COLORS } from '../utils/colors';
import { getLoadingContainerStyles, getErrorContainerStyles } from '../utils/styles';
import { useConfidenceThresholds } from '../utils/hooks';
import { type FieldDetail, type PredictionEnvelope, type CandidateSpan } from '../utils/types';

// Point to the worker file copied by CopyWebpackPlugin into dist/
pdfjsLib.GlobalWorkerOptions.workerSrc = '/public/plugins/invoicex-labeling-app/pdf.worker.min.js';

interface DocumentData {
  doc_id: string;
  sha256: string;
  pages: number;
  predictions?: Record<string, FieldPrediction>;
  predictionStatus?: 'ready' | 'pending';
  predictionPendingReason?: string;
  source_id: string | null;
  doc: import('../utils/api').DocData | null;
  filename: string | null;
  /** Corpus-wide candidate spans from docs.payload->'all_candidates', keyed by page_idx.
   * Wave A adds this to the SQL projection; optional here for backward-compat. */
  all_candidates?: CandidateSpan[];
}

interface FieldProvenance {
  page: number;
  bbox: number[]; // flat [x0, y0, x1, y1], normalized 0–1
  token_span: string[];
}

interface Candidate {
  raw_text: string;
  page_idx: number;
  bbox_norm_x0: number;
  bbox_norm_y0: number;
  bbox_norm_x1: number;
  bbox_norm_y1: number;
  total_score?: number;
}

interface FieldPrediction {
  value: string | null;
  confidence: number;
  status: 'PREDICTED' | 'ABSTAIN' | 'MISSING' | 'DEFAULT';
  provenance: FieldProvenance | null;
  raw_text: string | null;
  /** Candidate list from doc_evaluations. Secondary canvas hit-test layer:
   * clicking a candidate bbox selects that candidate's text as the correction
   * value when no Doc token matches at the click point. */
  candidates?: Candidate[];
  /** ISO timestamp from MAX(approvals.created_at) LEFT JOIN per (doc_id, field).
   * Non-null = field was approved; render greyed/locked. Threaded by the queue
   * projection (F7 backend, commit 268a804). The per-doc detail endpoint does not
   * yet thread this field — greying activates once that backend gap is closed. */
  approved_at?: string | null;
}

/**
 * Actions sent to the server's /label endpoint.
 * Must match the Literal type in api_models.LabelSubmission.action.
 */
type ServerLabelAction = 'correct' | 'not_applicable' | 'not_in_document' | 'reject';

/** Local-only UI states that never hit the server. */
type LocalAction = 'approve' | 'skip';

interface FieldLabel {
  field: string;
  action: ServerLabelAction | LocalAction;
  correct_value?: string | null;
  correct_bbox?: number[][];
}

interface Token {
  text: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}


interface DocumentReviewItem {
  doc_id: string | null;
  sha256: string;
  imported_at: string | null;
  field_count: number;
  filename: string | null;
  // Inference fields absent post-Wave-5A — kept optional for backward-compat
  lowest_confidence?: number | null;
  field_values?: Record<string, string | null> | null;
}

function shortDocId(docId: string | null | undefined): string {
  if (!docId) { return '—'; }
  return docId.replace(/^fs:/, '').slice(0, 8);
}

interface SidebarInvoiceItemProps {
  item: DocumentReviewItem;
  isActive: boolean;
  onClick: () => void;
}

function SidebarInvoiceItem({ item, isActive, onClick }: SidebarInvoiceItemProps) {
  const styles = useStyles2(getStyles);
  const docTitle = item.filename || `(unnamed: ${shortDocId(item.doc_id)})`;

  return (
    <div
      className={`${styles.sidebarNavItem} ${isActive ? styles.sidebarNavItemActive : ''}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } }}
    >
      <div className={styles.sidebarNavHead}>
        <span
          className={`${styles.sidebarNavId} ${isActive ? styles.sidebarNavIdActive : ''}`}
          title={`${docTitle} · ${item.doc_id}`}
        >
          {docTitle}
        </span>
      </div>
      <div className={styles.sidebarNavMeta}>
        <span className={styles.sidebarNavDate}>{shortDocId(item.doc_id)}</span>
        {item.imported_at && (
          <span className={styles.sidebarNavDate}>
            {new Date(item.imported_at).toLocaleDateString()}
          </span>
        )}
      </div>
    </div>
  );
}

interface LabelLogEntry {
  field: string;
  action: 'approve' | 'correct' | 'reject' | 'not_in_document';
  value?: string;
  ts: number;
}

const LABEL_LOG_MAX = 20;

function formatTimeAgo(ts: number): string {
  const diffSec = Math.floor((Date.now() - ts) / 1000);
  if (diffSec < 5) { return 'just now'; }
  if (diffSec < 60) { return `${diffSec}s ago`; }
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) { return `${diffMin}m ago`; }
  return `${Math.floor(diffMin / 60)}h ago`;
}

/** Normalize a bbox read from sessionStorage draft — handles legacy flat shape.
 * Legacy: [x0,y0,x1,y1]   → [[x0,y0,x1,y1]]
 * New:    [[x0,y0,x1,y1],…] → as-is
 * Absent/invalid:           → null
 */
function normalizeDraftBbox(raw: unknown): number[][] | null {
  if (!Array.isArray(raw) || raw.length === 0) { return null; }
  if (typeof raw[0] === 'number') { return [raw as number[]]; }
  if (Array.isArray(raw[0])) { return raw as number[][]; }
  return null;
}

export function LabelPage() {
  const styles = useStyles2(getStyles);
  const navigate = useNavigate();
  const location = useLocation();
  const { documentId } = useParams<{ documentId: string }>();

  // Horizontal split between PDF panel and fields panel (percentage of mainCol width)
  const [splitPct, setSplitPct] = useState(62);
  const [isDragging, setIsDragging] = useState(false);
  const mainColRef = useRef<HTMLDivElement>(null);

  const handleSplitterMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);

    const onMouseMove = (ev: MouseEvent) => {
      const col = mainColRef.current;
      if (!col) { return; }
      const rect = col.getBoundingClientRect();
      const pct = ((ev.clientX - rect.left) / rect.width) * 100;
      setSplitPct(Math.min(80, Math.max(35, pct)));
    };

    const onMouseUp = () => {
      setIsDragging(false);
      window.document.removeEventListener('mousemove', onMouseMove);
      window.document.removeEventListener('mouseup', onMouseUp);
    };

    window.document.addEventListener('mousemove', onMouseMove);
    window.document.addEventListener('mouseup', onMouseUp);
  }, []);
  // Support both ?filter= (new) and ?mode= (legacy back-compat)
  const _searchParams = new URLSearchParams(location.search);
  const queueFilter = _searchParams.get('filter') ?? null;
  const queueMode = queueFilter
    ? (queueFilter === 'ready' ? 'needs_review' : queueFilter === 'all' ? 'all' : 'all_processed')
    : (_searchParams.get('mode') ?? 'needs_review');
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  const offscreenCanvasRefs = useRef<Map<number, HTMLCanvasElement>>(new Map());
  // Logical (CSS-pixel) dimensions for each rendered page — set during PDF rasterize,
  // consumed by compositePageToCanvas to assign canvas.style.width/height so retina
  // displays get a crisp 1:1 buffer-to-device-pixel mapping instead of a blurry upscale.
  const pageCssDimsRef = useRef<Map<number, { w: number; h: number }>>(new Map());
  const renderTasksRef = useRef<Map<HTMLCanvasElement, { cancel: () => void; promise: Promise<void> }>>(new Map());
  const pendingRendersRef = useRef<Map<HTMLCanvasElement, Promise<void>>>(new Map());
  const rafRef = useRef<number | null>(null);
  const compositeRef = useRef<(pageNum: number) => void>(() => {});

  const [document, setDocument] = useState<DocumentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // D.1: per-field validation errors. Server 422s with detail mentioning the
  // bad field land here; UI renders them inline next to the field, not as a
  // top-page banner. Cleared at submit-start so stale errors never persist.
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [labels, setLabels] = useState<Record<string, FieldLabel>>({});
  const [corrections, setCorrections] = useState<Record<string, { value: string; bbox: number[][] | null }>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitFeedback, setSubmitFeedback] = useState<string | null>(null);
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfDoc, setPdfDoc] = useState<pdfjsLib.PDFDocumentProxy | null>(null);
  const [pdfScale, setPdfScale] = useState(1.0);
  const pdfScaleRef = useRef(1.0);
  useEffect(() => { pdfScaleRef.current = pdfScale; }, [pdfScale]);
  const [pdfZoomMode, setPdfZoomMode] = useState<'fit' | 'manual'>('fit');
  const [tokensByPage, setTokensByPage] = useState<Record<number, Token[]>>({});
  const [hoveredToken, setHoveredToken] = useState<{ page: number; idx: number } | null>(null);
  const [selectedTokens, setSelectedTokens] = useState<Map<number, Set<number>>>(new Map());
  const [schemaFields, setSchemaFields] = useState<FieldDetail[]>([]);
  const [lastClickedToken, setLastClickedToken] = useState<string | null>(null);
  const [showPredictions, setShowPredictions] = useState<boolean>(true);
  const { autoApproveThreshold, mediumThreshold } = useConfidenceThresholds();

  const [labelLog, setLabelLog] = useState<LabelLogEntry[]>([]);

  const appendToLog = useCallback((entry: Omit<LabelLogEntry, 'ts'>) => {
    setLabelLog(prev => {
      const next = [{ ...entry, ts: Date.now() }, ...prev];
      return next.length > LABEL_LOG_MAX ? next.slice(0, LABEL_LOG_MAX) : next;
    });
  }, []);

  const fitPdfToWidth = useCallback(async (pdf: pdfjsLib.PDFDocumentProxy | null = pdfDoc) => {
    const container = containerRef.current;
    if (!pdf || !container) { return; }
    const page = await pdf.getPage(1);
    const viewport = page.getViewport({ scale: 1 });
    const availableWidth = Math.max(container.clientWidth - 24, 320);
    const nextScale = availableWidth / (viewport.width * 1.5);
    setPdfScale(Number(Math.min(1.4, Math.max(0.45, nextScale)).toFixed(2)));
  }, [pdfDoc]);

  const DRAFT_KEY = useMemo(() => `invoicex-draft-${documentId}`, [documentId]);
  const draftRef = useRef<{ labels: Record<string, FieldLabel>; corrections: Record<string, { value: string; bbox: number[][] | null }> }>({ labels: {}, corrections: {} });
  const [showDraftBanner, setShowDraftBanner] = useState(false);

  // Re-extract (this document) state
  const [reextracting, setReextracting] = useState(false);
  const [reextractResult, setReextractResult] = useState<{message: string; severity: 'success' | 'info' | 'error'} | null>(null);
  // Track the active re-extract run so polling no-ops if user navigates away

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarItems, setSidebarItems] = useState<DocumentReviewItem[]>([]);
  const sidebarItemsRef = useRef<DocumentReviewItem[]>([]);
  useEffect(() => { sidebarItemsRef.current = sidebarItems; }, [sidebarItems]);
  const [sidebarLoading, setSidebarLoading] = useState(false);
  const [sidebarError, setSidebarError] = useState<string | null>(null);
  const [sidebarFilter, setSidebarFilter] = useState('');

  // Stable refs for keyboard handler — updated each render so handler never
  // closes over stale state without needing the handler to re-register.
  const labelsRef = useRef<Record<string, FieldLabel>>({});
  useEffect(() => { labelsRef.current = labels; }, [labels]);
  const correctionsRef = useRef<Record<string, { value: string; bbox: number[][] | null }>>({});
  useEffect(() => { correctionsRef.current = corrections; }, [corrections]);
  const documentRef = useRef<DocumentData | null>(null);
  useEffect(() => { documentRef.current = document; }, [document]);
  const visibleFieldsRef = useRef<string[]>([]);
  // submitLabelsRef is set after submitLabels is defined below
  const submitLabelsRef = useRef<() => Promise<void>>(async () => {});
  // Refs for keyboard UX: scroll-into-view on j/k, focus correction input on 'c'
  const fieldsListRef = useRef<HTMLDivElement>(null);
  const fieldCardRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const correctionInputRefs = useRef<Map<string, HTMLInputElement>>(new Map());

  // Fetch schema fields from latest_contract_schema RPC on mount.
  // The contract_schema table payload carries the field definitions as
  // {fields: {[name]: FieldDetail, ...}}. Falls back to empty list on error.
  useEffect(() => {
    const fetchSchemaFields = async () => {
      try {
        const response = await fetchWithRetry(`${API_BASE}/rpc/latest_contract_schema`);
        if (!response.ok) {throw new Error('Failed to fetch field schema');}
        const data = await response.json();
        // latest_contract_schema returns the contract_schema row: {sequence_id, version, payload, created_at}
        // payload.field_definitions is the field map; flatten to array for the UI.
        const fieldsMap: Record<string, FieldDetail> = data?.payload?.field_definitions ?? {};
        const fieldsArr: FieldDetail[] = Object.entries(fieldsMap).map(([name, def]) => ({
          ...(def as FieldDetail),
          name,
        }));
        setSchemaFields(fieldsArr);
      } catch (err) {
        console.error('Field schema fetch error:', err);
        setSchemaFields([]);
      }
    };

    fetchSchemaFields();
  }, []);

  // Sidebar fetch — callable from mount/queueMode-change and from the
  // "couldn't refresh" pill so a transient failure doesn't strand the user.
  const fetchSidebar = useCallback(async (): Promise<DocumentReviewItem[]> => {
    setSidebarLoading(true);
    setSidebarError(null);
    try {
      // Map legacy queueMode to queue_list filter param.
      const filter = queueMode === 'needs_review' ? 'ready' : 'all';
      const data = await queueList({ sort: 'recent', filter, page_size: 50 });
      const items: DocumentReviewItem[] = (data.items || []).map((item) => ({
        doc_id: item.doc_id,
        sha256: item.sha256,
        imported_at: item.imported_at,
        field_count: item.field_count,
        filename: item.filename,
      }));
      items.sort((a, b) => {
        if (!a.imported_at && !b.imported_at) { return 0; }
        if (!a.imported_at) { return 1; }
        if (!b.imported_at) { return -1; }
        return b.imported_at.localeCompare(a.imported_at);
      });
      setSidebarItems(items);
      return items;
    } catch (err) {
      setSidebarError(err instanceof Error ? err.message : 'Failed to load queue');
      return sidebarItemsRef.current; // fall back to stale list on transient failure
    } finally {
      setSidebarLoading(false);
    }
  }, [queueMode]);

  useEffect(() => {
    fetchSidebar();
  }, [fetchSidebar]);

  // Restore draft from sessionStorage on mount
  useEffect(() => {
    // Reset per-document draft state on doc change so a previous doc's
    // corrections/labels can't bleed into the next document.
    setLabels({});
    setCorrections({});
    setShowDraftBanner(false);
    try {
      const saved = sessionStorage.getItem(DRAFT_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.labels && Object.keys(parsed.labels).length > 0) {
          // Normalize any legacy flat-bbox stored in FieldLabel.correct_bbox
          const normalizedLabels: Record<string, FieldLabel> = {};
          for (const [k, v] of Object.entries(parsed.labels as Record<string, FieldLabel>)) {
            const bb = (v as FieldLabel).correct_bbox;
            normalizedLabels[k] = {
              ...v,
              correct_bbox: normalizeDraftBbox(bb) ?? undefined,
            };
          }
          setLabels(normalizedLabels);
          setShowDraftBanner(true);
        }
        if (parsed.corrections && Object.keys(parsed.corrections).length > 0) {
          // Normalize any legacy flat-bbox stored in corrections
          const normalizedCorrections: Record<string, { value: string; bbox: number[][] | null }> = {};
          for (const [k, v] of Object.entries(parsed.corrections as Record<string, { value: string; bbox: unknown }>)) {
            normalizedCorrections[k] = { value: v.value, bbox: normalizeDraftBbox(v.bbox) };
          }
          setCorrections(normalizedCorrections);
        }
      }
    } catch (_) {}
  }, [DRAFT_KEY]);

  useEffect(() => {
    if (documentId) {
      fetchDocument(documentId);
    }
  }, [documentId]);

  // Debounced draft auto-save
  useEffect(() => {
    draftRef.current = { labels, corrections };
    if (Object.keys(labels).length === 0 && Object.keys(corrections).length === 0) {return;}
    const timer = setTimeout(() => {
      sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draftRef.current));
    }, 300);
    return () => clearTimeout(timer);
  }, [labels, corrections, DRAFT_KEY]);

  // Auto-clear token click affordance after 2s
  useEffect(() => {
    if (!lastClickedToken) {return;}
    const t = setTimeout(() => setLastClickedToken(null), 2000);
    return () => clearTimeout(t);
  }, [lastClickedToken]);

  // Keyboard shortcuts — scoped to this page, ignored when focus is in an input.
  // a = approve, r = reject (wrong), c = correct (fix), n = not_in_document
  // j/ArrowDown = next field, k/ArrowUp = prev field
  // Enter = submit review, Backspace = undo label on selected field
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable) { return; }

      const getActiveField = (): string | null => {
        if (selectedField) { return selectedField; }
        return visibleFieldsRef.current.find(f => !labelsRef.current[f]) ?? null;
      };

      switch (e.key) {
        case 'a': {
          const field = getActiveField();
          if (field) {
            setSelectedField(field);
            setLabels(prev => ({ ...prev, [field]: { field, action: 'approve' } }));
          }
          break;
        }
        case 'r': {
          const field = getActiveField();
          if (field) {
            setSelectedField(field);
            const predValue = documentRef.current?.predictions?.[field]?.value ?? undefined;
            setLabels(prev => ({ ...prev, [field]: { field, action: 'reject', correct_value: predValue } }));
          }
          break;
        }
        case 'c': {
          const field = getActiveField();
          if (field) {
            setSelectedField(field);
            const correction = correctionsRef.current[field];
            setLabels(prev => ({
              ...prev,
              [field]: {
                field,
                action: 'correct',
                correct_value: correction?.value || '',
                correct_bbox: correction?.bbox || undefined,
              }
            }));
            // Focus the correction input so the user can type immediately
            requestAnimationFrame(() => {
              correctionInputRefs.current.get(field)?.focus();
            });
          }
          break;
        }
        case 'n': {
          const field = getActiveField();
          if (field) {
            setSelectedField(field);
            setLabels(prev => ({ ...prev, [field]: { field, action: 'not_in_document', correct_value: null } }));
          }
          break;
        }
        case 'j':
        case 'ArrowDown': {
          e.preventDefault();
          const fields = visibleFieldsRef.current;
          if (fields.length === 0) { break; }
          const idx = selectedField ? fields.indexOf(selectedField) : -1;
          const next = fields[Math.min(idx + 1, fields.length - 1)];
          if (next) { setSelectedField(next); }
          break;
        }
        case 'k':
        case 'ArrowUp': {
          e.preventDefault();
          const fields = visibleFieldsRef.current;
          if (fields.length === 0) { break; }
          const idx = selectedField ? fields.indexOf(selectedField) : fields.length;
          const prev = fields[Math.max(idx - 1, 0)];
          if (prev) { setSelectedField(prev); }
          break;
        }
        case 'Enter': {
          e.preventDefault();
          void submitLabelsRef.current();
          break;
        }
        case 'Backspace': {
          const field = getActiveField();
          if (field && labelsRef.current[field]) {
            setLabels(prev => {
              const next = { ...prev };
              delete next[field];
              return next;
            });
            setCorrections(prev => {
              const next = { ...prev };
              delete next[field];
              return next;
            });
            setSelectedTokens(new Map());
          }
          break;
        }
        default:
          break;
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedField]);

  // Load PDF via the Go backend resource handler (/api/plugins/invoicex-labeling-app/resources/pdf/{sha}).
  // Same-origin fetch — no CORS negotiation. getBackendSrv carries Grafana auth automatically.
  useEffect(() => {
    if (!document?.sha256) {return;}

    const loadPdf = async () => {
      setPdfLoading(true);
      try {
        const obs = getBackendSrv().fetch<ArrayBuffer>({
          url: `/api/plugins/invoicex-labeling-app/resources/pdf/${document.sha256}`,
          responseType: 'arraybuffer',
        });
        const result = await lastValueFrom(obs);
        const pdf = await pdfjsLib.getDocument({ data: result.data }).promise;
        canvasRefs.current = new Array(pdf.numPages).fill(null);
        setPdfDoc(pdf);
      } catch (err) {
        console.error('PDF load error:', err);
        setPdfDoc(null);
      } finally {
        setPdfLoading(false);
      }
    };

    loadPdf();

    // Capture ref values at effect-run time so the cleanup closure sees a stable snapshot
    const renderTasks = renderTasksRef.current;
    const offscreenCanvases = offscreenCanvasRefs.current;
    const cssDims = pageCssDimsRef.current;

    return () => {
      setPdfDoc(prev => {
        prev?.destroy();
        return null;
      });
      renderTasks.forEach(task => task.cancel());
      renderTasks.clear();
      // Release bitmap memory before dropping Map references — GC cannot reclaim
      // canvas backing stores immediately; zeroing width/height does it eagerly.
      offscreenCanvases.forEach(c => { c.width = 0; c.height = 0; });
      offscreenCanvases.clear();
      cssDims.clear();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [document?.sha256]);

  // Token overlay: build per-page token map from the persisted Doc JSON.
  // Doc spatial geometry is now stored in docs.payload->'doc' and projected
  // by get_document — no re-running the pipeline needed.
  useEffect(() => {
    if (!document?.doc?.pages) {
      setTokensByPage({});
      return;
    }
    const byPage: Record<number, Array<{ text: string; x0: number; y0: number; x1: number; y1: number }>> = {};
    for (const page of document.doc.pages) {
      byPage[page.page_idx] = page.tokens.map(t => ({
        text: t.text,
        x0: t.bbox_norm_x0,
        y0: t.bbox_norm_y0,
        x1: t.bbox_norm_x1,
        y1: t.bbox_norm_y1,
      }));
    }
    setTokensByPage(byPage);
  // document.doc.pages is derived from document — re-reading on sha256 change is correct.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [document?.sha256]);

  // Token overlays are now DOM divs (position:absolute) — no canvas drawing needed.
  // This stub satisfies compositePageToCanvas's call site without canvas mutations.
  const drawTokenOverlaysForPage = useCallback((_ctx: CanvasRenderingContext2D, _w: number, _h: number, _page: number) => {
    // no-op: tokens rendered via DOM overlays, not canvas
  }, []);

  // Draw prediction bboxes for a given page
  const drawPredictionOverlaysForPage = useCallback((ctx: CanvasRenderingContext2D, canvasWidth: number, canvasHeight: number, pageNum: number) => {
    if (!showPredictions || !document) { return; }

    ctx.save();

    const predictions = document.predictions ?? {};
    for (const [fieldName, prediction] of Object.entries(predictions)) {
      const prov = prediction.provenance;
      if (!prov) { continue; }
      if (prov.page !== pageNum) { continue; }
      const bbox = prov.bbox;
      if (!bbox || bbox.length < 4) { continue; }
      // Skip zero bboxes (DEFAULT, derived, computed fields with no real location)
      if (!bbox.some((v: number) => v !== 0)) { continue; }

      const [x0, y0, x1, y1] = bbox;
      const x = x0 * canvasWidth;
      const y = y0 * canvasHeight;
      const w = (x1 - x0) * canvasWidth;
      const h = (y1 - y0) * canvasHeight;

      // Deterministic hue from field name
      let hash = 0;
      for (let i = 0; i < fieldName.length; i++) {
        hash = (hash * 31 + fieldName.charCodeAt(i)) >>> 0;
      }
      const hue = hash % 360;
      const color = `hsl(${hue}, 70%, 50%)`;

      // Opacity scaled by confidence, clamped to [0.3, 1.0]
      const opacity = Math.min(1.0, Math.max(0.3, prediction.confidence));

      ctx.globalAlpha = opacity;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 2]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      // Label: "FieldName XX%"
      const label = `${fieldName} ${Math.round(prediction.confidence * 100)}%`;
      ctx.font = '10px sans-serif';
      ctx.globalAlpha = Math.min(1.0, opacity + 0.2);
      ctx.fillStyle = color;
      ctx.fillText(label, x + 2, y > 12 ? y - 2 : y + 10);
    }

    ctx.restore();
  }, [showPredictions, document]);

  // Composite: copy offscreen PDF render + draw overlays onto visible canvas (Bug 1 fix)
  const compositePageToCanvas = useCallback((pageNum: number) => {
    const canvas = canvasRefs.current[pageNum];
    const offscreen = offscreenCanvasRefs.current.get(pageNum);
    if (!canvas || !offscreen) {return;}

    const ctx = canvas.getContext('2d');
    if (!ctx) {return;}

    canvas.width = offscreen.width;
    canvas.height = offscreen.height;
    // Apply logical CSS dimensions so the browser displays the physical buffer at
    // device-pixel resolution (crisp on retina) rather than stretching it.
    const cssDims = pageCssDimsRef.current.get(pageNum);
    if (cssDims) {
      // Set only width — let height follow the canvas's natural aspect ratio.
      // Setting height inline while max-width CSS can shrink the width breaks aspect ratio.
      canvas.style.width = `${cssDims.w}px`;
      canvas.style.height = '';
    }
    ctx.drawImage(offscreen, 0, 0);
    drawTokenOverlaysForPage(ctx, canvas.width, canvas.height, pageNum);
    // Prediction overlays moved to SVG layer in JSX — no canvas draw here.
  }, [drawTokenOverlaysForPage]);
  compositeRef.current = compositePageToCanvas;

  // Render PDF page to offscreen canvas (Bug 3 fix: render cancellation + race sentinel)
  const renderPageToOffscreen = useCallback(async (pdf: pdfjsLib.PDFDocumentProxy, pageNum: number) => {
    // Resolve the offscreen canvas BEFORE any await so the per-canvas guard is
    // in place before we check.
    let offscreen = offscreenCanvasRefs.current.get(pageNum);
    if (!offscreen) {
      offscreen = window.document.createElement('canvas');
      offscreenCanvasRefs.current.set(pageNum, offscreen);
    }

    const capturedOffscreen = offscreen;

    // Snapshot the predecessor BEFORE we build renderPromise. Each concurrent
    // caller chains off whoever was last registered at the time it entered —
    // not off a single A that all callers captured at entry (which is the
    // original race: B and C both awaited A, then both raced into page.render).
    const predecessor = pendingRendersRef.current.get(capturedOffscreen);

    const renderPromise = (async () => {
      // Wait for the direct predecessor (if any) to complete or be cancelled.
      // Because we set pendingRendersRef synchronously below, every new caller
      // sees the latest entry — so the chain is always: A → B → C, never a fork.
      if (predecessor) {
        try { await predecessor; } catch (_) { /* cancelled — proceed */ }
      }

      // Cancel any in-flight render on THIS canvas (not just this page index).
      // Keying on the canvas DOM node catches the case where the same canvas is
      // reused across page switches or concurrent scale-change renders.
      const existingTask = renderTasksRef.current.get(capturedOffscreen);
      if (existingTask) {
        existingTask.cancel();
        try { await existingTask.promise; } catch (_) { /* RenderingCancelledException — expected */ }
      }

      const page = await pdf.getPage(pageNum + 1); // pdfjs is 1-indexed
      // Scale by devicePixelRatio so the rasterised buffer matches physical screen pixels.
      // compositePageToCanvas then sets canvas.style.width/height to logical CSS pixels,
      // giving crisp text on retina/HiDPI displays instead of a browser-upscaled blur.
      const dpr = window.devicePixelRatio || 1;
      const viewport = page.getViewport({ scale: pdfScaleRef.current * 1.5 * dpr });
      pageCssDimsRef.current.set(pageNum, { w: viewport.width / dpr, h: viewport.height / dpr });

      capturedOffscreen.width = viewport.width;
      capturedOffscreen.height = viewport.height;

      const task = page.render({ canvas: capturedOffscreen, viewport });
      renderTasksRef.current.set(capturedOffscreen, task);

      try {
        await task.promise;
      } catch (e: unknown) {
        if (e && typeof e === 'object' && 'name' in e && (e as { name: string }).name === 'RenderingCancelledException') {return;}
        throw e;
      }
      renderTasksRef.current.delete(capturedOffscreen);

      // After successful render, composite to visible canvas
      compositeRef.current(pageNum);
    })();

    // Register synchronously — before any await — so the next concurrent caller
    // sees this promise as its predecessor rather than A's (the original race).
    pendingRendersRef.current.set(capturedOffscreen, renderPromise);
    try {
      await renderPromise;
    } finally {
      // Only the tail of the chain clears the entry. If a newer call has already
      // replaced our entry, leave it alone — clearing it would drop the live gate.
      if (pendingRendersRef.current.get(capturedOffscreen) === renderPromise) {
        pendingRendersRef.current.delete(capturedOffscreen);
      }
    }
  }, []);  // Stable — compositeRef breaks the dep chain

  // Render all pages to offscreen canvases when the PDF loads.
  // Token changes only need a composite pass (handled by the effect below) — not a
  // full PDF re-rasterize, so tokensByPage is intentionally absent from this dep array.
  useEffect(() => {
    if (!pdfDoc) {return;}

    for (let i = 0; i < pdfDoc.numPages; i++) {
      renderPageToOffscreen(pdfDoc, i);
    }
  }, [pdfDoc, renderPageToOffscreen]); // eslint-disable-line react-hooks/exhaustive-deps

  // Composite overlays when hover/selection/prediction visibility changes (fast path — no PDF re-render)
  useEffect(() => {
    if (!pdfDoc) {return;}

    for (let i = 0; i < pdfDoc.numPages; i++) {
      compositePageToCanvas(i);
    }
  }, [pdfDoc, compositePageToCanvas, hoveredToken, selectedTokens, showPredictions]);

  // Fit-to-width on PDF load and on container resize (when in fit mode)
  useEffect(() => {
    if (!pdfDoc || pdfZoomMode !== 'fit') { return; }

    void fitPdfToWidth(pdfDoc);
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === 'undefined') { return; }

    const observer = new ResizeObserver(() => {
      void fitPdfToWidth(pdfDoc);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [pdfDoc, pdfZoomMode, fitPdfToWidth]);

  // Re-render all pages when pdfScale changes
  useEffect(() => {
    if (!pdfDoc) { return; }
    for (let i = 0; i < pdfDoc.numPages; i++) {
      void renderPageToOffscreen(pdfDoc, i);
    }
  }, [pdfScale, pdfDoc, renderPageToOffscreen]);

  // Cleanup render tasks on unmount
  useEffect(() => {
    const renderTasks = renderTasksRef.current;
    const pendingRenders = pendingRendersRef.current;
    return () => {
      renderTasks.forEach(task => task.cancel());
      renderTasks.clear();
      pendingRenders.clear();
    };
  }, []);

  const fieldOrder = useMemo(() => schemaFields.map(f => f.name), [schemaFields]);
  const schemaByName = useMemo(() => Object.fromEntries(schemaFields.map(f => [f.name, f])), [schemaFields]);
  const visibleFields = useMemo(() => {
    if (!document) { return []; }
    const predictions = document.predictions ?? {};
    const hasPredictions = Object.keys(predictions).length > 0;
    if (!hasPredictions) {
      // label-from-zero: no predictions yet — show all schema fields
      return fieldOrder;
    }
    return fieldOrder.length > 0
      ? fieldOrder.filter(f => f in predictions)
      : Object.keys(predictions);
  }, [document, fieldOrder]);
  // Keep ref in sync so keyboard handler always sees current visibleFields
  useEffect(() => { visibleFieldsRef.current = visibleFields; }, [visibleFields]);

  // Auto-scroll selected field card into view when selection changes via j/k
  useEffect(() => {
    if (!selectedField) { return; }
    const card = fieldCardRefs.current.get(selectedField);
    if (card) {
      card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [selectedField]);

  // handleTokenClick: DOM overlay event — browser does hit-test natively via event.target.
  // Token overlays are absolutely-positioned divs floating above the canvas; no coord math needed.
  const handleCorrectionChange = useCallback((field: string, value: string, bbox: number[][] | null = null) => {
    setCorrections(prev => ({ ...prev, [field]: { value, bbox } }));
    setLabels(prev => {
      if (prev[field]?.action !== 'correct') {return prev;}
      return { ...prev, [field]: { ...prev[field], correct_value: value, correct_bbox: bbox || undefined } };
    });
  }, []);

  const handleTokenClick = useCallback((pageNum: number, tokenIdx: number, e: React.MouseEvent) => {
    if (!document) {return;}
    e.stopPropagation();

    let activeField = selectedField;
    if (!activeField) {
      const firstUnlabeled = visibleFields.find(f => !labels[f]);
      if (!firstUnlabeled) {return;}
      activeField = firstUnlabeled;
      setSelectedField(activeField);
    }

    const token = (tokensByPage[pageNum] || [])[tokenIdx];
    if (!token) {return;}
    setLastClickedToken(token.text);
    const currentCorrection = corrections[activeField];
    const currentValue = currentCorrection?.value || '';

    if (e.detail === 2) {
      handleCorrectionChange(activeField, token.text, [[token.x0, token.y0, token.x1, token.y1]]);
      setSelectedTokens(new Map([[pageNum, new Set([tokenIdx])]]));
    } else {
      const newValue = currentValue ? `${currentValue} ${token.text}` : token.text;
      const newMap = new Map(selectedTokens);
      const pageSet = new Set(newMap.get(pageNum) || []);
      pageSet.add(tokenIdx);
      newMap.set(pageNum, pageSet);
      const spanBboxes: number[][] = [];
      newMap.forEach((indices, pg) => {
        (tokensByPage[pg] || []).forEach((t, idx) => {
          if (indices.has(idx)) { spanBboxes.push([t.x0, t.y0, t.x1, t.y1]); }
        });
      });
      handleCorrectionChange(activeField, newValue, spanBboxes.length > 0 ? spanBboxes : null);
      setSelectedTokens(newMap);
    }
  }, [document, selectedField, visibleFields, labels, tokensByPage, corrections, selectedTokens, handleCorrectionChange]);

  // handleCandidateClick: secondary DOM overlay — fires when clicking a candidate bbox div.
  const handleCandidateClick = useCallback((fieldName: string, cand: Candidate, e: React.MouseEvent) => {
    if (!document) {return;}
    e.stopPropagation();
    setSelectedField(fieldName);
    setLastClickedToken(cand.raw_text);
    const bbox: number[][] = [[cand.bbox_norm_x0, cand.bbox_norm_y0, cand.bbox_norm_x1, cand.bbox_norm_y1]];
    if (e.detail === 2) {
      handleCorrectionChange(fieldName, cand.raw_text, bbox);
      setSelectedTokens(new Map());
    } else {
      const currentCorrection = corrections[fieldName];
      const newValue = currentCorrection?.value
        ? `${currentCorrection.value} ${cand.raw_text}`
        : cand.raw_text;
      handleCorrectionChange(fieldName, newValue, [...(currentCorrection?.bbox ?? []), ...bbox]);
      setSelectedTokens(new Map());
    }
  }, [document, corrections, handleCorrectionChange]);

  // Token hover handled by CSS :hover on overlay divs — no mousemove math needed.

  // Filtered sidebar list — metadata search only (filename + doc_id)
  const filteredSidebarItems = useMemo(() => {
    const q = sidebarFilter.trim().toLowerCase();
    if (!q) { return sidebarItems; }
    return sidebarItems.filter((item) => {
      const filename = item.filename ?? '';
      return (
        (item.doc_id ?? '').toLowerCase().includes(q) ||
        filename.toLowerCase().includes(q)
      );
    });
  }, [sidebarItems, sidebarFilter]);

  /** Synchronously flush the live draft, reset local state, then navigate. */
  const handleSidebarClick = useCallback((docId: string) => {
    if (docId === documentId) { return; }
    // 1. Flush current draft synchronously (beats 300ms debounce window)
    try {
      sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draftRef.current));
    } catch { /* quota */ }
    // 2. Reset stale state so the debounce effect can't corrupt the new doc's draft
    setLabels({});
    setCorrections({});
    setSelectedTokens(new Map());
    // 3. Navigate (preserve filter so sidebar keeps showing the same queue context)
    const navParam = queueFilter ? `filter=${queueFilter}` : `mode=${queueMode}`;
    navigate(`/label/${docId}?${navParam}`);
  }, [documentId, DRAFT_KEY, navigate, queueMode, queueFilter]);

  const fetchDocument = async (docId: string) => {
    try {
      setLoading(true);
      const data = await getDocument(docId);
      if (!data) { throw new Error('Document not found'); }

      // get_document returns predictions as a bare dict (never a v2 envelope).
      const rawPredictions = data.predictions;
      const predictions = rawPredictions as Record<string, FieldPrediction> | undefined;
      // 'pending' when the dict is absent or empty — extraction not yet done.
      // 'ready' only when at least one prediction exists.
      const hasPredictions = predictions != null && Object.keys(predictions).length > 0;
      const predictionStatus: 'ready' | 'pending' = hasPredictions ? 'ready' : 'pending';

      setDocument({
        doc_id: data.doc_id,
        sha256: data.sha256,
        pages: data.pages,
        filename: data.filename,
        source_id: data.source_id ?? null,
        doc: data.doc ?? null,
        predictions,
        predictionStatus,
        // all_candidates is projected by get_document (init.sql:396, docs.payload->'candidates').
        // Null for unextracted docs — empty overlay is correct silent behaviour.
        all_candidates: data.all_candidates ?? [],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = (field: string) => {
    setLabels(prev => ({ ...prev, [field]: { field, action: 'approve' } }));
    appendToLog({ field, action: 'approve' });
  };

  const handleCorrect = (field: string) => {
    const correction = corrections[field];
    setLabels(prev => ({
      ...prev,
      [field]: {
        field,
        action: 'correct',
        correct_value: correction?.value || '',
        correct_bbox: correction?.bbox || undefined,
      }
    }));
    appendToLog({ field, action: 'correct', value: correction?.value || undefined });
  };

  const handleNotInDocument = (field: string) => {
    setLabels(prev => ({ ...prev, [field]: { field, action: 'not_in_document', correct_value: null } }));
    appendToLog({ field, action: 'not_in_document' });
  };

  const handleReject = (field: string) => {
    const predValue = document?.predictions?.[field]?.value ?? undefined;
    setLabels(prev => ({ ...prev, [field]: { field, action: 'reject', correct_value: predValue } }));
    appendToLog({ field, action: 'reject' });
  };

  const handleClearLabel = (field: string) => {
    setLabels(prev => {
      const next = { ...prev };
      delete next[field];
      return next;
    });
    setCorrections(prev => {
      const next = { ...prev };
      delete next[field];
      return next;
    });
    setSelectedTokens(new Map());
  };

  const handleFieldFocus = (field: string) => {
    setSelectedField(field);
    setSelectedTokens(new Map());
  };

  /** Submit one labeling action and return { ok, errorMessage } per field.
   *
   * All four action types route to /api/rpc/submit_label — the real
   * PostgREST RPC.  No separate approve endpoint.
   *
   * D.2: never throws on 4xx/5xx — returns the error so the caller can collect
   * per-field results and render them inline. No "fail loudly with retry"
   * banner that loses the labels the user already committed.
   */
  const sendOneLabel = async (label: FieldLabel, docId: string, grafanaUser: string): Promise<{ ok: boolean; errorMessage: string | null }> => {
    try {
      // Build action-specific payload.
      const actionPayload: Record<string, unknown> = {};
      if (label.action === 'correct') {
        if (label.correct_value !== undefined) { actionPayload.correct_value = label.correct_value; }
        if (label.correct_bbox) { actionPayload.correct_bbox = label.correct_bbox; }
      } else if (label.action === 'reject') {
        if (label.correct_value !== undefined) { actionPayload.correct_value = label.correct_value; }
      }
      // approve and not_in_document carry empty payload.

      await submitLabel({
        doc_id: docId,
        field: label.field,
        action: label.action as 'approve' | 'correct' | 'not_in_document' | 'reject',
        payload: actionPayload,
        submitted_by: grafanaUser,
      });
      return { ok: true, errorMessage: null };
    } catch (err) {
      return { ok: false, errorMessage: err instanceof Error ? err.message : 'Network error' };
    }
  };

  /** Find the next doc in a given list after currentDocId, wrapping to first if at end.
   * Accepts an explicit list so callers can pass a freshly-refetched array without
   * relying on setState having flushed (React state updates are async). */
  const findNextSidebarDocId = (currentDocId: string, list: DocumentReviewItem[]): string | null => {
    const idx = list.findIndex((item) => item.sha256 === currentDocId);
    if (idx < 0) {
      // Current doc no longer in list (e.g. it just passed the confidence threshold
      // and left the review queue). Advance to first item of fresh list.
      return list.length > 0 ? list[0].sha256 : null;
    }
    for (let i = idx + 1; i < list.length; i++) {
      return list[i].sha256;
    }
    // No further entries — fall back to first if we're not already there.
    return idx > 0 && list.length > 0 ? list[0].sha256 : null;
  };

  /** Submit only the labels that have real actions (approve, correct, not_in_document).
   *
   * D.1 + D.2 + D.3: parallel sends, per-field result tracking, inline errors,
   * auto-advance on full success.
   */
  const submitLabels = async (labelsToSend: FieldLabel[]) => {
    if (!document) {return;}

    setSubmitting(true);
    setFieldErrors({});  // clear stale per-field errors at submit-start
    setError(null);
    const grafanaUser = config.bootData.user?.login || config.bootData.user?.email || 'unknown';

    // Fire all label requests in parallel. Per-field success/failure is
    // tracked by index. The backend already handles each label idempotently
    // (corrections.unique_key), so a partial success on retry just no-ops the
    // already-committed rows.
    const docId = document.sha256;
    const results = await Promise.all(labelsToSend.map((label) => sendOneLabel(label, docId, grafanaUser)));

    const errors: Record<string, string> = {};
    results.forEach((res, i) => {
      if (!res.ok && res.errorMessage) { errors[labelsToSend[i].field] = res.errorMessage; }
    });

    setSubmitting(false);

    if (Object.keys(errors).length === 0) {
      // All saved. Navigate to next doc via get_next_doc (created_at DESC walk,
      // no status/schema filter). Sidebar refetch runs concurrently for freshness
      // but the navigation target comes from the RPC, not the sidebar list.
      sessionStorage.removeItem(DRAFT_KEY);
      setShowDraftBanner(false); // dismiss so the prior draft can't resurface on the next doc
      void fetchSidebar(); // fire-and-forget sidebar refresh

      // Correction legibility: surface a brief summary of what was corrected.
      // Shown transiently before navigation; clears itself so it never persists on the next doc.
      const correctedLabels = labelsToSend.filter(l => l.action === 'correct');
      if (correctedLabels.length > 0) {
        const names = correctedLabels.map(l => l.field).join(', ');
        const feedbackMsg = correctedLabels.length === 1
          ? `1 correction saved: ${names}`
          : `${correctedLabels.length} corrections saved: ${names}`;
        setSubmitFeedback(feedbackMsg);
        setTimeout(() => setSubmitFeedback(null), 3000);
      }

      try {
        const next = await getNextDoc(docId);
        if (next?.sha256) {
          const navParam = queueFilter ? `filter=${queueFilter}` : `mode=${queueMode}`;
          navigate(`/label/${next.sha256}?${navParam}`);
        } else {
          navigate('/queue');
        }
      } catch (navErr) {
        console.warn('[submitLabels] get_next_doc failed, falling back to queue', navErr);
        navigate('/queue');
      }
      return;
    }

    // Partial failure: render errors inline at the bad fields. Stay on this
    // doc so the labeler can fix and resubmit the failed ones; successes already
    // landed and won't be re-sent.
    setFieldErrors(errors);
  };

  const handleSubmit = async () => {
    const predictions = document?.predictions ?? {};
    const labelsToSubmit = Object.values(labels).filter(l =>
      l.action !== 'skip' && !predictions[l.field]?.approved_at
    );
    await submitLabels(labelsToSubmit);
  };
  // Keep ref in sync so keyboard Enter handler always calls the current version
  submitLabelsRef.current = handleSubmit;

  const formatFieldName = (field: string) => {
    return field.replace(/([a-z])([A-Z])/g, '$1 $2');
  };

  const handleReextractThisDoc = async () => {
    if (!document?.sha256) { return; }
    const sha256 = document.sha256;
    setReextracting(true);
    setReextractResult(null);

    try {
      await reextractDoc(sha256);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not connect to the server. Please try again.';
      setReextractResult({ message, severity: 'error' });
      setReextracting(false);
      return;
    }

    setReextractResult({
      message: 'Re-extracting this document — returning to queue.',
      severity: 'info',
    });

    // Worker re-claims the row on next ledger poll. Bounce to queue so the
    // user sees fresh state without us spinning a polling loop.
    window.setTimeout(() => {
      if (mountedRef.current) {
        navigate('/queue');
      }
    }, 1500);
  };

  const labeledCount = Object.keys(labels).length;
  const totalFields = visibleFields.length;
  const numPages = pdfDoc?.numPages || document?.pages || 0;

  if (loading) {
    return (
      <div className={getLoadingContainerStyles(styles.theme)}>
        <Spinner size="xl" />
        <p>Loading document...</p>
      </div>
    );
  }

  if (error || !document) {
    return (
      <div className={getErrorContainerStyles(styles.theme)}>
        <Alert severity="error" title="Error">
          {error || 'Document not found'}
        </Alert>
        <Button onClick={() => navigate('/queue')}>
          Back to Queue
        </Button>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {showDraftBanner && (
        <Alert
          severity='info'
          title='Draft restored'
          onRemove={() => setShowDraftBanner(false)}
        >
          Restored your previous draft. Changes not yet submitted.
        </Alert>
      )}
      {/* Header */}
      <div className={styles.header}>

        <div className={styles.headerLeft}>
          <Button
            variant="secondary"
            icon="arrow-left"
            onClick={() => navigate('/queue')}
          >
            Back
          </Button>
          <Button
            variant="secondary"
            icon="graph-bar"
            onClick={() => navigate('/model')}
            title="Model status and training"
          >
            Model
          </Button>
          <div className={styles.docInfo}>
            <h1 className={styles.title}>Document Review</h1>
            <span className={styles.docId} title={document.doc_id ?? undefined}>
              {document.filename ?? `(unnamed: ${shortDocId(document.doc_id)})`}
            </span>
          </div>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.progress}>
            <span>{labeledCount} / {totalFields} fields reviewed</span>
            <div className={styles.progressBar}>
              <div
                className={styles.progressFill}
                style={{ width: `${(labeledCount / totalFields) * 100}%` }}
              />
            </div>
          </div>
          <Button
            variant="secondary"
            icon="repeat"
            onClick={handleReextractThisDoc}
            disabled={reextracting}
            title="Re-extract this document with the current schema"
          >
            {reextracting ? 'Re-extracting...' : 'Re-extract'}
          </Button>
          <Button
            variant="primary"
            icon="check"
            onClick={handleSubmit}
            disabled={submitting || labeledCount === 0}
          >
            {submitting ? 'Submitting...' : 'Submit Review'}
          </Button>
        </div>
      </div>

      {reextractResult && (
        <Alert
          title={
            reextractResult.severity === 'error'
              ? 'Re-extraction Failed'
              : reextractResult.severity === 'success'
                ? 'Re-extraction Complete'
                : 'Re-extracting Document'
          }
          severity={reextractResult.severity}
          onRemove={() => setReextractResult(null)}
        >
          {reextractResult.message}
        </Alert>
      )}

      {/* Document metadata strip — forward-compat: renders any primitive key on document */}
      <div className={styles.docMetaStrip}>
        {([
          ['SHA-256', document.sha256 ? document.sha256.slice(0, 16) + '…' : null, document.sha256],
          ['Pages', document.pages != null ? String(document.pages) : null, null],
          ...(Object.entries(document as unknown as Record<string, unknown>)
            .filter(([k]) => !['doc_id','sha256','pages','predictions','pdf_url','filename','predictionStatus','predictionPendingReason'].includes(k))
            .filter(([, v]) => v !== null && v !== undefined && typeof v !== 'object')
            .map(([k, v]) => [k.replace(/_/g, ' '), String(v), null])
          ),
        ] as Array<[string, string | null, string | null]>)
          .filter(([, val]) => val !== null)
          .map(([label, val, title]) => (
            <span key={label} className={styles.docMetaItem} title={title ?? undefined}>
              <span className={styles.docMetaLabel}>{label}:</span>
              <span className={styles.docMetaValue}>{val}</span>
            </span>
          ))}
      </div>

      {/* Body: sidebar + main column */}
      <div className={styles.body}>
        {/* Sidebar */}
        <div className={sidebarOpen ? styles.sidebar : styles.sidebarCollapsed}>
          {sidebarOpen ? (
            <>
              <div className={styles.sidebarHeader}>
                <span className={styles.sidebarTitle}>Queue</span>
                <Button
                  variant="secondary"
                  fill="text"
                  size="sm"
                  onClick={() => setSidebarOpen(false)}
                  title="Collapse sidebar"
                >
                  <Icon name="angle-left" />
                </Button>
              </div>
              <div className={styles.sidebarSearch}>
                <Input
                  placeholder="Filter invoices..."
                  prefix={<Icon name="search" />}
                  value={sidebarFilter}
                  onChange={(e) => setSidebarFilter(e.currentTarget.value)}
                />
              </div>
              <div className={styles.sidebarList}>
                {sidebarLoading && sidebarItems.length === 0 && (
                  <div className={styles.sidebarState}>
                    <Spinner size="sm" />
                  </div>
                )}
                {sidebarError && (
                  /* Non-blocking pill: cached sidebarItems stay visible below.
                     Click to retry; the user's labelling work isn't blocked. */
                  <div
                    style={{ padding: '6px 8px', fontSize: '0.85em', color: '#a66', cursor: 'pointer' }}
                    onClick={() => { void fetchSidebar(); }}
                    title="Click to retry"
                  >
                    Couldn’t refresh — showing last fetched
                  </div>
                )}
                {!sidebarLoading && !sidebarError && filteredSidebarItems.length === 0 && (
                  <div className={styles.sidebarState}>
                    <span className={styles.sidebarEmptyText}>No results</span>
                  </div>
                )}
                {filteredSidebarItems.map((item) => (
                  <SidebarInvoiceItem
                    key={item.sha256 || item.doc_id || ''}
                    item={item}
                    isActive={item.sha256 === documentId}
                    onClick={() => item.sha256 && handleSidebarClick(item.sha256)}
                  />
                ))}
              </div>
            </>
          ) : (
            <div className={styles.sidebarRail}>
              <Button
                variant="secondary"
                fill="text"
                size="sm"
                onClick={() => setSidebarOpen(true)}
                title="Expand sidebar"
              >
                <Icon name="angle-right" />
              </Button>
            </div>
          )}
        </div>

        {/* Main column: PDF + fields */}
        <div className={styles.mainCol} ref={mainColRef}>

      {/* PDF Viewer -- left section */}
      <div className={styles.pdfPanel} style={{ flex: `0 0 ${splitPct}%`, minWidth: 0 }}>
        <div className={styles.pdfHeader}>
          <div className={styles.pdfHeaderLeft}>
            <span>PDF Preview</span>
            <Checkbox
              label="Show predictions"
              value={showPredictions}
              onChange={(e) => setShowPredictions(e.currentTarget.checked)}
            />
            {lastClickedToken && (
              <span className={styles.tokenClickAffordance}>
                Token: {lastClickedToken}
              </span>
            )}
          </div>
          <div className={styles.pdfHeaderRight}>
            <Button
              size="sm"
              variant="secondary"
              fill="text"
              icon="minus"
              onClick={() => {
                setPdfZoomMode('manual');
                setPdfScale((scale) => Math.max(0.25, Number((scale - 0.1).toFixed(2))));
              }}
              title="Zoom out"
            />
            <span className={styles.zoomValue}>{Math.round(pdfScale * 100)}%</span>
            <Button
              size="sm"
              variant="secondary"
              fill="text"
              icon="plus"
              onClick={() => {
                setPdfZoomMode('manual');
                setPdfScale((scale) => Math.min(4.0, Number((scale + 0.1).toFixed(2))));
              }}
              title="Zoom in"
            />
            <Button
              size="sm"
              variant="secondary"
              fill="text"
              onClick={() => {
                setPdfZoomMode('fit');
                void fitPdfToWidth();
              }}
              title="Fit to width"
            >
              Fit width
            </Button>
            <Button
              size="sm"
              variant="secondary"
              fill="text"
              onClick={async () => {
                const container = containerRef.current;
                if (!pdfDoc || !container) { return; }
                const page = await pdfDoc.getPage(1);
                const viewport = page.getViewport({ scale: 1 });
                const availableHeight = Math.max(container.clientHeight - 24, 320);
                const nextScale = availableHeight / (viewport.height * 1.5);
                setPdfScale(Number(Math.min(4.0, Math.max(0.25, nextScale)).toFixed(2)));
                setPdfZoomMode('manual');
              }}
              title="Fit to page height"
            >
              Fit page
            </Button>
            <Button
              size="sm"
              variant="secondary"
              fill="text"
              onClick={() => {
                setPdfScale(1.0);
                setPdfZoomMode('manual');
              }}
              title="Reset to 1:1"
            >
              1:1
            </Button>
            <span>{numPages} page{numPages !== 1 ? 's' : ''}</span>
          </div>
        </div>
        <div className={styles.pdfContainer} ref={containerRef}>
          {pdfLoading ? (
            <div className={styles.pdfLoadingState}>
              <Spinner />
              <span>Loading PDF...</span>
            </div>
          ) : (
            Array.from({ length: numPages }, (_, pageNum) => (
              <div key={pageNum} style={{ position: 'relative', display: 'inline-block' }}>
                <canvas
                  ref={(el) => { canvasRefs.current[pageNum] = el; }}
                  className={styles.pdfCanvas}
                  style={{ display: 'block' }}
                />
                {/* Token overlay — position:absolute divs; browser hit-tests natively */}
                <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
                  {(tokensByPage[pageNum] || []).map((token, idx) => {
                    const pageSelected = selectedTokens.get(pageNum);
                    const isSelected = pageSelected?.has(idx);
                    return (
                      <div
                        key={idx}
                        className={`${styles.tokenBox} ${selectedField ? styles.tokenBoxField : ''} ${isSelected ? styles.tokenBoxSelected : ''}`}
                        style={{
                          left: `${token.x0 * 100}%`,
                          top: `${token.y0 * 100}%`,
                          width: `${(token.x1 - token.x0) * 100}%`,
                          height: `${(token.y1 - token.y0) * 100}%`,
                        }}
                        onClick={(e) => handleTokenClick(pageNum, idx, e)}
                        title={token.text}
                      />
                    );
                  })}
                </div>
                {/* Candidate overlay — corpus-wide spans from all_candidates, filtered per page */}
                <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
                  {document && (document.all_candidates ?? [])
                    .filter(cand => cand.page_idx === pageNum)
                    .map((cand, ci) => {
                      const activeFieldName = selectedField ?? visibleFields[0] ?? '';
                      return (
                        <div
                          key={`corpus-${ci}`}
                          className={styles.candidateBox}
                          style={{
                            left: `${cand.bbox_norm_x0 * 100}%`,
                            top: `${cand.bbox_norm_y0 * 100}%`,
                            width: `${(cand.bbox_norm_x1 - cand.bbox_norm_x0) * 100}%`,
                            height: `${(cand.bbox_norm_y1 - cand.bbox_norm_y0) * 100}%`,
                          }}
                          onClick={(e) => handleCandidateClick(activeFieldName, cand, e)}
                          title={cand.raw_text}
                        />
                      );
                    })}
                </div>
                {/* SVG prediction bbox overlay — normalized 0-1 viewBox so no coord math needed */}
                {showPredictions && document && (() => {
                  const predictions = document.predictions ?? {};
                  const entries = Object.entries(predictions);
                  if (entries.length === 0) { return null; }
                  // Palette for cycling non-selected field colors
                  const palette = [
                    '#4C9BE8', '#E8854C', '#6DB86D', '#B56DBF', '#E8C84C',
                    '#4CE8C8', '#E84C6D', '#8B8BE8', '#E88B4C', '#4CE84C',
                  ];
                  return (
                    <svg
                      style={{
                        position: 'absolute',
                        inset: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none',
                        overflow: 'visible',
                      }}
                      viewBox="0 0 1 1"
                      preserveAspectRatio="none"
                    >
                      {entries.map(([fieldName, prediction], fieldIdx) => {
                        const prov = prediction.provenance;
                        if (!prov || prov.page !== pageNum) { return null; }
                        const bbox = prov.bbox;
                        if (!bbox || bbox.length < 4) { return null; }
                        if (!bbox.some((v: number) => v !== 0)) { return null; }
                        const [x0, y0, x1, y1] = bbox;
                        const isSelected = fieldName === selectedField;
                        const baseColor = (() => {
                          const conf = prediction.confidence;
                          if (isSelected) {
                            return getConfidenceColor(conf, autoApproveThreshold, mediumThreshold);
                          }
                          return palette[fieldIdx % palette.length];
                        })();
                        const fillOpacity = isSelected ? 0.15 : 0.08;
                        const strokeOpacity = isSelected ? 0.9 : 0.55;
                        const strokeWidth = isSelected ? 0.003 : 0.002;
                        const strokeDasharray = isSelected ? undefined : '0.012 0.006';
                        const labelText = `${fieldName} ${Math.round(prediction.confidence * 100)}%`;
                        // Font size in SVG user units (0–1 space): ~10px / canvas height
                        // We use a fixed small value; the browser scales with the SVG
                        const fontSize = 0.025;
                        return (
                          <g key={`pred-${fieldName}`}>
                            <rect
                              x={x0}
                              y={y0}
                              width={x1 - x0}
                              height={y1 - y0}
                              fill={baseColor}
                              fillOpacity={fillOpacity}
                              stroke={baseColor}
                              strokeOpacity={strokeOpacity}
                              strokeWidth={strokeWidth}
                              strokeDasharray={strokeDasharray}
                            />
                            <text
                              x={x0 + 0.004}
                              y={y0 > fontSize + 0.005 ? y0 - 0.004 : y0 + fontSize + 0.004}
                              fontSize={fontSize}
                              fill={baseColor}
                              fillOpacity={Math.min(1, strokeOpacity + 0.1)}
                              style={{ fontFamily: 'sans-serif' }}
                            >
                              {labelText}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  );
                })()}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Drag-to-resize splitter */}
      <div
        className={`${styles.splitter} ${isDragging ? styles.splitterActive : ''}`}
        onMouseDown={handleSplitterMouseDown}
      />

      {/* Fields Panel -- right section */}
      <div className={styles.fieldsPanel} style={{ flex: `0 0 ${100 - splitPct}%`, minWidth: 0 }}>
        <div>
          <h2 className={styles.fieldsPanelTitle}>Extracted Fields</h2>
          <p className={styles.fieldsPanelSubtitle}>
            Review each field: approve, fix, or mark as not in document
          </p>
        </div>

        {/* Keyboard shortcut hint strip */}
        <div className={styles.kbHintStrip}>
          <kbd>j/k</kbd> navigate &nbsp;&middot;&nbsp;
          <kbd>a</kbd> approve &nbsp;&middot;&nbsp;
          <kbd>r</kbd> wrong &nbsp;&middot;&nbsp;
          <kbd>n</kbd> not in doc &nbsp;&middot;&nbsp;
          <kbd>c</kbd> fix
        </div>

        <div className={styles.fieldsList} ref={fieldsListRef}>
            {document.predictionStatus === 'pending' && (
              <div style={{ padding: '8px 0 12px', fontSize: '0.85em', color: 'var(--color-text-secondary)' }}>
                Extraction in progress — heuristic values shown where available.
                {document.predictionPendingReason ? ` (${document.predictionPendingReason})` : ''}
              </div>
            )}
            {visibleFields.length === 0 && (
              <div style={{ padding: '24px 0', textAlign: 'center' }}>
                <p style={{ color: 'var(--color-text-secondary)', marginBottom: 12 }}>
                  No fields defined yet. Add fields in Field Manager to start labeling.
                </p>
                <Button
                  variant="secondary"
                  icon="plus"
                  onClick={() => navigate('/schema')}
                >
                  Add Field
                </Button>
              </div>
            )}
            {visibleFields.map(field => {
              const prediction = (document.predictions ?? {})[field] ?? {
                value: null,
                confidence: 0,
                status: 'MISSING' as const,
                provenance: null,
                raw_text: null,
              };
              const label = labels[field];
              const isLabeled = !!label;
              const hasValue = !!prediction.value;
              const hasCorrection = !!corrections[field]?.value?.trim();
              const isNotInDoc = label?.action === 'not_in_document';
              const isRejected = label?.action === 'reject';
              const fieldSchema = schemaByName[field];
              const isComputed = !!(fieldSchema?.computed && fieldSchema.computed_fn && fieldSchema.computed_from?.length);
              const confidenceColor = getConfidenceColor(prediction.confidence, autoApproveThreshold, mediumThreshold);
              // approved_at is threaded by the queue projection (F7/268a804).
              // The per-doc detail endpoint does not yet carry it — greying is
              // dormant until that backend gap is closed (see TODO in routes/active_learning.py).
              const isApproved = !!prediction.approved_at;
              const approvedLabel = isApproved
                ? `Approved ${new Date(prediction.approved_at!).toLocaleString()}`
                : undefined;

              return (
                <div
                  key={field}
                  ref={(el) => { if (el) { fieldCardRefs.current.set(field, el); } else { fieldCardRefs.current.delete(field); } }}
                  className={`${styles.fieldCard} ${isApproved ? styles.fieldCardApproved : ''} ${isLabeled && !isNotInDoc && !isRejected ? styles.fieldCardLabeled : ''} ${isNotInDoc ? styles.fieldCardNotInDoc : ''} ${isRejected ? styles.fieldCardRejected : ''} ${selectedField === field ? styles.fieldCardSelected : ''}`}
                  onClick={() => { if (!isApproved) { setSelectedField(field); } }}
                >
                  <div className={styles.fieldHeader}>
                    <div className={`${styles.fieldName} ${isNotInDoc ? styles.fieldNameStruck : ''}`}>
                      {formatFieldName(field)}
                    </div>
                    <div className={styles.fieldBadges}>
                      {isComputed && (
                        <Badge text="Computed" color="purple" icon="link" />
                      )}
                      <Badge
                        text={prediction.status}
                        color={getStatusColor(prediction.status)}
                      />
                    </div>
                  </div>

                  {isComputed && fieldSchema.computed_from && (
                    <div className={styles.provenanceLine}>
                      Derived from {fieldSchema.computed_from.join(' + ')} via {fieldSchema.computed_fn}
                    </div>
                  )}

                  <div className={`${styles.modelGuess} ${isNotInDoc ? styles.fieldValueStruck : ''}`}>
                    <span className={styles.modelGuessLabel}>Model:</span>
                    {hasValue
                      ? <strong className={styles.modelGuessValue} style={{ color: confidenceColor }}>{prediction.value}</strong>
                      : <em className={styles.noValue}>nothing extracted</em>
                    }
                    {hasValue && (
                      <span className={styles.modelGuessConf} style={{ color: confidenceColor }}>
                        {Math.round(prediction.confidence * 100)}%
                      </span>
                    )}
                  </div>

                  <div className={styles.confidenceRow}>
                    <span className={styles.confidenceLabel}>Confidence:</span>
                    <div className={styles.confidenceBarSmall}>
                      <div
                        className={styles.confidenceBarFill}
                        style={{
                          width: `${prediction.confidence * 100}%`,
                          backgroundColor: confidenceColor,
                        }}
                      />
                    </div>
                    <span style={{ color: confidenceColor }}>
                      {formatConfidencePercent(prediction.confidence)}%
                    </span>
                  </div>

                  {/* Per-field schema metadata */}
                  {fieldSchema && (
                    <div className={styles.fieldSchemaRow}>
                      {fieldSchema.base_type && (
                        <span className={styles.fieldSchemaBit} title="Base type">{fieldSchema.base_type}</span>
                      )}
                      {fieldSchema.normalizer && (
                        <span className={styles.fieldSchemaBit} title="Normalizer">{fieldSchema.normalizer}</span>
                      )}
                      {fieldSchema.dataverse_column && (
                        <span className={styles.fieldSchemaBit} title="Dataverse column">{fieldSchema.dataverse_column}</span>
                      )}
                      {fieldSchema.status && fieldSchema.status !== 'active' && (
                        <span className={`${styles.fieldSchemaBit} ${styles.fieldSchemaStatus}`} title="Field status">{fieldSchema.status}</span>
                      )}
                      {prediction.raw_text && (
                        <span className={styles.fieldSchemaBit} title="Raw extracted text">raw: {prediction.raw_text}</span>
                      )}
                    </div>
                  )}

                  {/* Correction input -- disabled for approved fields */}
                  <div className={styles.correctionInput}>
                    <Input
                      ref={(el: HTMLInputElement | null) => { if (el) { correctionInputRefs.current.set(field, el); } else { correctionInputRefs.current.delete(field); } }}
                      placeholder={isApproved ? '' : 'Type or click tokens on PDF...'}
                      value={isApproved ? '' : (corrections[field]?.value || '')}
                      disabled={isApproved}
                      onChange={(e) => handleCorrectionChange(field, e.currentTarget.value, corrections[field]?.bbox || null)}
                      onFocus={() => { if (!isApproved) { handleFieldFocus(field); } }}
                      onClick={(e) => e.stopPropagation()}
                    />
                    {fieldErrors[field] && (
                      <div style={{ color: '#d44', fontSize: '0.85em', marginTop: 4 }}>
                        {fieldErrors[field]}
                      </div>
                    )}
                  </div>

                  {/* Action buttons -- hidden for approved fields */}
                  {isApproved ? (
                    <div className={styles.labelResult} title={approvedLabel}>
                      <span className={styles.labelText} style={{ color: 'var(--color-success-text, #6ccf8e)' }}>
                        Approved
                      </span>
                    </div>
                  ) : !isLabeled ? (
                    <div className={styles.fieldActions}>
                      {hasValue && (
                        <Button
                          size="sm"
                          variant="success"
                          icon="check"
                          onClick={(e) => { e.stopPropagation(); handleApprove(field); }}
                          title="Mark this prediction as correct"
                        >
                          Approve
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="primary"
                        icon="pen"
                        onClick={(e) => { e.stopPropagation(); handleCorrect(field); }}
                        disabled={!hasCorrection}
                        title={hasCorrection ? 'Submit your correction' : 'Type a correction above first'}
                      >
                        Fix
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={(e) => { e.stopPropagation(); handleNotInDocument(field); }}
                        title="This field is not present in the document"
                      >
                        Not in doc
                      </Button>
                      <Button
                        size="sm"
                        variant="destructive"
                        icon="times"
                        onClick={(e) => { e.stopPropagation(); handleReject(field); }}
                        title="This prediction is wrong"
                      >
                        Wrong
                      </Button>
                    </div>
                  ) : (
                    <div className={styles.labelResult}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
                        <span className={`${styles.labelText} ${isNotInDoc ? styles.labelTextMuted : ''} ${isRejected ? styles.labelTextRejected : ''}`}>
                          {label.action === 'approve' && 'Marked correct'}
                          {label.action === 'correct' && `Fixed: ${label.correct_value}`}
                          {label.action === 'not_in_document' && 'Not in document'}
                          {label.action === 'reject' && 'Marked wrong'}
                        </span>
                        {label.action === 'correct' && prediction.value && label.correct_value &&
                          label.correct_value.trim() !== prediction.value.trim() && (
                          <span className={styles.correctionDelta}>
                            <span className={styles.correctionDeltaBefore}>{prediction.value}</span>
                            {' → '}
                            <span className={styles.correctionDeltaAfter}>{label.correct_value}</span>
                          </span>
                        )}
                      </div>
                      <Button
                        size="sm"
                        variant="secondary"
                        fill="text"
                        onClick={(e) => { e.stopPropagation(); handleClearLabel(field); }}
                      >
                        Undo
                      </Button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

        {/* Label Log — compact timeline at bottom of fields panel */}
        {labelLog.length > 0 && (
          <div className={styles.labelLogSection}>
            <div className={styles.labelLogHeader}>
              <span className={styles.labelLogTitle}>Label Log</span>
              <span className={styles.labelLogBadge}>{labelLog.length}</span>
            </div>
            <div className={styles.labelLogList}>
              {labelLog.map((entry, i) => (
                <div key={i} className={styles.labelLogRow}>
                  <span
                    className={styles.labelLogDot}
                    style={{
                      background:
                        entry.action === 'approve' ? '#6ccf8e' :
                        entry.action === 'correct' ? '#5794f2' :
                        entry.action === 'reject'  ? '#f2495c' :
                        '#8e8e8e',
                    }}
                  />
                  <span className={styles.labelLogField}>{entry.field}</span>
                  <span className={styles.labelLogAction}>
                    {entry.action === 'approve' && 'approved'}
                    {entry.action === 'correct' && (entry.value ? `→ ${entry.value}` : 'fixed')}
                    {entry.action === 'reject' && 'wrong'}
                    {entry.action === 'not_in_document' && 'not in doc'}
                  </span>
                  <span className={styles.labelLogTime}>{formatTimeAgo(entry.ts)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {submitFeedback && (
          <div className={styles.submitFeedbackMsg} aria-live="polite">
            {submitFeedback}
          </div>
        )}

        <div className={styles.fieldsFooter}>
          <div style={{ fontSize: '0.78em', color: 'var(--color-text-secondary)', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <span><kbd>a</kbd> approve</span>
            <span><kbd>r</kbd> wrong</span>
            <span><kbd>c</kbd> fix</span>
            <span><kbd>n</kbd> not in doc</span>
            <span><kbd>j/k</kbd> next/prev</span>
            <span><kbd>Enter</kbd> submit</span>
            <span><kbd>Bksp</kbd> undo</span>
          </div>
        </div>
      </div>
        </div>{/* end mainCol */}
      </div>{/* end body */}
    </div>
  );
}


const getStyles = (theme: GrafanaTheme2) => ({
  theme, // Pass theme for shared style functions
  container: css`
    height: calc(100vh - 80px);
    display: flex;
    flex-direction: column;
  `,
  header: css`
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: ${theme.spacing(2)} ${theme.spacing(3)};
    background: ${theme.colors.background.secondary};
    border-bottom: 1px solid ${theme.colors.border.weak};
  `,
  headerLeft: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(2)};
  `,
  headerRight: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(3)};
  `,
  docInfo: css``,
  title: css`
    margin: 0;
    font-size: 20px;
  `,
  docId: css`
    font-family: monospace;
    font-size: 12px;
    color: ${theme.colors.text.secondary};
  `,
  progress: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(1)};
    font-size: 14px;
    color: ${theme.colors.text.secondary};
  `,
  progressBar: css`
    width: 120px;
    height: 8px;
    background: ${theme.colors.background.canvas};
    border-radius: 4px;
    overflow: hidden;
  `,
  progressFill: css`
    height: 100%;
    background: linear-gradient(90deg, ${theme.colors.primary.main}, ${theme.colors.primary.shade});
    transition: width 0.3s ease;
  `,
  pdfPanel: css`
    display: flex;
    flex-direction: column;
    /* flex basis is set via inline style (splitPct); keep height full */
    height: 100%;
    min-height: 0;
    background: ${theme.colors.background.secondary};
    border-radius: ${theme.shape.radius.default};
    overflow: hidden;
  `,
  tokenBox: css`
    position: absolute;
    cursor: text;
    background: transparent;
    border: 1px solid transparent;
    pointer-events: auto;
    &:hover {
      background: ${COLORS.tokenHover};
      border-color: ${COLORS.tokenHoverBorder};
    }
  `,
  candidateBox: css`
    position: absolute;
    cursor: pointer;
    pointer-events: auto;
    border: 1px dashed rgba(100, 160, 255, 0.5);
    background: transparent;
    border-radius: 1px;
    &:hover {
      background: rgba(100, 160, 255, 0.18);
      border-color: rgba(100, 160, 255, 0.95);
      border-style: solid;
    }
  `,
  tokenBoxField: css`
    cursor: crosshair;
  `,
  tokenBoxSelected: css`
    background: ${COLORS.tokenSelected};
    border-color: ${COLORS.tokenSelectedBorder};
    &:hover {
      background: ${COLORS.tokenSelected};
      border-color: ${COLORS.tokenSelectedBorder};
    }
  `,
  pdfHeader: css`
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: ${theme.spacing(1.5)} ${theme.spacing(2)};
    background: ${theme.colors.background.canvas};
    border-bottom: 1px solid ${theme.colors.border.weak};
    font-weight: 600;
  `,
  pdfHeaderLeft: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(1.5)};
  `,
  pdfHeaderRight: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(0.5)};
    font-size: 13px;
    color: ${theme.colors.text.secondary};
  `,
  zoomValue: css`
    font-size: 12px;
    min-width: 36px;
    text-align: center;
    font-variant-numeric: tabular-nums;
    color: ${theme.colors.text.primary};
  `,
  pdfContainer: css`
    flex: 1;
    position: relative;
    background: #525659;
    overflow: auto;
    padding: ${theme.spacing(1)};
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: ${theme.spacing(1)};
  `,
  pdfCanvas: css`
    display: block;
    flex-shrink: 0;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
  `,
  pdfLoadingState: css`
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: ${theme.spacing(1)};
    color: ${theme.colors.text.secondary};
    padding: ${theme.spacing(4)};
  `,
  fieldsPanel: css`
    display: flex;
    flex-direction: column;
    /* flex basis is set via inline style (100 - splitPct); keep height full */
    height: 100%;
    min-height: 0;
    background: ${theme.colors.background.secondary};
    border-radius: ${theme.shape.radius.default};
    overflow: hidden;
  `,
  fieldsPanelTitle: css`
    margin: 0;
    padding: ${theme.spacing(2)} ${theme.spacing(2)} 0;
    font-size: 18px;
  `,
  fieldsPanelSubtitle: css`
    margin: 0;
    padding: ${theme.spacing(0.5)} ${theme.spacing(2)} ${theme.spacing(2)};
    font-size: 13px;
    color: ${theme.colors.text.secondary};
  `,
  fieldsList: css`
    flex: 1;
    overflow-y: auto;
    padding: 0 ${theme.spacing(2)} ${theme.spacing(2)};
  `,
  fieldCard: css`
    padding: ${theme.spacing(2)};
    margin-bottom: ${theme.spacing(1.5)};
    background: ${theme.colors.background.primary};
    border: 1px solid ${theme.colors.border.weak};
    border-radius: ${theme.shape.radius.default};
    cursor: pointer;
    transition: all 0.2s ease;

    &:hover {
      border-color: ${theme.colors.border.medium};
    }
  `,
  fieldCardLabeled: css`
    border-color: ${theme.colors.success.border};
    background: ${theme.colors.success.transparent};
  `,
  fieldCardNotInDoc: css`
    border-color: ${theme.colors.border.weak};
    background: ${theme.colors.background.secondary};
    opacity: 0.6;
  `,
  fieldCardRejected: css`
    border-color: ${theme.colors.error.border};
    background: ${theme.colors.error.transparent};
    opacity: 0.8;
  `,
  fieldCardSelected: css`
    border-color: ${theme.colors.primary.border};
    box-shadow: 0 0 0 1px ${theme.colors.primary.border};
  `,
  fieldCardApproved: css`
    opacity: 0.55;
    pointer-events: none;
    border-color: ${theme.colors.border.weak};
    background: ${theme.colors.background.canvas};
  `,
  fieldHeader: css`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: ${theme.spacing(1)};
  `,
  fieldBadges: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(0.75)};
  `,
  provenanceLine: css`
    font-size: 11px;
    color: ${theme.colors.text.secondary};
    margin-bottom: ${theme.spacing(1)};
    font-style: italic;
  `,
  fieldName: css`
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: ${theme.colors.text.secondary};
  `,
  fieldNameStruck: css`
    text-decoration: line-through;
    color: ${theme.colors.text.disabled};
  `,
  fieldValue: css`
    font-size: 16px;
    font-weight: 600;
    margin-bottom: ${theme.spacing(1)};
    color: ${theme.colors.text.primary};
  `,
  modelGuess: css`
    display: flex;
    align-items: baseline;
    gap: ${theme.spacing(0.75)};
    margin-bottom: ${theme.spacing(0.5)};
    min-height: 24px;
  `,
  modelGuessLabel: css`
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: ${theme.colors.text.secondary};
    flex-shrink: 0;
  `,
  modelGuessValue: css`
    font-size: 15px;
    font-weight: 700;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  `,
  modelGuessConf: css`
    font-size: 13px;
    font-weight: 600;
    flex-shrink: 0;
  `,
  fieldValueStruck: css`
    text-decoration: line-through;
    color: ${theme.colors.text.disabled};
  `,
  noValue: css`
    color: ${theme.colors.text.disabled};
    font-weight: normal;
  `,
  confidenceRow: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(1)};
    margin-bottom: ${theme.spacing(1.5)};
    font-size: 12px;
  `,
  confidenceLabel: css`
    color: ${theme.colors.text.secondary};
  `,
  confidenceBarSmall: css`
    flex: 1;
    height: 6px;
    background: ${theme.colors.background.canvas};
    border-radius: 3px;
    overflow: hidden;
  `,
  confidenceBarFill: css`
    height: 100%;
    border-radius: 3px;
  `,
  correctionInput: css`
    margin-bottom: ${theme.spacing(1.5)};
  `,
  fieldActions: css`
    display: flex;
    gap: ${theme.spacing(1)};
  `,
  labelResult: css`
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: ${theme.spacing(1)};
    border-top: 1px solid ${theme.colors.border.weak};
  `,
  labelText: css`
    font-size: 12px;
    color: ${theme.colors.success.text};
    font-weight: 600;
  `,
  labelTextMuted: css`
    color: ${theme.colors.text.disabled};
  `,
  labelTextRejected: css`
    color: ${theme.colors.error.text};
  `,
  /** Inline before/after diff shown when labeler corrects a predicted value. */
  correctionDelta: css`
    font-size: 11px;
    color: ${theme.colors.text.secondary};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
  `,
  correctionDeltaBefore: css`
    color: ${theme.colors.text.disabled};
    text-decoration: line-through;
  `,
  correctionDeltaAfter: css`
    color: ${theme.colors.primary.text};
    font-weight: 600;
  `,
  /** Transient success message shown briefly after submit, before navigation. */
  submitFeedbackMsg: css`
    margin: 0 ${theme.spacing(2)} ${theme.spacing(1)};
    padding: ${theme.spacing(0.75)} ${theme.spacing(1.5)};
    font-size: 12px;
    color: ${theme.colors.success.text};
    background: ${theme.colors.success.transparent};
    border: 1px solid ${theme.colors.success.border};
    border-radius: ${theme.shape.radius.default};
    animation: invoicex-feedback-fadein 0.15s ease;
    @keyframes invoicex-feedback-fadein {
      from { opacity: 0; transform: translateY(4px); }
      to   { opacity: 1; transform: translateY(0); }
    }
  `,
  kbHintStrip: css`
    padding: ${theme.spacing(0.75)} ${theme.spacing(2)};
    font-size: 11px;
    color: ${theme.colors.text.disabled};
    background: ${theme.colors.background.canvas};
    border-bottom: 1px solid ${theme.colors.border.weak};
    flex-shrink: 0;
    kbd {
      font-family: ${theme.typography.fontFamilyMonospace};
      font-size: 10px;
      padding: 1px 4px;
      border-radius: 3px;
      background: ${theme.colors.background.secondary};
      border: 1px solid ${theme.colors.border.medium};
      color: ${theme.colors.text.secondary};
    }
  `,
  fieldsFooter: css`
    display: flex;
    justify-content: center;
    padding: ${theme.spacing(2)};
    border-top: 1px solid ${theme.colors.border.weak};
  `,
  tokenClickAffordance: css`
    font-size: 12px;
    color: ${theme.colors.primary.text};
    background: ${theme.colors.primary.transparent};
    border: 1px solid ${theme.colors.primary.border};
    border-radius: ${theme.shape.radius.pill};
    padding: 2px ${theme.spacing(1)};
    max-width: 240px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    animation: invoicex-token-fadein 0.15s ease;
    @keyframes invoicex-token-fadein {
      from { opacity: 0; transform: translateY(-4px); }
      to   { opacity: 1; transform: translateY(0); }
    }
  `,
  // Layout
  body: css`
    display: flex;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  `,
  mainCol: css`
    display: flex;
    flex-direction: row;
    flex: 1;
    min-width: 0;
    overflow: hidden;
  `,
  // Sidebar (open)
  sidebar: css`
    display: flex;
    flex-direction: column;
    width: 240px;
    flex-shrink: 0;
    background: ${theme.colors.background.canvas};
    border-right: 1px solid ${theme.colors.border.weak};
    overflow: hidden;
  `,
  // Sidebar (collapsed rail)
  sidebarCollapsed: css`
    display: flex;
    flex-direction: column;
    width: 32px;
    flex-shrink: 0;
    background: ${theme.colors.background.canvas};
    border-right: 1px solid ${theme.colors.border.weak};
    overflow: hidden;
  `,
  sidebarRail: css`
    display: flex;
    justify-content: center;
    padding-top: ${theme.spacing(1)};
  `,
  sidebarHeader: css`
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: ${theme.spacing(1)} ${theme.spacing(1.5)};
    border-bottom: 1px solid ${theme.colors.border.weak};
    flex-shrink: 0;
  `,
  sidebarTitle: css`
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: ${theme.colors.text.secondary};
  `,
  sidebarSearch: css`
    padding: ${theme.spacing(1)};
    border-bottom: 1px solid ${theme.colors.border.weak};
    flex-shrink: 0;
  `,
  sidebarList: css`
    flex: 1;
    overflow-y: auto;
  `,
  sidebarState: css`
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: ${theme.spacing(1)};
    padding: ${theme.spacing(2)};
  `,
  sidebarErrorText: css`
    font-size: 12px;
    color: ${theme.colors.error.text};
    text-align: center;
  `,
  sidebarEmptyText: css`
    font-size: 12px;
    color: ${theme.colors.text.disabled};
    text-align: center;
  `,
  // SidebarInvoiceItem nav entries
  sidebarNavItem: css`
    padding: ${theme.spacing(1)} ${theme.spacing(1.5)};
    border-left: 3px solid transparent;
    background: transparent;
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease;
    &:hover {
      background: ${theme.colors.action.hover};
    }
  `,
  sidebarNavItemActive: css`
    border-left-color: ${theme.colors.primary.main};
    background: ${theme.colors.primary.transparent};
    &:hover {
      background: ${theme.colors.primary.transparent};
    }
  `,
  sidebarNavHead: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(0.75)};
    margin-bottom: ${theme.spacing(0.25)};
  `,
  sidebarNavId: css`
    font-size: 12px;
    font-weight: 600;
    color: ${theme.colors.text.primary};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
    flex: 1;
  `,
  sidebarNavIdActive: css`
    color: ${theme.colors.primary.text};
  `,
  sidebarNavVendor: css`
    font-size: 12px;
    font-weight: 600;
    color: ${theme.colors.text.primary};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  `,
  sidebarNavInvoice: css`
    font-size: 11px;
    color: ${theme.colors.text.secondary};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  `,
  sidebarNavMeta: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(0.75)};
    margin-top: ${theme.spacing(0.25)};
  `,
  // Small circular dot indicating confidence level for each invoice item in the sidebar nav list.
  sidebarNavDot: css`
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  `,
  sidebarNavDate: css`
    font-size: 11px;
    color: ${theme.colors.text.secondary};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
  `,
  docMetaStrip: css`
    display: flex;
    flex-wrap: wrap;
    gap: ${theme.spacing(2)};
    padding: ${theme.spacing(1)} ${theme.spacing(2)};
    background: ${theme.colors.background.secondary};
    border-bottom: 1px solid ${theme.colors.border.weak};
    font-size: 12px;
  `,
  docMetaItem: css`
    display: flex;
    gap: ${theme.spacing(0.5)};
    align-items: center;
  `,
  docMetaLabel: css`
    color: ${theme.colors.text.secondary};
    text-transform: capitalize;
  `,
  docMetaValue: css`
    color: ${theme.colors.text.primary};
    font-weight: 500;
  `,
  fieldSchemaRow: css`
    display: flex;
    flex-wrap: wrap;
    gap: ${theme.spacing(0.5)};
    margin-bottom: ${theme.spacing(1)};
  `,
  fieldSchemaBit: css`
    font-size: 11px;
    padding: 1px 6px;
    border-radius: ${theme.shape.radius.pill};
    background: ${theme.colors.background.canvas};
    border: 1px solid ${theme.colors.border.weak};
    color: ${theme.colors.text.secondary};
    font-family: ${theme.typography.fontFamilyMonospace};
  `,
  fieldSchemaStatus: css`
    border-color: ${theme.colors.warning.border};
    color: ${theme.colors.warning.text};
    background: ${theme.colors.warning.transparent};
  `,
  labelLogSection: css`
    border-top: 1px solid ${theme.colors.border.weak};
    flex-shrink: 0;
    background: ${theme.colors.background.canvas};
  `,
  labelLogHeader: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(0.75)};
    padding: ${theme.spacing(0.75)} ${theme.spacing(2)};
    border-bottom: 1px solid ${theme.colors.border.weak};
  `,
  labelLogTitle: css`
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: ${theme.colors.text.secondary};
  `,
  labelLogBadge: css`
    font-size: 10px;
    font-weight: 600;
    padding: 1px 5px;
    border-radius: ${theme.shape.radius.pill};
    background: ${theme.colors.background.secondary};
    border: 1px solid ${theme.colors.border.weak};
    color: ${theme.colors.text.disabled};
    font-variant-numeric: tabular-nums;
  `,
  labelLogList: css`
    max-height: 180px;
    overflow-y: auto;
    padding: ${theme.spacing(0.5)} 0;
  `,
  labelLogRow: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(0.75)};
    padding: ${theme.spacing(0.5)} ${theme.spacing(2)};
    font-size: 11px;
    &:hover {
      background: ${theme.colors.action.hover};
    }
  `,
  labelLogDot: css`
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  `,
  labelLogField: css`
    font-weight: 600;
    color: ${theme.colors.text.primary};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 90px;
    flex-shrink: 0;
    font-family: ${theme.typography.fontFamilyMonospace};
    font-size: 10px;
  `,
  labelLogAction: css`
    flex: 1;
    color: ${theme.colors.text.secondary};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 11px;
  `,
  labelLogTime: css`
    flex-shrink: 0;
    font-size: 10px;
    color: ${theme.colors.text.disabled};
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  `,
  splitter: css`
    width: 6px;
    flex-shrink: 0;
    cursor: col-resize;
    background: ${theme.colors.border.weak};
    border-left: 1px solid ${theme.colors.border.weak};
    border-right: 1px solid ${theme.colors.border.weak};
    transition: background 0.15s ease;
    &:hover {
      background: ${theme.colors.primary.border};
    }
  `,
  splitterActive: css`
    background: ${theme.colors.primary.main};
  `,
});
