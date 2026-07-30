import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { css } from '@emotion/css';
import { GrafanaTheme2, SelectableValue } from '@grafana/data';
import { useStyles2, Card, Spinner, Button, Alert, Select, Checkbox, Icon, Input } from '@grafana/ui';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  queueList,
  reextractDoc,
  triggerRetrain,
  wakePipeline,
  getLatestModelRunId,
  type SortAxis,
} from '../utils/api';
import { COLORS } from '../utils/colors';
import { getLoadingContainerStyles, getErrorContainerStyles } from '../utils/styles';
import {
  type QueueFilter,
  isQueueFilter,
  filterToMode,
} from '../utils/types';

interface DocumentReviewItem {
  doc_id: string;
  sha256: string;
  imported_at: string | null;
  filename: string | null;
  field_count: number;
  // Metadata from ledger/ingest — inference fields absent post-Wave-5A
  source?: string | null;
  source_id?: string | null;
  pages?: number | null;
  deleted_at?: string | null;
  error_message?: string | null;
}

interface QueueStats {
  total: number;
}

const DISMISSED_KEY = 'invoicex-dismissed-queue';

function loadDismissed(): Set<string> {
  try {
    const raw = sessionStorage.getItem(DISMISSED_KEY);
    return raw ? new Set(JSON.parse(raw) as string[]) : new Set();
  } catch {
    return new Set();
  }
}

function saveDismissed(ids: Set<string>): void {
  sessionStorage.setItem(DISMISSED_KEY, JSON.stringify(Array.from(ids)));
}

function shortDocId(docId: string): string {
  return docId.replace(/^fs:/, '').slice(0, 8);
}

/** Parse ?filter= (or legacy ?mode=) from the URL, fall back to 'ready'. */
function initialFilter(search: string): QueueFilter {
  const params = new URLSearchParams(search);
  const raw = params.get('filter');
  if (isQueueFilter(raw)) { return raw; }
  // Legacy ?mode= back-compat: needs_review → ready, all_processed → all, all → all
  const mode = params.get('mode');
  if (mode === 'needs_review') { return 'ready'; }
  if (mode === 'all') { return 'all'; }
  return 'ready';
}

/** Build canonical queue query params — used by both initial fetch and loadMore. */
export function buildQueueParams({
  sort,
  filter,
  cursor,
}: {
  sort: SortAxis;
  filter: QueueFilter;
  cursor?: string | null;
}): URLSearchParams {
  const mode = filterToMode(filter);
  const p: Record<string, string> = { page_size: '50', sort, mode, filter };
  if (cursor) { p['cursor'] = cursor; }
  return new URLSearchParams(p);
}

/** Relative timestamp — e.g. "2 days ago". */
function relativeTime(iso: string | null | undefined): string | null {
  if (!iso) { return null; }
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 2) { return 'just now'; }
  if (mins < 60) { return `${mins}m ago`; }
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) { return `${hrs}h ago`; }
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export function QueuePage() {
  const styles = useStyles2(getStyles);
  const navigate = useNavigate();
  const location = useLocation();
  const [items, setItems] = useState<DocumentReviewItem[]>([]);
  const [stats, setStats] = useState<QueueStats>({ total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<{success: boolean; message: string; severity: 'success' | 'info' | 'error'} | null>(null);
  const [retryingDocs, setRetryingDocs] = useState<Set<string>>(new Set());
  const [retryResult, setRetryResult] = useState<{message: string; severity: 'success' | 'info' | 'error'} | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [wakeResult, setWakeResult] = useState<{message: string; severity: 'success' | 'info' | 'error'} | null>(null);
  const [sortBy, setSortBy] = useState<SortAxis>('recent');
  // Bootstrap: honour ?filter= (or legacy ?mode=) on first mount; default = 'ready'.
  const [queueFilter, setQueueFilter] = useState<QueueFilter>(() => initialFilter(location.search));
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [dismissed, setDismissed] = useState<Set<string>>(loadDismissed);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');

  // Unmount guard — prevents setState after component is gone
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Debounce search input — 300ms delay before firing server query
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(searchQuery), 300);
    return () => clearTimeout(t);
  }, [searchQuery]);

  // Active training poll ref — new click cancels the previous interval
  const trainingPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll refs for sync+run and per-doc retry — same cancel/timeout pattern as trainingPollRef
  const syncPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retryPollRefs = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());


  // Fetch-request lock: incremented on each queue fetch. stale responses
  // (from a previous mode/sort selection) are dropped when they settle.
  // Doubles as a double-click guard — the AbortController cancels in-flight XHR.
  const fetchSeqRef = useRef(0);
  const fetchAbortRef = useRef<AbortController | null>(null);

  // Mirror retryingDocs into a ref so per-card poll closures can check cancellation
  // without being recreated each render.
  const retryingDocsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    retryingDocsRef.current = retryingDocs;
  }, [retryingDocs]);

  const computeStats = (_buckets: unknown, total: number): QueueStats => ({ total });

  const fetchQueue = useCallback(async (clearDismissed = false, sort: SortAxis = 'recent', filter: QueueFilter = 'ready', search = '') => {
    // Cancel any in-flight fetch from a previous call
    fetchAbortRef.current?.abort();
    const controller = new AbortController();
    fetchAbortRef.current = controller;

    // Stamp this fetch; if a newer one starts before we settle, we discard
    fetchSeqRef.current += 1;
    const seq = fetchSeqRef.current;

    try {
      setLoading(true);
      const data = await queueList({ sort, filter, page_size: 50, searchQuery: search || undefined });

      // Stale-response guard: discard if a newer fetch has already started
      if (!mountedRef.current || fetchSeqRef.current !== seq) { return; }

      const docs: DocumentReviewItem[] = (data.items || []).map((item) => ({
        doc_id: item.doc_id,
        sha256: item.sha256,
        imported_at: item.imported_at,
        filename: item.filename,
        field_count: item.field_count,
        source: item.source,
        source_id: item.source_id,
      }));
      setItems(docs);
      setNextCursor(data.next_cursor || null);
      setStats(computeStats(null, data.total_count ?? docs.length));
      if (clearDismissed) {
        sessionStorage.removeItem(DISMISSED_KEY);
        setDismissed(new Set());
        setSelected(new Set());
      }
    } catch (err) {
      // AbortError is not a real error — silently ignore it
      if (err instanceof DOMException && err.name === 'AbortError') { return; }
      if (!mountedRef.current || fetchSeqRef.current !== seq) { return; }
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      if (mountedRef.current && fetchSeqRef.current === seq) {
        setLoading(false);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // Use the bootstrapped queueFilter (may have been parsed from ?filter= URL param)
    fetchQueue(false, sortBy, queueFilter);
  // Run once on mount — sortBy and queueFilter are initialised before this fires
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchQueue]);

  // Re-fetch when debounced search term changes (after initial mount)
  const isMountedSearchRef = useRef(false);
  useEffect(() => {
    if (!isMountedSearchRef.current) {
      isMountedSearchRef.current = true;
      return;
    }
    fetchQueue(false, sortBy, queueFilter, debouncedSearch);
  // fetchQueue is stable (useCallback with [] deps); sortBy/queueFilter/debouncedSearch are the triggers
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSearch]);

  const handleDismiss = (docId: string) => {
    setDismissed((prev) => {
      const next = new Set(prev);
      next.add(docId);
      saveDismissed(next);
      return next;
    });
  };

  const handleBulkDismiss = () => {
    if (selected.size === 0) {return;}
    setDismissed((prev) => {
      const next = new Set([...prev, ...selected]);
      saveDismissed(next);
      return next;
    });
    setSelected(new Set());
  };

  const handleToggleSelect = (docId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) {
        next.delete(docId);
      } else {
        next.add(docId);
      }
      return next;
    });
  };

  const handleSelectAll = () => {
    const allSelected = visibleIds.length > 0 && visibleIds.every((id) => selected.has(id));
    if (allSelected) {
      // All visible selected → deselect all
      setSelected(new Set());
    } else {
      // None or partial → select all visible
      setSelected(new Set(visibleIds));
    }
  };

  const loadMore = async () => {
    if (!nextCursor || loadingMore) {return;}
    try {
      setLoadingMore(true);
      const data = await queueList({ sort: sortBy, filter: queueFilter, page_size: 50, cursor: nextCursor });
      const newDocs: DocumentReviewItem[] = (data.items || []).map((item) => ({
        doc_id: item.doc_id,
        sha256: item.sha256,
        imported_at: item.imported_at,
        filename: item.filename,
        field_count: item.field_count,
        source: item.source,
        source_id: item.source_id,
      }));
      setItems((prev) => [...prev, ...newDocs]);
      setStats((s) => computeStats(null, s.total));
      setNextCursor(data.next_cursor || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoadingMore(false);
    }
  };

  /** Shared POST helper that handles non-JSON error responses gracefully. */
  const postAction = async (url: string): Promise<{ ok: boolean; data: any }> => {
    const response = await fetch(url, { method: 'POST' });
    let data: any;
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      data = await response.json();
    } else {
      const text = await response.text();
      data = { message: text || `Request failed (${response.status})` };
    }
    return { ok: response.ok, data };
  };

  const handleRetrain = async () => {
    // Stop any existing poll before starting a new one.
    if (trainingPollRef.current !== null) {
      clearInterval(trainingPollRef.current);
      trainingPollRef.current = null;
    }
    setRetraining(true);
    setRetrainResult(null);

    // Capture baseline run_id BEFORE enqueuing. Schema-edit appends also bump
    // version, so we key on run_id (present only in retrain appends) to avoid
    // false completions when the user saves schema changes during the poll window.
    let baselineRunId: string | null;
    try {
      baselineRunId = await getLatestModelRunId();
    } catch {
      // If we can't read the baseline, null is safe: any non-null run_id seen
      // later will fire completion correctly (null → non-null is the first-ever
      // retrain case, which is also correct).
      baselineRunId = null;
    }

    try {
      await triggerRetrain();
    } catch (err) {
      setRetrainResult({
        success: false,
        message: err instanceof Error ? err.message : 'Could not connect to the server. Please try again.',
        severity: 'error',
      });
      setRetraining(false);
      return;
    }

    // Enqueue succeeded — keep retraining=true while we poll for completion.
    setRetrainResult({
      success: true,
      message: 'Training queued — watching for model completion…',
      severity: 'info',
    });

    // Safety timeout: stop polling after 10 minutes.
    const POLL_TIMEOUT_MS = 10 * 60 * 1000;
    const pollStartedAt = Date.now();

    trainingPollRef.current = setInterval(async () => {
      // Unmount guard
      if (!mountedRef.current) {
        clearInterval(trainingPollRef.current!);
        trainingPollRef.current = null;
        return;
      }

      // Safety timeout
      if (Date.now() - pollStartedAt > POLL_TIMEOUT_MS) {
        clearInterval(trainingPollRef.current!);
        trainingPollRef.current = null;
        if (mountedRef.current) {
          setRetrainResult({
            success: true,
            message: 'Training is taking longer than expected — check back shortly.',
            severity: 'info',
          });
          setRetraining(false);
        }
        return;
      }

      let latest: string | null;
      try {
        latest = await getLatestModelRunId();
      } catch {
        // Network blip — swallow and retry on next tick
        return;
      }

      if (latest !== null && latest !== baselineRunId) {
        // A new retrain run_id has been registered — model is active.
        clearInterval(trainingPollRef.current!);
        trainingPollRef.current = null;
        if (mountedRef.current) {
          setRetrainResult({
            success: true,
            message: 'New model active — re-scoring your labeled docs',
            severity: 'success',
          });
          await fetchQueue(false, sortBy, queueFilter, debouncedSearch);
          setRetraining(false);
        }
      }
    }, 4500);
  };

  // Cleanup all poll intervals on unmount.
  useEffect(() => {
    const retryPollMap = retryPollRefs.current;
    return () => {
      if (trainingPollRef.current !== null) {
        clearInterval(trainingPollRef.current);
      }
      if (syncPollRef.current !== null) {
        clearInterval(syncPollRef.current);
      }
      retryPollMap.forEach((id) => clearInterval(id));
      retryPollMap.clear();
    };
  }, []);

  const handleSyncAndRun = async () => {
    // Cancel any prior sync poll before starting a new one
    if (syncPollRef.current !== null) {
      clearInterval(syncPollRef.current);
      syncPollRef.current = null;
    }
    setSyncing(true);
    setWakeResult(null);

    try {
      await wakePipeline();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not connect to the server. Please try again.';
      setWakeResult({ message: `Extraction trigger failed — ${message}`, severity: 'error' });
      await fetchQueue(true, sortBy, queueFilter, debouncedSearch);
      setSyncing(false);
      return;
    }

    // Capture baseline ready count to detect any new docs completing extraction
    let baselineReadyCount: number;
    try {
      const baseline = await queueList({ sort: sortBy, filter: 'ready', page_size: 50 });
      baselineReadyCount = baseline.total_count ?? 0;
    } catch {
      baselineReadyCount = 0;
    }

    setWakeResult({ message: 'Sync + extraction queued — watching for new ready documents…', severity: 'info' });

    const POLL_TIMEOUT_MS = 2 * 60 * 1000;
    const pollStartedAt = Date.now();

    syncPollRef.current = setInterval(async () => {
      if (!mountedRef.current) {
        clearInterval(syncPollRef.current!);
        syncPollRef.current = null;
        return;
      }

      if (Date.now() - pollStartedAt > POLL_TIMEOUT_MS) {
        clearInterval(syncPollRef.current!);
        syncPollRef.current = null;
        if (mountedRef.current) {
          setWakeResult({ message: 'Sync queued — extraction is taking longer than expected, check back shortly', severity: 'info' });
          await fetchQueue(true, sortBy, queueFilter, debouncedSearch);
          setSyncing(false);
        }
        return;
      }

      let latest: number;
      try {
        const res = await queueList({ sort: sortBy, filter: 'ready', page_size: 50 });
        latest = res.total_count ?? 0;
      } catch {
        return;
      }

      if (latest > baselineReadyCount) {
        clearInterval(syncPollRef.current!);
        syncPollRef.current = null;
        if (mountedRef.current) {
          setWakeResult({ message: 'Sync complete · new documents ready for review', severity: 'success' });
          await fetchQueue(true, sortBy, queueFilter, debouncedSearch);
          setSyncing(false);
        }
      }
    }, 4500);
  };

  const handleRetryDoc = async (sha256: string, docId: string) => {
    // Guard against double-click on the same card
    if (retryingDocsRef.current.has(sha256)) {
      return;
    }
    setRetryingDocs((prev) => {
      const next = new Set(prev);
      next.add(sha256);
      return next;
    });
    setRetryResult(null);

    const clearRetrying = () => {
      if (!mountedRef.current) {
        return;
      }
      setRetryingDocs((prev) => {
        const next = new Set(prev);
        next.delete(sha256);
        return next;
      });
    };

    try {
      await reextractDoc(sha256);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not connect to the server. Please try again.';
      setRetryResult({ message, severity: 'error' });
      clearRetrying();
      return;
    }

    setRetryResult({
      message: `Re-extracting ${shortDocId(docId)}… — watching for completion`,
      severity: 'info',
    });

    // Poll for the doc to appear in 'ready' (payload ? 'doc' = true = extraction complete).
    // 'failed' is an alias for 'pending' in the DB — no distinct failure state exists.
    // Timeout after 2 min with an honest uncertainty message.
    const POLL_TIMEOUT_MS = 2 * 60 * 1000;
    const pollStartedAt = Date.now();

    const pollId = setInterval(async () => {
      if (!mountedRef.current) {
        clearInterval(pollId);
        retryPollRefs.current.delete(sha256);
        return;
      }

      if (Date.now() - pollStartedAt > POLL_TIMEOUT_MS) {
        clearInterval(pollId);
        retryPollRefs.current.delete(sha256);
        if (mountedRef.current) {
          setRetryResult({
            message: `Re-extraction of ${shortDocId(docId)} is taking longer than expected — check back shortly`,
            severity: 'info',
          });
          await fetchQueue(true, sortBy, queueFilter);
          clearRetrying();
        }
        return;
      }

      let found = false;
      try {
        const res = await queueList({ sort: 'recent', filter: 'ready', page_size: 50 });
        found = (res.items ?? []).some((item) => item.sha256 === sha256);
      } catch {
        return;
      }

      if (found) {
        clearInterval(pollId);
        retryPollRefs.current.delete(sha256);
        if (mountedRef.current) {
          setRetryResult({
            message: `${shortDocId(docId)} re-extracted successfully`,
            severity: 'success',
          });
          await fetchQueue(true, sortBy, queueFilter);
          clearRetrying();
        }
      }
    }, 4500);

    retryPollRefs.current.set(sha256, pollId);
  };

  const handleReview = (sha256: string) => {
    navigate(`/label/${sha256}?filter=${queueFilter}`);
  };

  // Sort options — 'priority' and 'confidence' removed (inference-derived, Wave 5A)
  const sortOptions: Array<SelectableValue<SortAxis>> = [
    { label: 'Recent', value: 'recent', description: 'Most recently imported' },
    { label: 'Alphabetical', value: 'alphabetical', description: 'By filename A–Z' },
  ];

  const handleSortChange = (v: SelectableValue<SortAxis>) => {
    if (!v.value) {return;}
    const next = v.value;
    setSortBy(next);
    setNextCursor(null);
    fetchQueue(false, next, queueFilter, debouncedSearch);
  };

  const handleFilterChange = (f: QueueFilter) => {
    setQueueFilter(f);
    setNextCursor(null);
    navigate(`?filter=${f}`, { replace: true });
    // Clear dismissed set on filter switch — dismissed is session-local and stale across filters.
    fetchQueue(true, sortBy, f, debouncedSearch);
  };

  // Filter dismissed docs — ordering comes from the server
  const sortedItems = useMemo(() => {
    if (dismissed.size === 0) { return items; }
    return items.filter((d) => !dismissed.has(d.doc_id));
  }, [items, dismissed]);

  // Count only IDs that are dismissed AND present in the currently loaded items.
  // dismissed.size alone overstates when server has removed docs this session.
  const hiddenCount = useMemo(
    () => items.filter((d) => dismissed.has(d.doc_id)).length,
    [items, dismissed]
  );

  // Select-all checkbox tri-state derived from visible set
  const visibleIds = useMemo(() => sortedItems.map((d) => d.doc_id), [sortedItems]);
  const allVisibleSelected = visibleIds.length > 0 && visibleIds.every((id) => selected.has(id));
  const someVisibleSelected = !allVisibleSelected && visibleIds.some((id) => selected.has(id));

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <h1 className={styles.title}>Documents</h1>
          <p className={styles.subtitle}>All documents the system knows about.</p>
        </div>
        <div className={styles.headerActions}>
          <Button onClick={() => navigate('/model')} icon="graph-bar" variant="secondary">
            Model
          </Button>
          <Button onClick={() => navigate('/schema')} icon="cog" variant="secondary">
            Field Manager
          </Button>
          <Button onClick={handleRetrain} icon="process" variant="primary" disabled={retraining}>
            {retraining ? 'Training…' : 'Retrain'}
          </Button>
          <Button onClick={handleSyncAndRun} icon="sync" variant="secondary" disabled={syncing} title="Sync ledger and trigger extraction">
            {syncing ? 'Syncing…' : 'Sync + Run'}
          </Button>
        </div>
      </div>

      {/* Status indicators */}
      <div className={styles.statusRow}>
        <span className={styles.statusItem}>
          <span className={styles.statusCount}>{stats.total}</span> in queue
        </span>
      </div>

      {retrainResult && (
        <Alert
          title={
            retrainResult.severity === 'success'
              ? 'Learning Complete'
              : retrainResult.severity === 'info'
                ? 'Training Progress'
                : 'Learning Failed'
          }
          severity={retrainResult.severity}
          onRemove={() => setRetrainResult(null)}
        >
          {retrainResult.message}
        </Alert>
      )}

      {wakeResult && (
        <Alert
          title={
            wakeResult.severity === 'error'
              ? 'Extraction Trigger Failed'
              : wakeResult.severity === 'success'
                ? 'Sync Complete'
                : 'Sync In Progress'
          }
          severity={wakeResult.severity}
          onRemove={() => setWakeResult(null)}
        >
          {wakeResult.message}
        </Alert>
      )}

      {retryResult && (
        <Alert
          title={
            retryResult.severity === 'error'
              ? 'Retry Failed'
              : retryResult.severity === 'success'
                ? 'Document Re-extracted'
                : 'Retry In Progress'
          }
          severity={retryResult.severity}
          onRemove={() => setRetryResult(null)}
        >
          {retryResult.message}
        </Alert>
      )}

      {/* Queue List */}
      <div className={styles.queueSection}>
        <div className={styles.queueHeader}>
          <h2 className={styles.sectionTitle}>
            Documents
          </h2>
          <div className={styles.queueControls}>
            <Input
              placeholder="Search by filename or vendor..."
              prefix={<Icon name="search" />}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.currentTarget.value)}
              width={32}
            />
            <div className={styles.sortControl}>
              <span className={styles.sortLabel}>Sort by</span>
              <Select
                options={sortOptions}
                value={sortOptions.find((o) => o.value === sortBy)}
                onChange={handleSortChange}
                width={20}
                menuPlacement="bottom"
              />
            </div>
          </div>
        </div>

        {/* Filter chips — Ready / Pending / Failed / Labeled / All */}
        <div className={styles.filterChips}>
          {([
            { key: 'ready',   label: 'Ready to process' },
            { key: 'pending', label: 'Pending' },
            { key: 'failed',  label: 'Failed' },
            { key: 'labeled', label: 'Labeled' },
            { key: 'all',     label: 'All' },
          ] as Array<{ key: QueueFilter; label: string }>).map(({ key, label }) => (
            <button
              key={key}
              className={`${styles.filterChip} ${queueFilter === key ? styles.filterChipActive : ''}`}
              onClick={() => handleFilterChange(key)}
              type="button"
            >
              {label}
            </button>
          ))}
        </div>

        {/* Bulk selection toolbar — visible only when items are loaded */}
        {!loading && !error && items.length > 0 && (
          <div className={styles.bulkToolbar}>
            <Checkbox
              label={`Select all (${visibleIds.length})`}
              checked={allVisibleSelected}
              indeterminate={someVisibleSelected}
              onChange={handleSelectAll}
            />
            {selected.size > 0 && (
              <Button
                variant="secondary"
                size="sm"
                icon="times"
                onClick={handleBulkDismiss}
              >
                Dismiss {selected.size} selected
              </Button>
            )}
            {hiddenCount > 0 && (
              <span className={styles.hiddenBanner}>
                {hiddenCount} hidden
                <Button
                  variant="secondary"
                  fill="text"
                  size="sm"
                  onClick={() => fetchQueue(true, sortBy, queueFilter)}
                >
                  Show all
                </Button>
              </span>
            )}
          </div>
        )}

        {loading && (
          <div className={getLoadingContainerStyles(styles.theme)}>
            <Spinner size="xl" />
            <p>Loading queue...</p>
          </div>
        )}

        {error && (
          <div className={getErrorContainerStyles(styles.theme)}>
            <p>Error loading queue: {error}</p>
            <Button onClick={() => fetchQueue()}>Retry</Button>
          </div>
        )}

        {!loading && !error && items.length === 0 && (
          <div className={styles.emptyContainer}>
            <p>
              {queueFilter === 'all'
                ? 'No documents yet. Click Sync + Run to ingest from SharePoint.'
                : queueFilter === 'ready'
                ? 'No ready documents.'
                : queueFilter === 'pending'
                ? 'No pending documents.'
                : queueFilter === 'labeled'
                ? 'No labeled documents yet. Label some documents to see them here.'
                : 'No failed documents.'}
            </p>
          </div>
        )}

        {!loading && !error && items.length > 0 && (
          <>
            <div className={styles.queueList}>
              {sortedItems.map((doc) => {
                const docTitle = doc.filename || `(unnamed: ${shortDocId(doc.doc_id)})`;
                // F1: derive from the real in-progress state (keyed by sha256, same as retryingDocs)
                const isReextracting = retryingDocs.has(doc.sha256);
                const importedAgo = relativeTime(doc.imported_at);
                return (
                  <Card key={doc.doc_id} className={`${styles.queueCard} ${doc.deleted_at ? styles.queueCardDeleted : ''}`}>
                    <Card.Heading>
                      <div className={styles.cardHeader}>
                        {/* stopPropagation: prevent checkbox click from bubbling to card */}
                        <span onClick={(e) => e.stopPropagation()}>
                          <Checkbox
                            checked={selected.has(doc.doc_id)}
                            onChange={() => handleToggleSelect(doc.doc_id)}
                            aria-label={`Select document ${docTitle}`}
                          />
                        </span>
                        <span className={styles.docTitle} title={`${docTitle} · ${doc.doc_id}`}>
                          {docTitle}
                        </span>
                        {doc.deleted_at && (
                          <Icon name="trash-alt" title="Soft-deleted" style={{ marginLeft: 6, opacity: 0.5 }} />
                        )}
                        {isReextracting && (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, marginLeft: 8, opacity: 0.7 }}>
                            <Spinner size="sm" />
                            <span style={{ fontSize: '0.8em' }}>Re-extracting…</span>
                          </span>
                        )}
                      </div>
                    </Card.Heading>
                    <Card.Description>
                      <div className={styles.cardContent}>
                        {/* doc id · imported time · status */}
                        <div className={styles.cardMeta}>
                          <span title={doc.doc_id}>{shortDocId(doc.doc_id)}</span>
                          {importedAgo && <span>{importedAgo}</span>}
                          {doc.source && <span title={doc.source_id ?? undefined}>{doc.source}</span>}
                          {doc.pages != null && <span>{doc.pages}p</span>}
                        </div>
                        {/* Field count */}
                        <div className={styles.fieldInfo}>
                          <span className={styles.fieldLabel}>Fields:</span>
                          <span className={styles.fieldValue}>{doc.field_count}</span>
                        </div>
                      </div>
                    </Card.Description>
                    <Card.Actions>
                      <Button
                        variant="primary"
                        onClick={() => handleReview(doc.sha256)}
                        icon="eye"
                        disabled={isReextracting}
                      >
                        Review
                      </Button>
                      <Button
                        variant="secondary"
                        icon="repeat"
                        onClick={() => handleRetryDoc(doc.sha256, doc.doc_id)}
                        disabled={retryingDocs.has(doc.sha256) || isReextracting}
                        title="Re-extract this document"
                      >
                        {retryingDocs.has(doc.sha256) ? 'Retrying...' : 'Retry'}
                      </Button>
                      <Button
                        variant="secondary"
                        fill="text"
                        icon="times"
                        onClick={() => handleDismiss(doc.doc_id)}
                        title="Hide this document until next refresh"
                      >
                        Dismiss
                      </Button>
                    </Card.Actions>
                  </Card>
                );
              })}
            </div>
            {nextCursor && (
              <div className={styles.loadMoreContainer}>
                <Button onClick={loadMore} variant="secondary" disabled={loadingMore}>
                  {loadingMore ? 'Loading...' : 'Load More'}
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

const getStyles = (theme: GrafanaTheme2) => ({
  theme, // Pass theme for shared style functions
  container: css`
    padding: ${theme.spacing(3)};
    max-width: 1400px;
    margin: 0 auto;
  `,
  header: css`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: ${theme.spacing(4)};
    padding: ${theme.spacing(3)};
    background: linear-gradient(135deg, ${theme.colors.primary.main} 0%, ${theme.colors.primary.shade} 100%);
    border-radius: ${theme.shape.radius.default};
    color: ${theme.colors.primary.contrastText};
  `,
  headerContent: css``,
  headerActions: css`
    display: flex;
    gap: ${theme.spacing(1)};
    align-items: center;
  `,
  statusRow: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(1.5)};
    padding: ${theme.spacing(1)} ${theme.spacing(2)};
    margin-bottom: ${theme.spacing(2)};
    font-size: 12px;
    color: ${theme.colors.text.secondary};
  `,
  statusItem: css`
    white-space: nowrap;
  `,
  statusCount: css`
    font-weight: 600;
    color: ${theme.colors.text.primary};
  `,
  statusDot: css`
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: ${theme.colors.text.disabled};
  `,
  title: css`
    margin: 0;
    font-size: 28px;
    font-weight: 600;
  `,
  subtitle: css`
    margin: ${theme.spacing(1)} 0 0 0;
    opacity: 0.9;
  `,
  statsRow: css`
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: ${theme.spacing(2)};
    margin-bottom: ${theme.spacing(4)};
  `,
  statCard: css`
    padding: ${theme.spacing(3)};
    border-radius: ${theme.shape.radius.default};
    text-align: center;
    background: ${theme.colors.background.secondary};
    border: 1px solid ${theme.colors.border.weak};
    cursor: pointer;
    transition: all 0.2s ease;
    opacity: 0.7;

    &:hover {
      transform: translateY(-1px);
      box-shadow: ${theme.shadows.z2};
    }
  `,
  statCardActive: css`
    opacity: 1;
    border-width: 2px;
    box-shadow: ${theme.shadows.z1};
  `,
  statUrgent: css`
    border-left: 4px solid ${COLORS.error};
  `,
  statMedium: css`
    border-left: 4px solid ${COLORS.warning};
  `,
  statLow: css`
    border-left: 4px solid ${COLORS.success};
  `,
  statTotal: css`
    border-left: 4px solid ${theme.colors.primary.main};
  `,
  statValue: css`
    font-size: 36px;
    font-weight: 700;
    color: ${theme.colors.text.primary};
  `,
  statLabel: css`
    font-size: 14px;
    color: ${theme.colors.text.secondary};
    text-transform: uppercase;
    letter-spacing: 0.5px;
  `,
  queueSection: css``,
  queueHeader: css`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: ${theme.spacing(2)};
    gap: ${theme.spacing(2)};
  `,
  queueControls: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(2)};
    flex-shrink: 0;
  `,
  sectionTitle: css`
    font-size: 20px;
    margin: 0;
    display: flex;
    align-items: center;
    gap: ${theme.spacing(1)};
  `,
  sortControl: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(1)};
    flex-shrink: 0;
  `,
  sortLabel: css`
    font-size: 13px;
    color: ${theme.colors.text.secondary};
    white-space: nowrap;
  `,
  bulkToolbar: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(2)};
    padding: ${theme.spacing(1)} ${theme.spacing(1)};
    margin-bottom: ${theme.spacing(2)};
    min-height: 36px;
  `,
  hiddenBanner: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(0.5)};
    margin-left: auto;
    font-size: 13px;
    color: ${theme.colors.text.secondary};
  `,
  emptyContainer: css`
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: ${theme.spacing(6)};
    background: ${theme.colors.success.transparent};
    border-radius: ${theme.shape.radius.default};
    color: ${theme.colors.success.text};
  `,
  emptyIcon: css`
    font-size: 48px;
    margin-bottom: ${theme.spacing(2)};
  `,
  emptySwitchHint: css`
    margin-top: ${theme.spacing(2)};
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: ${theme.spacing(1)};
    font-size: 13px;
    opacity: 0.85;
  `,
  queueList: css`
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: ${theme.spacing(2)};
  `,
  queueCard: css`
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    &:hover {
      transform: translateY(-2px);
      box-shadow: ${theme.shadows.z3};
    }
  `,
  cardHeader: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(1)};
  `,
  docTitle: css`
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: ${theme.colors.text.primary};
    font-weight: 600;
  `,
  cardContent: css`
    margin-top: ${theme.spacing(1)};
  `,
  fieldInfo: css`
    display: flex;
    gap: ${theme.spacing(1)};
    margin-bottom: ${theme.spacing(0.5)};
  `,
  fieldLabel: css`
    color: ${theme.colors.text.secondary};
    font-size: 13px;
  `,
  fieldValue: css`
    font-weight: 600;
    font-size: 13px;
  `,
  filenameValue: css`
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 220px;
  `,
  confidenceBar: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(1)};
    margin-top: ${theme.spacing(1)};
  `,
  confidenceLabel: css`
    color: ${theme.colors.text.secondary};
    font-size: 12px;
  `,
  barContainer: css`
    flex: 1;
    height: 8px;
    background: ${theme.colors.background.canvas};
    border-radius: 4px;
    overflow: hidden;
  `,
  barFill: css`
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
  `,
  confidenceValue: css`
    font-size: 12px;
    font-weight: 600;
    min-width: 36px;
    text-align: right;
  `,
  reason: css`
    font-size: 13px;
    color: ${theme.colors.text.secondary};
    margin: 0;
  `,
  loadMoreContainer: css`
    display: flex;
    justify-content: center;
    margin-top: ${theme.spacing(3)};
  `,
  filterChips: css`
    display: flex;
    gap: ${theme.spacing(1)};
    margin-bottom: ${theme.spacing(2)};
    flex-wrap: wrap;
  `,
  filterChip: css`
    padding: ${theme.spacing(0.5)} ${theme.spacing(1.5)};
    border-radius: ${theme.shape.radius.pill};
    border: 1px solid ${theme.colors.border.medium};
    background: ${theme.colors.background.secondary};
    color: ${theme.colors.text.secondary};
    font-size: 13px;
    cursor: pointer;
    line-height: 1.6;
    transition: all 0.15s ease;
    &:hover {
      border-color: ${theme.colors.primary.border};
      color: ${theme.colors.text.primary};
    }
  `,
  filterChipActive: css`
    background: ${theme.colors.primary.transparent};
    border-color: ${theme.colors.primary.border};
    color: ${theme.colors.primary.text};
    font-weight: 600;
  `,
  cardMeta: css`
    display: flex;
    gap: ${theme.spacing(1)};
    font-size: 12px;
    color: ${theme.colors.text.secondary};
    margin-bottom: ${theme.spacing(0.5)};
    flex-wrap: wrap;
    align-items: center;
  `,
  cardMetaStrong: css`
    font-weight: 600;
    color: ${theme.colors.text.primary};
  `,
  worstField: css`
    font-size: 12px;
    color: ${theme.colors.text.disabled};
    margin-left: ${theme.spacing(0.5)};
  `,
  queueCardDeleted: css`
    opacity: 0.6;
    border-style: dashed;
  `,
});
