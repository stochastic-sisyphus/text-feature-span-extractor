import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { css } from '@emotion/css';
import { GrafanaTheme2 } from '@grafana/data';
import { useStyles2, useTheme2, Alert, Button, Spinner, Icon, Switch } from '@grafana/ui';
import { useNavigate } from 'react-router-dom';
import { fetchExtractionResults, type ExtractionResultRow } from '../utils/api';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function confidenceColor(theme: GrafanaTheme2, conf: number | null): string {
  if (conf === null) { return theme.colors.text.disabled; }
  if (conf >= 0.8) { return theme.colors.success.main; }
  if (conf >= 0.5) { return theme.colors.warning.main; }
  return theme.colors.error.main;
}

function confidenceLabel(conf: number | null): string {
  if (conf === null) { return '—'; }
  return `${Math.round(conf * 100)}%`;
}

function shortFilename(name: string | null): string {
  if (!name) { return '(no name)'; }
  return name.length > 40 ? name.slice(0, 38) + '…' : name;
}

// ---------------------------------------------------------------------------
// Pivot: rows = unique docs, columns = unique fields
// ---------------------------------------------------------------------------

interface PivotTable {
  fields: string[];
  rows: Array<{
    doc_id: string;
    filename: string | null;
    evaluated_at: string;
    cells: Record<string, { value: string | null; confidence: number | null }>;
  }>;
}

function pivotResults(rows: ExtractionResultRow[]): PivotTable {
  const fieldSet = new Set<string>();
  const docMap = new Map<string, PivotTable['rows'][0]>();

  for (const row of rows) {
    fieldSet.add(row.field);
    if (!docMap.has(row.doc_id)) {
      docMap.set(row.doc_id, {
        doc_id: row.doc_id,
        filename: row.filename,
        evaluated_at: row.evaluated_at,
        cells: {},
      });
    }
    const doc = docMap.get(row.doc_id)!;
    doc.cells[row.field] = {
      value: row.extracted_value,
      confidence: row.confidence,
    };
    // keep latest evaluated_at so the table sorts by most-recent evaluation
    if (row.evaluated_at > doc.evaluated_at) {
      doc.evaluated_at = row.evaluated_at;
    }
  }

  const fields = Array.from(fieldSet).sort();
  const pivotRows = Array.from(docMap.values()).sort(
    (a, b) => b.evaluated_at.localeCompare(a.evaluated_at)
  );

  return { fields, rows: pivotRows };
}

function relativeTime(iso: string): string {
  try {
    const parsed = new Date(iso).getTime();
    if (isNaN(parsed)) { return iso; }
    const diffMs = Date.now() - parsed;
    const diffMins = Math.floor(diffMs / (1000 * 60));
    if (diffMins < 1) { return 'just now'; }
    if (diffMins < 60) { return `${diffMins}m ago`; }
    const diffHrs = Math.floor(diffMins / 60);
    if (diffHrs < 24) { return `${diffHrs}h ago`; }
    return `${Math.floor(diffHrs / 24)}d ago`;
  } catch {
    return iso;
  }
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

function EmptyState({ styles }: { styles: ReturnType<typeof getStyles> }) {
  return (
    <div className={styles.emptyOuter}>
      <div className={styles.emptyCard}>
        <Icon name="document-info" size="xxl" />
        <h3 className={styles.emptyHeadline}>No extractions yet</h3>
        <p className={styles.emptyText}>
          Once invoices are processed, extracted field values will appear here.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ResultsPage() {
  const styles = useStyles2(getStyles);
  const theme = useTheme2();
  const navigate = useNavigate();

  const [rows, setRows] = useState<ExtractionResultRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastLoadedAt, setLastLoadedAt] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const mountedRef = useRef(true);
  const inFlightRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // load(background=false) for manual refresh — shows spinner + resets error.
  // load(background=true) for polling — silently swaps data, preserves scroll.
  const load = useCallback(async (background = false) => {
    if (inFlightRef.current) { return; }
    inFlightRef.current = true;
    if (!background) {
      setLoading(true);
      setError(null);
    }
    try {
      const data = await fetchExtractionResults();
      if (mountedRef.current) {
        setRows(data);
        setLastLoadedAt(new Date());
        if (!background) { setError(null); }
      }
    } catch (err) {
      if (mountedRef.current && !background) {
        setError(err instanceof Error ? err.message : 'Failed to load results');
      }
    } finally {
      inFlightRef.current = false;
      if (mountedRef.current && !background) { setLoading(false); }
    }
  }, []);

  // Initial load
  useEffect(() => { load(false); }, [load]);

  // Auto-refresh polling — 15s interval, paused when autoRefresh=false
  useEffect(() => {
    if (!autoRefresh) { return; }
    const id = setInterval(() => { load(true); }, 15_000);
    return () => clearInterval(id);
  }, [autoRefresh, load]);

  const pivot = useMemo(() => pivotResults(rows), [rows]);

  const hasData = !loading && !error && pivot.rows.length > 0;
  const isEmpty = !loading && !error && pivot.rows.length === 0;

  // -------------------------------------------------------------------------
  // Navigation
  // -------------------------------------------------------------------------
  const handleRowClick = useCallback((doc_id: string) => {
    navigate(`/label/${doc_id}`);
  }, [navigate]);

  const handleRowKeyDown = useCallback((e: React.KeyboardEvent, doc_id: string) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      navigate(`/label/${doc_id}`);
    }
  }, [navigate]);

  // -------------------------------------------------------------------------
  // Export helpers (client-side, no fetch)
  // -------------------------------------------------------------------------
  const exportJSON = useCallback(() => {
    const data = pivot.rows.map((row) => {
      const obj: Record<string, string | null> = {
        document: row.filename ?? row.doc_id,
        doc_id: row.doc_id,
        evaluated_at: row.evaluated_at,
      };
      for (const f of pivot.fields) {
        obj[f] = row.cells[f]?.value ?? null;
      }
      return obj;
    });
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `extraction-results-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [pivot]);

  const exportCSV = useCallback(() => {
    const escape = (v: string | null | undefined): string => {
      if (v === null || v === undefined) { return ''; }
      let s = String(v);
      // Neutralize spreadsheet formula injection (CSV injection): force text on
      // values that a spreadsheet would interpret as a formula.
      if (/^[=+\-@\t\r]/.test(s)) { s = "'" + s; }
      if (s.includes(',') || s.includes('"') || s.includes('\n')) {
        return '"' + s.replace(/"/g, '""') + '"';
      }
      return s;
    };
    const header = ['Document', 'doc_id', 'Evaluated', ...pivot.fields].map(escape).join(',');
    const bodyLines = pivot.rows.map((row) => {
      const cols = [
        escape(row.filename ?? row.doc_id),
        escape(row.doc_id),
        escape(row.evaluated_at),
        ...pivot.fields.map((f) => escape(row.cells[f]?.value ?? null)),
      ];
      return cols.join(',');
    });
    const csv = [header, ...bodyLines].join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `extraction-results-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [pivot]);

  const avgConf = useMemo(() => {
    const vals = rows.map((r) => r.confidence).filter((c): c is number => c !== null);
    if (!vals.length) { return null; }
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }, [rows]);

  return (
    <div className={styles.container}>

      {/* Header */}
      <div className={styles.headerStrip}>
        <div>
          <h1 className={styles.headerTitle}>Extraction Results</h1>
          {!loading && rows.length > 0 && (
            <p className={styles.headerMeta}>
              {pivot.rows.length} docs · {pivot.fields.length} fields · avg confidence{' '}
              {avgConf !== null ? `${Math.round(avgConf * 100)}%` : '—'}
            </p>
          )}
        </div>
        <div className={styles.headerActions}>
          {lastLoadedAt && (
            <span className={styles.lastUpdated}>
              Updated {relativeTime(lastLoadedAt.toISOString())}
            </span>
          )}
          <div className={styles.autoRefreshToggle}>
            <span className={styles.autoRefreshLabel}>Auto-refresh</span>
            <Switch value={autoRefresh} onChange={() => setAutoRefresh((v) => !v)} />
          </div>
          <Button variant="secondary" icon="sync" size="sm" onClick={() => load(false)} disabled={loading}>
            {loading ? <><Spinner size="sm" /> Loading…</> : 'Refresh'}
          </Button>
          {hasData && (
            <>
              <Button variant="secondary" icon="download-alt" size="sm" onClick={exportCSV}>
                Export CSV
              </Button>
              <Button variant="secondary" icon="download-alt" size="sm" onClick={exportJSON}>
                Export JSON
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <Alert severity="error" title="Failed to load extraction results">
          {error}{' '}
          <Button size="sm" variant="secondary" onClick={() => load(false)}>Retry</Button>
        </Alert>
      )}

      {/* Loading */}
      {loading && (
        <div className={styles.loadingRow}>
          <Spinner size="lg" />
        </div>
      )}

      {/* Empty */}
      {isEmpty && <EmptyState styles={styles} />}

      {/* Table */}
      {hasData && (
        <div className={styles.tableWrapper}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th className={styles.thSticky}>Document</th>
                <th className={styles.th}>Evaluated</th>
                {pivot.fields.map((f) => (
                  <th key={f} className={styles.th} title={f}>
                    {f.replace(/_/g, ' ')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pivot.rows.map((row) => (
                <tr
                  key={row.doc_id}
                  className={styles.trClickable}
                  role="button"
                  tabIndex={0}
                  onClick={() => handleRowClick(row.doc_id)}
                  onKeyDown={(e) => handleRowKeyDown(e, row.doc_id)}
                >
                  <td className={styles.tdFilename} title={row.filename ?? row.doc_id}>
                    <span className={styles.filename}>
                      {shortFilename(row.filename)}
                      <Icon name="angle-right" size="sm" className={styles.openIcon} />
                    </span>
                    <span className={styles.docHash} title={row.doc_id}>
                      {row.doc_id.slice(0, 8)}
                    </span>
                  </td>
                  <td className={styles.tdMeta}>
                    {relativeTime(row.evaluated_at)}
                  </td>
                  {pivot.fields.map((f) => {
                    const cell = row.cells[f];
                    if (!cell) {
                      return <td key={f} className={styles.tdEmpty}>—</td>;
                    }
                    const color = confidenceColor(theme, cell.confidence);
                    return (
                      <td key={f} className={styles.td}>
                        <div className={styles.cellValue} title={cell.value ?? ''}>
                          {cell.value ? (
                            cell.value.length > 22
                              ? cell.value.slice(0, 20) + '…'
                              : cell.value
                          ) : (
                            <span className={styles.noValue}>—</span>
                          )}
                        </div>
                        <div
                          className={styles.confChip}
                          style={{ color, borderColor: color }}
                          title={`Confidence: ${cell.confidence !== null ? cell.confidence.toFixed(3) : 'n/a'}`}
                        >
                          {confidenceLabel(cell.confidence)}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const getStyles = (theme: GrafanaTheme2) => ({
  container: css`
    padding: ${theme.spacing(3)};
    max-width: 1600px;
    margin: 0 auto;
  `,
  headerStrip: css`
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: ${theme.spacing(2)};
    margin-bottom: ${theme.spacing(3)};
    padding: ${theme.spacing(2.5)} ${theme.spacing(3)};
    background: linear-gradient(135deg, ${theme.colors.primary.main} 0%, ${theme.colors.primary.shade} 100%);
    border-radius: ${theme.shape.radius.default};
    color: ${theme.colors.primary.contrastText};
  `,
  headerTitle: css`
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    color: ${theme.colors.primary.contrastText};
  `,
  headerMeta: css`
    margin: ${theme.spacing(0.5)} 0 0 0;
    font-size: 13px;
    opacity: 0.85;
  `,
  loadingRow: css`
    display: flex;
    justify-content: center;
    padding: ${theme.spacing(6)};
  `,
  tableWrapper: css`
    overflow: auto;
    border: 1px solid ${theme.colors.border.weak};
    border-radius: ${theme.shape.radius.default};
    max-height: calc(100vh - 220px);
  `,
  table: css`
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    white-space: nowrap;
  `,
  thSticky: css`
    text-align: left;
    padding: ${theme.spacing(1)} ${theme.spacing(1.5)};
    border-bottom: 2px solid ${theme.colors.border.medium};
    color: ${theme.colors.text.secondary};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    position: sticky;
    top: 0;
    left: 0;
    z-index: 3;
    background: ${theme.colors.background.primary};
    min-width: 180px;
  `,
  th: css`
    text-align: left;
    padding: ${theme.spacing(1)} ${theme.spacing(1.5)};
    border-bottom: 2px solid ${theme.colors.border.medium};
    color: ${theme.colors.text.secondary};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    position: sticky;
    top: 0;
    z-index: 2;
    background: ${theme.colors.background.primary};
    min-width: 100px;
  `,
  tr: css`
    &:hover td {
      background: ${theme.colors.background.secondary};
    }
  `,
  trClickable: css`
    cursor: pointer;
    &:hover td {
      background: ${theme.colors.action.hover};
    }
    &:focus {
      outline: 2px solid ${theme.colors.primary.main};
      outline-offset: -2px;
    }
  `,
  openIcon: css`
    margin-left: ${theme.spacing(0.5)};
    opacity: 0.5;
    vertical-align: middle;
  `,
  headerActions: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(1)};
    flex-wrap: wrap;
  `,
  lastUpdated: css`
    font-size: 12px;
    opacity: 0.75;
    white-space: nowrap;
  `,
  autoRefreshToggle: css`
    display: flex;
    align-items: center;
    gap: ${theme.spacing(0.75)};
  `,
  autoRefreshLabel: css`
    font-size: 12px;
    opacity: 0.85;
    white-space: nowrap;
  `,
  tdFilename: css`
    padding: ${theme.spacing(0.75)} ${theme.spacing(1.5)};
    border-bottom: 1px solid ${theme.colors.border.weak};
    position: sticky;
    left: 0;
    background: ${theme.colors.background.canvas};
    z-index: 1;
    min-width: 180px;
    max-width: 220px;
  `,
  filename: css`
    display: block;
    color: ${theme.colors.text.primary};
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  `,
  docHash: css`
    display: block;
    color: ${theme.colors.text.disabled};
    font-family: ${theme.typography.fontFamilyMonospace};
    font-size: 10px;
    margin-top: 1px;
  `,
  tdMeta: css`
    padding: ${theme.spacing(0.75)} ${theme.spacing(1.5)};
    border-bottom: 1px solid ${theme.colors.border.weak};
    color: ${theme.colors.text.secondary};
    font-size: 11px;
    min-width: 80px;
  `,
  td: css`
    padding: ${theme.spacing(0.5)} ${theme.spacing(1.5)};
    border-bottom: 1px solid ${theme.colors.border.weak};
    vertical-align: middle;
  `,
  tdEmpty: css`
    padding: ${theme.spacing(0.75)} ${theme.spacing(1.5)};
    border-bottom: 1px solid ${theme.colors.border.weak};
    color: ${theme.colors.text.disabled};
    text-align: center;
  `,
  cellValue: css`
    color: ${theme.colors.text.primary};
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 140px;
  `,
  noValue: css`
    color: ${theme.colors.text.disabled};
  `,
  confChip: css`
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 8px;
    border: 1px solid;
    margin-top: 2px;
    line-height: 1.4;
    opacity: 0.85;
  `,
  emptyOuter: css`
    display: flex;
    justify-content: center;
    padding: ${theme.spacing(8)} ${theme.spacing(2)};
  `,
  emptyCard: css`
    text-align: center;
    color: ${theme.colors.text.secondary};
  `,
  emptyHeadline: css`
    margin: ${theme.spacing(1)} 0 0 0;
    color: ${theme.colors.text.primary};
    font-size: ${theme.typography.h4.fontSize};
  `,
  emptyText: css`
    margin: ${theme.spacing(1)} 0 0 0;
    font-size: 14px;
  `,
});
