import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import DataGrid, { type Column, type RenderEditCellProps, type SortColumn } from 'react-data-grid';
import 'react-data-grid/lib/styles.css';
import { css } from '@emotion/css';
import { SelectableValue, GrafanaTheme2 } from '@grafana/data';
import { useStyles2, Spinner, Button, Alert, Icon, MultiSelect, ConfirmModal, Modal, Select } from '@grafana/ui';
import Ajv, { type ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import { API_BASE } from '../utils/api';

// ---------- contract_schema helpers ----------
// All field mutations use a load-mutate-append pattern:
// 1. GET /rpc/latest_contract_schema  →  current payload
// 2. Mutate payload.fields in-memory
// 3. POST /rpc/append_contract_schema  →  new row (monotonic version)

async function loadSchemaPayload(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/rpc/latest_contract_schema`);
  if (!res.ok) { return {}; }
  const row = await res.json();
  return typeof row?.payload === 'object' && row.payload !== null ? row.payload : {};
}

async function appendSchemaPayload(payload: Record<string, unknown>): Promise<void> {
  const res = await fetch(`${API_BASE}/rpc/append_contract_schema`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify({ p_payload: payload }),
  });
  if (!res.ok) {
    const ct = res.headers.get('content-type') || '';
    let msg = `append_contract_schema failed (${res.status})`;
    if (ct.includes('application/json')) {
      const data = await res.json();
      msg = data?.message || data?.detail || msg;
    }
    throw new Error(msg);
  }
}
import { type ComputedFn, type FieldDetail, type AnchorFamily, BASE_TYPES, NORMALIZERS, ANCHOR_FAMILIES } from '../utils/types';

// ---------- types ----------

type RowStatus = 'clean' | 'new' | 'dirty' | 'saving' | 'error';

interface Row extends FieldDetail {
  __id: string;
  __isNew: boolean;
  __status: RowStatus;
  __error?: string;
  __original?: FieldDetail;
}

const NAME_PATTERN = /^[A-Za-z][A-Za-z0-9_]*$/;

const NEW_ROW_DEFAULTS: Omit<FieldDetail, 'name' | 'base_type' | 'normalizer' | 'anchor_family'> = {
  required: false,
  description: '',
  dataverse_column: '',
  computed: false,
  computed_fn: null,
  computed_from: null,
  default_value: null,
  is_vendor: false,
  confidence_threshold: 0.75,
  importance: 0.5,
  priority_bonus: 0.0,
  keyword_proximal: false,
  status: 'active',
  deprecation_reason: null,
};

// Editable fields on existing rows (non-structural tunables)
const EXISTING_ROW_EDITABLE: Set<keyof FieldDetail> = new Set([
  'description', 'dataverse_column', 'confidence_threshold', 'importance',
  'priority_bonus', 'keyword_proximal',
]);

const COMPUTED_FN_OPTIONS: ComputedFn[] = ['infer_currency', 'concat_strip', 'coalesce'];

// ---------- helpers ----------

const toRow = (detail: FieldDetail): Row => {
  const normalized: FieldDetail = {
    ...detail,
    anchor_family: detail.anchor_family ?? null,
    dataverse_column: detail.dataverse_column ?? '',
    computed_fn: detail.computed_fn ?? null,
    computed_from: detail.computed_from ?? null,
    confidence_threshold: detail.confidence_threshold ?? 0.75,
    importance: detail.importance ?? 0.5,
    priority_bonus: detail.priority_bonus ?? 0.0,
    keyword_proximal: detail.keyword_proximal ?? false,
    is_vendor: detail.is_vendor ?? false,
    status: detail.status ?? 'active',
    deprecation_reason: detail.deprecation_reason ?? null,
  };
  return {
    ...normalized,
    __id: `db:${detail.name}`,
    __isNew: false,
    __status: 'clean',
    __original: { ...normalized },
  };
};


const fullForPost = (row: Row): Record<string, unknown> => ({
  name: row.name,
  base_type: row.base_type,
  normalizer: row.normalizer,
  anchor_family: row.anchor_family || null,
  keyword_proximal: row.keyword_proximal,
  required: row.required,
  description: row.description,
  dataverse_column: row.dataverse_column || null,
  computed: row.computed,
  computed_fn: row.computed_fn || null,
  computed_from: row.computed_from && row.computed_from.length > 0 ? row.computed_from : null,
  default_value: row.default_value || null,
  is_vendor: row.is_vendor ?? false,
});

// ---------- cell editors ----------

function TextEditor({ row, column, onRowChange, onClose }: RenderEditCellProps<Row>) {
  const key = column.key as keyof Row;
  return (
    <input
      className={editorInputClass}
      autoFocus
      value={String(row[key] ?? '')}
      onChange={(e) => onRowChange({ ...row, [key]: e.target.value } as Row)}
      onBlur={() => onClose(true)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {onClose(true);}
      }}
    />
  );
}

function SelectEditor(options: string[], allowEmpty = false) {
  return function SelectEditorInner({ row, column, onRowChange, onClose }: RenderEditCellProps<Row>) {
    const key = column.key as keyof Row;
    return (
      <select
        className={editorInputClass}
        autoFocus
        value={String(row[key] ?? '')}
        onChange={(e) => onRowChange({ ...row, [key]: e.target.value } as Row, true)}
        onBlur={() => onClose(true)}
      >
        {allowEmpty && <option value="">— none —</option>}
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    );
  };
}

function ComputedFromEditor(allRows: Row[]) {
  return function ComputedFromEditorInner({ row, onRowChange, onClose }: RenderEditCellProps<Row>) {
    const options: Array<SelectableValue<string>> = allRows
      .filter((r) => r.name.trim() !== '' && r.name !== row.name)
      .map((r) => ({ label: r.name, value: r.name }));

    const current: Array<SelectableValue<string>> = (row.computed_from ?? []).map((v) => ({
      label: v,
      value: v,
    }));

    return (
      <div className={multiSelectWrapClass}>
        <MultiSelect
          options={options}
          value={current}
          menuShouldPortal
          autoFocus
          onChange={(vals) => {
            onRowChange({ ...row, computed_from: vals.map((v) => v.value as string) });
          }}
          onBlur={() => onClose(true)}
        />
      </div>
    );
  };
}

// ---------- column order persistence ----------

const COL_ORDER_KEY = 'invoicex-field-manager-col-order';

function loadColOrder(): string[] | null {
  try {
    const raw = localStorage.getItem(COL_ORDER_KEY);
    return raw ? (JSON.parse(raw) as string[]) : null;
  } catch {
    return null;
  }
}

function saveColOrder(order: string[]): void {
  localStorage.setItem(COL_ORDER_KEY, JSON.stringify(order));
}

// ---------- page ----------

// Minimal placeholder schema for ajv when /schema/meta-schema is not yet live.
// C2-expanded owns the real endpoint; this keeps saves working in the interim.
const PLACEHOLDER_META_SCHEMA = {
  type: 'object',
  properties: {
    field_definitions: { type: 'object' },
  },
  required: ['field_definitions'],
} as const;

export function SchemaEditorPage() {
  const styles = useStyles2(getStyles);
  const [rows, setRows] = useState<Row[]>([]);
  const [sortColumns, setSortColumns] = useState<readonly SortColumn[]>([]);
  const [columnOrder, setColumnOrder] = useState<string[] | null>(loadColOrder);
  const [axes, setAxes] = useState<{ base_type: string[]; normalizer: string[]; anchor_family: string[] }>({
    base_type: ['str'],
    normalizer: ['passthrough'],
    anchor_family: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [alert, setAlert] = useState<{ severity: 'success' | 'error' | 'info'; message: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [showDeprecated, setShowDeprecated] = useState(false);

  // Rename modal state
  const [renameRow, setRenameRow] = useState<Row | null>(null);
  const [renameNewName, setRenameNewName] = useState('');
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);

  // Type-change modal state
  const [typeChangeRow, setTypeChangeRow] = useState<Row | null>(null);
  const [typeChangeNewType, setTypeChangeNewType] = useState<string>('str');
  const [typeChanging, setTypeChanging] = useState(false);

  // ajv schema validator — compiled once on mount against PLACEHOLDER_META_SCHEMA.
  // The dedicated /schema/meta-schema endpoint (C2-expanded) is not yet live;
  // latest_contract_schema returns a data row ({version,payload,...}), NOT a JSON
  // Schema, so compiling it as one made pre-save validation silently a no-op.
  // PLACEHOLDER_META_SCHEMA enforces {fields: object} at the document level;
  // per-field validation runs in the per-row path below (~:490).
  const schemaValidator = useRef<ValidateFunction | null>(null);

  useEffect(() => {
    const ajv = new Ajv({ allErrors: true });
    addFormats(ajv);
    schemaValidator.current = ajv.compile(PLACEHOLDER_META_SCHEMA);
  }, []);

  const load = useCallback(async (_inclDeprecated = false) => {
    // Field schema is loaded from latest_contract_schema (append-only event log).
    // payload.fields is a {[name]: FieldDetail} map written by append_contract_schema.
    // Falls back to empty list — page renders a notice when no schema has been saved.
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE}/rpc/latest_contract_schema`);
      if (!res.ok) {
        // No schema seeded yet — treat as empty rather than hard error.
        setRows([]);
        return;
      }
      const schemaRow = await res.json();
      const fieldsMap: Record<string, FieldDetail> = schemaRow?.payload?.field_definitions ?? {};
      const fieldRows = Object.entries(fieldsMap).map(([name, def]) =>
        toRow({ ...(def as FieldDetail), name })
      );
      setRows(fieldRows);
      // Derive axis option lists from the loaded fields rather than a separate /schema/axes endpoint.
      const baseTypes = [...new Set(fieldRows.map((r) => r.base_type).filter(Boolean))] as string[];
      const normalizers = [...new Set(fieldRows.map((r) => r.normalizer).filter(Boolean))] as string[];
      const anchorFamilies = [...new Set(fieldRows.map((r) => r.anchor_family).filter((v) => v !== null))] as string[];
      setAxes({
        base_type: baseTypes.length > 0 ? baseTypes : ['str', 'decimal', 'date', 'int'],
        normalizer: normalizers.length > 0 ? normalizers : ['passthrough', 'amount', 'date', 'id', 'name', 'currency', 'email', 'phone'],
        anchor_family: anchorFamilies,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Load failed');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(showDeprecated);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  const addRow = () => {
    const id = `new:${Date.now()}:${Math.random().toString(36).slice(2, 7)}`;
    setRows((prev) => [
      ...prev,
      {
        __id: id,
        __isNew: true,
        __status: 'new',
        name: '',
        base_type: '' as unknown as FieldDetail['base_type'],
        normalizer: '' as unknown as FieldDetail['normalizer'],
        anchor_family: null,
        ...NEW_ROW_DEFAULTS,
      },
    ]);
  };

  const deprecateRow = useCallback(async (row: Row) => {
    const reason = window.prompt(
      `Deprecate field "${row.name}"?\n\nLabels in the corrections table are preserved. The field will be hidden from active extraction but visible via the "Show deprecated" toggle.\n\nEnter a reason for deprecation (required):`,
      ''
    );
    if (reason === null) {return;} // cancelled
    if (!reason.trim()) {
      setAlert({ severity: 'error', message: 'Deprecation reason is required.' });
      return;
    }
    try {
      // Load-mutate-append: mark field deprecated in contract_schema payload.
      const payload = await loadSchemaPayload();
      const fields = (payload.field_definitions ?? {}) as Record<string, Record<string, unknown>>;
      if (!fields[row.name]) { throw new Error(`Field "${row.name}" not found in schema`); }
      fields[row.name] = { ...fields[row.name], status: 'deprecated', deprecation_reason: reason.trim() };
      await appendSchemaPayload({ ...payload, field_definitions: fields });
      setRows((prev) =>
        prev.map((r) =>
          r.__id === row.__id
            ? { ...r, status: 'deprecated', deprecation_reason: reason.trim(), __status: 'clean' }
            : r
        )
      );
      setAlert({ severity: 'success', message: `Field "${row.name}" deprecated.` });
    } catch (e) {
      setRows((prev) =>
        prev.map((r) =>
          r.__id === row.__id
            ? { ...r, __status: 'error', __error: e instanceof Error ? e.message : 'Deprecation failed' }
            : r
        )
      );
    }
  }, []);

  const handleRename = async () => {
    if (!renameRow) {return;}
    const trimmed = renameNewName.trim();
    if (!trimmed) { setRenameError('New name is required.'); return; }
    if (!NAME_PATTERN.test(trimmed)) { setRenameError('Must start with a letter (letters/digits/underscore).'); return; }
    if (trimmed === renameRow.name) { setRenameError('New name must differ from current name.'); return; }
    setRenaming(true);
    setRenameError(null);
    try {
      // Load-mutate-append: rename key in contract_schema payload.
      const payload = await loadSchemaPayload();
      const fields = (payload.field_definitions ?? {}) as Record<string, Record<string, unknown>>;
      if (!fields[renameRow.name]) { throw new Error(`Field "${renameRow.name}" not found in schema`); }
      if (fields[trimmed]) { throw new Error(`Field "${trimmed}" already exists`); }
      fields[trimmed] = { ...fields[renameRow.name], name: trimmed };
      delete fields[renameRow.name];
      await appendSchemaPayload({ ...payload, field_definitions: fields });
      const updated: FieldDetail = { ...(fields[trimmed] as unknown as FieldDetail), name: trimmed };
      setRows((prev) => prev.map((r) => r.__id === renameRow.__id ? toRow(updated) : r));
      setAlert({ severity: 'success', message: `Field renamed "${renameRow.name}" → "${trimmed}".` });
      setRenameRow(null);
    } catch (e) {
      setRenameError(e instanceof Error ? e.message : 'Rename failed');
    } finally {
      setRenaming(false);
    }
  };

  const handleTypeChange = async (_cascade: 'coerce' | 'reject') => {
    if (!typeChangeRow) {return;}
    setTypeChanging(true);
    try {
      // Load-mutate-append: update base_type in contract_schema payload.
      const payload = await loadSchemaPayload();
      const fields = (payload.field_definitions ?? {}) as Record<string, Record<string, unknown>>;
      if (!fields[typeChangeRow.name]) { throw new Error(`Field "${typeChangeRow.name}" not found in schema`); }
      fields[typeChangeRow.name] = { ...fields[typeChangeRow.name], base_type: typeChangeNewType };
      await appendSchemaPayload({ ...payload, field_definitions: fields });
      const updated: FieldDetail = { ...(fields[typeChangeRow.name] as unknown as FieldDetail), name: typeChangeRow.name };
      setRows((prev) => prev.map((r) => r.__id === typeChangeRow.__id ? toRow(updated) : r));
      setAlert({ severity: 'success', message: `Field "${typeChangeRow.name}" type changed to ${typeChangeNewType}.` });
      setTypeChangeRow(null);
    } catch (e) {
      setAlert({ severity: 'error', message: e instanceof Error ? e.message : 'Type change failed' });
      setTypeChangeRow(null);
    } finally {
      setTypeChanging(false);
    }
  };

  const revertRow = (id: string) => {
    setRows((prev) =>
      prev.flatMap((r) => {
        if (r.__id !== id) {return [r];}
        if (r.__isNew) {return [];} // drop new rows
        return [{ ...r.__original!, __id: r.__id, __isNew: false, __status: 'clean' as RowStatus, __original: r.__original }];
      })
    );
  };

  const revertAll = () => {
    setRows((prev) =>
      prev.flatMap((r) => {
        if (r.__isNew) {return [];}
        return [{ ...r.__original!, __id: r.__id, __isNew: false, __status: 'clean' as RowStatus, __original: r.__original }];
      })
    );
    setAlert(null);
  };

  const onRowsChange = (newRows: Row[]) => {
    setRows(
      newRows.map((r) => {
        const next = { ...r };
        if (r.__isNew) {
          next.__status = 'new';
        } else if (r.__original) {
          // Mark dirty if any editable field changed from original
          const changed = (Array.from(EXISTING_ROW_EDITABLE) as Array<keyof FieldDetail>).some(
            (k) => r[k] !== r.__original![k]
          );
          next.__status = changed ? 'dirty' : 'clean';
          if (!changed) {next.__error = undefined;}
        }
        return next;
      })
    );
  };

  const saveAll = async () => {
    const toSave = rows.filter((r) => r.__status === 'new' || r.__status === 'dirty');
    if (toSave.length === 0) {
      setAlert({ severity: 'info', message: 'Nothing to save.' });
      return;
    }

    // --- AJV meta-schema validation (document-level, blocks save if invalid) ---
    if (schemaValidator.current) {
      // Build a snapshot of the full field set as it would exist after saves apply.
      // Rows marked 'error' are excluded — they won't be submitted anyway.
      const pendingFields: Record<string, unknown> = {};
      rows
        .filter((r) => r.__status !== 'error' && r.name.trim() !== '')
        .forEach((r) => {
          pendingFields[r.name] = {
            base_type: r.base_type,
            normalizer: r.normalizer,
            anchor_family: r.anchor_family,
            required: r.required,
            description: r.description,
            keyword_proximal: r.keyword_proximal,
            confidence_threshold: r.confidence_threshold,
            importance: r.importance,
            priority_bonus: r.priority_bonus,
            computed: r.computed,
            computed_fn: r.computed_fn ?? null,
            computed_from: r.computed_from ?? null,
            default_value: r.default_value ?? null,
            dataverse_column: r.dataverse_column || null,
            is_vendor: r.is_vendor,
          };
        });
      const doc = { fields: pendingFields };
      const valid = schemaValidator.current(doc);
      if (!valid) {
        const errs = (schemaValidator.current.errors ?? [])
          .map((e) => `${e.instancePath || '(root)'}: ${e.message}`)
          .join('\n');
        setAlert({ severity: 'error', message: `Schema validation failed:\n${errs}` });
        return;
      }
    }

    // --- Validate new rows (per-row errors, don't abort all) ---
    const existingNames = new Set(rows.filter((r) => !r.__isNew).map((r) => r.name));
    let hasNewErrors = false;
    setRows((prev) =>
      prev.map((r) => {
        if (!r.__isNew) {return r;}
        const nameEmpty = !r.name.trim();
        const nameBad = !nameEmpty && !NAME_PATTERN.test(r.name);
        const nameDup = !nameEmpty && !nameBad && existingNames.has(r.name);
        const axesMissing = !r.base_type || !r.normalizer;
        const kwpIncoherent = r.keyword_proximal && !r.anchor_family;
        if (nameEmpty || nameBad || nameDup || axesMissing || kwpIncoherent) {
          hasNewErrors = true;
          const msg = nameEmpty
            ? 'Name is required'
            : axesMissing
            ? 'base_type and normalizer are required'
            : kwpIncoherent
            ? 'keyword_proximal=true requires anchor_family to be set'
            : nameBad
            ? `Invalid name — must start with a letter (letters/digits/underscore)`
            : `Duplicate name "${r.name}"`;
          return { ...r, __status: 'error' as RowStatus, __error: msg };
        }
        if (!existingNames.has(r.name)) {existingNames.add(r.name);}
        return r;
      })
    );
    if (hasNewErrors) {
      setAlert({ severity: 'error', message: 'Some new rows have errors — fix them before saving.' });
      return;
    }

    // --- Mark saving ---
    const validToSave = rows.filter((r) => r.__status === 'new' || r.__status === 'dirty');
    setSaving(true);
    setAlert(null);
    setRows((prev) => prev.map((r) => (validToSave.find((t) => t.__id === r.__id) ? { ...r, __status: 'saving', __error: undefined } : r)));

    // Load-mutate-append: apply all pending changes to contract_schema in one append.
    // New rows are added; dirty rows are updated (only changed editable fields).
    let savePayload: Record<string, unknown>;
    try {
      savePayload = await loadSchemaPayload();
    } catch (e) {
      setAlert({ severity: 'error', message: e instanceof Error ? e.message : 'Failed to load schema for save' });
      setSaving(false);
      setRows((prev) => prev.map((r) => validToSave.find((t) => t.__id === r.__id) ? { ...r, __status: 'clean' } : r));
      return;
    }
    // BUG FIX: read field_definitions (the spec dict), not fields (the name list).
    // savePayload.fields is string[], not the keyed spec object the loop mutates.
    const saveFields = (savePayload.field_definitions ?? {}) as Record<string, Record<string, unknown>>;
    // Keep the fields name-list in sync: pipeline.py drives schema_fields from it.
    const saveFieldNames: string[] = Array.isArray(savePayload.fields) ? [...(savePayload.fields as string[])] : [];

    const results = await Promise.allSettled(
      validToSave.map(async (r) => {
        if (r.__isNew) {
          const fieldData = fullForPost(r);
          saveFields[r.name] = fieldData as Record<string, unknown>;
          // Add to name list only if not already present (idempotent on retry).
          if (!saveFieldNames.includes(r.name)) { saveFieldNames.push(r.name); }
          return { id: r.__id, detail: { ...fieldData, name: r.name } as FieldDetail };
        } else {
          const orig = r.__original!;
          const patch: Record<string, unknown> = {};
          (Array.from(EXISTING_ROW_EDITABLE) as Array<keyof FieldDetail>).forEach((k) => {
            if (r[k] !== orig[k]) { patch[k] = r[k]; }
          });
          saveFields[r.name] = { ...(saveFields[r.name] ?? {}), ...patch };
          return { id: r.__id, detail: { ...orig, ...patch } as FieldDetail };
        }
      })
    );

    // Commit the accumulated field mutations in a single append.
    try {
      await appendSchemaPayload({ ...savePayload, fields: saveFieldNames, field_definitions: saveFields });
    } catch (e) {
      setAlert({ severity: 'error', message: e instanceof Error ? e.message : 'Save failed' });
      setSaving(false);
      return;
    }

    setRows((prev) =>
      prev.map((r) => {
        const idx = validToSave.findIndex((t) => t.__id === r.__id);
        if (idx < 0) {return r;}
        const result = results[idx];
        if (result.status === 'fulfilled') {
          const { detail } = result.value;
          return toRow(detail);
        } else {
          return { ...r, __status: 'error', __error: result.reason instanceof Error ? result.reason.message : String(result.reason) };
        }
      })
    );

    const ok = results.filter((r) => r.status === 'fulfilled').length;
    const fail = results.length - ok;
    setSaving(false);
    if (fail === 0) {
      setAlert({ severity: 'success', message: `Saved ${ok} row${ok === 1 ? '' : 's'}.` });
    } else {
      setAlert({ severity: 'error', message: `${ok} saved, ${fail} failed. Hover the error badge on a row for details.` });
    }
  };

  const columns = useMemo<Array<Column<Row>>>(() => {
    const LOCKED_TOOLTIP = 'Immutable on existing fields — deprecate and re-add to change this.';

    // Helper for text columns editable on new rows only (structural fields)
    const textCol = (key: keyof FieldDetail, name: string, width: number): Column<Row> => ({
      key,
      name,
      width,
      resizable: true,
      sortable: true,
      draggable: true,
      editable: (r) => r.__isNew,
      renderEditCell: TextEditor,
      renderCell: ({ row }) => (
        <span className={styles.cellText}>{String(row[key] ?? '')}</span>
      ),
    });

    // Helper for text columns editable on ALL rows (tunables / description / dataverse)
    const editableTextCol = (key: keyof FieldDetail, name: string, width: number): Column<Row> => ({
      key,
      name,
      width,
      resizable: true,
      sortable: true,
      draggable: true,
      editable: (r) => r.status !== 'deprecated',
      renderEditCell: TextEditor,
      renderCell: ({ row }) => (
        <span className={styles.cellText}>{String(row[key] ?? '')}</span>
      ),
    });

    // Helper for numeric tunable columns editable on all non-deprecated rows
    const numericCol = (key: keyof FieldDetail, name: string, width: number, format: (v: unknown) => string): Column<Row> => ({
      key,
      name,
      width,
      resizable: true,
      sortable: true,
      draggable: true,
      editable: (r) => r.status !== 'deprecated',
      renderEditCell: TextEditor,
      renderCell: ({ row }) => (
        <span className={styles.cellText}>{format(row[key])}</span>
      ),
    });

    return [
      {
        key: '__status',
        name: '',
        width: 32,
        frozen: true,
        renderCell: ({ row }) => (
          <span title={row.__error ?? row.__status} className={styles.statusDot}>
            {row.__status === 'new' && <Icon name="plus-circle" />}
            {row.__status === 'dirty' && <Icon name="pen" />}
            {row.__status === 'saving' && <Spinner size="sm" />}
            {row.__status === 'error' && <Icon name="exclamation-triangle" />}
            {row.__status === 'clean' && row.status === 'deprecated' && <Icon name="minus-circle" />}
            {row.__status === 'clean' && row.status !== 'deprecated' && <Icon name="check" />}
          </span>
        ),
      },
      // name: always editable
      textCol('name', 'Name', 160),
      // base_type: locked on existing rows (structural axis)
      {
        key: 'base_type',
        name: 'Base Type',
        width: 100,
        resizable: true,
        sortable: true,
        draggable: true,
        editable: (r) => r.__isNew,
        renderEditCell: SelectEditor([...BASE_TYPES]),
        renderCell: ({ row }) => (
          <span title={!row.__isNew ? LOCKED_TOOLTIP : (row.__isNew && !row.base_type ? 'Base Type required' : undefined)}
            style={row.__isNew && !row.base_type ? { border: '1px solid var(--color-error-border, #e05252)', display: 'block', height: '100%' } : undefined}>
            {row.base_type || <em style={{ opacity: 0.5 }}>select…</em>}
          </span>
        ),
      },
      // normalizer: locked on existing rows (structural axis)
      {
        key: 'normalizer',
        name: 'Normalizer',
        width: 110,
        resizable: true,
        sortable: true,
        draggable: true,
        editable: (r) => r.__isNew,
        renderEditCell: SelectEditor([...NORMALIZERS]),
        renderCell: ({ row }) => (
          <span title={!row.__isNew ? LOCKED_TOOLTIP : (row.__isNew && !row.normalizer ? 'Normalizer required' : undefined)}
            style={row.__isNew && !row.normalizer ? { border: '1px solid var(--color-error-border, #e05252)', display: 'block', height: '100%' } : undefined}>
            {row.normalizer || <em style={{ opacity: 0.5 }}>select…</em>}
          </span>
        ),
      },
      // anchor_family: locked on existing rows (structural axis), nullable
      {
        key: 'anchor_family',
        name: 'Anchor',
        width: 90,
        resizable: true,
        sortable: true,
        draggable: true,
        editable: (r) => r.__isNew,
        renderEditCell: ({ row, onRowChange, onClose }: RenderEditCellProps<Row>) => {
          return (
            <select
              className={editorInputClass}
              autoFocus
              value={String(row.anchor_family ?? '')}
              onChange={(e) => {
                const v = e.target.value;
                onRowChange({ ...row, anchor_family: (v === '' ? null : v) as AnchorFamily | null }, true);
              }}
              onBlur={() => onClose(true)}
            >
              <option value="">— none —</option>
              {ANCHOR_FAMILIES.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          );
        },
        renderCell: ({ row }) => (
          <span title={!row.__isNew ? LOCKED_TOOLTIP : undefined}>
            {row.anchor_family ?? '—'}
          </span>
        ),
      },
      // computed_fn: dropdown, new rows only
      {
        key: 'computed_fn',
        name: 'Computed Fn',
        width: 150,
        resizable: true,
        sortable: true,
        draggable: true,
        editable: (r) => r.__isNew,
        renderEditCell: SelectEditor(COMPUTED_FN_OPTIONS, true),
        renderCell: ({ row }) => (
          <span className={styles.cellText}>{row.computed_fn ?? ''}</span>
        ),
      },
      // computed_from: multi-select, new rows only
      {
        key: 'computed_from',
        name: 'Computed From',
        width: 200,
        resizable: true,
        sortable: true,
        draggable: true,
        editable: (r) => r.__isNew,
        renderEditCell: ComputedFromEditor(rows),
        renderCell: ({ row }) => (
          <span className={styles.cellText}>
            {row.computed_from && row.computed_from.length > 0 ? row.computed_from.join(', ') : ''}
          </span>
        ),
      },
      // default_value: new rows only (structural default)
      textCol('default_value', 'Default', 120),
      // description: editable on all rows
      editableTextCol('description', 'Description', 280),
      // dataverse_column: editable on all rows
      editableTextCol('dataverse_column', 'Dataverse Col', 160),
      // Tunables — editable on all non-deprecated rows
      numericCol('confidence_threshold', 'Conf Threshold', 120, (v) => Number(v).toFixed(2)),
      numericCol('importance', 'Importance', 100, (v) => Number(v).toFixed(2)),
      numericCol('priority_bonus', 'Priority Bonus', 110, (v) => Number(v).toFixed(2)),
      // keyword_proximal: editable on all non-deprecated rows
      {
        key: 'keyword_proximal',
        name: 'Kw Proximal',
        width: 100,
        resizable: true,
        sortable: true,
        draggable: true,
        editable: (r) => r.status !== 'deprecated',
        renderCell: ({ row, onRowChange }) => (
          <input
            type="checkbox"
            checked={Boolean(row.keyword_proximal)}
            disabled={row.status === 'deprecated'}
            onChange={(e) => row.status !== 'deprecated' && onRowChange({ ...row, keyword_proximal: e.target.checked })}
          />
        ),
      },
      // is_vendor: new rows only (single-field invariant — dangerous to edit inline)
      {
        key: 'is_vendor',
        name: 'Is Vendor',
        width: 90,
        resizable: true,
        sortable: true,
        draggable: true,
        editable: (r) => r.__isNew,
        renderCell: ({ row, onRowChange }) => (
          <input
            type="checkbox"
            checked={Boolean(row.is_vendor)}
            disabled={!row.__isNew}
            onChange={(e) => row.__isNew && onRowChange({ ...row, is_vendor: e.target.checked })}
          />
        ),
      },
      {
        key: '__actions',
        name: '',
        width: 190,
        renderCell: ({ row }) => {
          if (row.__status === 'saving') {return null;}
          if (row.__status === 'dirty') {
            return (
              <Button size="sm" variant="secondary" fill="text" onClick={() => revertRow(row.__id)}>
                Revert
              </Button>
            );
          }
          if (row.__status === 'clean' && !row.__isNew && row.status !== 'deprecated') {
            return (
              <span style={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <Button
                  size="sm"
                  variant="secondary"
                  fill="text"
                  title="Rename field (cascades to all labels)"
                  onClick={() => { setRenameRow(row); setRenameNewName(''); setRenameError(null); }}
                >
                  <Icon name="pen" />
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  fill="text"
                  title="Change base type"
                  onClick={() => { setTypeChangeRow(row); setTypeChangeNewType(row.base_type); }}
                >
                  <Icon name="cog" />
                </Button>
                <Button size="sm" variant="destructive" fill="text" onClick={() => deprecateRow(row)}>
                  Deprecate
                </Button>
              </span>
            );
          }
          if (row.__status === 'clean') {return null;}
          return (
            <Button size="sm" variant="secondary" fill="text" onClick={() => revertRow(row.__id)}>
              {row.__isNew ? 'Remove' : 'Revert'}
            </Button>
          );
        },
      },
    ];
  }, [rows, styles.cellText, styles.statusDot, deprecateRow]);

  // Derive column list respecting user's drag-reorder preference.
  // __status (frozen) and __actions stay pinned; middle columns are reorderable.
  const orderedColumns = useMemo(() => {
    const statusCol = columns[0]; // __status — always first
    const actionsCol = columns[columns.length - 1]; // __actions — always last
    const moveable = columns.slice(1, -1);

    if (!columnOrder) {return columns;}

    const colMap = new Map(moveable.map((c) => [c.key, c]));
    // Keep only keys that exist in current moveable columns, in stored order
    const reordered = columnOrder
      .filter((k) => colMap.has(k))
      .map((k) => colMap.get(k)!);
    // Append any new columns not yet in the stored order
    const seen = new Set(columnOrder);
    moveable.forEach((c) => { if (!seen.has(c.key)) {reordered.push(c);} });

    return [statusCol, ...reordered, actionsCol];
  }, [columns, columnOrder]);

  const handleColumnsReorder = useCallback((sourceKey: string, targetKey: string) => {
    // Recompute the moveable column order from orderedColumns, then swap
    const moveable = orderedColumns.slice(1, -1);
    const sourceIdx = moveable.findIndex((c) => c.key === sourceKey);
    const targetIdx = moveable.findIndex((c) => c.key === targetKey);
    if (sourceIdx === -1 || targetIdx === -1) {return;}

    const next = [...moveable];
    next.splice(targetIdx, 0, ...next.splice(sourceIdx, 1));
    const newOrder = next.map((c) => c.key);
    saveColOrder(newOrder);
    setColumnOrder(newOrder);
  }, [orderedColumns]);

  const sortedRows = useMemo<Row[]>(() => {
    if (sortColumns.length === 0) {return rows;}
    return [...rows].sort((a, b) => {
      for (const { columnKey, direction } of sortColumns) {
        const aVal = a[columnKey as keyof Row];
        const bVal = b[columnKey as keyof Row];
        // Nulls last
        if (aVal == null && bVal == null) {continue;}
        if (aVal == null) {return 1;}
        if (bVal == null) {return -1;}
        let cmp = 0;
        if (typeof aVal === 'boolean' && typeof bVal === 'boolean') {
          cmp = (aVal === bVal) ? 0 : aVal ? -1 : 1; // true first
        } else if (typeof aVal === 'number' && typeof bVal === 'number') {
          cmp = aVal - bVal;
        } else {
          cmp = String(aVal).localeCompare(String(bVal));
        }
        if (cmp !== 0) {return direction === 'ASC' ? cmp : -cmp;}
      }
      return 0;
    });
  }, [rows, sortColumns]);

  const dirtyCount = rows.filter((r) => r.__status === 'new' || r.__status === 'dirty' || r.__status === 'error').length;
  const activeRows = sortedRows.filter((r) => r.status !== 'deprecated');
  const deprecatedRows = sortedRows.filter((r) => r.status === 'deprecated');

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Field Manager</h1>
          <p className={styles.subtitle}>Edit schema fields inline — like a spreadsheet. Save All commits every change.</p>
        </div>
        <div className={styles.headerActions}>
          <Button onClick={addRow} icon="plus" variant="secondary" disabled={saving}>Add Row</Button>
          <Button onClick={saveAll} icon="save" variant="primary" disabled={saving || dirtyCount === 0}>
            {saving ? 'Saving...' : `Save All${dirtyCount ? ` (${dirtyCount})` : ''}`}
          </Button>
          <Button onClick={revertAll} variant="secondary" fill="text" disabled={saving || dirtyCount === 0}>Revert</Button>
          <Button
            onClick={() => {
              const next = !showDeprecated;
              setShowDeprecated(next);
              load(next);
            }}
            variant="secondary"
            fill="text"
            title={showDeprecated ? 'Hide deprecated fields' : 'Show deprecated fields'}
          >
            {showDeprecated ? 'Hide Deprecated' : 'Show Deprecated'}
          </Button>
          <Button onClick={() => load(showDeprecated)} icon="sync" variant="secondary" fill="text" title="Reload from server" />
        </div>
      </div>

      {alert && (
        <Alert title="" severity={alert.severity} onRemove={() => setAlert(null)}>
          {alert.message}
        </Alert>
      )}

      {loading && (
        <div className={styles.centered}>
          <Spinner size="xl" />
          <p>Loading fields...</p>
        </div>
      )}

      {error && (
        <div className={styles.centered}>
          <p className={styles.errorText}>{error}</p>
          <Button onClick={() => load(showDeprecated)}>Retry</Button>
        </div>
      )}

      {!loading && !error && (
        <>
          <div className={styles.gridWrapper}>
            <DataGrid
              columns={orderedColumns}
              rows={activeRows}
              rowKeyGetter={(r) => r.__id}
              onRowsChange={onRowsChange}
              sortColumns={sortColumns}
              onSortColumnsChange={setSortColumns}
              onColumnsReorder={handleColumnsReorder}
              rowHeight={32}
              headerRowHeight={36}
              className={styles.grid}
              rowClass={(r) =>
                r.__status === 'error' ? styles.rowError :
                r.__status === 'new' || r.__status === 'dirty' ? styles.rowDirty : ''
              }
            />
          </div>
          {showDeprecated && deprecatedRows.length > 0 && (
            <div className={styles.deprecatedSection}>
              <h3 className={styles.deprecatedTitle}>
                Deprecated Fields ({deprecatedRows.length})
              </h3>
              <DataGrid
                columns={orderedColumns}
                rows={deprecatedRows}
                rowKeyGetter={(r) => r.__id}
                onRowsChange={onRowsChange}
                sortColumns={sortColumns}
                onSortColumnsChange={setSortColumns}
                onColumnsReorder={handleColumnsReorder}
                rowHeight={32}
                headerRowHeight={36}
                className={styles.gridDeprecated}
                rowClass={() => styles.rowDeprecated}
              />
            </div>
          )}
        </>
      )}

      {/* Rename modal */}
      <ConfirmModal
        isOpen={renameRow !== null}
        title={`Rename field "${renameRow?.name}"`}
        body={
          <div>
            <p>Renaming cascades to all existing labels in the corrections table. Enter the new field name:</p>
            <input
              className={editorInputClass}
              style={{ display: 'block', width: '100%', marginTop: 8, padding: '6px 8px' }}
              value={renameNewName}
              autoFocus
              placeholder="new_field_name"
              onChange={(e) => { setRenameNewName(e.target.value); setRenameError(null); }}
              onKeyDown={(e) => { if (e.key === 'Enter') { handleRename(); } }}
            />
            {renameError && <p style={{ color: 'red', marginTop: 4, fontSize: 12 }}>{renameError}</p>}
          </div>
        }
        confirmText={renaming ? 'Renaming…' : 'Rename'}
        onConfirm={handleRename}
        onDismiss={() => { setRenameRow(null); setRenameNewName(''); setRenameError(null); }}
      />

      {/* Type-change modal */}
      <Modal
        title={`Change type for "${typeChangeRow?.name}"`}
        isOpen={typeChangeRow !== null}
        onDismiss={() => setTypeChangeRow(null)}
      >
        <div>
          <p>Select a new base type, then choose how to handle existing extracted values:</p>
          <div style={{ marginBottom: 12 }}>
            <Select
              options={BASE_TYPES.map((t) => ({ label: t, value: t }))}
              value={typeChangeNewType}
              onChange={(v) => setTypeChangeNewType(v.value as string)}
            />
          </div>
          <p style={{ fontSize: 12, marginBottom: 16, opacity: 0.8 }}>
            <strong>Coerce</strong> — cast existing label values to the new type where possible.<br />
            <strong>Reject</strong> — refuse the change if any existing label value is incompatible.
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            <Button
              variant="destructive"
              disabled={typeChanging || typeChangeNewType === typeChangeRow?.base_type}
              onClick={() => handleTypeChange('coerce')}
            >
              {typeChanging ? 'Changing…' : 'Coerce'}
            </Button>
            <Button
              variant="destructive"
              fill="outline"
              disabled={typeChanging || typeChangeNewType === typeChangeRow?.base_type}
              onClick={() => handleTypeChange('reject')}
            >
              {typeChanging ? 'Changing…' : 'Reject'}
            </Button>
            <Button variant="secondary" fill="text" onClick={() => setTypeChangeRow(null)}>Cancel</Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

// ---------- styles ----------

const editorInputClass = css`
  width: 100%;
  height: 100%;
  border: 0;
  padding: 0 6px;
  font: inherit;
  background: transparent;
  color: inherit;
  outline: 2px solid #4a9eff;
  outline-offset: -2px;
`;

const multiSelectWrapClass = css`
  width: 100%;
  min-width: 200px;
`;

const getStyles = (theme: GrafanaTheme2) => ({
  container: css`
    padding: ${theme.spacing(3)};
    max-width: 1600px;
    margin: 0 auto;
  `,
  header: css`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: ${theme.spacing(3)};
    padding: ${theme.spacing(3)};
    background: linear-gradient(135deg, ${theme.colors.primary.main} 0%, ${theme.colors.primary.shade} 100%);
    border-radius: ${theme.shape.radius.default};
    color: ${theme.colors.primary.contrastText};
  `,
  headerActions: css`
    display: flex;
    gap: ${theme.spacing(1)};
    align-items: center;
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
  centered: css`
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: ${theme.spacing(6)};
    gap: ${theme.spacing(2)};
  `,
  errorText: css`
    color: ${theme.colors.error.text};
  `,
  gridWrapper: css`
    background: ${theme.colors.background.secondary};
    border: 1px solid ${theme.colors.border.weak};
    border-radius: ${theme.shape.radius.default};
    overflow: hidden;

    /* Re-theme react-data-grid CSS custom properties for Grafana. */
    .rdg {
      --rdg-color: ${theme.colors.text.primary};
      --rdg-background-color: ${theme.colors.background.secondary};
      --rdg-header-background-color: ${theme.colors.background.canvas};
      --rdg-row-hover-background-color: ${theme.colors.action.hover};
      --rdg-border-color: ${theme.colors.border.weak};
      --rdg-summary-border-color: ${theme.colors.border.weak};
      --rdg-selection-color: ${theme.colors.primary.main};
      block-size: 100%;
      border: none;
      font-size: 13px;
    }
  `,
  grid: css`
    block-size: 70vh;
    min-height: 400px;
  `,
  statusDot: css`
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    color: ${theme.colors.text.secondary};
  `,
  cellText: css`
    font-family: monospace;
    font-size: 12px;
  `,
  cellReadonly: css`
    font-family: monospace;
    font-size: 12px;
    color: ${theme.colors.text.disabled};
  `,
  rowDirty: css`
    background-color: ${theme.colors.warning.transparent} !important;
  `,
  rowError: css`
    background-color: ${theme.colors.error.transparent} !important;
  `,
  rowDeprecated: css`
    opacity: 0.55;
  `,
  deprecatedSection: css`
    margin-top: ${theme.spacing(3)};
  `,
  deprecatedTitle: css`
    font-size: 16px;
    font-weight: 500;
    color: ${theme.colors.text.secondary};
    margin-bottom: ${theme.spacing(1)};
  `,
  gridDeprecated: css`
    block-size: auto;
    max-height: 40vh;
    min-height: 100px;
  `,
});
