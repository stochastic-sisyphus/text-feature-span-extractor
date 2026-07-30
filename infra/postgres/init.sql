-- Wave E: init.sql — 4 JSONB tables + GIN indexes + RPCs + PostgREST roles
-- Runs against the 'invoicex' database (created in init-db.sh).
-- Field names match Pydantic models in src/invoices/models.py (source of truth).
-- SQL types are governed by this file; Pydantic does not dictate SQL types.

-- ---------------------------------------------------------------------------
-- PostgREST schema cache: automatic reload on every DDL change
-- Canonical pattern: docs.postgrest.org/en/v12/references/schema_cache.html
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pgrst_watch() RETURNS event_trigger
LANGUAGE plpgsql AS $$
BEGIN
  NOTIFY pgrst, 'reload schema';
END;
$$;

DROP EVENT TRIGGER IF EXISTS pgrst_watch;
CREATE EVENT TRIGGER pgrst_watch
  ON ddl_command_end
  EXECUTE PROCEDURE pgrst_watch();

-- ---------------------------------------------------------------------------
-- docs: content-addressed PDF store, keyed by sha256
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS docs (
    sha256        text PRIMARY KEY,
    doc_schema_version text NOT NULL DEFAULT '',
    payload       jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS docs_payload_gin ON docs USING gin (payload jsonb_path_ops);

-- Wave R: drop status column — doc readiness derived from payload ? 'doc'.
ALTER TABLE docs DROP COLUMN IF EXISTS status;
DROP INDEX IF EXISTS docs_status_idx;

-- ---------------------------------------------------------------------------
-- labels: human-applied labels, append-only
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS labels (
    id            bigserial PRIMARY KEY,
    doc_id        text NOT NULL REFERENCES docs(sha256) ON DELETE CASCADE,
    field         text NOT NULL DEFAULT '',
    action        text NOT NULL DEFAULT '',
    submitted_by  text,
    payload       jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS labels_payload_gin ON labels USING gin (payload jsonb_path_ops);
CREATE INDEX IF NOT EXISTS labels_doc_id_idx ON labels (doc_id);

-- ---------------------------------------------------------------------------
-- doc_evaluations: model scores per doc, JSONB payload
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS doc_evaluations (
    id                bigserial PRIMARY KEY,
    doc_id            text NOT NULL REFERENCES docs(sha256) ON DELETE CASCADE,
    field             text NOT NULL DEFAULT '',
    evaluator_version text NOT NULL DEFAULT 'cc2-v1',
    payload           jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS doc_evaluations_payload_gin ON doc_evaluations USING gin (payload jsonb_path_ops);
CREATE INDEX IF NOT EXISTS doc_evaluations_doc_id_idx ON doc_evaluations (doc_id);

-- ---------------------------------------------------------------------------
-- contract_schema: append-only event log
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS contract_schema (
    sequence_id   bigserial PRIMARY KEY,
    version       int NOT NULL,
    payload       jsonb NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS contract_schema_version_idx ON contract_schema (version DESC);

-- ---------------------------------------------------------------------------
-- model_runs: training-run projection (rule #6 — retrain is async, so it
-- writes a state row the UI subscribes to). Model state lives in MLflow;
-- this is a denormalized read-projection of each run so model_status() has a
-- Postgres-local source. NOT contract_schema — a schema edit must never look
-- like a retrain.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    id            bigserial PRIMARY KEY,
    run_id        text NOT NULL,
    payload       jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS model_runs_created_idx ON model_runs (created_at DESC);

-- ---------------------------------------------------------------------------
-- Backfill: legacy contract_schema rows carried the field map under
-- `payload.fields`; the canonical key is `payload.field_definitions` (used by
-- SchemaEditorPage, LabelPage, and the Python pipeline). Older rows would
-- otherwise show "No fields defined yet" even when fields exist. Idempotent
-- (WHERE clause filters out already-migrated rows), safe on every deploy.
-- ---------------------------------------------------------------------------
UPDATE contract_schema
   SET payload = payload || jsonb_build_object('field_definitions', payload->'fields')
 WHERE payload ? 'fields'
   AND NOT (payload ? 'field_definitions');

-- ---------------------------------------------------------------------------
-- PostgREST roles
-- authenticator: the LOGIN role used in PGRST_DB_URI — switches into web_anon
-- web_anon: NOLOGIN anonymous role — the read surface exposed by PostgREST
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticator') THEN
        CREATE ROLE authenticator NOINHERIT LOGIN NOCREATEDB NOCREATEROLE NOSUPERUSER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'web_anon') THEN
        CREATE ROLE web_anon NOLOGIN;
    END IF;
END
$$;

GRANT web_anon TO authenticator;

GRANT USAGE ON SCHEMA public TO web_anon;
GRANT SELECT ON docs, labels, doc_evaluations, contract_schema TO web_anon;

-- ---------------------------------------------------------------------------
-- RPCs: write-with-side-effect via PostgREST /rpc/<function>
-- Only SQL functions that require server-side compute (idempotency, sequencing).
-- No application code embeds raw SQL strings; all writes flow through these.
-- ---------------------------------------------------------------------------

-- ingest_doc: merge upsert — re-extraction enriches existing rows.
-- Single doc-write surface. No ordering dependency between callers —
-- ingest_doc creates the row if absent or merges into it if present.
-- Readiness derived from payload ? 'doc' (set by handle_reextract).
CREATE OR REPLACE FUNCTION ingest_doc(
    p_sha256 text,
    p_doc_schema_version text,
    p_payload jsonb
)
RETURNS docs LANGUAGE SQL AS $$
    INSERT INTO docs (sha256, doc_schema_version, payload)
    VALUES (p_sha256, p_doc_schema_version, p_payload)
    ON CONFLICT (sha256) DO UPDATE
        SET payload            = docs.payload || EXCLUDED.payload,
            doc_schema_version = EXCLUDED.doc_schema_version,
            updated_at         = now()
    RETURNING *;
$$;

-- append_contract_schema: monotonic version; no UPDATE/DELETE path exists
CREATE OR REPLACE FUNCTION append_contract_schema(p_payload jsonb)
RETURNS contract_schema LANGUAGE SQL SECURITY DEFINER SET search_path = public, pg_temp AS $$
    INSERT INTO contract_schema (version, payload)
    VALUES ((SELECT COALESCE(MAX(version), 0) + 1 FROM contract_schema), p_payload)
    RETURNING *;
$$;

-- latest_contract_schema: most recent contract_schema row. No filter, no gate —
-- contract_schema is the schema log only (model state lives in MLflow, decoupled),
-- so the latest row is always the current schema.
CREATE OR REPLACE FUNCTION latest_contract_schema()
RETURNS contract_schema LANGUAGE SQL AS $$
    SELECT * FROM contract_schema
    ORDER BY version DESC
    LIMIT 1;
$$;

-- latest_model_run_id: run_id of the most recent row in model_runs.
-- Reads model_runs (the retrain projection), not contract_schema.
-- Returns NULL when no training run has been recorded yet.
CREATE OR REPLACE FUNCTION latest_model_run_id()
RETURNS text LANGUAGE SQL SECURITY DEFINER SET search_path = public, pg_temp AS $$
    SELECT run_id
    FROM model_runs
    ORDER BY id DESC
    LIMIT 1;
$$;

-- ---------------------------------------------------------------------------
-- pgqueuer enqueue RPCs — callable by PostgREST /rpc/<function>
--
-- pgqueuer installs its schema at worker boot (queries.install()), NOT here.
-- LANGUAGE plpgsql + EXECUTE defers table-name resolution past parse time so
-- these functions can be created before the pgqueuer table exists.
-- The pgqueuer trigger fires pg_notify on INSERT — no manual NOTIFY needed.
-- ---------------------------------------------------------------------------



-- ---------------------------------------------------------------------------
-- queue_health: operational snapshot of pgqueuer state for the admin UI.
--
-- Uses EXECUTE for all pgqueuer/pgqueuer_log SELECTs — pgqueuer installs its
-- schema at worker boot, not here. Parse-time resolution would fail on first
-- deploy.
--
-- queue_health raises undefined_table until the worker installs the pgqueuer
-- schema. That is the correct teaching signal — empty arrays would mask
-- 'worker not running' as 'no jobs ever ran'.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION queue_health()
RETURNS json LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE
    v_counts     json;
    v_exceptions json;
    v_recent     json;
BEGIN
    -- counts_by_status: live queue state
    EXECUTE $q$
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
        FROM (
            SELECT status::text AS status, COUNT(*) AS count
            FROM pgqueuer
            GROUP BY status
        ) t
    $q$ INTO v_counts;

    -- recent_exceptions: last 20 job failures with traceback
    EXECUTE $q$
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
        FROM (
            SELECT
                job_id,
                entrypoint,
                created,
                traceback->>'exception_type'                                   AS exception_type,
                substring(traceback->>'exception_message' FROM 1 FOR 500)      AS exception_message
            FROM pgqueuer_log
            WHERE traceback IS NOT NULL
            ORDER BY created DESC
            LIMIT 20
        ) t
    $q$ INTO v_exceptions;

    -- recent_jobs: last 50 completed jobs
    EXECUTE $q$
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
        FROM (
            SELECT
                job_id,
                entrypoint,
                status::text AS status,
                created
            FROM pgqueuer_log
            ORDER BY created DESC
            LIMIT 50
        ) t
    $q$ INTO v_recent;

    RETURN json_build_object(
        'counts_by_status',  v_counts,
        'recent_exceptions', v_exceptions,
        'recent_jobs',       v_recent
    );
END;
$$;

-- submit_label: append a labeling event (one of four user actions)
-- action ∈ {'approve','correct','not_in_document','reject'}
-- payload carries action-specific shape:
--   approve:         {}
--   correct:         {correct_value, correct_bbox}
--   reject:          {correct_value}  (model's value the user is rejecting)
--   not_in_document: {}
CREATE OR REPLACE FUNCTION submit_label(
    p_doc_id       text,
    p_field        text,
    p_action       text,
    p_payload      jsonb,
    p_submitted_by text DEFAULT NULL
)
RETURNS labels LANGUAGE SQL AS $$
    INSERT INTO labels (doc_id, field, action, submitted_by, payload)
    VALUES (p_doc_id, p_field, p_action, p_submitted_by, COALESCE(p_payload, '{}'::jsonb))
    RETURNING *;
$$;

-- stale_count: docs with doc_evaluations older than their last label
-- Returns plain JSON for PostgREST /rpc/stale_count consumption.
-- Filtered to prediction events (evaluator_version = 'cc2-v1') so approval
-- rows in doc_evaluations (future) do not skew the stale signal.
CREATE OR REPLACE FUNCTION stale_count()
RETURNS json LANGUAGE SQL AS $$
    SELECT json_build_object(
        'count',
        COUNT(DISTINCT l.doc_id)
    )
    FROM labels l
    WHERE NOT EXISTS (
        SELECT 1 FROM doc_evaluations e
        WHERE e.doc_id = l.doc_id
          AND e.created_at > l.created_at
    );
$$;

-- ---------------------------------------------------------------------------
-- queue_list: paginated document queue for the labeling UI
--
-- Returns docs joined with their latest doc_evaluations summary, ordered by
-- the requested sort axis. Cursor-based pagination via docs.created_at.
-- filter ∈ {'ready','labeled','pending','failed','all'}
--   ready   — extracted (payload ? 'doc') AND not yet labeled — the work queue
--   labeled — extracted AND has labels — docs already worked (labeled lane)
--   pending — docs where NOT payload ? 'doc' (not yet extracted)
--   failed  — alias for pending (no failure state; UI label is cosmetic)
--   all     — all docs
-- Readiness derived from payload ? 'doc'; no status column.
-- ---------------------------------------------------------------------------
-- Wave-V signature change: queue_list went from 4 to 5 params (added p_search).
-- CREATE OR REPLACE only updates same-signature; the old 4-param overload would
-- linger and confuse PostgREST function resolution. Explicit DROP clears it.
DROP FUNCTION IF EXISTS queue_list(jsonb);
DROP FUNCTION IF EXISTS queue_list(text, text, int, text, text);
DROP FUNCTION IF EXISTS queue_list(text, text, int, text);
CREATE OR REPLACE FUNCTION queue_list(jsonb DEFAULT '{}'::jsonb)
RETURNS json
LANGUAGE SQL
AS $$
    WITH
    p AS (
        SELECT
            COALESCE($1->>'p_sort',        'recent') AS p_sort,
            COALESCE($1->>'p_filter',      'ready')  AS p_filter,
            COALESCE(($1->>'p_page_size')::int, 50)  AS p_page_size,
            $1->>'p_cursor'                          AS p_cursor,
            $1->>'p_search'                          AS p_search
    ),
    ranked AS (
        SELECT
            d.sha256                              AS doc_id,
            d.sha256,
            d.created_at                          AS imported_at,
            d.payload->>'filename'                AS filename,
            d.payload->>'source'                  AS source,
            d.payload->>'source_id'               AS source_id,
            COUNT(DISTINCT e.field)               AS field_count,
            (d.payload ? 'doc')                   AS is_ready,
            EXISTS (SELECT 1 FROM labels l WHERE l.doc_id = d.sha256) AS has_labels
        FROM docs d
        LEFT JOIN doc_evaluations e ON e.doc_id = d.sha256
        GROUP BY d.sha256, d.created_at, d.payload
    ),
    filtered AS (
        SELECT * FROM ranked, p
        WHERE
            CASE p.p_filter
                WHEN 'ready'   THEN is_ready = TRUE AND has_labels = FALSE
                WHEN 'labeled' THEN is_ready = TRUE AND has_labels = TRUE
                WHEN 'pending' THEN is_ready = FALSE
                WHEN 'failed'  THEN is_ready = FALSE
                ELSE TRUE
            END
            AND (
                p.p_search IS NULL
                OR filename ILIKE '%' || p.p_search || '%'
                OR EXISTS (
                    SELECT 1 FROM doc_evaluations e2
                    WHERE e2.doc_id = sha256
                      AND e2.payload::text ILIKE '%' || p.p_search || '%'
                )
            )
            AND (
                p.p_cursor IS NULL
                OR (p.p_sort = 'recent'      AND imported_at < (SELECT d3.created_at FROM docs d3 WHERE d3.sha256 = p.p_cursor))
                OR (p.p_sort = 'alphabetical' AND (filename, sha256) > (
                        (SELECT d2.payload->>'filename' FROM docs d2 WHERE d2.sha256 = p.p_cursor),
                        p.p_cursor
                   ))
                OR (p.p_sort NOT IN ('recent','alphabetical') AND imported_at < (SELECT d4.created_at FROM docs d4 WHERE d4.sha256 = p.p_cursor))
            )
        ORDER BY
            CASE WHEN p.p_sort = 'alphabetical' THEN filename END ASC NULLS LAST,
            CASE WHEN p.p_sort = 'recent' OR p.p_sort NOT IN ('alphabetical') THEN imported_at END DESC NULLS LAST,
            sha256 ASC
        LIMIT (SELECT p_page_size FROM p) + 1
    ),
    page AS (
        SELECT * FROM filtered LIMIT (SELECT p_page_size FROM p)
    )
    SELECT json_build_object(
        'items',        COALESCE(json_agg(row_to_json(page.*)), '[]'::json),
        'total_count',  (SELECT COUNT(*) FROM filtered),
        'next_cursor',  (
            CASE WHEN (SELECT COUNT(*) FROM filtered) > (SELECT p_page_size FROM p)
                THEN (SELECT sha256 FROM filtered LIMIT 1 OFFSET (SELECT p_page_size FROM p))
                ELSE NULL
            END
        )
    )
    FROM page;
$$;

-- ---------------------------------------------------------------------------
-- get_document_source: narrow projection for the Go PDF proxy backend.
-- Returns only the two fields needed to resolve the SharePoint download URL:
--   source_id (SharePoint item ID) and drive_id (SharePoint drive ID).
-- Consumers: grafana-plugin Go backend resource handler.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_document_source(p_sha text)
RETURNS json LANGUAGE SQL AS $$
    SELECT json_build_object(
        'source_id', payload->>'source_id',
        'drive_id',  payload->>'drive_id'
    )
    FROM docs
    WHERE sha256 = p_sha;
$$;

-- ---------------------------------------------------------------------------
-- get_document: single-doc detail for the labeling UI.
-- Returns doc metadata + Doc JSON (page/token spatial geometry) +
-- per-field prediction payload (latest eval per field).
-- Write path (handle_reextract) stores:
--   payload->>'source_id'  — SharePoint item ID
--   payload->>'drive_id'   — SharePoint drive ID (from connector.config.drive_id)
--   payload->'doc'         — full Doc JSON (result.doc.model_dump(mode="json"))
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_document(p_sha text)
RETURNS json LANGUAGE SQL AS $$
    SELECT json_build_object(
        'doc_id',        d.sha256,
        'sha256',        d.sha256,
        'filename',      d.payload->>'filename',
        'source_id',     d.payload->>'source_id',
        'drive_id',      d.payload->>'drive_id',
        'doc',           d.payload->'doc',
        'pages',         COALESCE(jsonb_array_length(d.payload->'doc'->'pages'), 0),
        'all_candidates', d.payload->'candidates',
        'predictions', (
            SELECT json_object_agg(
                e.field,
                json_build_object(
                    'value',       e.payload->'candidates'->0->>'raw_text',
                    'confidence',  COALESCE((e.payload->>'priority_score')::float, 0),
                    'status',      'PREDICTED',
                    'provenance',  json_build_object(
                        'page_idx',     (e.payload->'candidates'->0->>'page_idx')::int,
                        'bbox_norm_x0', (e.payload->'candidates'->0->>'bbox_norm_x0')::float,
                        'bbox_norm_y0', (e.payload->'candidates'->0->>'bbox_norm_y0')::float,
                        'bbox_norm_x1', (e.payload->'candidates'->0->>'bbox_norm_x1')::float,
                        'bbox_norm_y1', (e.payload->'candidates'->0->>'bbox_norm_y1')::float
                    ),
                    'raw_text',    e.payload->'candidates'->0->>'raw_text',
                    'candidates',  e.payload->'candidates',
                    'approved_at', (
                        SELECT MAX(l.created_at)
                        FROM labels l
                        WHERE l.doc_id = p_sha
                          AND l.field  = e.field
                          AND l.action IN ('approve', 'correct')
                    )
                )
            )
            FROM (
                SELECT DISTINCT ON (doc_id, field) doc_id, field, payload
                FROM doc_evaluations
                WHERE doc_id = p_sha
                ORDER BY doc_id, field, created_at DESC
            ) e
        )
    )
    FROM docs d
    WHERE d.sha256 = p_sha;
$$;

-- ---------------------------------------------------------------------------
-- get_next_doc: cursor-advance for the labeling UI — next doc older than
-- current (created_at DESC walk). No status or schema filters; every doc
-- is a candidate regardless of extraction state or whether predictions exist.
-- Returns null when the current doc is the oldest or the table is empty.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_next_doc(p_current_sha text DEFAULT NULL)
RETURNS json LANGUAGE sql AS $$
    SELECT json_build_object(
        'sha256',   sha256,
        'doc_id',   sha256,
        'filename', payload->>'filename'
    )
    FROM docs
    WHERE p_current_sha IS NULL
       OR created_at < (SELECT created_at FROM docs WHERE sha256 = p_current_sha)
    ORDER BY created_at DESC
    LIMIT 1;
$$;

-- ---------------------------------------------------------------------------
-- pgqueuer schema (inlined from pgqueuer==1.0.2 `pgq install --dry-run`)
-- Idempotent: every primitive is IF NOT EXISTS / CREATE OR REPLACE.
-- Library pin: pyproject.toml exact-pins pgqueuer so this DDL never drifts.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'pgqueuer_status') THEN
    CREATE TYPE pgqueuer_status AS ENUM ('queued', 'picked', 'successful', 'exception', 'canceled', 'deleted', 'failed');
  END IF;
END
$$;

CREATE TABLE IF NOT EXISTS pgqueuer (
    id SERIAL PRIMARY KEY,
    priority INT NOT NULL,
    queue_manager_id UUID,
    created TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    execute_after TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    status pgqueuer_status NOT NULL,
    entrypoint TEXT NOT NULL,
    dedupe_key TEXT,
    payload BYTEA,
    headers JSONB,
    attempts INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS pgqueuer_priority_id_id1_idx ON pgqueuer (priority ASC, id DESC) INCLUDE (id) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS pgqueuer_updated_id_id1_idx ON pgqueuer (updated ASC, id DESC) INCLUDE (id) WHERE status = 'picked';
CREATE INDEX IF NOT EXISTS pgqueuer_queue_manager_id_idx ON pgqueuer (queue_manager_id) WHERE queue_manager_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS pgqueuer_unique_dedupe_key ON pgqueuer (dedupe_key) WHERE ((status IN ('queued', 'picked') AND dedupe_key IS NOT NULL));

-- 1.0.x in-place upgrade for pre-existing 0.26.x deployments (idempotent).
-- PG16 permits ALTER TYPE ... ADD VALUE inside a transaction block as long as
-- the new value is not used in the same transaction (init.sql never uses 'failed').
ALTER TABLE pgqueuer ADD COLUMN IF NOT EXISTS attempts INT NOT NULL DEFAULT 0;
ALTER TYPE pgqueuer_status ADD VALUE IF NOT EXISTS 'failed';

CREATE TABLE IF NOT EXISTS pgqueuer_log (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    created TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    job_id BIGINT NOT NULL,
    status pgqueuer_status NOT NULL,
    priority INT NOT NULL,
    entrypoint TEXT NOT NULL,
    traceback JSONB DEFAULT NULL,
    aggregated BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS pgqueuer_log_not_aggregated ON pgqueuer_log ((1)) WHERE not aggregated;
CREATE INDEX IF NOT EXISTS pgqueuer_log_created ON pgqueuer_log (created);
CREATE INDEX IF NOT EXISTS pgqueuer_log_status ON pgqueuer_log (status);
CREATE INDEX IF NOT EXISTS pgqueuer_log_job_id_status ON pgqueuer_log (job_id, created DESC);

CREATE TABLE IF NOT EXISTS pgqueuer_statistics (
    id SERIAL PRIMARY KEY,
    created TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT DATE_TRUNC('sec', NOW() at time zone 'UTC'),
    count BIGINT NOT NULL,
    priority INT NOT NULL,
    status pgqueuer_status NOT NULL,
    entrypoint TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS pgqueuer_statistics_unique_count ON pgqueuer_statistics (priority, DATE_TRUNC('sec', created at time zone 'UTC'), status, entrypoint);

CREATE TABLE IF NOT EXISTS pgqueuer_schedules (
    id SERIAL PRIMARY KEY,
    expression TEXT NOT NULL,
    entrypoint TEXT NOT NULL,
    heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    created TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    next_run TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    last_run TIMESTAMP WITH TIME ZONE,
    status pgqueuer_status DEFAULT 'queued',
    UNIQUE (expression, entrypoint)
);

CREATE OR REPLACE FUNCTION fn_pgqueuer_changed() RETURNS TRIGGER AS $$
DECLARE
    to_emit BOOLEAN := false;
BEGIN
    IF TG_OP = 'UPDATE' AND OLD IS DISTINCT FROM NEW THEN
        to_emit := true;
    ELSIF TG_OP = 'DELETE' THEN
        to_emit := true;
    ELSIF TG_OP = 'INSERT' THEN
        to_emit := true;
    ELSIF TG_OP = 'TRUNCATE' THEN
        to_emit := true;
    END IF;
    IF to_emit THEN
        PERFORM pg_notify(
            'ch_pgqueuer',
            json_build_object(
                'channel', 'ch_pgqueuer',
                'operation', lower(TG_OP),
                'sent_at', NOW(),
                'table', TG_TABLE_NAME,
                'type', 'table_changed_event'
            )::text
        );
    END IF;
    IF TG_OP IN ('INSERT', 'UPDATE') THEN
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NULL;
    END IF;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tg_pgqueuer_changed ON pgqueuer;
CREATE TRIGGER tg_pgqueuer_changed
AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON pgqueuer
EXECUTE FUNCTION fn_pgqueuer_changed();

-- ---------------------------------------------------------------------------
-- fn_pgqueuer_enqueue — PostgREST RPC wrapper; web_anon does not need direct
-- INSERT on pgqueuer. Dedupe collision (23505) is silently swallowed to match
-- Python-side DuplicateJobError handling. NOTIFY is handled by tg_pgqueuer_changed.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION fn_pgqueuer_enqueue(
    entrypoint  TEXT,
    payload     TEXT    DEFAULT NULL,
    dedupe_key  TEXT    DEFAULT NULL
) RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    WITH inserted AS (
        INSERT INTO pgqueuer (priority, entrypoint, payload, execute_after, dedupe_key, headers, status)
        VALUES (
            0,
            fn_pgqueuer_enqueue.entrypoint,
            CASE WHEN fn_pgqueuer_enqueue.payload IS NOT NULL
                 THEN decode(fn_pgqueuer_enqueue.payload, 'base64')
                 ELSE NULL END,
            NOW(),
            fn_pgqueuer_enqueue.dedupe_key,
            NULL,
            'queued'
        )
        RETURNING pgqueuer.id, pgqueuer.entrypoint, pgqueuer.status, pgqueuer.priority
    )
    INSERT INTO pgqueuer_log (job_id, status, entrypoint, priority)
    SELECT inserted.id, 'queued', inserted.entrypoint, inserted.priority FROM inserted;
EXCEPTION WHEN unique_violation THEN
    RETURN;
END;
$$;

-- ---------------------------------------------------------------------------
-- model_status: projection of current model state + training history + label
-- activity for the admin UI. Pure read from model_runs + labels — no
-- dynamic SQL needed (both tables exist at parse time, unlike pgqueuer).
--
-- current: latest training run from model_runs; NULL on first-run.
-- history: last 20 training runs, newest first.
-- recent_labels: 7-day rolling window of labeling activity by action.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION model_status()
RETURNS json LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE
    v_current json;
    v_history json;
    v_labels  json;
BEGIN
    -- current: most recent training run (model_runs) or NULL on first-run
    SELECT json_build_object(
        'version',             mr.id,
        'run_id',              mr.run_id,
        'trained_at',          mr.created_at,
        'n_docs',              (mr.payload->>'n_groups')::int,
        'n_samples',           (mr.payload->>'n_samples')::int,
        'n_positive',          (mr.payload->>'n_positive')::int,
        'n_features',          (mr.payload->>'n_features')::int,
        'mlflow_model_prefix', mr.payload->>'mlflow_model_prefix'
    )
    INTO v_current
    FROM model_runs mr
    ORDER BY mr.id DESC
    LIMIT 1;

    -- history: last 20 training runs, newest first
    SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
    INTO v_history
    FROM (
        SELECT
            id                             AS version,
            run_id,
            created_at                     AS trained_at,
            (payload->>'n_groups')::int    AS n_docs,
            (payload->>'n_positive')::int  AS n_positive
        FROM model_runs
        ORDER BY id DESC
        LIMIT 20
    ) t;

    -- recent_labels: 7-day activity summary
    SELECT json_build_object(
        'total',         COUNT(*),
        'approve_count', COUNT(*) FILTER (WHERE action = 'approve'),
        'correct_count', COUNT(*) FILTER (WHERE action = 'correct'),
        'reject_count',  COUNT(*) FILTER (WHERE action = 'reject')
    )
    INTO v_labels
    FROM labels
    WHERE created_at >= now() - interval '7 days';

    RETURN json_build_object(
        'current',       v_current,
        'history',       v_history,
        'recent_labels', v_labels
    );
END;
$$;

-- ---------------------------------------------------------------------------
-- PostgREST function grants — must come after all CREATE OR REPLACE FUNCTION
-- ---------------------------------------------------------------------------
GRANT INSERT ON labels TO web_anon;
GRANT USAGE, SELECT ON SEQUENCE labels_id_seq TO web_anon;

-- ---------------------------------------------------------------------------
-- extraction_results_view: pivot doc_evaluations into a per-doc/field table
-- for the Results page.  Returns the latest evaluation per doc+field pair.
-- DISTINCT ON with ORDER BY id DESC picks the most-recent row per (doc, field)
-- without a correlated subquery, giving a better query plan as the table grows.
-- Payload structure mirrors get_document(): value from candidates[0].raw_text,
-- confidence from priority_score.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION extraction_results_view()
RETURNS json LANGUAGE SQL SECURITY DEFINER SET search_path = public, pg_temp AS $$
    SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
    FROM (
        SELECT * FROM (
            SELECT DISTINCT ON (e.doc_id, e.field)
                d.payload->>'filename'                           AS filename,
                e.field,
                e.payload->'candidates'->0->>'raw_text'          AS extracted_value,
                COALESCE((e.payload->>'priority_score')::float, 0) AS confidence,
                e.evaluator_version,
                e.created_at                                     AS evaluated_at,
                d.sha256                                         AS doc_id
            FROM doc_evaluations e
            JOIN docs d ON d.sha256 = e.doc_id
            ORDER BY e.doc_id, e.field, e.id DESC
        ) s
        ORDER BY s.evaluated_at DESC, s.doc_id, s.field
    ) t;
$$;

-- ---------------------------------------------------------------------------
-- dump_docs / dump_labels / dump_doc_evaluations: full-table exports for the
-- dump-state workflow. Routed through /api/rpc/ (nginx → PostgREST) so the
-- dump runner never needs direct PostgREST port access (port 3000 is
-- expose-only, not published to the host).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION dump_docs()
RETURNS json LANGUAGE SQL SECURITY DEFINER SET search_path = public, pg_temp AS $$
    SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
    FROM (SELECT sha256, doc_schema_version, created_at, updated_at FROM docs ORDER BY created_at DESC) t;
$$;

CREATE OR REPLACE FUNCTION dump_labels()
RETURNS json LANGUAGE SQL SECURITY DEFINER SET search_path = public, pg_temp AS $$
    SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
    FROM (
        SELECT id, doc_id, field, action, submitted_by, payload, created_at
        FROM labels ORDER BY created_at DESC
    ) t;
$$;

CREATE OR REPLACE FUNCTION dump_doc_evaluations()
RETURNS json LANGUAGE SQL SECURITY DEFINER SET search_path = public, pg_temp AS $$
    SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
    FROM (
        SELECT id, doc_id, field, evaluator_version, payload, created_at
        FROM doc_evaluations ORDER BY created_at DESC
    ) t;
$$;

GRANT EXECUTE ON FUNCTION
    ingest_doc(text, text, jsonb),
    append_contract_schema(jsonb),
    latest_contract_schema(),
    submit_label(text, text, text, jsonb, text),
    stale_count(),
    queue_list(jsonb),
    get_document(text),
    get_document_source(text),
    queue_health(),
    get_next_doc(text),
    fn_pgqueuer_enqueue(text, text, text),
    latest_model_run_id(),
    model_status(),
    extraction_results_view(),
    dump_docs(),
    dump_labels(),
    dump_doc_evaluations()
TO web_anon;
