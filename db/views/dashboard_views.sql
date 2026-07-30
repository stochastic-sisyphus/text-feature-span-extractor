-- dashboard_views.sql
--
-- Stable read-model views for Grafana.  Every Grafana panel/dashboard
-- reads from a `dashboard_*` view, never from a base table.  Base
-- tables can evolve underneath; the views absorb the drift.
--
-- This is ADDITIVE.  It creates new views over existing tables.  No
-- existing data is touched.  No tables are altered.  Apply via:
--
--     psql ... -f dashboard_views.sql
--
-- (or paste in pgAdmin if no CLI access — every CREATE OR REPLACE is
--  idempotent, so re-running is safe.)
--
-- Adding a column the dashboards need
-- -----------------------------------
-- 1. Add the column to the underlying table.
-- 2. Add it to the relevant CREATE OR REPLACE VIEW below.
-- 3. Re-run this file.
-- 4. Update the Grafana panel JSON to read the new column.
--
-- Removing or renaming a column
-- -----------------------------
-- 1. Update the CREATE OR REPLACE VIEW below: COALESCE the old column
--    name to the new value, OR add `NULL::TYPE AS old_name` so the
--    dashboard still gets the column.
-- 2. Re-run.  Dashboard keeps working.
-- 3. Now you have a window to update the dashboard panel without
--    breaking it.
-- 4. Once panels are migrated, drop the COALESCE/NULL in the view.
--
-- The dashboard contract is the view, not the table.

-- ---------------------------------------------------------------
-- dashboard_review_queue
-- ---------------------------------------------------------------
-- The "what's left to label" surface.  Joins ingest_index → docs →
-- doc_evaluations and projects a stable column set.  Missing columns
-- in doc_evaluations (added later, removed later) materialize as NULL
-- here so dashboards never break on schema drift.

CREATE OR REPLACE VIEW dashboard_review_queue AS
SELECT
    ii.sha256                              AS sha256,
    ii.doc_id                              AS doc_id,
    ii.filename                            AS filename,
    ii.imported_at                         AS imported_at,
    de.field                               AS field,
    -- The next four are columns that have been added/removed historically
    -- on doc_evaluations.  Coalesce to keep dashboards stable.
    COALESCE(de.reason,            NULL)   AS reason,
    COALESCE(de.priority_score,    NULL)   AS priority_score,
    COALESCE(de.signal_disagreement, FALSE) AS signal_disagreement,
    COALESCE(de.used_ml_model,     FALSE)  AS used_ml_model,
    COALESCE(de.evaluator_version, 'live') AS evaluator_version,
    -- recency_at: prefer doc_evaluations.created_at if present, else
    -- ingest imported_at.  Resilient to either column being absent
    -- in older snapshots.
    COALESCE(de.created_at, ii.imported_at) AS recency_at,
    NOT EXISTS (
        SELECT 1 FROM approvals a WHERE a.doc_id = ii.doc_id
    )                                      AS unapproved
FROM ingest_index ii
LEFT JOIN doc_evaluations de ON de.doc_id = ii.doc_id;


-- ---------------------------------------------------------------
-- dashboard_ledger_status
-- ---------------------------------------------------------------
-- Stable surface for "what state is each doc in?" panels.

CREATE OR REPLACE VIEW dashboard_ledger_status AS
SELECT
    l.sha256             AS sha256,
    l.doc_id             AS doc_id,
    l.source_id          AS source_id,
    l.status             AS status,
    l.retry_count        AS retry_count,
    l.error_message      AS error_message,
    l.extraction_version AS extraction_version,
    l.processed_at       AS processed_at,
    -- Friendly bucket for the status pie chart.
    CASE l.status
        WHEN 'processed' THEN 'processed'
        WHEN 'pending'   THEN 'pending'
        WHEN 'failed'    THEN 'failed'
        ELSE 'other'
    END                  AS status_bucket
FROM ledger l;


-- ---------------------------------------------------------------
-- dashboard_doc_inventory
-- ---------------------------------------------------------------
-- "How many docs do we have, by what we know about them."

CREATE OR REPLACE VIEW dashboard_doc_inventory AS
SELECT
    ii.sha256                AS sha256,
    ii.doc_id                AS doc_id,
    ii.filename              AS filename,
    ii.imported_at           AS imported_at,
    (d.payload IS NOT NULL)  AS has_payload,
    COALESCE(d.doc_schema_version, 'unknown') AS doc_schema_version,
    l.status                 AS ledger_status,
    EXISTS (
        SELECT 1 FROM doc_evaluations de WHERE de.doc_id = ii.doc_id
    )                        AS has_evaluations,
    EXISTS (
        SELECT 1 FROM approvals a WHERE a.doc_id = ii.doc_id
    )                        AS has_approval
FROM ingest_index ii
LEFT JOIN docs   d ON d.sha256 = ii.sha256
LEFT JOIN ledger l ON l.sha256 = ii.sha256;


-- ---------------------------------------------------------------
-- dashboard_label_activity
-- ---------------------------------------------------------------
-- Time series of human labeling activity.  Stable so a panel can read
-- "labels per day" without ever caring how `corrections` is laid out
-- internally.

CREATE OR REPLACE VIEW dashboard_label_activity AS
SELECT
    DATE_TRUNC('hour', c.created_at) AS hour_bucket,
    COUNT(*)                          AS n_corrections,
    COUNT(DISTINCT c.doc_id)          AS n_docs_touched,
    COUNT(DISTINCT c.field)           AS n_distinct_fields
FROM corrections c
GROUP BY 1
ORDER BY 1 DESC;


-- ===============================================================
-- PHASE 2 — ADMIN EXPOSURE VIEWS
-- ===============================================================
-- Eight additional read-model views for full admin-surface Grafana
-- dashboards.  Same contract: CREATE OR REPLACE, fully idempotent,
-- no table altered, COALESCE on every nullable column.
--
-- FLAG: doc_evaluations has no predicted_value column.
-- dashboard_corrections_full exposes the evaluator snapshot at
-- correction time (evaluator_version, priority_score, reason) for
-- lineage, but cannot project a predicted_value — the column does
-- not exist.  If a predicted_value column is added to doc_evaluations
-- in a future migration, add it to dashboard_corrections_full then.
-- ===============================================================


-- ---------------------------------------------------------------
-- dashboard_doc_full
-- ---------------------------------------------------------------
-- Full document surface: ingest_index + ledger + docs with rollup
-- subqueries for approvals, corrections, and evaluations counts.

CREATE OR REPLACE VIEW dashboard_doc_full AS
SELECT
    ii.sha256                                       AS sha256,
    ii.doc_id                                       AS doc_id,
    COALESCE(ii.filename,    '')                    AS filename,
    ii.imported_at                                  AS imported_at,
    -- ledger columns
    COALESCE(l.status,             'unknown')       AS ledger_status,
    COALESCE(l.source_id,          '')              AS source_id,
    COALESCE(l.extraction_version, 'unknown')       AS extraction_version,
    COALESCE(l.retry_count,        0)               AS retry_count,
    COALESCE(l.error_message,      '')              AS error_message,
    l.processed_at                                  AS processed_at,
    l.dataverse_id                                  AS dataverse_id,
    -- docs columns
    (d.payload IS NOT NULL)                         AS has_payload,
    COALESCE(d.doc_schema_version, 'unknown')       AS doc_schema_version,
    d.created_at                                    AS doc_created_at,
    d.updated_at                                    AS doc_updated_at,
    -- rollup subqueries
    (
        SELECT COUNT(*)
        FROM approvals a
        WHERE a.doc_id = ii.doc_id
    )                                               AS n_approvals,
    (
        SELECT COUNT(*)
        FROM corrections c
        WHERE c.doc_id = ii.doc_id
    )                                               AS n_corrections,
    (
        SELECT COUNT(*)
        FROM doc_evaluations de
        WHERE de.doc_id = ii.doc_id
    )                                               AS n_evaluations
FROM ingest_index ii
LEFT JOIN ledger l ON l.sha256 = ii.sha256
LEFT JOIN docs   d ON d.sha256 = ii.sha256;


-- ---------------------------------------------------------------
-- dashboard_evaluations_full
-- ---------------------------------------------------------------
-- Full evaluation surface: doc_evaluations + ingest_index metadata.
-- Priority bucket for panel segmentation.

CREATE OR REPLACE VIEW dashboard_evaluations_full AS
SELECT
    de.doc_id                                           AS doc_id,
    de.field                                            AS field,
    COALESCE(de.evaluator_version,     'unknown')       AS evaluator_version,
    COALESCE(de.priority_score,        0.0)             AS priority_score,
    COALESCE(de.reason,                '')              AS reason,
    COALESCE(de.signal_disagreement,   FALSE)           AS signal_disagreement,
    COALESCE(de.used_ml_model,         FALSE)           AS used_ml_model,
    de.created_at                                       AS created_at,
    -- ingest_index join
    COALESCE(ii.filename,              '')              AS filename,
    ii.imported_at                                      AS imported_at,
    ii.sha256                                           AS sha256,
    -- priority bucket for pie/bar panels
    CASE
        WHEN COALESCE(de.priority_score, 0.0) >= 0.75 THEN 'high'
        WHEN COALESCE(de.priority_score, 0.0) >= 0.40 THEN 'medium'
        ELSE                                                'low'
    END                                                 AS priority_bucket
FROM doc_evaluations de
LEFT JOIN ingest_index ii ON ii.doc_id = de.doc_id;


-- ---------------------------------------------------------------
-- dashboard_corrections_full
-- ---------------------------------------------------------------
-- Full corrections surface with ingest_index and doc_evaluations
-- lineage (evaluator snapshot at correction time by doc+field).
--
-- NOTE: doc_evaluations has no predicted_value column.  Lineage is
-- limited to evaluator_version, priority_score, and reason.
-- See FLAG note in section header above.

CREATE OR REPLACE VIEW dashboard_corrections_full AS
SELECT
    c.id                                                AS correction_id,
    c.doc_id                                            AS doc_id,
    c.field                                             AS field,
    COALESCE(c.correct_value,   '')                     AS correct_value,
    COALESCE(c.entity_name,     '')                     AS entity_name,
    COALESCE(c.schema_version,  'unknown')              AS schema_version,
    COALESCE(c.notes,           '')                     AS notes,
    COALESCE(c.action,          'correct')              AS action,
    c.created_at                                        AS created_at,
    -- ingest_index join
    COALESCE(ii.filename,       '')                     AS filename,
    ii.imported_at                                      AS imported_at,
    ii.sha256                                           AS sha256,
    -- doc_evaluations lineage snapshot (latest evaluator version for this doc+field)
    COALESCE(de.evaluator_version, 'unknown')           AS evaluator_version,
    COALESCE(de.priority_score,    0.0)                 AS evaluator_priority_score,
    COALESCE(de.reason,            '')                  AS evaluator_reason,
    COALESCE(de.signal_disagreement, FALSE)             AS evaluator_signal_disagreement
FROM corrections c
LEFT JOIN ingest_index   ii ON ii.doc_id = c.doc_id
LEFT JOIN doc_evaluations de ON de.doc_id = c.doc_id
                             AND de.field = c.field;


-- ---------------------------------------------------------------
-- dashboard_approvals_full
-- ---------------------------------------------------------------
-- Full approvals surface with ingest_index and per-doc rollup
-- subqueries for corrections count and labeler count at approval time.

CREATE OR REPLACE VIEW dashboard_approvals_full AS
SELECT
    a.id                                                AS approval_id,
    a.doc_id                                            AS doc_id,
    a.field                                             AS field,
    COALESCE(a.entity_name,    '')                      AS entity_name,
    COALESCE(a.schema_version, 'unknown')               AS schema_version,
    COALESCE(a.notes,          '')                      AS notes,
    a.created_at                                        AS created_at,
    -- ingest_index join
    COALESCE(ii.filename,      '')                      AS filename,
    ii.imported_at                                      AS imported_at,
    ii.sha256                                           AS sha256,
    -- corrections at time of approval (for this doc+field)
    (
        SELECT COUNT(*)
        FROM corrections c
        WHERE c.doc_id = a.doc_id
          AND c.field  = a.field
          AND c.created_at <= a.created_at
    )                                                   AS n_corrections_at_approval,
    -- distinct labelers who touched this doc (approvals table, all fields)
    (
        SELECT COUNT(DISTINCT a2.entity_name)
        FROM approvals a2
        WHERE a2.doc_id = a.doc_id
          AND a2.entity_name IS NOT NULL
    )                                                   AS n_labelers
FROM approvals a
LEFT JOIN ingest_index ii ON ii.doc_id = a.doc_id;


-- ---------------------------------------------------------------
-- dashboard_pipeline_runs_full
-- ---------------------------------------------------------------
-- Pipeline runs with COALESCE(ended_at, NOW()) for in-progress runs
-- and computed duration_seconds.

CREATE OR REPLACE VIEW dashboard_pipeline_runs_full AS
SELECT
    pr.run_id                                               AS run_id,
    COALESCE(pr.status,    'unknown')                       AS status,
    pr.started_at                                           AS started_at,
    pr.ended_at                                             AS ended_at,
    COALESCE(pr.ended_at, NOW())                            AS ended_at_or_now,
    COALESCE(pr.doc_count, 0)                               AS doc_count,
    COALESCE(pr.error,     '')                              AS error,
    EXTRACT(
        EPOCH FROM (COALESCE(pr.ended_at, NOW()) - pr.started_at)
    )::BIGINT                                               AS duration_seconds
FROM pipeline_runs pr;


-- ---------------------------------------------------------------
-- dashboard_model_state_full
-- ---------------------------------------------------------------
-- UNION ALL across all four ML state singleton tables.
-- payload_excerpt is the first 500 chars of the JSONB column cast to
-- text, for quick inspection without pulling the whole blob.

CREATE OR REPLACE VIEW dashboard_model_state_full AS
SELECT
    'tuned_weights'                                     AS source_table,
    tw.id::TEXT                                         AS record_id,
    tw.updated_at                                       AS updated_at,
    LEFT(tw.weights::TEXT, 500)                         AS payload_excerpt
FROM tuned_weights tw

UNION ALL

SELECT
    'calibration_mapping'                               AS source_table,
    cm.id::TEXT                                         AS record_id,
    cm.updated_at                                       AS updated_at,
    LEFT(cm.mapping::TEXT, 500)                         AS payload_excerpt
FROM calibration_mapping cm

UNION ALL

SELECT
    'learned_anchors'                                   AS source_table,
    la.id::TEXT                                         AS record_id,
    la.updated_at                                       AS updated_at,
    LEFT(la.anchors::TEXT, 500)                         AS payload_excerpt
FROM learned_anchors la

UNION ALL

SELECT
    'field_blend_weights'                               AS source_table,
    fb.id::TEXT                                         AS record_id,
    fb.updated_at                                       AS updated_at,
    LEFT(fb.weights::TEXT, 500)                         AS payload_excerpt
FROM field_blend_weights fb;


-- ---------------------------------------------------------------
-- dashboard_inflight
-- ---------------------------------------------------------------
-- Ledger rows in 'pending' or 'failed' states with age in seconds.

CREATE OR REPLACE VIEW dashboard_inflight AS
SELECT
    l.sha256                                            AS sha256,
    l.doc_id                                            AS doc_id,
    COALESCE(l.source_id,          '')                  AS source_id,
    l.status                                            AS status,
    COALESCE(l.retry_count,        0)                   AS retry_count,
    COALESCE(l.error_message,      '')                  AS error_message,
    COALESCE(l.extraction_version, 'unknown')           AS extraction_version,
    l.processed_at                                      AS processed_at,
    EXTRACT(
        EPOCH FROM (NOW() - l.processed_at)
    )::BIGINT                                           AS seconds_since_status
FROM ledger l
WHERE l.status IN ('pending', 'failed');


-- ---------------------------------------------------------------
-- dashboard_queue_view
-- ---------------------------------------------------------------
-- Richer queue surface: ingest_index + doc_evaluations with full
-- COALESCE on nullable evaluation columns and unapproved flag.
-- Superset of dashboard_review_queue; panels needing extra columns
-- should migrate here.

CREATE OR REPLACE VIEW dashboard_queue_view AS
SELECT
    ii.sha256                                           AS sha256,
    ii.doc_id                                           AS doc_id,
    COALESCE(ii.filename,              '')              AS filename,
    ii.imported_at                                      AS imported_at,
    -- evaluation columns (all nullable — COALESCE to safe defaults)
    COALESCE(de.field,                 '')              AS field,
    COALESCE(de.evaluator_version,     'unknown')       AS evaluator_version,
    COALESCE(de.priority_score,        0.0)             AS priority_score,
    COALESCE(de.reason,                '')              AS reason,
    COALESCE(de.signal_disagreement,   FALSE)           AS signal_disagreement,
    COALESCE(de.used_ml_model,         FALSE)           AS used_ml_model,
    COALESCE(de.created_at, ii.imported_at)             AS recency_at,
    -- unapproved flag: TRUE when no approval row exists for this doc
    NOT EXISTS (
        SELECT 1 FROM approvals a WHERE a.doc_id = ii.doc_id
    )                                                   AS unapproved
FROM ingest_index ii
LEFT JOIN doc_evaluations de ON de.doc_id = ii.doc_id;
