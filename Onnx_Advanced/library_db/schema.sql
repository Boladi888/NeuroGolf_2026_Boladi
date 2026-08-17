-- Onnx_Advanced generated SQLite library.
-- Source of truth remains the routed Markdown library. This schema is a compiled
-- agent lookup surface with provenance/status fields on every fact-like row.

PRAGMA foreign_keys = ON;

CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE source_files (
  path TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  short_sha TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  mtime_utc TEXT NOT NULL
);

CREATE TABLE techniques (
  section TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '',
  formerly TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'current',
  source_path TEXT NOT NULL,
  source_anchor TEXT NOT NULL,
  source_line INTEGER
);

CREATE TABLE task_routes (
  route_id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL,
  cost INTEGER,
  family TEXT NOT NULL,
  technique_class TEXT NOT NULL,
  route_text TEXT NOT NULL,
  exemplar TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'current',
  source_path TEXT NOT NULL,
  source_line INTEGER
);

CREATE TABLE tasks (
  task_id TEXT PRIMARY KEY,
  best_cost INTEGER,
  route_count INTEGER NOT NULL,
  families TEXT NOT NULL,
  technique_classes TEXT NOT NULL,
  best_route_text TEXT NOT NULL,
  source_path TEXT NOT NULL
);

CREATE TABLE ort_ops (
  op TEXT PRIMARY KEY,
  accepted_dtypes TEXT NOT NULL,
  rejected_dtypes TEXT NOT NULL,
  notes TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'current',
  confidence TEXT NOT NULL DEFAULT 'probed',
  source_path TEXT NOT NULL,
  source_line INTEGER
);

CREATE TABLE traps (
  trap_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  summary TEXT NOT NULL,
  trap_kind TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'current',
  source_path TEXT NOT NULL,
  source_line INTEGER
);

CREATE TABLE cost_floors (
  task_id TEXT PRIMARY KEY,
  note TEXT NOT NULL,
  cost INTEGER,
  status TEXT NOT NULL DEFAULT 'current',
  source_path TEXT NOT NULL,
  source_line INTEGER
);

CREATE TABLE facts (
  fact_id TEXT PRIMARY KEY,
  fact_type TEXT NOT NULL,
  claim TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'current',
  confidence TEXT NOT NULL DEFAULT 'documented',
  task_id TEXT,
  op TEXT,
  technique_section TEXT,
  source_path TEXT NOT NULL,
  source_line INTEGER,
  source_anchor TEXT NOT NULL DEFAULT '',
  mutable INTEGER NOT NULL DEFAULT 0,
  reverify_after TEXT,
  superseded_by TEXT
);

CREATE TABLE search_index (
  rowid INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  ref_id TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  source_path TEXT NOT NULL,
  source_line INTEGER
);

CREATE VIRTUAL TABLE search_fts USING fts5(
  kind,
  ref_id,
  title,
  body,
  source_path,
  content='search_index',
  content_rowid='rowid'
);

CREATE TRIGGER search_index_ai AFTER INSERT ON search_index BEGIN
  INSERT INTO search_fts(rowid, kind, ref_id, title, body, source_path)
  VALUES (new.rowid, new.kind, new.ref_id, new.title, new.body, new.source_path);
END;

CREATE TRIGGER search_index_ad AFTER DELETE ON search_index BEGIN
  INSERT INTO search_fts(search_fts, rowid, kind, ref_id, title, body, source_path)
  VALUES ('delete', old.rowid, old.kind, old.ref_id, old.title, old.body, old.source_path);
END;

CREATE TRIGGER search_index_au AFTER UPDATE ON search_index BEGIN
  INSERT INTO search_fts(search_fts, rowid, kind, ref_id, title, body, source_path)
  VALUES ('delete', old.rowid, old.kind, old.ref_id, old.title, old.body, old.source_path);
  INSERT INTO search_fts(rowid, kind, ref_id, title, body, source_path)
  VALUES (new.rowid, new.kind, new.ref_id, new.title, new.body, new.source_path);
END;

CREATE INDEX idx_task_routes_task ON task_routes(task_id);
CREATE INDEX idx_task_routes_cost ON task_routes(cost);
CREATE INDEX idx_facts_task ON facts(task_id);
CREATE INDEX idx_facts_op ON facts(op);
CREATE INDEX idx_facts_type_status ON facts(fact_type, status);
CREATE INDEX idx_cost_floors_cost ON cost_floors(cost);
