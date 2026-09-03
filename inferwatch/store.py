"""SQLite storage: raw requests, samples, events, and re-aggregatable rollups.

Retention model (chosen deliberately):
  * raw `requests` rows are kept RAW_RETENTION_DAYS (default 7)
  * `rollup_1m` and `rollup_1h` are kept forever

Percentiles over long windows are the hard part of that model.  Storing a
bucket's p99 is a dead end -- percentiles of percentiles are not percentiles.
So each rollup row stores a fixed-bucket HISTOGRAM of latency and TTFT.
Histograms add, so a percentile over any range (an hour or a year) is computed
by summing buckets and walking to the target rank.  The result is exact to the
bucket width, and honest: `p_from_hist` reports the bucket's upper bound.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

SCHEMA_VERSION = 6

# Log-spaced upper bounds in ms.  Fine where local inference actually lives
# (0.1s-30s), coarse in the tail.  Last bucket is the overflow (inf).
HIST_BOUNDS_MS = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000,
                  3000, 5000, 7500, 10_000, 15_000, 20_000, 30_000, 60_000,
                  120_000, 300_000]
NBUCKETS = len(HIST_BOUNDS_MS) + 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- One row per HTTP request seen in the access log.  Token/timing columns are
-- NULL for non-inference requests and for inference requests that failed
-- before reaching the runner (there are no timings to attribute).
CREATE TABLE IF NOT EXISTS requests (
    id                  INTEGER PRIMARY KEY,
    ts                  REAL NOT NULL,   -- completion time, epoch seconds
    started_ts          REAL,            -- ts - latency
    model               TEXT,
    endpoint            TEXT NOT NULL,
    method              TEXT,
    class               TEXT NOT NULL,   -- inference|embed|admin|health
    status              INTEGER,
    client_ip           TEXT,
    latency_ms          REAL,            -- wall clock, from the access log
    ttft_ms             REAL,            -- prompt eval time (server-side TTFT)
    decode_ms           REAL,
    total_ms            REAL,            -- runner total
    queue_ms            REAL,            -- latency - total: scheduler overhead
    prompt_tokens       INTEGER,         -- uncached prompt tokens actually run
    prompt_tokens_total INTEGER,         -- full prompt incl. cache hits
    cached_tokens       INTEGER,
    output_tokens       INTEGER,
    prefill_tps         REAL,
    decode_tps          REAL,
    draft_accept        REAL,
    draft_mean_len      REAL,
    truncated           INTEGER,
    context_tokens      INTEGER,         -- tokens resident in the slot's KV at release
    n_ctx_slot          INTEGER,         -- that slot's KV capacity; the ratio is
                                         -- how full the live KV cache got
    slot_id             INTEGER,
    task_id             INTEGER,
    attribution         TEXT             -- exact|ambiguous|none
);
CREATE INDEX IF NOT EXISTS idx_req_ts    ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_req_class ON requests(class, ts);
CREATE INDEX IF NOT EXISTS idx_req_model ON requests(model, ts);

-- Periodic snapshots: GPU state and loaded models.
CREATE TABLE IF NOT EXISTS gpu_samples (
    ts        REAL NOT NULL,
    gpu_index INTEGER NOT NULL,
    name      TEXT,
    util_pct  REAL,
    mem_used  INTEGER,   -- MiB
    mem_total INTEGER,
    temp_c    REAL,
    power_w   REAL,
    PRIMARY KEY (ts, gpu_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS ps_samples (
    ts           REAL PRIMARY KEY,
    loaded_count INTEGER,
    models_json  TEXT,     -- /api/ps payload, trimmed
    inflight     INTEGER   -- requests in flight per the correlator
) WITHOUT ROWID;

-- ----------------------------------------------------------------------
-- Ollama's prompt cache, sampled from the log.
--
-- This is the pool of saved prompt states that lets a returning conversation
-- skip prefill, and `cache state` lines report its occupancy outright -- the
-- one gauge ollama publishes that resembles vLLM's kv_cache_usage_perc.
--
-- It gets its own table rather than columns on `ps_samples` because the two
-- have different clocks: ps_samples is polled on a fixed interval, while these
-- appear only when ollama runs a cache update.  Blending them would make one
-- series look like it had gaps and the other like it had duplicates.
--
-- The counter columns are DELTAS since the previous row, not running totals:
-- llama.cpp resets its own counters when a runner restarts, and a delta cannot
-- go negative across that the way a monotonic counter would.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ollama_cache_samples (
    ts               REAL PRIMARY KEY,
    model            TEXT,      -- most recently active model; the line names none
    prompts          INTEGER,   -- prompt states resident
    used_mib         REAL,
    limit_mib        REAL,
    usage            REAL,      -- used_mib / limit_mib, NULL when unbounded
    token_limit      INTEGER,
    est_tokens       INTEGER,
    ckpt_used        INTEGER,   -- checkpoint high-water since the last sample
    ckpt_total       INTEGER,   -- checkpoint cap ("2 of 32")
    update_ms        REAL,      -- cost of the maintenance pass, inside TTFT
    evictions        INTEGER,
    evict_crowded    INTEGER,   -- capacity pressure
    evict_invalidated INTEGER,  -- cached positions no longer applied
    restores         INTEGER,
    ckpt_created     INTEGER,
    saves            INTEGER
) WITHOUT ROWID;

-- Model load/unload/warning/error timeline.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,   -- model_loaded|unload|problem|truncation|...
    level       TEXT,
    model       TEXT,
    source      TEXT,
    msg         TEXT,
    duration_ms REAL,
    detail_json TEXT,
    dedupe_key  TEXT
);
CREATE INDEX IF NOT EXISTS idx_ev_ts   ON events(ts);
CREATE INDEX IF NOT EXISTS idx_ev_kind ON events(kind, ts);

-- Rollups.  bucket = unix time floored to the period.  Histograms are JSON
-- arrays of NBUCKETS ints so they can be summed across rows.
CREATE TABLE IF NOT EXISTS rollup_1m (
    bucket        INTEGER NOT NULL,
    model         TEXT    NOT NULL,   -- '' when unknown
    class         TEXT    NOT NULL,
    req_count     INTEGER NOT NULL,
    err_count     INTEGER NOT NULL,
    in_tokens     INTEGER NOT NULL,
    out_tokens    INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    decode_ms_sum REAL    NOT NULL,   -- for tokens/sec = out_tokens/decode_s
    ttft_sum      REAL    NOT NULL,
    ttft_n        INTEGER NOT NULL,
    ttft_max      REAL,
    lat_sum       REAL    NOT NULL,
    lat_n         INTEGER NOT NULL,
    lat_max       REAL,
    queue_sum     REAL    NOT NULL,
    queue_n       INTEGER NOT NULL,
    ttft_hist     TEXT    NOT NULL,
    lat_hist      TEXT    NOT NULL,
    PRIMARY KEY (bucket, model, class)
) WITHOUT ROWID;

-- UI-editable configuration.  Outranked by environment and command line.
CREATE TABLE IF NOT EXISTS config (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,      -- JSON scalar
    updated_at REAL NOT NULL
);

-- Monitored engines.  A variable-length list the settings screen edits, so it
-- gets a table rather than a config key.
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,      -- ollama | vllm
    name        TEXT NOT NULL UNIQUE,
    enabled     INTEGER NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

-- ----------------------------------------------------------------------
-- vLLM lives in its own tables, deliberately.
--
-- vLLM's /metrics exposes pre-aggregated counters, gauges and histograms and
-- carries NO per-request identity, so there is nothing to put in `requests`.
-- Its histogram bucket bounds are also not a refinement of the ones used for
-- ollama, so re-bucketing into the shared rollups would mean interpolating --
-- inventing numbers.  Instead each scrape stores vLLM's own bounds alongside
-- its own counts, and percentiles are computed against those.
--
-- `metric` is a free-text column so a vLLM upgrade that adds metrics needs no
-- migration.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vllm_samples (
    ts     REAL NOT NULL,
    source TEXT NOT NULL,
    metric TEXT NOT NULL,
    labels TEXT NOT NULL DEFAULT '',   -- canonical "k=v,k=v", '' when none
    model  TEXT,
    value  REAL,                       -- gauge value, or counter delta
    rate   REAL,                       -- per-second rate for counters
    PRIMARY KEY (ts, source, metric, labels)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_vllm_s_metric ON vllm_samples(source, metric, ts);

CREATE TABLE IF NOT EXISTS vllm_hist (
    ts        REAL NOT NULL,
    source    TEXT NOT NULL,
    metric    TEXT NOT NULL,
    model     TEXT,
    bounds    TEXT NOT NULL,   -- JSON array of le values, vLLM's own
    counts    TEXT NOT NULL,   -- JSON array of per-interval bucket deltas
    observations INTEGER,      -- delta of _count
    sum_value REAL,            -- delta of _sum
    PRIMARY KEY (ts, source, metric)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_vllm_h_metric ON vllm_hist(source, metric, ts);

-- One row per instance: what it is, and when its engine last started (used to
-- detect a restart, after which counters begin again from zero).
CREATE TABLE IF NOT EXISTS vllm_instances (
    source       TEXT PRIMARY KEY,
    last_seen    REAL,
    engine_start REAL,
    model        TEXT,
    reachable    INTEGER,
    error        TEXT,
    info_json    TEXT,
    -- GPU indices this instance's processes actually hold, as a JSON array.
    -- NULL means attribution was not possible (remote instance, no cgroup
    -- visibility, or nvidia-smi cannot be asked) -- which is different from
    -- "[]", meaning asked and holding none.
    gpu_indices  TEXT,
    -- How that was resolved ("cgroup:vllm-qwen38.service", "port:8000", ...)
    -- and when.  Both exist because a stale attribution used to be
    -- indistinguishable from a current one: a failed resolution was COALESCEd
    -- over the last good answer, so a topology change never propagated and the
    -- UI kept presenting weeks-old indices as fact.
    gpu_source   TEXT,
    gpu_ts       REAL
);

CREATE TABLE IF NOT EXISTS rollup_1h (
    bucket        INTEGER NOT NULL,
    model         TEXT    NOT NULL,
    class         TEXT    NOT NULL,
    req_count     INTEGER NOT NULL,
    err_count     INTEGER NOT NULL,
    in_tokens     INTEGER NOT NULL,
    out_tokens    INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    decode_ms_sum REAL    NOT NULL,
    ttft_sum      REAL    NOT NULL,
    ttft_n        INTEGER NOT NULL,
    ttft_max      REAL,
    lat_sum       REAL    NOT NULL,
    lat_n         INTEGER NOT NULL,
    lat_max       REAL,
    queue_sum     REAL    NOT NULL,
    queue_n       INTEGER NOT NULL,
    ttft_hist     TEXT    NOT NULL,
    lat_hist      TEXT    NOT NULL,
    PRIMARY KEY (bucket, model, class)
) WITHOUT ROWID;
"""


def _f(v) -> str:
    """Render a value for a dedupe key; None and missing collapse to ''."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def request_key(r: dict) -> str:
    """Stable identity for a request row.

    Built from the journal timestamp (microsecond resolution) plus the fields
    that identify the exchange.  Two genuinely distinct requests would have to
    share a microsecond, an endpoint, a client, a status, a latency and a task
    id to collide.
    """
    return "|".join(_f(r.get(k)) for k in
                    ("ts", "endpoint", "method", "client_ip", "status",
                     "latency_ms", "task_id"))


def event_key(e: dict) -> str:
    return "|".join(_f(e.get(k)) for k in ("ts", "kind", "model", "msg", "duration_ms"))


def hist_index(ms: float) -> int:
    for i, b in enumerate(HIST_BOUNDS_MS):
        if ms <= b:
            return i
    return NBUCKETS - 1


def hist_add(acc: list[int], ms: float | None) -> None:
    if ms is not None:
        acc[hist_index(ms)] += 1


def p_from_hist(hist: list[int], q: float) -> float | None:
    """Percentile from summed histogram buckets.

    Returns the upper bound of the bucket containing the target rank -- an
    upper estimate, never an interpolated guess between bounds.  The final
    overflow bucket has no upper bound, so it reports the last finite bound
    as a floor (callers should render it as '>300s').
    """
    total = sum(hist)
    if total == 0:
        return None
    target = q * total
    cum = 0
    for i, c in enumerate(hist):
        cum += c
        if cum >= target:
            return float(HIST_BOUNDS_MS[i]) if i < len(HIST_BOUNDS_MS) else float(HIST_BOUNDS_MS[-1])
    return float(HIST_BOUNDS_MS[-1])


def sum_hists(rows: list[str]) -> list[int]:
    acc = [0] * NBUCKETS
    for raw in rows:
        try:
            h = json.loads(raw)
        except (TypeError, ValueError):
            continue
        for i, c in enumerate(h[:NBUCKETS]):
            acc[i] += c
    return acc


class Store:
    """Thread-safe-enough SQLite wrapper (one connection guarded by a lock).

    WAL mode lets the API and MCP server read while the collector writes.
    """

    def __init__(self, path: str, read_only: bool = False):
        self.path = path
        self.read_only = read_only
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if read_only:
            uri = f"file:{os.path.abspath(path)}?mode=ro"
            self.db = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5)
        else:
            self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        if not read_only:
            with self.lock:
                self.db.execute("PRAGMA journal_mode=WAL")
                self.db.execute("PRAGMA synchronous=NORMAL")
                self.db.executescript(DDL)
                self._migrate()
                self.db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),))
                self.db.commit()
        else:
            self.db.execute("PRAGMA query_only=ON")

    def _migrate(self) -> None:
        """Bring a pre-schema-2 database up to date, in place.

        Schema 2 adds dedupe_key + a unique index to requests and events.  Rows
        written before it have the column NULL, and SQLite counts NULLs as
        distinct, so they are backfilled first; any duplicates already present
        (from a replayed journal) are collapsed before the index is built,
        because CREATE UNIQUE INDEX would otherwise fail.
        """
        # schema 4: per-instance GPU attribution
        cols = {c["name"] for c in self.db.execute("PRAGMA table_info(vllm_instances)")}
        if cols and "gpu_indices" not in cols:
            self.db.execute("ALTER TABLE vllm_instances ADD COLUMN gpu_indices TEXT")

        # schema 6: how and when GPU attribution was resolved.
        cols = {c["name"] for c in self.db.execute("PRAGMA table_info(vllm_instances)")}
        if cols:
            for col, decl in (("gpu_source", "TEXT"), ("gpu_ts", "REAL")):
                if col not in cols:
                    self.db.execute(
                        f"ALTER TABLE vllm_instances ADD COLUMN {col} {decl}")

        # schema 5: the slot KV capacity a request ran against.  Rows written
        # before it keep NULL, so live-KV occupancy is simply unavailable for
        # them rather than being back-computed from a capacity we never saw.
        cols = {c["name"] for c in self.db.execute("PRAGMA table_info(requests)")}
        if cols and "n_ctx_slot" not in cols:
            self.db.execute("ALTER TABLE requests ADD COLUMN n_ctx_slot INTEGER")

        for table, keyfn, index in (("requests", request_key, "idx_req_dedupe"),
                                    ("events", event_key, "idx_ev_dedupe")):
            cols = {c["name"] for c in self.db.execute(f"PRAGMA table_info({table})")}
            if "dedupe_key" not in cols:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN dedupe_key TEXT")
            if self._has_index(index):
                # Already migrated.  The unique index makes both steps below
                # impossible to need, and they are the expensive ones: the
                # backfill scans every NULL row and the collapse is a full scan
                # plus a GROUP BY over the whole table.  Running them on every
                # startup cost that for nothing.
                continue
            self._backfill_dedupe(table, keyfn)
            # Collapse duplicates already present (from a replayed journal),
            # keeping the earliest row, or CREATE UNIQUE INDEX below would fail.
            self.db.execute(
                f"DELETE FROM {table} WHERE id NOT IN"
                f" (SELECT MIN(id) FROM {table} GROUP BY dedupe_key)")
        self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_req_dedupe ON requests(dedupe_key)")
        self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ev_dedupe ON events(dedupe_key)")

    def _has_index(self, name: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (name,)).fetchone()
        return row is not None

    def _backfill_dedupe(self, table: str, keyfn, batch: int = 5000) -> int:
        """Fill dedupe_key on pre-schema-2 rows, a bounded batch at a time.

        The key is computed in Python, so this cannot be one UPDATE statement.
        Reading the whole table with fetchall() could mean a multi-hundred-MB
        result and a long startup holding the write lock, so rows are taken in
        batches instead; each pass shrinks the candidate set because the rows it
        wrote no longer match.
        """
        done = 0
        while True:
            rows = self.db.execute(
                f"SELECT * FROM {table} WHERE dedupe_key IS NULL LIMIT ?",
                (batch,)).fetchall()
            if not rows:
                return done
            self.db.executemany(
                f"UPDATE {table} SET dedupe_key = ? WHERE id = ?",
                [(keyfn(dict(r)), r["id"]) for r in rows])
            done += len(rows)

    # -- config --------------------------------------------------------------

    def get_config(self, key: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_config(self, key: str, value: str) -> None:
        with self.lock:
            self.db.execute(
                "INSERT INTO config(key,value,updated_at) VALUES(?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " updated_at=excluded.updated_at", (key, value, time.time()))

    def delete_config(self, key: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM config WHERE key=?", (key,))

    def all_config(self) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT key, value FROM config").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # -- sources -------------------------------------------------------------

    def list_sources(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT id,kind,name,enabled,config_json,created_at,updated_at"
                " FROM sources ORDER BY kind, name").fetchall()
        out = []
        for r in rows:
            try:
                cfg = json.loads(r["config_json"])
            except ValueError:
                cfg = {}
            out.append({"id": r["id"], "kind": r["kind"], "name": r["name"],
                        "enabled": bool(r["enabled"]), "config": cfg,
                        "created_at": r["created_at"], "updated_at": r["updated_at"]})
        return out

    def add_source(self, kind: str, name: str, config: dict, enabled: bool = True) -> int:
        now = time.time()
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO sources(kind,name,enabled,config_json,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?)",
                (kind, name, 1 if enabled else 0, json.dumps(config), now, now))
            self.db.commit()
            return cur.lastrowid

    def update_source(self, source_id: int, name: str | None = None,
                      config: dict | None = None, enabled: bool | None = None) -> None:
        sets, params = [], []
        if name is not None:
            sets.append("name=?"); params.append(name)
        if config is not None:
            sets.append("config_json=?"); params.append(json.dumps(config))
        if enabled is not None:
            sets.append("enabled=?"); params.append(1 if enabled else 0)
        if not sets:
            return
        sets.append("updated_at=?"); params.append(time.time())
        params.append(source_id)
        with self.lock:
            self.db.execute(f"UPDATE sources SET {','.join(sets)} WHERE id=?", tuple(params))
            self.db.commit()

    def delete_source(self, source_id: int) -> None:
        with self.lock:
            self.db.execute("DELETE FROM sources WHERE id=?", (source_id,))
            self.db.commit()

    # -- vllm ----------------------------------------------------------------

    def insert_vllm_samples(self, rows: list[tuple]) -> None:
        """rows of (ts, source, metric, labels, model, value, rate)."""
        if not rows:
            return
        with self.lock:
            self.db.executemany(
                "INSERT OR REPLACE INTO vllm_samples"
                " (ts,source,metric,labels,model,value,rate) VALUES (?,?,?,?,?,?,?)", rows)

    def insert_vllm_hist(self, rows: list[tuple]) -> None:
        """rows of (ts, source, metric, model, bounds, counts, observations, sum_value)."""
        if not rows:
            return
        with self.lock:
            self.db.executemany(
                "INSERT OR REPLACE INTO vllm_hist"
                " (ts,source,metric,model,bounds,counts,observations,sum_value)"
                " VALUES (?,?,?,?,?,?,?,?)", rows)

    def upsert_vllm_instance(self, source: str, gpu_known: bool = False,
                             **fields) -> None:
        """Record one scrape's view of an instance.

        `gpu_known` says whether this caller actually looked at the GPUs.  It
        matters because the two failure modes need opposite handling:

          * an unreachable scrape looked at nothing, so it must not wipe the
            attribution learned while the engine was up (gpu_known=False, the
            stored value is kept);
          * a reachable scrape that tried and could not attribute has produced a
            real answer -- "no longer knowable" -- and must be allowed to CLEAR
            the old one (gpu_known=True).

        Collapsing those was the bug: a topology change turned resolution into a
        permanent NULL, every NULL was COALESCEd away, and the UI went on
        displaying the indices from before the change.
        """
        cols = ("last_seen", "engine_start", "model", "reachable", "error",
                "info_json", "gpu_indices", "gpu_source", "gpu_ts")
        vals = {c: fields.get(c) for c in cols}
        # Only an authoritative look may overwrite with NULL.
        gpu_set = ("gpu_indices=excluded.gpu_indices,"
                   " gpu_source=excluded.gpu_source, gpu_ts=excluded.gpu_ts"
                   if gpu_known else
                   "gpu_indices=COALESCE(excluded.gpu_indices, gpu_indices),"
                   " gpu_source=COALESCE(excluded.gpu_source, gpu_source),"
                   " gpu_ts=COALESCE(excluded.gpu_ts, gpu_ts)")
        with self.lock:
            self.db.execute(
                "INSERT INTO vllm_instances(source,last_seen,engine_start,model,"
                "reachable,error,info_json,gpu_indices,gpu_source,gpu_ts)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(source) DO UPDATE SET last_seen=excluded.last_seen,"
                " engine_start=COALESCE(excluded.engine_start, engine_start),"
                " model=COALESCE(excluded.model, model),"
                " reachable=excluded.reachable, error=excluded.error,"
                " info_json=COALESCE(excluded.info_json, info_json),"
                f" {gpu_set}",
                (source, vals["last_seen"], vals["engine_start"], vals["model"],
                 vals["reachable"], vals["error"], vals["info_json"],
                 vals["gpu_indices"], vals["gpu_source"], vals["gpu_ts"]))

    # -- meta ----------------------------------------------------------------

    def get_meta(self, key: str, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))

    # -- writes --------------------------------------------------------------

    def insert_request(self, r: dict) -> None:
        cols = ("ts", "started_ts", "model", "endpoint", "method", "class", "status",
                "client_ip", "latency_ms", "ttft_ms", "decode_ms", "total_ms", "queue_ms",
                "prompt_tokens", "prompt_tokens_total", "cached_tokens", "output_tokens",
                "prefill_tps", "decode_tps", "draft_accept", "draft_mean_len", "truncated",
                "context_tokens", "n_ctx_slot", "slot_id", "task_id", "attribution",
                "dedupe_key")
        r = dict(r)
        r["dedupe_key"] = request_key(r)
        placeholders = ",".join("?" * len(cols))
        quoted = ",".join(f'"{c}"' for c in cols)
        with self.lock:
            # OR IGNORE: re-reading the same journal lines is a no-op.
            self.db.execute(
                f"INSERT OR IGNORE INTO requests ({quoted}) VALUES ({placeholders})",
                tuple(r.get(c) for c in cols))

    def insert_event(self, e: dict) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR IGNORE INTO events"
                " (ts,kind,level,model,source,msg,duration_ms,detail_json,dedupe_key)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (e.get("ts"), e.get("kind"), e.get("level"), e.get("model"), e.get("source"),
                 e.get("msg"), e.get("duration_ms"),
                 json.dumps(e["detail"]) if e.get("detail") else None,
                 event_key(e)))

    CACHE_COLS = ("ts", "model", "prompts", "used_mib", "limit_mib", "usage",
                  "token_limit", "est_tokens", "ckpt_used", "ckpt_total", "update_ms",
                  "evictions", "evict_crowded", "evict_invalidated", "restores",
                  "ckpt_created", "saves")

    def insert_cache_sample(self, row: dict) -> None:
        """One prompt-cache gauge sample.

        REPLACE rather than IGNORE on the timestamp: re-reading the same log
        lines must be a no-op, and a second read produces an identical row.
        """
        cols = ",".join(self.CACHE_COLS)
        placeholders = ",".join("?" * len(self.CACHE_COLS))
        with self.lock:
            self.db.execute(
                f"INSERT OR REPLACE INTO ollama_cache_samples ({cols})"
                f" VALUES ({placeholders})",
                tuple(row.get(c) for c in self.CACHE_COLS))

    def insert_gpu_samples(self, ts: float, gpus: list[dict]) -> None:
        with self.lock:
            self.db.executemany(
                "INSERT OR REPLACE INTO gpu_samples"
                " (ts,gpu_index,name,util_pct,mem_used,mem_total,temp_c,power_w)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [(ts, g["index"], g.get("name"), g.get("util_pct"), g.get("mem_used"),
                  g.get("mem_total"), g.get("temp_c"), g.get("power_w")) for g in gpus])

    def insert_ps_sample(self, ts: float, loaded_count: int, models: list, inflight: int) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO ps_samples (ts,loaded_count,models_json,inflight)"
                " VALUES (?,?,?,?)", (ts, loaded_count, json.dumps(models), inflight))

    def commit(self) -> None:
        with self.lock:
            self.db.commit()

    def commit_nowait(self) -> bool:
        """Commit only if the lock is free.  Safe to call from a signal handler.

        A signal handler runs on the main thread between bytecodes, so blocking
        on a lock a worker thread happens to hold could deadlock.  Skipping is
        safe here: writes are committed every second anyway, and inserts are
        idempotent, so at worst a restart re-reads a second of journal and
        discards the duplicates.
        """
        if not self.lock.acquire(blocking=False):
            return False
        try:
            self.db.commit()
            return True
        finally:
            self.lock.release()

    # -- rollups -------------------------------------------------------------

    def rebuild_rollups(self, since: float, until: float) -> int:
        """(Re)build 1m and 1h rollups for [since, until) from raw requests.

        Idempotent: a bucket is fully recomputed, so re-running over a window
        that was already rolled up changes nothing.  Cheap enough to re-run the
        trailing few minutes on every tick, which is how late-arriving rows
        (the correlator can lag a request by seconds) get folded in.
        """
        written = 0
        for table, period in (("rollup_1m", 60), ("rollup_1h", 3600)):
            b0 = int(since // period) * period
            b1 = int(until // period) * period + period
            with self.lock:
                rows = self.db.execute(
                    "SELECT * FROM requests WHERE ts >= ? AND ts < ?", (b0, b1)).fetchall()
            agg: dict[tuple, dict] = {}
            for r in rows:
                key = (int(r["ts"] // period) * period, r["model"] or "", r["class"])
                a = agg.get(key)
                if a is None:
                    a = agg[key] = {
                        "req": 0, "err": 0, "in": 0, "out": 0, "cached": 0, "dec_ms": 0.0,
                        "ttft_sum": 0.0, "ttft_n": 0, "ttft_max": None,
                        "lat_sum": 0.0, "lat_n": 0, "lat_max": None,
                        "q_sum": 0.0, "q_n": 0,
                        "ttft_h": [0] * NBUCKETS, "lat_h": [0] * NBUCKETS,
                    }
                a["req"] += 1
                if r["status"] and r["status"] >= 400:
                    a["err"] += 1
                a["in"] += r["prompt_tokens"] or 0
                a["out"] += r["output_tokens"] or 0
                a["cached"] += r["cached_tokens"] or 0
                a["dec_ms"] += r["decode_ms"] or 0.0
                if r["ttft_ms"] is not None:
                    a["ttft_sum"] += r["ttft_ms"]; a["ttft_n"] += 1
                    a["ttft_max"] = max(a["ttft_max"] or 0, r["ttft_ms"])
                    hist_add(a["ttft_h"], r["ttft_ms"])
                if r["latency_ms"] is not None:
                    a["lat_sum"] += r["latency_ms"]; a["lat_n"] += 1
                    a["lat_max"] = max(a["lat_max"] or 0, r["latency_ms"])
                    hist_add(a["lat_h"], r["latency_ms"])
                if r["queue_ms"] is not None:
                    a["q_sum"] += r["queue_ms"]; a["q_n"] += 1

            with self.lock:
                self.db.execute(f"DELETE FROM {table} WHERE bucket >= ? AND bucket < ?", (b0, b1))
                self.db.executemany(
                    f"INSERT INTO {table} (bucket,model,class,req_count,err_count,in_tokens,"
                    "out_tokens,cached_tokens,decode_ms_sum,ttft_sum,ttft_n,ttft_max,lat_sum,"
                    "lat_n,lat_max,queue_sum,queue_n,ttft_hist,lat_hist)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(k[0], k[1], k[2], a["req"], a["err"], a["in"], a["out"], a["cached"],
                      a["dec_ms"], a["ttft_sum"], a["ttft_n"], a["ttft_max"], a["lat_sum"],
                      a["lat_n"], a["lat_max"], a["q_sum"], a["q_n"],
                      json.dumps(a["ttft_h"]), json.dumps(a["lat_h"]))
                     for k, a in agg.items()])
                self.db.commit()
            written += len(agg)
        return written

    def prune(self, raw_retention_days: float, sample_retention_days: float = 30.0) -> dict:
        """Drop raw rows past retention.  Rollups are never pruned."""
        now = time.time()
        raw_cut = now - raw_retention_days * 86400
        samp_cut = now - sample_retention_days * 86400
        with self.lock:
            n_req = self.db.execute("DELETE FROM requests WHERE ts < ?", (raw_cut,)).rowcount
            n_gpu = self.db.execute("DELETE FROM gpu_samples WHERE ts < ?", (samp_cut,)).rowcount
            n_ps = self.db.execute("DELETE FROM ps_samples WHERE ts < ?", (samp_cut,)).rowcount
            n_cache = self.db.execute("DELETE FROM ollama_cache_samples WHERE ts < ?",
                                      (samp_cut,)).rowcount
            n_ev = self.db.execute("DELETE FROM events WHERE ts < ?", (samp_cut,)).rowcount
            n_vs = self.db.execute("DELETE FROM vllm_samples WHERE ts < ?", (samp_cut,)).rowcount
            n_vh = self.db.execute("DELETE FROM vllm_hist WHERE ts < ?", (samp_cut,)).rowcount
            self.db.commit()
        return {"requests": n_req, "gpu_samples": n_gpu, "ps_samples": n_ps,
                "ollama_cache_samples": n_cache, "events": n_ev,
                "vllm_samples": n_vs, "vllm_hist": n_vh}

    # -- reads ---------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, params).fetchall()
