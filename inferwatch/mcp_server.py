"""MCP server exposing the inferwatch database over stdio.

Read-only by construction: the SQLite connection is opened with mode=ro and
PRAGMA query_only, so nothing reachable from here can modify the collector's
data even if a tool were misused.

Every tool answers through `inferwatch.metrics`, the same module the dashboard
uses, so a number reported here always matches the number on screen.  Results
carry a `notes` field describing provenance (exact rows vs. stored histograms)
because that distinction changes how the numbers should be read.

Run:  python -m inferwatch.mcp_server [--db PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import metrics, vllm_metrics
from .store import Store

from .main import default_db, env

srv = MCPServer(
    name="inferwatch",
    title="Ollama monitoring",
    instructions=(
        "Metrics for locally served LLMs. Two engines, deliberately different "
        "tool sets, because they expose different things:\n\n"
        "OLLAMA (tools without a prefix) is reconstructed from its debug log, so "
        "PER-REQUEST detail exists: time-to-first-token, token counts, client "
        "address, HTTP status, queue wait, prompt-cache reuse, model load times. "
        "Percentiles are exact inside the raw-retention window. Its prompt-cache "
        "occupancy and eviction pressure come from cache_status(), sampled from "
        "the log on ollama's own schedule rather than on a fixed interval.\n\n"
        "vLLM (tools prefixed vllm_) comes from its Prometheus /metrics endpoint, "
        "which is PRE-AGGREGATED and carries no request identity. There are no "
        "per-request rows, no client addresses and no exact percentiles for vLLM; "
        "its percentile figures are the upper bound of vLLM's own histogram "
        "bucket, while the means are exact -- prefer the mean. In exchange it "
        "reports evenly sampled gauges ollama does not: preemptions, batch "
        "occupancy and speculative-decoding acceptance.\n\n"
        "Windows are relative durations like '15m', '6h', '24h', '7d', '30d'. "
        "Every result carries `exact` and/or `notes` saying how it was derived.\n\n"
        "Ollama's per-request and cache lines come from its runner at high log "
        "verbosity, which OLLAMA_DEBUG=1 guarantees; call health() to check. Use "
        "list_sources() to see which engines are configured."
    ),
    version="1.0.0",
)

_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        raise RuntimeError("store not initialised")
    return _store


def _win(window: str) -> tuple[float, float]:
    return metrics.bounds(window)


def _note(exact: bool) -> str:
    return ("Percentiles are exact (computed from raw request rows)." if exact else
            "Range exceeds raw retention: percentiles come from stored histograms "
            "and are the UPPER BOUND of the containing bucket, not interpolated.")


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

@srv.tool(
    title="Get metrics summary",
    description="Headline metrics for a time window: request rate, TTFT "
                "percentiles, token throughput in and out, decode speed, "
                "prompt-cache hit rate, queue wait, and error rate.",
)
def get_summary(window: str = "1h", model: str | None = None) -> dict:
    """Summarise Ollama activity over a window.

    Args:
        window: Relative duration, e.g. '15m', '1h', '6h', '24h', '7d', '30d'.
        model: Optional model name (e.g. 'llama3.2:3b') to filter to.
    """
    start, end = _win(window)
    s = metrics.summary(store(), start, end, model)
    s["notes"] = _note(s["exact"])
    s["window"] = window
    if s.get("queue_ms_mean") and s.get("requests"):
        s["notes"] += (" queue_ms is wall latency minus runner time: time a request "
                       "spent waiting rather than generating.")
    return s


@srv.tool(
    title="Get metric timeseries",
    description="Bucketed timeseries for charting or trend analysis: request "
                "rate, error rate, TTFT p50/p90/p99, input and output tokens/sec, "
                "decode tokens/sec, mean queue wait, and prompt-cache hit rate.",
)
def get_timeseries(window: str = "1h", step_seconds: int | None = None,
                   model: str | None = None, metric: str | None = None) -> dict:
    """Return time-bucketed series.

    Args:
        window: Relative duration, e.g. '1h', '24h', '7d'.
        step_seconds: Bucket width; omitted picks a width giving ~60-240 points.
        model: Optional model name filter.
        metric: Optional single series name to return instead of all. One of
            req_per_s, err_per_s, ttft_p50, ttft_p90, ttft_p99, out_tps,
            in_tps, decode_tps, queue_ms, cache_hit_rate.
    """
    start, end = _win(window)
    ts = metrics.timeseries(store(), start, end, step_seconds, model)
    if metric:
        if metric not in ts["series"]:
            raise ValueError(f"unknown metric {metric!r}; available: "
                             f"{', '.join(sorted(ts['series']))}")
        ts["series"] = {metric: ts["series"][metric]}
    ts["notes"] = _note(ts["exact"]) + (
        " Null entries are buckets with no traffic -- gaps, not zeroes.")
    ts["window"] = window
    return ts


@srv.tool(
    title="Compare models",
    description="Per-model breakdown for a window: request counts, errors, "
                "mean and max TTFT, mean queue wait, decode speed, and token volumes.",
)
def compare_models(window: str = "24h") -> dict:
    """Compare every model that served traffic in the window.

    Args:
        window: Relative duration, e.g. '1h', '24h', '7d'.
    """
    start, end = _win(window)
    rows = metrics.by_model(store(), start, end)
    return {"window": window, "models": rows, "count": len(rows),
            "notes": "decode_tps is output tokens divided by time actually spent "
                     "decoding, so it measures model speed and excludes queueing."}


@srv.tool(
    title="List models",
    description="Models seen in the metrics history and models currently "
                "resident in VRAM, with their VRAM footprint and expiry.",
)
def list_models(days: float = 30.0) -> dict:
    """List known and currently loaded models.

    Args:
        days: How far back to look for models that served traffic.
    """
    st = store()
    return {"seen": metrics.model_names(st, days), "resident": metrics.loaded_models(st)}


@srv.tool(
    title="Get recent errors",
    description="Failed requests (HTTP status >= 400) with endpoint, model, "
                "client IP, status and latency.",
)
def recent_errors(window: str = "24h", limit: int = 50) -> dict:
    """List recent failed requests.

    Args:
        window: Relative duration to search.
        limit: Maximum rows to return.
    """
    start, end = _win(window)
    rows = metrics.recent_errors(store(), start, end, min(limit, 500))
    return {"window": window, "count": len(rows), "errors": _humanise(rows),
            "statuses": metrics.status_breakdown(store(), start, end)}


@srv.tool(
    title="Get slowest requests",
    description="Worst requests in a window ranked by a chosen timing: TTFT, "
                "total latency, queue wait, runner time, or decode time. Useful "
                "for telling 'the model is slow' apart from 'the request waited'.",
)
def slowest_requests(window: str = "1h", by: str = "ttft_ms", limit: int = 20) -> dict:
    """Rank requests by a timing column.

    Args:
        window: Relative duration to search.
        by: One of ttft_ms, latency_ms, queue_ms, total_ms, decode_ms.
        limit: Maximum rows to return.
    """
    start, end = _win(window)
    rows = metrics.slowest(store(), start, end, by, min(limit, 200))
    return {"window": window, "sorted_by": by, "count": len(rows),
            "requests": _humanise(rows),
            "notes": "queue_ms = latency_ms - total_ms (waiting, not working). "
                     "ttft_ms is llama.cpp's prompt eval time."}


@srv.tool(
    title="Get recent requests",
    description="Most recent individual inference requests with full per-request "
                "metrics: TTFT, queue wait, latency, token counts, cache reuse, "
                "decode rate and attribution confidence.",
)
def recent_requests(window: str = "1h", limit: int = 50,
                    include_health_checks: bool = False) -> dict:
    """List recent requests.

    Args:
        window: Relative duration to search.
        limit: Maximum rows to return.
        include_health_checks: Include HEAD / and /api/ps polling traffic,
            which is normally excluded because it dominates request counts.
    """
    start, end = _win(window)
    rows = metrics.recent_requests(store(), start, end, min(limit, 500),
                                   include_health_checks)
    return {"window": window, "count": len(rows), "requests": _humanise(rows),
            "notes": "attribution: 'exact' = timings definitely belong to this "
                     "request; 'ambiguous' = two requests finished together and "
                     "the pairing is a best guess; 'none' = failed before "
                     "reaching the runner; 'orphan' = timings with no access line."}


@srv.tool(
    title="Get runner events",
    description="Model load and eviction timeline plus ollama warnings and "
                "errors: cold-load durations, keep-alive expiries, context "
                "truncations, GPU discovery failures.",
)
def get_events(window: str = "24h", kind: str | None = None, limit: int = 50) -> dict:
    """List model lifecycle and problem events.

    Args:
        window: Relative duration to search.
        kind: Optional filter -- model_loaded, unload, idle_timer, truncation, problem.
        limit: Maximum rows to return.
    """
    start, end = _win(window)
    rows = metrics.events(store(), start, end, kind, min(limit, 500))
    for r in rows:
        r["when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
        if r.get("detail_json"):
            try:
                r["detail"] = json.loads(r.pop("detail_json"))
            except ValueError:
                r.pop("detail_json", None)
        else:
            r.pop("detail_json", None)
    return {"window": window, "count": len(rows), "events": rows}


@srv.tool(
    title="Get GPU status",
    description="GPU utilisation, VRAM use, temperature and power over a window, "
                "per device, plus the most recent sample.",
)
def gpu_status(window: str = "1h", step_seconds: int | None = None) -> dict:
    """Return GPU timeseries and the latest reading.

    Args:
        window: Relative duration to search.
        step_seconds: Bucket width in seconds.
    """
    start, end = _win(window)
    g = metrics.gpu_series(store(), start, end, step_seconds)
    latest = []
    for gpu in g["gpus"]:
        last = {"index": gpu["index"], "name": gpu["name"], "mem_total_mib": gpu["mem_total"]}
        for key in ("util_pct", "mem_used", "temp_c", "power_w"):
            vals = [v for v in gpu[key] if v is not None]
            last[key] = vals[-1] if vals else None
        latest.append(last)
    return {"window": window, "latest": latest, "series": g}


@srv.tool(
    title="Get per-client statistics",
    description="Per-client breakdown of Ollama traffic: which models each "
                "client address requested, how large its prompts are, how close "
                "it runs to its context limit, tokens, TTFT and error counts. "
                "Raw-window only -- the rollups carry no client column -- and "
                "requests whose model the log never named are counted as "
                "`unattributed` rather than dropped.",
)
def client_stats(window: str = "1h", limit: int = 25) -> dict:
    """Report who is calling, for which models, at what context size.

    Args:
        window: Relative duration to search.
        limit: Maximum clients to return, busiest first.
    """
    st = store()
    start, end = _win(window)
    rows = metrics.by_client(st, start, end, limit)
    covers_from = metrics.raw_coverage(st)
    complete = covers_from is not None and covers_from <= start
    out: dict[str, Any] = {
        "window": window,
        "clients": _humanise([dict(c) for c in rows]),
        "complete": complete,
        "covers_from": _when(covers_from),
        "notes": [
            "Client identity exists only on raw request rows, so this reaches "
            "back only as far as raw retention (7 days by default); the rollups "
            "aggregate by model and class and carry no client address.",
            "`prompt_tokens_max` is what the client sends; `n_ctx_max` is the "
            "capacity of the slot it ran in. `ctx_usage_max` is the peak ratio "
            "of the two computed per request, which is the client's closest "
            "approach to a context shift.",
        ],
    }
    if not complete:
        out["warning"] = (
            f"this window starts before the oldest surviving raw request row "
            f"({out['covers_from']}), so the earlier part of it has no "
            f"per-client detail at all -- raise retention.raw_days or ask for a "
            f"shorter window rather than reading this as the full picture.")
    stale = sum(c["unattributed"] for c in rows)
    total = sum(c["requests"] for c in rows)
    if stale:
        out["attribution_note"] = (
            f"{stale} of {total} requests have no model attributed. Ollama names "
            f"the model on a per-request scheduler line; when that line is "
            f"absent from the log the request is still counted, with its model "
            f"left null rather than guessed.")
    return out


@srv.tool(
    title="Get KV / prompt cache status",
    description="Ollama's prompt-cache occupancy over a window -- how full the "
                "saved-prompt pool is, how often it evicted, and what its "
                "maintenance pass cost -- plus how full the live KV context got. "
                "This is the closest ollama equivalent of vLLM's kv_cache_usage; "
                "unlike vLLM it is sampled from the log on ollama's own schedule, "
                "so a window in which no cache update ran legitimately has no "
                "samples -- which is not the same as an empty cache.",
)
def cache_status(window: str = "1h", step_seconds: int | None = None) -> dict:
    """Report prompt-cache occupancy, eviction pressure and live KV usage.

    Args:
        window: Relative duration to search.
        step_seconds: Bucket width in seconds for the series.
    """
    st = store()
    start, end = _win(window)
    summ = metrics.cache_summary(st, start, end)
    out: dict[str, Any] = {
        "window": window,
        "prompt_cache": summ,
        # A different cache: how close requests came to filling the slot's
        # context, which is the distance to a context shift or truncation.
        "live_kv": metrics.summary(st, start, end).get("ctx_usage"),
        "series": metrics.cache_series(st, start, end, step_seconds),
        "notes": [
            "prompt_cache figures are exact over the samples that exist, but "
            "ollama logs a sample only when it runs a cache update, so the "
            "sampling is uneven -- counters are totals, never rates.",
            "live_kv is derived from per-request rows and is therefore "
            "available only inside the raw retention window.",
        ],
    }
    if not summ["samples"]:
        out["warning"] = summ.get("note")
    elif summ["under_pressure"]:
        out["warning"] = (
            f"the prompt cache peaked at {summ['usage']['max']:.1%} of its "
            f"{summ['limit_mib']:.0f} MiB limit with {summ['evictions']} "
            f"eviction(s) in this window; evicted prompts have to be prefilled "
            f"again, which shows up as higher TTFT.")
    return out


@srv.tool(
    title="Check collector health",
    description="Collector status: journal lines read, rows stored, database "
                "size, whether OLLAMA_DEBUG timing lines are available, and "
                "the time of the last observed request.",
)
def health() -> dict:
    """Report whether collection is working and what data exists."""
    st = store()
    row = st.query("SELECT COUNT(*) n, MIN(ts) first, MAX(ts) last FROM requests")[0]
    counts = {}
    for table in ("requests", "events", "gpu_samples", "ps_samples",
                  "ollama_cache_samples", "rollup_1m", "rollup_1h"):
        counts[table] = st.query(f"SELECT COUNT(*) n FROM {table}")[0]["n"]
    out: dict[str, Any] = {
        "db_path": st.path,
        "db_bytes": os.path.getsize(st.path) if os.path.exists(st.path) else None,
        "rows": counts,
        "first_request": _when(row["first"]),
        "last_request": _when(row["last"]),
        "seconds_since_last_request": (time.time() - row["last"]) if row["last"] else None,
        "resident_models": metrics.loaded_models(st),
    }
    if not row["n"]:
        out["warning"] = ("No requests recorded yet. Either no traffic has hit "
                          "ollama since collection began, or the collector is "
                          "not running.")
    return out


@srv.tool(
    title="List vLLM instances",
    description="Configured vLLM instances, whether each is currently "
                "reachable, which model it serves, and when it was last scraped.",
)
def vllm_instances() -> dict:
    """List vLLM instances and their reachability."""
    inst = vllm_metrics.instances(store())
    return {"instances": inst, "count": len(inst),
            "notes": "An unreachable instance keeps its previously collected "
                     "history; `error` says why the last scrape failed."}


@srv.tool(
    title="Get vLLM summary",
    description="Headline metrics for one vLLM instance: request and token "
                "throughput, TTFT and end-to-end latency, queue wait, KV-cache "
                "occupancy, batch occupancy, preemptions, completion outcomes "
                "and speculative-decoding acceptance.",
)
def vllm_summary(source: str | None = None, window: str = "1h") -> dict:
    """Summarise a vLLM instance over a window.

    Args:
        source: Instance name; omit to use the only configured instance.
        window: Relative duration, e.g. '15m', '1h', '6h', '24h', '7d'.
    """
    src = _resolve_vllm_source(source)
    start, end = _win(window)
    out = vllm_metrics.summary(store(), src, start, end)
    out["window"] = window
    out["notes"] = (
        "vLLM publishes aggregates only, so there are no per-request rows. "
        "Latency entries carry an exact `mean` plus p50/p90/p99 that are the "
        "UPPER BOUND of vLLM's histogram bucket (its buckets step 10 -> 20 -> "
        "40s), so prefer `mean` and treat percentiles as 'at most'. "
        "`preemptions` above zero means requests were evicted under KV-cache "
        "pressure, which shows up as latency. "
        "Note that `output_tokens` and `requests` are NOT per-request aligned: "
        "vLLM advances its token counters as tokens stream but its request "
        "counter only on completion, so over a short window they describe "
        "overlapping but different sets of requests -- do not divide one by the "
        "other to get tokens per request."
    )
    return out


@srv.tool(
    title="Get vLLM timeseries",
    description="Bucketed vLLM series for trend analysis: request rate, prompt "
                "and generation tokens/sec, TTFT percentiles, queue and "
                "end-to-end latency, KV-cache usage, running and waiting "
                "request counts, and preemptions.",
)
def vllm_timeseries(source: str | None = None, window: str = "1h",
                    step_seconds: int | None = None, metric: str | None = None) -> dict:
    """Return time-bucketed vLLM series.

    Args:
        source: Instance name; omit to use the only configured instance.
        window: Relative duration, e.g. '1h', '24h', '7d'.
        step_seconds: Bucket width. Stored rows are per-minute, so anything
            below 60 is clamped to 60.
        metric: Optional single series name. One of req_per_s, out_tok_per_s,
            in_tok_per_s, ttft_p50, ttft_p90, ttft_p99, queue_p90, e2e_p90,
            kv_cache_usage, running, waiting, preemptions, tpot_p50.
    """
    src = _resolve_vllm_source(source)
    start, end = _win(window)
    ts = vllm_metrics.timeseries(store(), src, start, end, step_seconds)
    if metric:
        if metric not in ts["series"]:
            raise ValueError(f"unknown metric {metric!r}; available: "
                             f"{', '.join(sorted(ts['series']))}")
        ts["series"] = {metric: ts["series"][metric]}
    ts["window"] = window
    ts["notes"] = ("Nulls are buckets with no data -- gaps, not zeroes. "
                   "Percentile series are histogram bucket upper bounds.")
    return ts


@srv.tool(
    title="List configured sources",
    description="Which engines this collector is monitoring: Ollama sources "
                "with their log reader, and vLLM sources with their URL.",
)
def list_sources() -> dict:
    """List monitored engines and how each is collected."""
    rows = store().list_sources()
    return {"sources": rows, "count": len(rows),
            "notes": "Ollama metrics come from the log reader shown; vLLM from "
                     "the URL's /metrics endpoint. One Ollama source runs at a "
                     "time; vLLM instances may be many."}


@srv.tool(
    title="Get settings",
    description="Effective configuration with the origin of each value "
                "(default, saved in the database, environment variable, or "
                "command-line flag). Read-only.",
)
def get_settings() -> dict:
    """Report effective settings and where each value came from."""
    from .config import SPEC
    st = store()
    out = []
    for setting in SPEC:
        raw = st.get_config(setting.key)
        value = setting.default
        origin = "default"
        if raw is not None:
            try:
                value = setting.coerce(json.loads(raw))
                origin = "database"
            except (ValueError, TypeError):
                pass
        out.append({"key": setting.key, "label": setting.label, "value": value,
                    "origin": origin, "restart_required": setting.restart})
    return {"settings": out,
            "notes": "This is the database view; a value pinned by an "
                     "environment variable or command-line flag in the running "
                     "collector overrides it. The dashboard's Settings tab shows "
                     "the fully resolved value."}


def _resolve_vllm_source(source: str | None) -> str:
    inst = vllm_metrics.instances(store())
    if source:
        if any(i["source"] == source for i in inst):
            return source
        raise ValueError(f"no vLLM instance named {source!r}; "
                         f"known: {[i['source'] for i in inst] or 'none'}")
    if not inst:
        raise ValueError("no vLLM instances are configured; add one in the "
                         "dashboard's Settings tab")
    if len(inst) > 1:
        raise ValueError("several vLLM instances are configured; pass source="
                         f"{[i['source'] for i in inst]}")
    return inst[0]["source"]


_SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


@srv.tool(
    title="Run a read-only SQL query",
    description="Escape hatch for questions the other tools do not cover. "
                "Runs a SELECT against the metrics database. Tables: requests, "
                "events, gpu_samples, ps_samples, rollup_1m, rollup_1h.",
)
def run_sql(sql: str, limit: int = 200) -> dict:
    """Execute a read-only SELECT.

    The connection is opened read-only, so writes cannot succeed; statements
    that are not SELECT/WITH are rejected before execution.

    Args:
        sql: A single SELECT (or WITH ... SELECT) statement.
        limit: Maximum rows returned to the caller.
    """
    if not _SELECT_ONLY.match(sql or ""):
        raise ValueError("only SELECT / WITH queries are allowed")
    if ";" in sql.strip().rstrip(";"):
        raise ValueError("only one statement at a time")
    rows = store().query(sql)
    truncated = len(rows) > limit
    return {"columns": list(rows[0].keys()) if rows else [],
            "rows": [dict(r) for r in rows[:limit]],
            "row_count": len(rows[:limit]),
            "truncated": truncated}


@srv.tool(
    title="Describe the schema",
    description="Column names, types and semantics for every table in the "
                "metrics database. Read this before writing run_sql queries.",
)
def describe_schema() -> dict:
    """Return table and column definitions with units."""
    st = store()
    out: dict[str, Any] = {"tables": {}}
    for t in ("requests", "events", "gpu_samples", "ps_samples", "rollup_1m",
              "rollup_1h", "vllm_samples", "vllm_hist", "vllm_instances",
              "sources", "config"):
        cols = st.query(f"PRAGMA table_info({t})")
        out["tables"][t] = [{"name": c["name"], "type": c["type"]} for c in cols]
    out["semantics"] = {
        "ts": "epoch seconds; request completion time",
        "ttft_ms": "time to first token = llama.cpp prompt eval time",
        "queue_ms": "latency_ms - total_ms: waiting, not generating",
        "total_ms": "runner time for the request (prefill + decode)",
        "prompt_tokens": "prompt tokens actually evaluated (cache misses only)",
        "prompt_tokens_total": "full prompt length including cache hits",
        "cached_tokens": "prompt_tokens_total - prompt_tokens: prompt cache reuse",
        "output_tokens": "generated tokens",
        "decode_tps": "tokens/sec while decoding (model speed)",
        "class": "inference | embed | admin | health (health = HEAD / and /api/ps polling)",
        "attribution": "exact | ambiguous | none | orphan (join confidence)",
        "rollup_*.ttft_hist": "JSON array of counts over inferwatch.store.HIST_BOUNDS_MS",
        "vllm_samples.value": "gauge value, or the counter delta for that minute",
        "vllm_samples.rate": "per-second rate for counter rows",
        "vllm_hist.bounds": "JSON array of vLLM's own `le` bounds; the final "
                            "+Inf bucket is stored as null",
        "vllm_hist.counts": "per-minute bucket deltas, additive across rows",
        "vllm_hist.sum_value": "delta of the histogram's _sum, so mean is exact",
    }
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _when(ts: float | None) -> str | None:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else None


def _humanise(rows: list[dict]) -> list[dict]:
    """Add a readable timestamp; models read these better than epoch floats."""
    for r in rows:
        if r.get("ts"):
            r["when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
    return rows


def main(argv=None) -> int:
    global _store
    p = argparse.ArgumentParser(prog="inferwatch-mcp")
    p.add_argument("--db", default=env("DB") or default_db())
    p.add_argument("--transport", default="stdio",
                   choices=["stdio", "sse", "streamable-http"])
    args = p.parse_args(argv)
    if not os.path.exists(args.db):
        raise SystemExit(f"no metrics database at {args.db}\n"
                         f"start the collector first: python -m inferwatch.main serve")
    _store = Store(args.db, read_only=True)
    srv.run(args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
