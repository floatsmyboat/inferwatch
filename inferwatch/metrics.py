"""Query layer shared by the HTTP API and the MCP server.

Both front ends call these functions so a number quoted by the MCP server is
by construction the same number on the dashboard.

Source selection: raw `requests` rows give exact percentiles but only exist for
the retention window (7d default); beyond that we read the rollups' histograms.
Every response carries `exact: true|false` so a caller can tell which it got,
and percentile values from histograms are bucket upper bounds, never
interpolations.
"""

from __future__ import annotations

import re
import time

from .store import NBUCKETS, p_from_hist, sum_hists

# Ranges shorter than this read raw rows (exact); longer ranges read rollups.
RAW_WINDOW_S = 6 * 3600
INFERENCE_CLASSES = ("inference", "embed")

_WINDOW_RE = re.compile(r"^(\d+(?:\.\d+)?)([smhdw])$")
_WINDOW_MULT = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_window(window: str) -> float:
    """'15m' -> 900.0.  Raises ValueError on anything else."""
    m = _WINDOW_RE.match((window or "").strip().lower())
    if not m:
        raise ValueError(f"bad window {window!r}; use forms like 5m, 6h, 7d, 2w")
    return float(m.group(1)) * _WINDOW_MULT[m.group(2)]


def bounds(window: str = "1h", end: float | None = None) -> tuple[float, float]:
    e = end if end is not None else time.time()
    return e - parse_window(window), e


def pick_step(span_s: float) -> int:
    """Bucket width that keeps a chart near 60-240 points."""
    for step in (5, 15, 30, 60, 300, 900, 1800, 3600, 10800, 21600, 86400):
        if span_s / step <= 240:
            return step
    return 86400


def _quantiles(values: list[float], qs=(0.5, 0.9, 0.99)) -> dict:
    """Exact percentiles (nearest-rank) from raw values."""
    if not values:
        return {f"p{int(q * 100)}": None for q in qs}
    s = sorted(values)
    out = {}
    for q in qs:
        idx = min(len(s) - 1, max(0, int(round(q * len(s) + 0.5)) - 1))
        out[f"p{int(q * 100)}"] = s[idx]
    return out


def _model_clause(model: str | None, params: list) -> str:
    if not model:
        return ""
    params.append(model)
    return " AND model = ?"


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def summary(store, start: float, end: float, model: str | None = None) -> dict:
    """Headline numbers for a window: rates, tokens, latency, errors, queue."""
    span = max(1e-9, end - start)
    use_raw = span <= RAW_WINDOW_S

    if use_raw:
        params: list = [start, end]
        clause = _model_clause(model, params)
        rows = store.query(
            "SELECT class,status,ttft_ms,latency_ms,queue_ms,decode_ms,decode_tps,"
            "prompt_tokens,output_tokens,cached_tokens,context_tokens,n_ctx_slot,"
            "attribution"
            f" FROM requests WHERE ts >= ? AND ts < ?{clause}", tuple(params))
        inf = [r for r in rows if r["class"] in INFERENCE_CLASSES]
        errors = [r for r in inf if r["status"] and r["status"] >= 400]
        ttfts = [r["ttft_ms"] for r in inf if r["ttft_ms"] is not None]
        lats = [r["latency_ms"] for r in inf if r["latency_ms"] is not None]
        queues = [r["queue_ms"] for r in inf if r["queue_ms"] is not None]
        out_tok = sum(r["output_tokens"] or 0 for r in inf)
        in_tok = sum(r["prompt_tokens"] or 0 for r in inf)
        cached = sum(r["cached_tokens"] or 0 for r in inf)
        decode_s = sum(r["decode_ms"] or 0 for r in inf) / 1000.0
        ttft = _quantiles(ttfts)
        lat = _quantiles(lats)
        ambiguous = sum(1 for r in inf if r["attribution"] in ("ambiguous", "orphan", "none"))
        n_inf = len(inf)
        n_all = len(rows)
        n_err = len(errors)
        queue_mean = (sum(queues) / len(queues)) if queues else None
        queue_p90 = _quantiles(queues, (0.9,))["p90"] if queues else None
        ttft_max = max(ttfts) if ttfts else None
        # How full the LIVE KV cache got: tokens resident in the slot at
        # release, over that slot's capacity.  This is the distance to a
        # context shift, and it is a different cache from the prompt-cache
        # pool that cache_summary() reports on.
        ctx = [r["context_tokens"] / r["n_ctx_slot"] for r in inf
               if r["context_tokens"] and r["n_ctx_slot"]]
        ctx_usage = ({"mean": sum(ctx) / len(ctx), "max": max(ctx), "n": len(ctx)}
                     if ctx else None)
    else:
        table = "rollup_1h" if span > 7 * 86400 else "rollup_1m"
        params = [start, end]
        clause = _model_clause(model, params)
        rows = store.query(
            f"SELECT * FROM {table} WHERE bucket >= ? AND bucket < ?{clause}", tuple(params))
        inf = [r for r in rows if r["class"] in INFERENCE_CLASSES]
        n_inf = sum(r["req_count"] for r in inf)
        n_all = sum(r["req_count"] for r in rows)
        n_err = sum(r["err_count"] for r in inf)
        out_tok = sum(r["out_tokens"] for r in inf)
        in_tok = sum(r["in_tokens"] for r in inf)
        cached = sum(r["cached_tokens"] for r in inf)
        decode_s = sum(r["decode_ms_sum"] for r in inf) / 1000.0
        th = sum_hists([r["ttft_hist"] for r in inf])
        lh = sum_hists([r["lat_hist"] for r in inf])
        ttft = {f"p{int(q * 100)}": p_from_hist(th, q) for q in (0.5, 0.9, 0.99)}
        lat = {f"p{int(q * 100)}": p_from_hist(lh, q) for q in (0.5, 0.9, 0.99)}
        qn = sum(r["queue_n"] for r in inf)
        queue_mean = (sum(r["queue_sum"] for r in inf) / qn) if qn else None
        queue_p90 = None  # not reconstructable from rollups; raw window only
        ttft_max = max([r["ttft_max"] for r in inf if r["ttft_max"] is not None], default=None)
        ambiguous = None
        # The rollups aggregate per model and class, not per slot, so the
        # capacity a request ran against is not in them.  Null rather than a
        # figure derived from an assumed context size.
        ctx_usage = None

    ttft_n = ttft.get("p50") is not None
    return {
        "start": start, "end": end, "span_s": span, "exact": use_raw,
        "requests": n_inf,
        "requests_all": n_all,
        "req_per_s": n_inf / span,
        "errors": n_err,
        "error_rate": (n_err / n_inf) if n_inf else 0.0,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cached_tokens": cached,
        "cache_hit_rate": (cached / (cached + in_tok)) if (cached + in_tok) else None,
        # Two different, both-useful throughput numbers:
        #  - out_tok_per_s_wall: tokens/sec against wall clock (system output)
        #  - decode_tps_mean:    tokens/sec while actually decoding (model speed)
        "out_tok_per_s_wall": out_tok / span,
        "in_tok_per_s_wall": in_tok / span,
        "decode_tps_mean": (out_tok / decode_s) if decode_s > 0 else None,
        "ttft_ms": ttft if ttft_n else {"p50": None, "p90": None, "p99": None},
        "ttft_max_ms": ttft_max,
        "latency_ms": lat,
        "queue_ms_mean": queue_mean,
        "queue_ms_p90": queue_p90,
        # Live KV occupancy; raw window only, see above.
        "ctx_usage": ctx_usage,
        "ambiguous_rows": ambiguous,
    }


# --------------------------------------------------------------------------
# timeseries
# --------------------------------------------------------------------------

def timeseries(store, start: float, end: float, step: int | None = None,
               model: str | None = None) -> dict:
    """Bucketed series for the charts.

    Returns parallel arrays (one entry per bucket) so the client does no
    grouping.  Buckets with no traffic are present with nulls, which keeps
    gaps honest instead of drawing a line across an idle hour.
    """
    span = max(1.0, end - start)
    step = step or pick_step(span)
    nb = int(span // step) + 1
    base = int(start // step) * step

    series = {k: [None] * nb for k in (
        "req_per_s", "err_per_s", "ttft_p50", "ttft_p90", "ttft_p99",
        "out_tps", "in_tps", "decode_tps", "queue_ms", "cache_hit_rate")}
    counts = [0] * nb

    if span <= RAW_WINDOW_S:
        params: list = [start, end]
        clause = _model_clause(model, params)
        rows = store.query(
            "SELECT ts,class,status,ttft_ms,queue_ms,decode_ms,prompt_tokens,"
            "output_tokens,cached_tokens"
            f" FROM requests WHERE ts >= ? AND ts < ?{clause}"
            " AND class IN ('inference','embed')", tuple(params))
        buckets: list[dict] = [None] * nb  # type: ignore[assignment]
        for r in rows:
            i = int((r["ts"] - base) // step)
            if not 0 <= i < nb:
                continue
            b = buckets[i]
            if b is None:
                b = buckets[i] = {"n": 0, "err": 0, "ttft": [], "q": [], "out": 0,
                                  "in": 0, "cached": 0, "dec_ms": 0.0}
            b["n"] += 1
            if r["status"] and r["status"] >= 400:
                b["err"] += 1
            if r["ttft_ms"] is not None:
                b["ttft"].append(r["ttft_ms"])
            if r["queue_ms"] is not None:
                b["q"].append(r["queue_ms"])
            b["out"] += r["output_tokens"] or 0
            b["in"] += r["prompt_tokens"] or 0
            b["cached"] += r["cached_tokens"] or 0
            b["dec_ms"] += r["decode_ms"] or 0.0
        for i, b in enumerate(buckets):
            if not b:
                continue
            counts[i] = b["n"]
            q = _quantiles(b["ttft"])
            series["req_per_s"][i] = b["n"] / step
            series["err_per_s"][i] = b["err"] / step
            series["ttft_p50"][i] = q["p50"]
            series["ttft_p90"][i] = q["p90"]
            series["ttft_p99"][i] = q["p99"]
            series["out_tps"][i] = b["out"] / step
            series["in_tps"][i] = b["in"] / step
            series["decode_tps"][i] = (b["out"] / (b["dec_ms"] / 1000.0)) if b["dec_ms"] > 0 else None
            series["queue_ms"][i] = (sum(b["q"]) / len(b["q"])) if b["q"] else None
            denom = b["cached"] + b["in"]
            series["cache_hit_rate"][i] = (b["cached"] / denom) if denom else None
        exact = True
    else:
        table = "rollup_1h" if span > 7 * 86400 else "rollup_1m"
        params = [start, end]
        clause = _model_clause(model, params)
        rows = store.query(
            f"SELECT * FROM {table} WHERE bucket >= ? AND bucket < ?{clause}"
            " AND class IN ('inference','embed')", tuple(params))
        agg: dict[int, dict] = {}
        for r in rows:
            i = int((r["bucket"] - base) // step)
            if not 0 <= i < nb:
                continue
            a = agg.setdefault(i, {"n": 0, "err": 0, "out": 0, "in": 0, "cached": 0,
                                   "dec_ms": 0.0, "q_sum": 0.0, "q_n": 0,
                                   "th": [0] * NBUCKETS})
            a["n"] += r["req_count"]; a["err"] += r["err_count"]
            a["out"] += r["out_tokens"]; a["in"] += r["in_tokens"]
            a["cached"] += r["cached_tokens"]; a["dec_ms"] += r["decode_ms_sum"]
            a["q_sum"] += r["queue_sum"]; a["q_n"] += r["queue_n"]
            for j, c in enumerate(_load_hist(r["ttft_hist"])):
                a["th"][j] += c
        for i, a in agg.items():
            counts[i] = a["n"]
            series["req_per_s"][i] = a["n"] / step
            series["err_per_s"][i] = a["err"] / step
            series["ttft_p50"][i] = p_from_hist(a["th"], 0.5)
            series["ttft_p90"][i] = p_from_hist(a["th"], 0.9)
            series["ttft_p99"][i] = p_from_hist(a["th"], 0.99)
            series["out_tps"][i] = a["out"] / step
            series["in_tps"][i] = a["in"] / step
            series["decode_tps"][i] = (a["out"] / (a["dec_ms"] / 1000.0)) if a["dec_ms"] > 0 else None
            series["queue_ms"][i] = (a["q_sum"] / a["q_n"]) if a["q_n"] else None
            denom = a["cached"] + a["in"]
            series["cache_hit_rate"][i] = (a["cached"] / denom) if denom else None
        exact = False

    return {"start": base, "step": step, "n": nb, "exact": exact,
            "t": [base + i * step for i in range(nb)],
            "counts": counts, "series": series}


def _load_hist(raw) -> list[int]:
    import json
    try:
        h = json.loads(raw)
    except (TypeError, ValueError):
        return [0] * NBUCKETS
    return (h + [0] * NBUCKETS)[:NBUCKETS]


# --------------------------------------------------------------------------
# breakdowns
# --------------------------------------------------------------------------

def by_model(store, start: float, end: float) -> list[dict]:
    span = max(1e-9, end - start)
    if span <= RAW_WINDOW_S:
        rows = store.query(
            "SELECT model, COUNT(*) n,"
            " SUM(CASE WHEN status>=400 THEN 1 ELSE 0 END) err,"
            " SUM(prompt_tokens) in_tok, SUM(output_tokens) out_tok,"
            " SUM(cached_tokens) cached, SUM(decode_ms) dec_ms,"
            " AVG(ttft_ms) ttft_mean, MAX(ttft_ms) ttft_max, AVG(queue_ms) q_mean,"
            " AVG(decode_tps) tps_mean"
            " FROM requests WHERE ts >= ? AND ts < ? AND class IN ('inference','embed')"
            " GROUP BY model ORDER BY n DESC", (start, end))
    else:
        table = "rollup_1h" if span > 7 * 86400 else "rollup_1m"
        rows = store.query(
            "SELECT model, SUM(req_count) n, SUM(err_count) err,"
            " SUM(in_tokens) in_tok, SUM(out_tokens) out_tok, SUM(cached_tokens) cached,"
            " SUM(decode_ms_sum) dec_ms,"
            " CASE WHEN SUM(ttft_n)>0 THEN SUM(ttft_sum)/SUM(ttft_n) END ttft_mean,"
            " MAX(ttft_max) ttft_max,"
            " CASE WHEN SUM(queue_n)>0 THEN SUM(queue_sum)/SUM(queue_n) END q_mean,"
            " NULL tps_mean"
            f" FROM {table} WHERE bucket >= ? AND bucket < ? AND class IN ('inference','embed')"
            " GROUP BY model ORDER BY n DESC", (start, end))
    out = []
    for r in rows:
        dec_s = (r["dec_ms"] or 0) / 1000.0
        out.append({
            "model": r["model"] or "(unattributed)",
            "requests": r["n"], "errors": r["err"] or 0,
            "input_tokens": r["in_tok"] or 0, "output_tokens": r["out_tok"] or 0,
            "cached_tokens": r["cached"] or 0,
            "decode_tps": (r["out_tok"] / dec_s) if dec_s > 0 and r["out_tok"] else r["tps_mean"],
            "ttft_ms_mean": r["ttft_mean"], "ttft_ms_max": r["ttft_max"],
            "queue_ms_mean": r["q_mean"],
        })
    return out


def by_endpoint(store, start: float, end: float) -> list[dict]:
    rows = store.query(
        "SELECT endpoint, class, COUNT(*) n,"
        " SUM(CASE WHEN status>=400 THEN 1 ELSE 0 END) err, AVG(latency_ms) lat_mean"
        " FROM requests WHERE ts >= ? AND ts < ? GROUP BY endpoint, class"
        " ORDER BY n DESC LIMIT 50", (start, end))
    return [{"endpoint": r["endpoint"], "class": r["class"], "requests": r["n"],
             "errors": r["err"] or 0, "latency_ms_mean": r["lat_mean"]} for r in rows]


def raw_coverage(store) -> float | None:
    """Timestamp of the oldest surviving raw request row, or None if there are none.

    The honest bound for anything that can only be answered from raw rows.  It
    is NOT the same as RAW_WINDOW_S: that constant only decides when a query
    switches to the rollups for speed, while raw rows survive for
    `retention.raw_days` (7 by default).  A 7-day client breakdown is therefore
    complete even though `summary()['exact']` is False for that span, and
    conflating the two reports a full answer as a partial one.
    """
    row = store.query("SELECT MIN(ts) t FROM requests")
    return row[0]["t"] if row and row[0]["t"] is not None else None


def by_client(store, start: float, end: float, limit: int = 25) -> list[dict]:
    """Per-client detail: who is calling, for which models, at what context size.

    Client identity exists ONLY on raw request rows -- the rollups aggregate by
    model and class and carry no client column at all -- so this can only see as
    far back as raw retention.  Callers should pair it with `raw_coverage()` to
    tell a complete answer from one whose window starts before the oldest
    surviving row.

    Two notions of "context size" are reported because they answer different
    questions.  `prompt_tokens` is what the client actually sends (the full
    prompt, cache hits included); `n_ctx_max` is the capacity of the slot it
    landed in.  The ratio of the two is how close that client runs to a context
    shift, and it is computed per row before aggregating -- taking the maximum
    of each separately would pair a peak prompt with an unrelated capacity.
    """
    params = (start, end, limit)
    rows = store.query(
        "SELECT client_ip, COUNT(*) n,"
        " SUM(CASE WHEN status>=400 THEN 1 ELSE 0 END) err,"
        " SUM(CASE WHEN model IS NULL THEN 1 ELSE 0 END) unattributed,"
        " COUNT(DISTINCT model) n_models,"
        " SUM(prompt_tokens) in_tok, SUM(cached_tokens) cached,"
        " SUM(output_tokens) out_tok, SUM(decode_ms) dec_ms,"
        " AVG(prompt_tokens_total) prompt_mean, MAX(prompt_tokens_total) prompt_max,"
        " MAX(context_tokens) ctx_max, MAX(n_ctx_slot) n_ctx_max,"
        " MAX(CASE WHEN n_ctx_slot > 0 THEN CAST(context_tokens AS REAL) / n_ctx_slot END)"
        "   ctx_usage_max,"
        " AVG(ttft_ms) ttft_mean, MAX(ttft_ms) ttft_max, AVG(latency_ms) lat_mean,"
        " MIN(ts) first_seen, MAX(ts) last_seen"
        " FROM requests WHERE ts >= ? AND ts < ? AND class IN ('inference','embed')"
        " GROUP BY client_ip ORDER BY n DESC LIMIT ?", params)
    out = []
    for r in rows:
        dec_s = (r["dec_ms"] or 0) / 1000.0
        out.append({
            "client_ip": r["client_ip"],
            "requests": r["n"], "errors": r["err"] or 0,
            # Requests whose model the log never named.  Surfaced rather than
            # hidden: on a stretch where ollama did not log its per-request
            # scheduler line, this is most of the traffic, and a models list
            # that quietly omitted them would misrepresent the client.
            "unattributed": r["unattributed"] or 0,
            "models": [],          # filled in below
            "input_tokens": r["in_tok"] or 0,
            "cached_tokens": r["cached"] or 0,
            "output_tokens": r["out_tok"] or 0,
            "decode_tps": (r["out_tok"] / dec_s) if dec_s > 0 and r["out_tok"] else None,
            "prompt_tokens_mean": r["prompt_mean"],
            "prompt_tokens_max": r["prompt_max"],
            "context_tokens_max": r["ctx_max"],
            "n_ctx_max": r["n_ctx_max"],
            "ctx_usage_max": r["ctx_usage_max"],
            "ttft_ms_mean": r["ttft_mean"], "ttft_ms_max": r["ttft_max"],
            "latency_ms_mean": r["lat_mean"],
            "first_seen": r["first_seen"], "last_seen": r["last_seen"],
        })
    if not out:
        return out

    # Model mix, in one pass rather than a query per client.
    by_ip = {c["client_ip"]: c for c in out}
    mix = store.query(
        "SELECT client_ip, model, COUNT(*) n, MAX(prompt_tokens_total) prompt_max,"
        " MAX(n_ctx_slot) n_ctx"
        " FROM requests WHERE ts >= ? AND ts < ? AND class IN ('inference','embed')"
        " AND model IS NOT NULL GROUP BY client_ip, model ORDER BY n DESC", (start, end))
    for r in mix:
        c = by_ip.get(r["client_ip"])
        if c is not None:
            c["models"].append({"model": r["model"], "requests": r["n"],
                                "prompt_tokens_max": r["prompt_max"],
                                "n_ctx": r["n_ctx"]})
    return out


def status_breakdown(store, start: float, end: float) -> list[dict]:
    rows = store.query(
        "SELECT status, class, COUNT(*) n FROM requests"
        " WHERE ts >= ? AND ts < ? GROUP BY status, class ORDER BY n DESC", (start, end))
    return [{"status": r["status"], "class": r["class"], "count": r["n"]} for r in rows]


def recent_errors(store, start: float, end: float, limit: int = 50) -> list[dict]:
    rows = store.query(
        "SELECT ts,model,endpoint,method,status,client_ip,latency_ms,attribution"
        " FROM requests WHERE ts >= ? AND ts < ? AND status >= 400"
        " ORDER BY ts DESC LIMIT ?", (start, end, limit))
    return [dict(r) for r in rows]


def slowest(store, start: float, end: float, by: str = "ttft_ms", limit: int = 20) -> list[dict]:
    if by not in ("ttft_ms", "latency_ms", "queue_ms", "total_ms", "decode_ms"):
        raise ValueError(f"cannot sort by {by!r}")
    rows = store.query(
        "SELECT ts,model,endpoint,status,client_ip,ttft_ms,latency_ms,queue_ms,total_ms,"
        "prompt_tokens,prompt_tokens_total,output_tokens,decode_tps,truncated,attribution"
        f" FROM requests WHERE ts >= ? AND ts < ? AND {by} IS NOT NULL"
        f" ORDER BY {by} DESC LIMIT ?", (start, end, limit))
    return [dict(r) for r in rows]


def recent_requests(store, start: float, end: float, limit: int = 100,
                    include_health: bool = False) -> list[dict]:
    clause = "" if include_health else " AND class IN ('inference','embed')"
    rows = store.query(
        "SELECT ts,model,endpoint,method,class,status,client_ip,latency_ms,ttft_ms,"
        "queue_ms,total_ms,prompt_tokens,prompt_tokens_total,cached_tokens,output_tokens,"
        "decode_tps,prefill_tps,draft_accept,truncated,slot_id,task_id,attribution"
        f" FROM requests WHERE ts >= ? AND ts < ?{clause} ORDER BY ts DESC LIMIT ?",
        (start, end, limit))
    return [dict(r) for r in rows]


def events(store, start: float, end: float, kind: str | None = None,
           limit: int = 100) -> list[dict]:
    params: list = [start, end]
    clause = ""
    if kind:
        clause = " AND kind = ?"
        params.append(kind)
    params.append(limit)
    rows = store.query(
        "SELECT ts,kind,level,model,source,msg,duration_ms,detail_json FROM events"
        f" WHERE ts >= ? AND ts < ?{clause} ORDER BY ts DESC LIMIT ?", tuple(params))
    return [dict(r) for r in rows]


def gpu_series(store, start: float, end: float, step: int | None = None) -> dict:
    span = max(1.0, end - start)
    step = step or pick_step(span)
    rows = store.query(
        "SELECT ts,gpu_index,name,util_pct,mem_used,mem_total,temp_c,power_w"
        " FROM gpu_samples WHERE ts >= ? AND ts < ? ORDER BY ts", (start, end))
    base = int(start // step) * step
    nb = int(span // step) + 1
    gpus: dict[int, dict] = {}
    for r in rows:
        g = gpus.setdefault(r["gpu_index"], {
            "index": r["gpu_index"], "name": r["name"], "mem_total": r["mem_total"],
            "util_pct": [None] * nb, "mem_used": [None] * nb, "temp_c": [None] * nb,
            "power_w": [None] * nb, "_n": [0] * nb})
        i = int((r["ts"] - base) // step)
        if not 0 <= i < nb:
            continue
        # running mean within the bucket
        n = g["_n"][i]
        for key in ("util_pct", "mem_used", "temp_c", "power_w"):
            v = r[key]
            if v is None:
                continue
            prev = g[key][i]
            g[key][i] = v if prev is None else (prev * n + v) / (n + 1)
        g["_n"][i] = n + 1
    for g in gpus.values():
        g.pop("_n", None)
    return {"start": base, "step": step, "n": nb,
            "t": [base + i * step for i in range(nb)],
            "gpus": [gpus[k] for k in sorted(gpus)]}


# --------------------------------------------------------------------------
# ollama's prompt cache
# --------------------------------------------------------------------------

# Occupancy above this is worth flagging: the pool is about to start evicting,
# and every eviction is a prefill somebody pays for later.
CACHE_PRESSURE = 0.9

_CACHE_COUNTERS = ("evictions", "evict_crowded", "evict_invalidated", "restores",
                   "ckpt_created", "saves")


def _gauge(values: list) -> dict:
    """mean / max / last over a gauge's samples, ignoring nulls."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "max": None, "last": None}
    return {"mean": sum(vals) / len(vals), "max": max(vals), "last": vals[-1]}


def cache_summary(store, start: float, end: float) -> dict:
    """Prompt-cache occupancy and pressure over a window.

    A caveat that shapes every number here: ollama logs a `cache state` line
    only when it actually runs a cache update, so the sampling is ITS schedule,
    not a fixed interval.  An empty window means "no cache updates happened",
    which is not the same as "the cache was empty" -- so `samples` is reported
    alongside, and the counters are summed rather than turned into rates, since
    a rate over unevenly sampled deltas would be fiction.
    """
    span = max(1e-9, end - start)
    rows = store.query(
        "SELECT * FROM ollama_cache_samples WHERE ts >= ? AND ts < ? ORDER BY ts",
        (start, end))
    usage = _gauge([r["usage"] for r in rows])
    out = {
        "start": start, "end": end, "span_s": span, "exact": True,
        "samples": len(rows),
        "last_ts": rows[-1]["ts"] if rows else None,
        "usage": usage,
        "used_mib": _gauge([r["used_mib"] for r in rows]),
        "limit_mib": next((r["limit_mib"] for r in reversed(rows)
                           if r["limit_mib"] is not None), None),
        "prompts": _gauge([r["prompts"] for r in rows]),
        "token_limit": next((r["token_limit"] for r in reversed(rows)
                             if r["token_limit"] is not None), None),
        # The `est` field on the same log line is NOT summarised here: on an
        # empty cache llama.cpp reports a byte budget (2^33) in the token slot,
        # so a mean over it is meaningless.  The raw column keeps the value.
        # The maintenance pass is synchronous, so this is latency a request
        # paid, not background work.
        "update_ms": _gauge([r["update_ms"] for r in rows]),
        "checkpoints": {
            "used": _gauge([r["ckpt_used"] for r in rows]),
            "total": next((r["ckpt_total"] for r in reversed(rows)
                           if r["ckpt_total"] is not None), None),
        },
        "under_pressure": (usage["max"] or 0) >= CACHE_PRESSURE,
    }
    for key in _CACHE_COUNTERS:
        out[key] = sum(r[key] or 0 for r in rows)
    if not rows:
        out["note"] = ("no prompt-cache samples in this window. these lines come "
                       "from the llama-server runner at high log verbosity, and "
                       "only when ollama runs a cache update -- an instance that "
                       "served no traffic logs none.")
    return out


def cache_series(store, start: float, end: float, step: int | None = None) -> dict:
    """Bucketed prompt-cache series for the charts.

    Gauges are averaged within a bucket and counters summed, matching how the
    two are stored.  Empty buckets stay null so an idle stretch reads as "not
    sampled" rather than as a cache that emptied.
    """
    span = max(1.0, end - start)
    step = step or pick_step(span)
    nb = int(span // step) + 1
    base = int(start // step) * step
    gauges = ("usage", "used_mib", "prompts", "update_ms", "ckpt_used")
    series = {k: [None] * nb for k in gauges + _CACHE_COUNTERS}
    counts = [0] * nb

    rows = store.query(
        "SELECT * FROM ollama_cache_samples WHERE ts >= ? AND ts < ? ORDER BY ts",
        (start, end))
    acc: dict[int, dict] = {}
    for r in rows:
        i = int((r["ts"] - base) // step)
        if not 0 <= i < nb:
            continue
        a = acc.setdefault(i, {k: [] for k in gauges})
        for k in gauges:
            if r[k] is not None:
                a[k].append(r[k])
        for k in _CACHE_COUNTERS:
            series[k][i] = (series[k][i] or 0) + (r[k] or 0)
        counts[i] += 1
    for i, a in acc.items():
        for k in gauges:
            if a[k]:
                series[k][i] = sum(a[k]) / len(a[k])

    return {"start": base, "step": step, "n": nb, "exact": True,
            "t": [base + i * step for i in range(nb)],
            "counts": counts, "series": series}


def loaded_models(store) -> dict:
    rows = store.query("SELECT ts,loaded_count,models_json,inflight FROM ps_samples"
                       " ORDER BY ts DESC LIMIT 1")
    if not rows:
        return {"ts": None, "loaded_count": 0, "models": [], "inflight": 0}
    import json
    r = rows[0]
    try:
        models = json.loads(r["models_json"] or "[]")
    except ValueError:
        models = []
    return {"ts": r["ts"], "loaded_count": r["loaded_count"], "models": models,
            "inflight": r["inflight"]}


def model_names(store, days: float = 30.0) -> list[str]:
    rows = store.query(
        "SELECT DISTINCT model FROM rollup_1h WHERE bucket >= ? AND model <> ''"
        " ORDER BY model", (time.time() - days * 86400,))
    return [r["model"] for r in rows]
