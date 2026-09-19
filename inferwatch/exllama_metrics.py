"""Query layer for the exllama (tabbyAPI) tables.

Like NInfer, exllama publishes no /metrics, so everything comes from the log.
Unlike NInfer there is no periodic throughput gauge, so the timeseries is
built entirely from per-request rows.
"""

from __future__ import annotations

import json
import time

from . import metrics as _m


def _pct(values: list[float], q: float):
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def summary(store, source: str, start: float, end: float) -> dict:
    span = max(1e-9, end - start)
    rows = store.query(
        "SELECT ttft_ms, latency_ms, decode_tps, prefill_tps, prompt_tokens,"
        "       cached_tokens, prompt_tokens_total, output_tokens, finish,"
        "       draft_accept, paired"
        "  FROM exllama_requests WHERE source=? AND ts >= ? AND ts < ?",
        (source, start, end))
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    dec = [r["decode_tps"] for r in rows if r["decode_tps"] is not None]
    pre = [r["prefill_tps"] for r in rows if r["prefill_tps"] is not None]
    acc = [r["draft_accept"] for r in rows if r["draft_accept"] is not None]
    errors = sum(1 for r in rows if (r["finish"] or "") == "error")
    disc = sum(1 for r in rows if (r["finish"] or "") == "disconnected")
    in_tok = sum(r["prompt_tokens"] or 0 for r in rows)
    cached = sum(r["cached_tokens"] or 0 for r in rows)
    sent = sum(r["prompt_tokens_total"] or 0 for r in rows)
    out_tok = sum(r["output_tokens"] or 0 for r in rows)
    return {
        "requests": len(rows),
        "errors": errors,
        "disconnected": disc,
        "error_rate": (errors / len(rows)) if rows else 0.0,
        "req_per_s": len(rows) / span,
        "input_tokens": in_tok,
        "cached_tokens": cached,
        "output_tokens": out_tok,
        "cache_hit_rate": (cached / sent) if sent else 0.0,
        "ttft_ms": {"p50": _pct(ttft, 0.5), "p90": _pct(ttft, 0.9),
                    "p99": _pct(ttft, 0.99)},
        "ttft_max_ms": max(ttft) if ttft else None,
        "latency_ms": {"p50": _pct(lat, 0.5), "p90": _pct(lat, 0.9),
                       "p99": _pct(lat, 0.99)},
        "decode_tps_mean": (sum(dec) / len(dec)) if dec else None,
        "prefill_tps_mean": (sum(pre) / len(pre)) if pre else None,
        "draft_accept_mean": (sum(acc) / len(acc)) if acc else None,
        "unpaired": sum(1 for r in rows if not r["paired"]),
        "exact": True,
        "notes": "Percentiles are exact within the raw retention window: "
                 "tabbyAPI logs per-request timings.",
    }


def timeseries(store, source: str, start: float, end: float,
               step: int | None = None) -> dict:
    span = max(1.0, end - start)
    step = step or _m.pick_step(span)
    base = int(start // step) * step
    n = max(1, int((end - base) // step) + 1)
    t = [base + i * step for i in range(n)]

    def slot(ts):
        return min(n - 1, max(0, int((ts - base) // step)))

    req = [0] * n
    err = [0] * n
    out_tok = [0] * n
    in_tok = [0] * n
    ttft_sum = [0.0] * n
    ttft_n = [0] * n
    dec_sum = [0.0] * n
    dec_n = [0] * n
    for r in store.query(
            "SELECT ts, finish, output_tokens, prompt_tokens, ttft_ms,"
            "       decode_tps"
            "  FROM exllama_requests WHERE source=? AND ts >= ? AND ts < ?",
            (source, start, end)):
        i = slot(r["ts"])
        req[i] += 1
        if (r["finish"] or "") == "error":
            err[i] += 1
        out_tok[i] += r["output_tokens"] or 0
        in_tok[i] += r["prompt_tokens"] or 0
        if r["ttft_ms"] is not None:
            ttft_sum[i] += r["ttft_ms"]
            ttft_n[i] += 1
        if r["decode_tps"] is not None:
            dec_sum[i] += r["decode_tps"]
            dec_n[i] += 1

    return {
        "t": t, "step": step,
        "req_per_s": [c / step for c in req],
        "err_per_s": [c / step for c in err],
        "out_tps": [c / step for c in out_tok],
        "in_tps": [c / step for c in in_tok],
        "ttft_mean_ms": [(ttft_sum[i] / ttft_n[i]) if ttft_n[i] else None
                         for i in range(n)],
        "decode_tps": [(dec_sum[i] / dec_n[i]) if dec_n[i] else None
                       for i in range(n)],
    }


def finish_reasons(store, source: str, start: float, end: float) -> list[dict]:
    return [dict(r) for r in store.query(
        "SELECT COALESCE(finish,'unknown') finish, COUNT(*) count"
        "  FROM exllama_requests WHERE source=? AND ts >= ? AND ts < ?"
        " GROUP BY 1 ORDER BY count DESC", (source, start, end))]


def by_model(store, source: str, start: float, end: float) -> list[dict]:
    return [dict(r) for r in store.query(
        "SELECT COALESCE(model,'unknown') model, COUNT(*) requests,"
        "       AVG(ttft_ms) ttft_mean, AVG(decode_tps) decode_tps,"
        "       SUM(output_tokens) output_tokens, SUM(prompt_tokens) input_tokens,"
        "       MAX(ts) last_used"
        "  FROM exllama_requests WHERE source=? AND ts >= ? AND ts < ?"
        " GROUP BY 1 ORDER BY requests DESC", (source, start, end))]


def recent_requests(store, source: str, start: float, end: float,
                    limit: int = 100) -> list[dict]:
    return [dict(r) for r in store.query(
        "SELECT ts, req_id, model, finish, error, prompt_tokens, cached_tokens,"
        "       output_tokens, ttft_ms, latency_ms, decode_tps, prefill_tps,"
        "       draft_accept, stream, tool_calls, paired"
        "  FROM exllama_requests WHERE source=? AND ts >= ? AND ts < ?"
        " ORDER BY ts DESC LIMIT ?", (source, start, end, limit))]


def slowest(store, source: str, start: float, end: float,
            column: str = "ttft_ms", limit: int = 10) -> list[dict]:
    if column not in ("ttft_ms", "latency_ms"):
        raise ValueError("column must be ttft_ms or latency_ms")
    return [dict(r) for r in store.query(
        f"SELECT ts, req_id, model, finish, prompt_tokens, output_tokens,"  # noqa: S608
        f"       ttft_ms, latency_ms, decode_tps"
        f"  FROM exllama_requests WHERE source=? AND ts >= ? AND ts < ?"
        f"   AND {column} IS NOT NULL ORDER BY {column} DESC LIMIT ?",
        (source, start, end, limit))]


def clients(store, source: str, start: float, end: float,
            limit: int = 50) -> list[dict]:
    """Per-client request activity, from the proxy in front of tabbyAPI.

    tabbyAPI sees the proxy (loopback), not the caller, so its own log carries
    no client address. The proxy does: it logs the caller on every request.
    Only fields the proxy states on the REQ line itself appear here; there is
    deliberately no status, latency or output-token column, because those live
    on separate lines carrying no request id and cannot be tied to a client
    without guessing. `prompt_tokens` is exact -- the proxy tokenises upstream.
    """
    rows = store.query(
        "SELECT client,"
        "       COUNT(*)                 requests,"
        "       SUM(prompt_tokens)       prompt_tokens,"
        "       AVG(prompt_tokens)       prompt_mean,"
        "       MAX(prompt_tokens)       prompt_max,"
        "       AVG(messages)            messages_mean,"
        "       MAX(ts)                  last_seen,"
        "       COUNT(DISTINCT model)    models,"
        "       SUM(CASE WHEN stream=1 THEN 1 ELSE 0 END) streamed,"
        "       SUM(CASE WHEN prompt_tokens IS NULL THEN 1 ELSE 0 END) unsized,"
        "       MAX(context_limit)       context_limit"
        "  FROM vllm_client_requests"
        " WHERE source=? AND ts >= ? AND ts < ?"
        " GROUP BY client ORDER BY requests DESC LIMIT ?",
        (source, start, end, limit))
    out = []
    for r in rows:
        d = dict(r)
        d["stale_s"] = (time.time() - d["last_seen"]) if d["last_seen"] else None
        # Share of this client's largest prompt against the window it must fit.
        d["context_peak_pct"] = ((d["prompt_max"] / d["context_limit"])
                                 if d.get("prompt_max") and d.get("context_limit")
                                 else None)
        top = store.query(
            "SELECT model, COUNT(*) n FROM vllm_client_requests"
            " WHERE source=? AND ts >= ? AND ts < ? AND client=? AND model IS NOT NULL"
            " GROUP BY model ORDER BY n DESC LIMIT 1",
            (source, start, end, d["client"]))
        d["top_model"] = top[0]["model"] if top else None
        out.append(d)
    return out


def clients_available(store, source: str) -> bool:
    """Whether any proxy-derived row exists for this source at all.

    Lets the pane tell "no proxy configured" apart from "a proxy is configured
    and nobody has called it", which are different things to show.
    """
    row = store.query("SELECT 1 FROM vllm_client_requests WHERE source=? LIMIT 1",
                      (source,))
    return bool(row)


def cache_usage(store, source: str, start: float, end: float) -> dict:
    """KV-cache occupancy over the window, from the polled /v1/cache/stats.

    used_tokens/max_tokens is the true fraction of the KV pool held by active
    jobs (exllamav3's page table) -- the analog of vLLM's kv_cache_usage_perc,
    which tabbyAPI does not otherwise publish.
    """
    rows = store.query(
        "SELECT used_tokens, max_tokens FROM exllama_cache_samples"
        " WHERE source=? AND ts >= ? AND ts < ? AND max_tokens > 0"
        "   AND used_tokens IS NOT NULL", (source, start, end))
    fracs = [r["used_tokens"] / r["max_tokens"] for r in rows]
    last = store.query(
        "SELECT used_tokens, max_tokens, active_jobs, pending_jobs, hit_rate"
        " FROM exllama_cache_samples WHERE source=? AND ts >= ? AND ts < ?"
        " ORDER BY ts DESC LIMIT 1", (source, start, end))
    l = last[0] if last else None
    return {
        "mean": (sum(fracs) / len(fracs)) if fracs else None,
        "max": max(fracs) if fracs else None,
        "last": (l["used_tokens"] / l["max_tokens"])
                if l and l["max_tokens"] and l["used_tokens"] is not None
                else None,
        "active_jobs": l["active_jobs"] if l else None,
        "pending_jobs": l["pending_jobs"] if l else None,
        "hit_rate": l["hit_rate"] if l else None,
        "n": len(rows),
    }


def cache_timeseries(store, source: str, start: float, end: float,
                     step: int | None = None) -> dict:
    span = max(1.0, end - start)
    step = step or _m.pick_step(span)
    base = int(start // step) * step
    n = max(1, int((end - base) // step) + 1)
    t = [base + i * step for i in range(n)]

    def slot(ts):
        return min(n - 1, max(0, int((ts - base) // step)))

    used = [0.0] * n
    active = [0.0] * n
    pending = [0.0] * n
    cnt = [0] * n
    for r in store.query(
            "SELECT ts, used_tokens, max_tokens, active_jobs, pending_jobs"
            " FROM exllama_cache_samples"
            " WHERE source=? AND ts >= ? AND ts < ? AND max_tokens > 0"
            "   AND used_tokens IS NOT NULL", (source, start, end)):
        i = slot(r["ts"])
        used[i] += r["used_tokens"] / r["max_tokens"]
        active[i] += r["active_jobs"] or 0
        pending[i] += r["pending_jobs"] or 0
        cnt[i] += 1
    return {"t": t, "step": step,
            "used_frac": [(used[i] / cnt[i]) if cnt[i] else None
                          for i in range(n)],
            "active_jobs": [(active[i] / cnt[i]) if cnt[i] else None
                            for i in range(n)],
            "pending_jobs": [(pending[i] / cnt[i]) if cnt[i] else None
                             for i in range(n)]}


def instances(store) -> list[dict]:
    configured = {s["name"] for s in store.list_sources()
                  if s["kind"] == "exllama"}
    latest_cache = {
        r["source"]: r for r in store.query(
            "SELECT c.source, c.used_tokens, c.max_tokens, c.active_jobs,"
            "       c.pending_jobs, c.hit_rate, c.ts"
            "  FROM exllama_cache_samples c"
            "  JOIN (SELECT source, MAX(ts) mts FROM exllama_cache_samples"
            "        GROUP BY source) l ON l.source=c.source AND l.mts=c.ts")}
    out = []
    for r in store.query(
            "SELECT source,last_seen,model,reachable,error,max_seq_len,"
            "       cache_size,load_ms,gpu_indices,gpu_source,gpu_ts"
            "  FROM exllama_instances ORDER BY source"):
        if r["source"] not in configured:
            continue
        d = dict(r)
        raw = d.pop("gpu_indices", None)
        try:
            d["gpu_indices"] = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            d["gpu_indices"] = None
        d["reachable"] = bool(d["reachable"])
        d["stale_s"] = (time.time() - d["last_seen"]) if d["last_seen"] else None
        d["gpu_age_s"] = (time.time() - d["gpu_ts"]) if d.get("gpu_ts") else None
        c = latest_cache.get(r["source"])
        if c:
            d["kv_used_tokens"] = c["used_tokens"]
            d["kv_max_tokens"] = c["max_tokens"]
            d["kv_frac"] = (c["used_tokens"] / c["max_tokens"]) \
                if c["max_tokens"] and c["used_tokens"] is not None else None
            d["kv_active_jobs"] = c["active_jobs"]
            d["kv_pending_jobs"] = c["pending_jobs"]
            d["kv_hit_rate"] = c["hit_rate"]
            d["kv_ts"] = c["ts"]
        out.append(d)
    return out