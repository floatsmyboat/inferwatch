"""Query layer for the NInfer tables.

Separate from `metrics.py` and `vllm_metrics.py` for the reason the other two
are separate from each other: the engines do not expose the same things, and
one shared query surface would have to either drop what is unique or invent
what is missing.

NInfer is the one engine here that needs no caveat about which half it lacks.
Its percentiles are computed from stored per-request rows, so within retention
they are **exact** -- not vLLM's bucket upper bounds -- and its queue-depth and
batch-occupancy gauges are evenly sampled, which ollama has no equivalent of.
"""

from __future__ import annotations

import json
import time

from . import metrics as _m


def _pct(values: list[float], q: float):
    """Exact percentile by rank, since the raw values are stored."""
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
        "       draft_accept, paired, reuse"
        "  FROM ninfer_requests WHERE source=? AND ts >= ? AND ts < ?",
        (source, start, end))
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    dec = [r["decode_tps"] for r in rows if r["decode_tps"] is not None]
    pre = [r["prefill_tps"] for r in rows if r["prefill_tps"] is not None]
    acc = [r["draft_accept"] for r in rows if r["draft_accept"] is not None]
    errors = sum(1 for r in rows if (r["finish"] or "") == "error")
    in_tok = sum(r["prompt_tokens"] or 0 for r in rows)
    cached = sum(r["cached_tokens"] or 0 for r in rows)
    sent = sum(r["prompt_tokens_total"] or 0 for r in rows)
    out_tok = sum(r["output_tokens"] or 0 for r in rows)
    return {
        "requests": len(rows),
        "errors": errors,
        "error_rate": (errors / len(rows)) if rows else 0.0,
        "req_per_s": len(rows) / span,
        "input_tokens": in_tok,
        "cached_tokens": cached,
        "output_tokens": out_tok,
        # What prefix reuse saved, against everything the clients actually sent.
        "cache_hit_rate": (cached / sent) if sent else 0.0,
        "ttft_ms": {"p50": _pct(ttft, 0.5), "p90": _pct(ttft, 0.9),
                    "p99": _pct(ttft, 0.99)},
        "ttft_max_ms": max(ttft) if ttft else None,
        "latency_ms": {"p50": _pct(lat, 0.5), "p90": _pct(lat, 0.9),
                       "p99": _pct(lat, 0.99)},
        "decode_tps_mean": (sum(dec) / len(dec)) if dec else None,
        "prefill_tps_mean": (sum(pre) / len(pre)) if pre else None,
        "draft_accept_mean": (sum(acc) / len(acc)) if acc else None,
        # A completion whose submission was never read still counts; this says
        # how many, so a low figure is visible rather than implied.
        "unpaired": sum(1 for r in rows if not r["paired"]),
        # Exact, unlike vLLM's: these come from stored per-request values.
        "exact": True,
        "notes": "Percentiles are exact within the raw retention window: NInfer "
                 "logs per-request timings, so nothing here is a bucket bound.",
    }


def timeseries(store, source: str, start: float, end: float,
               step: int | None = None) -> dict:
    """Per-bucket request rates and token throughput, plus the gauges.

    The gauges come from NInfer's own 5s throughput line rather than being
    derived from the request rows: queue depth and batch occupancy are states
    between requests, which no per-request row can reconstruct.
    """
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
    for r in store.query(
            "SELECT ts, finish, output_tokens, prompt_tokens, ttft_ms"
            "  FROM ninfer_requests WHERE source=? AND ts >= ? AND ts < ?",
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

    running = [None] * n
    waiting = [None] * n
    batch = [None] * n
    dec_tps = [None] * n
    pre_tps = [None] * n
    acc: dict[int, list] = {}
    for r in store.query(
            "SELECT ts, running, waiting, avg_decode_batch, decode_tps, prefill_tps"
            "  FROM ninfer_samples WHERE source=? AND ts >= ? AND ts < ?",
            (source, start, end)):
        acc.setdefault(slot(r["ts"]), []).append(r)
    for i, rows in acc.items():
        def mean(key):
            vals = [x[key] for x in rows if x[key] is not None]
            return (sum(vals) / len(vals)) if vals else None
        running[i] = mean("running")
        waiting[i] = mean("waiting")
        batch[i] = mean("avg_decode_batch")
        dec_tps[i] = mean("decode_tps")
        pre_tps[i] = mean("prefill_tps")

    return {
        "t": t, "step": step,
        "req_per_s": [c / step for c in req],
        "err_per_s": [c / step for c in err],
        "out_tps": [c / step for c in out_tok],
        "in_tps": [c / step for c in in_tok],
        "ttft_mean_ms": [(ttft_sum[i] / ttft_n[i]) if ttft_n[i] else None
                         for i in range(n)],
        "running": running, "waiting": waiting, "avg_decode_batch": batch,
        "decode_tps": dec_tps, "prefill_tps": pre_tps,
    }


def finish_reasons(store, source: str, start: float, end: float) -> list[dict]:
    """How requests ended.  NInfer logs no HTTP status, so this is the closest
    thing to one -- and it is more informative, since `output_limit` and
    `tool_calls` are both 200s that mean different things."""
    return [dict(r) for r in store.query(
        "SELECT COALESCE(finish,'unknown') finish, COUNT(*) count"
        "  FROM ninfer_requests WHERE source=? AND ts >= ? AND ts < ?"
        " GROUP BY 1 ORDER BY count DESC", (source, start, end))]


def by_model(store, source: str, start: float, end: float) -> list[dict]:
    return [dict(r) for r in store.query(
        "SELECT COALESCE(model,'unknown') model, COUNT(*) requests,"
        "       AVG(ttft_ms) ttft_mean, AVG(decode_tps) decode_tps,"
        "       SUM(output_tokens) output_tokens, SUM(prompt_tokens) input_tokens,"
        "       MAX(ts) last_used"
        "  FROM ninfer_requests WHERE source=? AND ts >= ? AND ts < ?"
        " GROUP BY 1 ORDER BY requests DESC", (source, start, end))]


def recent_requests(store, source: str, start: float, end: float,
                    limit: int = 100) -> list[dict]:
    return [dict(r) for r in store.query(
        "SELECT ts, req_id, model, finish, error, prompt_tokens, cached_tokens,"
        "       output_tokens, ttft_ms, latency_ms, decode_tps, prefill_tps,"
        "       draft_accept, reuse, stream, tools, paired"
        "  FROM ninfer_requests WHERE source=? AND ts >= ? AND ts < ?"
        " ORDER BY ts DESC LIMIT ?", (source, start, end, limit))]


def slowest(store, source: str, start: float, end: float, column: str = "ttft_ms",
            limit: int = 10) -> list[dict]:
    if column not in ("ttft_ms", "latency_ms"):
        raise ValueError("column must be ttft_ms or latency_ms")
    return [dict(r) for r in store.query(
        f"SELECT ts, req_id, model, finish, prompt_tokens, output_tokens,"  # noqa: S608
        f"       ttft_ms, latency_ms, decode_tps"
        f"  FROM ninfer_requests WHERE source=? AND ts >= ? AND ts < ?"
        f"   AND {column} IS NOT NULL ORDER BY {column} DESC LIMIT ?",
        (source, start, end, limit))]


def instances(store) -> list[dict]:
    """Configured NInfer instances, filtered to sources that still exist.

    Same rule as the vLLM instance list: a registry row outliving its source is
    stale by definition, and would otherwise keep offering a deleted engine.
    """
    configured = {s["name"] for s in store.list_sources() if s["kind"] == "ninfer"}
    out = []
    for r in store.query(
            "SELECT source,last_seen,model,reachable,error,kv_tokens,load_ms,"
            "       gpu_indices,gpu_source,gpu_ts FROM ninfer_instances"
            " ORDER BY source"):
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
        out.append(d)
    return out
