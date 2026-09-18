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


def clients(store, source: str, start: float, end: float) -> dict:
    """Who was connected, and the requests that can honestly be pinned on them.

    tabbyAPI logs no client address, so identity comes from the socket table.
    A request is attributed only where the sampling makes it unambiguous:
    every connection sample taken while it was in flight showed exactly ONE
    client.
    """
    samples = store.query(
        "SELECT ts, client, conns FROM exllama_client_samples"
        " WHERE source=? AND ts >= ? AND ts < ? ORDER BY ts",
        (source, start, end))
    by_ts: dict[float, set] = {}
    seen: dict[str, dict] = {}
    for r in samples:
        by_ts.setdefault(r["ts"], set()).add(r["client"])
        d = seen.setdefault(r["client"], {"client": r["client"], "samples": 0,
                                          "peak_conns": 0, "first_seen": r["ts"],
                                          "last_seen": r["ts"]})
        d["samples"] += 1
        d["peak_conns"] = max(d["peak_conns"], r["conns"] or 0)
        d["first_seen"] = min(d["first_seen"], r["ts"])
        d["last_seen"] = max(d["last_seen"], r["ts"])

    stamps = sorted(by_ts)
    total_samples = len(stamps)

    def sole_client(a: float, b: float):
        covering = [t for t in stamps if a <= t <= b]
        if not covering:
            return None
        who = set()
        for t in covering:
            names = by_ts[t]
            if len(names) != 1:
                return None
            who |= names
        return next(iter(who)) if len(who) == 1 else None

    stats: dict[str, dict] = {}
    attributed = ambiguous = 0
    for r in store.query(
            "SELECT ts, started_ts, prompt_tokens, cached_tokens, output_tokens,"
            "       ttft_ms, latency_ms, finish"
            "  FROM exllama_requests WHERE source=? AND ts >= ? AND ts < ?",
            (source, start, end)):
        began = r["started_ts"] if r["started_ts"] is not None else r["ts"]
        who = sole_client(began, r["ts"])
        if who is None:
            ambiguous += 1
            continue
        attributed += 1
        d = stats.setdefault(who, {"requests": 0, "input_tokens": 0,
                                   "output_tokens": 0, "cached_tokens": 0,
                                   "errors": 0, "ttft": [], "latency": []})
        d["requests"] += 1
        d["input_tokens"] += r["prompt_tokens"] or 0
        d["output_tokens"] += r["output_tokens"] or 0
        d["cached_tokens"] += r["cached_tokens"] or 0
        if (r["finish"] or "") == "error":
            d["errors"] += 1
        if r["ttft_ms"] is not None:
            d["ttft"].append(r["ttft_ms"])
        if r["latency_ms"] is not None:
            d["latency"].append(r["latency_ms"])

    out = []
    for name, d in seen.items():
        s = stats.get(name) or {}
        out.append({
            **d,
            "presence": (d["samples"] / total_samples) if total_samples else None,
            "requests": s.get("requests", 0),
            "input_tokens": s.get("input_tokens", 0),
            "output_tokens": s.get("output_tokens", 0),
            "cached_tokens": s.get("cached_tokens", 0),
            "errors": s.get("errors", 0),
            "ttft_p50": _pct(s.get("ttft") or [], 0.5),
            "latency_p50": _pct(s.get("latency") or [], 0.5),
        })
    out.sort(key=lambda x: (-x["requests"], -x["samples"]))
    return {"clients": out, "attributed": attributed, "ambiguous": ambiguous,
            "samples": total_samples,
            "notes": "Connections are sampled from the socket table; tabbyAPI "
                     "logs no client address. A request is attributed only "
                     "when every sample taken while it ran showed exactly one "
                     "connected client."}


def instances(store) -> list[dict]:
    configured = {s["name"] for s in store.list_sources()
                  if s["kind"] == "exllama"}
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
        out.append(d)
    return out