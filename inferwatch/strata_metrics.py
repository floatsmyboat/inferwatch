"""Query layer for the Strata tables.

Strata keeps the numbers that matter for a mixture-of-experts engine whose
experts mostly live in system RAM: how often a token's routed experts were
already in the GPU's expert cache, and how much had to cross PCIe instead.
Those are per request (strata_requests); the queue, the context in use and the
live decode rate are between requests (strata_samples).

Percentiles are exact within raw retention: each request row is the server's
own record, so nothing here is a bucket bound.
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


def _wmean(pairs):
    """Mean of values weighted by tokens, so a 3-token reply does not count
    as much as a 3,000-token one.  pairs: [(value, weight)]."""
    num = sum(v * w for v, w in pairs if v is not None and w)
    den = sum(w for v, w in pairs if v is not None and w)
    return (num / den) if den else None


def summary(store, source: str, start: float, end: float) -> dict:
    span = max(1e-9, end - start)
    rows = store.query(
        "SELECT ttft_ms, latency_ms, decode_tps, prefill_tps, prompt_tokens,"
        "       cached_tokens, prompt_tokens_total, output_tokens, finish,"
        "       expert_hit_rate, pcie_share, drafts_offered, drafts_accepted,"
        "       file_blobs"
        "  FROM strata_requests WHERE source=? AND ts >= ? AND ts < ?",
        (source, start, end))
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    dec = [r["decode_tps"] for r in rows if r["decode_tps"] is not None]
    pre = [r["prefill_tps"] for r in rows if r["prefill_tps"] is not None]
    errors = sum(1 for r in rows if (r["finish"] or "") == "error")
    cancelled = sum(1 for r in rows if (r["finish"] or "") == "cancelled")
    in_tok = sum(r["prompt_tokens"] or 0 for r in rows)
    cached = sum(r["cached_tokens"] or 0 for r in rows)
    sent = sum(r["prompt_tokens_total"] or 0 for r in rows)
    out_tok = sum(r["output_tokens"] or 0 for r in rows)
    offered = sum(r["drafts_offered"] or 0 for r in rows)
    accepted = sum(r["drafts_accepted"] or 0 for r in rows)
    hit = [r["expert_hit_rate"] for r in rows if r["expert_hit_rate"] is not None]
    return {
        "requests": len(rows),
        "errors": errors,
        "cancelled": cancelled,
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
        # MTP drafts, totalled rather than averaged per request: a short reply
        # with 3 of 4 accepted is not 75% of anything that matters.
        "draft_accept_mean": (accepted / offered) if offered else None,
        # Weighted by tokens generated: the share of the work, not of requests.
        "expert_hit_rate": _wmean([(r["expert_hit_rate"], r["output_tokens"])
                                   for r in rows]),
        "expert_hit_min": min(hit) if hit else None,
        "pcie_share": _wmean([(r["pcie_share"], r["output_tokens"]) for r in rows]),
        "disk_reads": sum(1 for r in rows if (r["file_blobs"] or 0) > 0),
        "exact": True,
        "notes": "Percentiles are exact within the raw retention window: each row "
                 "is Strata's own record of the request. TTFT is the engine's "
                 "prompt time and excludes HTTP and tokenisation overhead.",
    }


def timeseries(store, source: str, start: float, end: float,
               step: int | None = None) -> dict:
    """Per-bucket request rates, token throughput, and expert-cache behaviour."""
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
    hit = [[] for _ in range(n)]
    pcie = [[] for _ in range(n)]
    for r in store.query(
            "SELECT ts, finish, output_tokens, prompt_tokens, ttft_ms,"
            "       decode_tps, expert_hit_rate, pcie_share"
            "  FROM strata_requests WHERE source=? AND ts >= ? AND ts < ?",
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
        hit[i].append((r["expert_hit_rate"], r["output_tokens"]))
        pcie[i].append((r["pcie_share"], r["output_tokens"]))

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
        "expert_hit_rate": [_wmean(hit[i]) for i in range(n)],
        "pcie_share": [_wmean(pcie[i]) for i in range(n)],
    }


def sample_timeseries(store, source: str, start: float, end: float,
                      step: int | None = None) -> dict:
    """The polled gauges, averaged per bucket (peak for the context in use)."""
    span = max(1.0, end - start)
    step = step or _m.pick_step(span)
    base = int(start // step) * step
    n = max(1, int((end - base) // step) + 1)
    t = [base + i * step for i in range(n)]
    acc: dict[int, list] = {}
    for r in store.query(
            "SELECT ts, running, queued, kv_tokens, kv_max, kv_resident,"
            "       decode_tps, prefill_tps, host_pcie_rx_mb, host_pcie_tx_mb,"
            "       cpu_pct, ram_used_bytes, ram_total_bytes"
            "  FROM strata_samples WHERE source=? AND ts >= ? AND ts < ?",
            (source, start, end)):
        acc.setdefault(min(n - 1, max(0, int((r["ts"] - base) // step))), []).append(r)

    def series(fn):
        return [fn(acc[i]) if i in acc else None for i in range(n)]

    def mean(key):
        def f(rows):
            vals = [x[key] for x in rows if x[key] is not None]
            return (sum(vals) / len(vals)) if vals else None
        return f

    def kv_frac(rows):
        vals = [x["kv_tokens"] / x["kv_max"] for x in rows
                if x["kv_tokens"] is not None and x["kv_max"]]
        return max(vals) if vals else None

    def streamed(rows):
        """Tokens of the running context past the VRAM-resident window, i.e.
        KV the engine is streaming from system RAM."""
        vals = [max(0, x["kv_tokens"] - x["kv_resident"]) for x in rows
                if x["kv_tokens"] is not None and x["kv_resident"] is not None]
        return max(vals) if vals else None

    return {
        "t": t, "step": step,
        "running": series(mean("running")),
        "queued": series(mean("queued")),
        "kv_frac": series(kv_frac),
        "kv_tokens": series(lambda rows: max((x["kv_tokens"] or 0) for x in rows)),
        "kv_streamed": series(streamed),
        "decode_tps": series(mean("decode_tps")),
        "prefill_tps": series(mean("prefill_tps")),
        "host_pcie_rx_mb": series(mean("host_pcie_rx_mb")),
        "host_pcie_tx_mb": series(mean("host_pcie_tx_mb")),
        "cpu_pct": series(mean("cpu_pct")),
        "ram_used_gib": series(lambda rows: (lambda v: v / 2**30 if v is not None else None)(
            mean("ram_used_bytes")(rows))),
    }


def live(store, source: str, start: float, end: float) -> dict:
    """The newest sample in the window, plus the context peak over it."""
    last = store.query(
        "SELECT * FROM strata_samples WHERE source=? AND ts >= ? AND ts < ?"
        " ORDER BY ts DESC LIMIT 1", (source, start, end))
    peak = store.query(
        "SELECT MAX(kv_tokens) kv_peak, MAX(queued) queued_peak, COUNT(*) n"
        "  FROM strata_samples WHERE source=? AND ts >= ? AND ts < ?",
        (source, start, end))
    d = dict(last[0]) if last else {}
    p = dict(peak[0]) if peak else {}
    d["kv_frac"] = (d["kv_tokens"] / d["kv_max"]) \
        if d.get("kv_tokens") is not None and d.get("kv_max") else None
    d["kv_peak"] = p.get("kv_peak")
    d["kv_peak_frac"] = (p["kv_peak"] / d["kv_max"]) \
        if p.get("kv_peak") is not None and d.get("kv_max") else None
    d["queued_peak"] = p.get("queued_peak")
    d["n"] = p.get("n") or 0
    return d


def finish_reasons(store, source: str, start: float, end: float) -> list[dict]:
    return [dict(r) for r in store.query(
        "SELECT COALESCE(finish,'unknown') finish, COUNT(*) count"
        "  FROM strata_requests WHERE source=? AND ts >= ? AND ts < ?"
        " GROUP BY 1 ORDER BY count DESC", (source, start, end))]


def by_model(store, source: str, start: float, end: float) -> list[dict]:
    return [dict(r) for r in store.query(
        "SELECT COALESCE(model,'unknown') model, COUNT(*) requests,"
        "       AVG(ttft_ms) ttft_mean, AVG(decode_tps) decode_tps,"
        "       SUM(output_tokens) output_tokens, SUM(prompt_tokens) input_tokens,"
        "       MAX(ts) last_used"
        "  FROM strata_requests WHERE source=? AND ts >= ? AND ts < ?"
        " GROUP BY 1 ORDER BY requests DESC", (source, start, end))]


def recent_requests(store, source: str, start: float, end: float,
                    limit: int = 100) -> list[dict]:
    return [dict(r) for r in store.query(
        "SELECT ts, started_ts, model, finish, prompt_tokens, cached_tokens,"
        "       prompt_tokens_total, output_tokens, ttft_ms, latency_ms,"
        "       decode_tps, prefill_tps, expert_hit_rate, pcie_share,"
        "       file_blobs, draft_accept"
        "  FROM strata_requests WHERE source=? AND ts >= ? AND ts < ?"
        " ORDER BY ts DESC LIMIT ?", (source, start, end, limit))]


def slowest(store, source: str, start: float, end: float,
            column: str = "ttft_ms", limit: int = 10) -> list[dict]:
    if column not in ("ttft_ms", "latency_ms"):
        raise ValueError("column must be ttft_ms or latency_ms")
    return [dict(r) for r in store.query(
        f"SELECT ts, model, finish, prompt_tokens, output_tokens,"  # noqa: S608
        f"       ttft_ms, latency_ms, decode_tps, expert_hit_rate"
        f"  FROM strata_requests WHERE source=? AND ts >= ? AND ts < ?"
        f"   AND {column} IS NOT NULL ORDER BY {column} DESC LIMIT ?",
        (source, start, end, limit))]


def instances(store, kind: str = "strata") -> list[dict]:
    """Configured instances of one kind (Strata, or Maya, which shares these
    tables), filtered to sources that still exist."""
    configured = {s["name"] for s in store.list_sources() if s["kind"] == kind}
    out = []
    for r in store.query("SELECT * FROM strata_instances ORDER BY source"):
        if r["source"] not in configured:
            continue
        d = dict(r)
        raw = d.pop("gpu_indices", None)
        try:
            d["gpu_indices"] = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            d["gpu_indices"] = None
        d["reachable"] = bool(d["reachable"])
        d["vision"] = bool(d["vision"]) if d.get("vision") is not None else None
        d["stale_s"] = (time.time() - d["last_seen"]) if d["last_seen"] else None
        d["gpu_age_s"] = (time.time() - d["gpu_ts"]) if d.get("gpu_ts") else None
        d["uptime_s"] = (time.time() - d["started"]) if d.get("started") else None
        out.append(d)
    return out
