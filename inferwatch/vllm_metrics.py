"""Query layer for the vLLM tables.

Kept separate from `metrics.py` because the two engines expose genuinely
different things.  Sharing one query surface would have meant either dropping
vLLM's own metrics (KV-cache occupancy, preemptions, waiting-by-reason) or
faking per-request rows it does not publish.

Latency histograms are stored in vLLM's native seconds; everything here returns
milliseconds so the dashboard and MCP layer speak one unit.
"""

from __future__ import annotations

import json
import time

from .vllm import SECONDS_HISTOGRAMS

# Metric shorthands the API and MCP layer use, so callers never type the
# `vllm:..._seconds` names.
LATENCY_METRICS = {
    "ttft": "vllm:time_to_first_token_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "inference": "vllm:request_inference_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
}
# Inter-token latency changed name across vLLM releases; the first of these
# that actually has observations in the window is used.
ITL_CANDIDATES = (
    "vllm:inter_token_latency_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:time_per_output_token_seconds",
)

SIZE_METRICS = {
    "prompt_tokens": "vllm:request_prompt_tokens",
    "generation_tokens": "vllm:request_generation_tokens",
    "iteration_tokens": "vllm:iteration_tokens_total",
}


def _loads(raw, default):
    try:
        v = json.loads(raw)
        return v if v is not None else default
    except (TypeError, ValueError):
        return default


def p_from_bounds(bounds: list, counts: list[float], q: float) -> float | None:
    """Percentile from bucket counts, reported as the bucket's upper bound.

    `bounds` is the list of Prometheus `le` values with the final +Inf stored as
    None.  A result landing in that overflow bucket has no upper bound, so the
    last finite bound is returned as a floor -- callers should render it as
    "greater than" rather than an exact figure.
    """
    total = sum(counts)
    if total <= 0:
        return None
    target = q * total
    cum = 0.0
    finite = [b for b in bounds if b is not None]
    for i, c in enumerate(counts):
        cum += c
        if cum >= target:
            b = bounds[i] if i < len(bounds) else None
            if b is None:
                return float(finite[-1]) if finite else None
            return float(b)
    return float(finite[-1]) if finite else None


def hist_range(store, source: str, metric: str, start: float, end: float) -> dict:
    """Sum a histogram's per-interval bucket deltas over a window.

    Bucket deltas are additive, so this is exact for the stored resolution.
    Rows whose bucket layout differs (vLLM restarted with different config) are
    skipped rather than misaligned into the wrong buckets.
    """
    rows = store.query(
        "SELECT bounds,counts,observations,sum_value FROM vllm_hist"
        " WHERE source=? AND metric=? AND ts >= ? AND ts < ? ORDER BY ts",
        (source, metric, start, end))
    bounds: list | None = None
    counts: list[float] = []
    n = 0.0
    total = 0.0
    skipped = 0
    for r in rows:
        b = _loads(r["bounds"], None)
        c = _loads(r["counts"], None)
        if not b or not c or len(b) != len(c):
            skipped += 1
            continue
        if bounds is None:
            bounds, counts = b, list(c)
        elif b != bounds:
            skipped += 1
            continue
        else:
            for i, v in enumerate(c):
                counts[i] += v
        n += r["observations"] or 0
        total += r["sum_value"] or 0.0
    scale = 1000.0 if metric in SECONDS_HISTOGRAMS else 1.0
    return {"bounds": bounds or [], "counts": counts, "observations": n,
            "sum": total * scale, "scale": scale, "rows_skipped": skipped,
            "mean": (total * scale / n) if n else None}


def percentiles(store, source: str, metric: str, start: float, end: float,
                qs=(0.5, 0.9, 0.99)) -> dict:
    """Percentiles plus the exact mean.

    vLLM's bucket bounds are coarse in the seconds range (they step 10 -> 20 ->
    40 -> 80s), so a p90 can only be reported as "at most 40s".  `_sum` and
    `_count` are exact, however, so `mean` is exact and is the more useful
    headline for these histograms.  `bucketed: True` marks the percentiles as
    bucket upper bounds so callers can label them honestly.
    """
    h = hist_range(store, source, metric, start, end)
    out = {"metric": metric, "observations": h["observations"], "mean": h["mean"],
           "bucketed": True}
    for q in qs:
        p = p_from_bounds(h["bounds"], h["counts"], q)
        out[f"p{int(q * 100)}"] = None if p is None else p * h["scale"]
    return out


def first_with_data(store, source: str, candidates, start: float, end: float) -> dict:
    """Percentiles for the first candidate metric that has observations.

    Lets one code path serve vLLM versions that spell a metric differently.
    """
    fallback = None
    for metric in candidates:
        got = percentiles(store, source, metric, start, end)
        if got["observations"]:
            return got
        fallback = fallback or got
    return fallback or {"metric": None, "observations": 0, "mean": None,
                        "bucketed": True, "p50": None, "p90": None, "p99": None}


def counter_total(store, source: str, metric: str, start: float, end: float,
                  group_by_label: bool = False):
    """Sum of per-interval counter deltas over a window."""
    if group_by_label:
        rows = store.query(
            "SELECT labels, SUM(value) v FROM vllm_samples"
            " WHERE source=? AND metric=? AND ts >= ? AND ts < ? GROUP BY labels",
            (source, metric, start, end))
        return {r["labels"]: r["v"] or 0.0 for r in rows}
    rows = store.query(
        "SELECT SUM(value) v FROM vllm_samples"
        " WHERE source=? AND metric=? AND ts >= ? AND ts < ?",
        (source, metric, start, end))
    return (rows[0]["v"] or 0.0) if rows else 0.0


def gauge_stats(store, source: str, metric: str, start: float, end: float) -> dict:
    rows = store.query(
        "SELECT AVG(value) mean, MIN(value) lo, MAX(value) hi, COUNT(*) n"
        " FROM vllm_samples WHERE source=? AND metric=? AND ts >= ? AND ts < ?",
        (source, metric, start, end))
    r = rows[0] if rows else None
    last = store.query(
        "SELECT value FROM vllm_samples WHERE source=? AND metric=?"
        " AND ts >= ? AND ts < ? ORDER BY ts DESC LIMIT 1", (source, metric, start, end))
    peak = store.query(
        "SELECT MAX(value) hi FROM vllm_samples WHERE source=? AND metric=?"
        " AND ts >= ? AND ts < ?", (source, metric + ":max", start, end))
    hi = r["hi"] if r else None
    if peak and peak[0]["hi"] is not None:
        hi = max(hi or 0, peak[0]["hi"])
    return {"mean": r["mean"] if r else None, "min": r["lo"] if r else None,
            "max": hi, "last": last[0]["value"] if last else None,
            "n": r["n"] if r else 0}


def finish_reasons(store, source: str, start: float, end: float) -> list[dict]:
    """Completion outcomes: stop / length / abort / error / repetition."""
    totals = counter_total(store, source, "vllm:request_success_total", start, end, True)
    out = []
    for labels, value in totals.items():
        reason = None
        for part in labels.split(","):
            if part.startswith("finished_reason="):
                reason = part.split("=", 1)[1]
        out.append({"finished_reason": reason or "unknown", "count": value})
    out.sort(key=lambda x: -x["count"])
    return out


def instances(store) -> list[dict]:
    rows = store.query(
        "SELECT source,last_seen,engine_start,model,reachable,error,info_json,"
        "gpu_indices,gpu_source,gpu_ts FROM vllm_instances ORDER BY source")
    out = []
    for r in rows:
        d = dict(r)
        d["info"] = _loads(d.pop("info_json", None), None)
        # None means "could not attribute", which the UI shows differently from
        # an empty list ("attributed, and it holds no GPU").
        raw = d.pop("gpu_indices", None)
        d["gpu_indices"] = _loads(raw, None) if raw else None
        # How old the attribution is, so the UI can show it rather than implying
        # every reading is current.
        d["gpu_age_s"] = (time.time() - d["gpu_ts"]) if d.get("gpu_ts") else None
        d["reachable"] = bool(d["reachable"])
        d["stale_s"] = (time.time() - d["last_seen"]) if d["last_seen"] else None
        out.append(d)
    return out


def summary(store, source: str, start: float, end: float) -> dict:
    span = max(1e-9, end - start)
    success = finish_reasons(store, source, start, end)
    total_requests = sum(x["count"] for x in success)
    errored = sum(x["count"] for x in success
                  if x["finished_reason"] in ("error", "abort"))
    prompt_tok = counter_total(store, source, "vllm:prompt_tokens_total", start, end)
    gen_tok = counter_total(store, source, "vllm:generation_tokens_total", start, end)
    pc_q = counter_total(store, source, "vllm:prefix_cache_queries_total", start, end)
    pc_h = counter_total(store, source, "vllm:prefix_cache_hits_total", start, end)
    cached_tok = counter_total(store, source, "vllm:prompt_tokens_cached_total", start, end)
    drafts = counter_total(store, source, "vllm:spec_decode_num_drafts_total", start, end)
    draft_tok = counter_total(store, source, "vllm:spec_decode_num_draft_tokens_total",
                              start, end)
    accepted = counter_total(store, source, "vllm:spec_decode_num_accepted_tokens_total",
                             start, end)
    preempt = counter_total(store, source, "vllm:num_preemptions_total", start, end)

    return {
        "source": source, "start": start, "end": end, "span_s": span,
        # Token counters advance as tokens stream; the request counter advances
        # only on completion.  Over a short window the two therefore describe
        # overlapping but different sets of requests, and dividing one by the
        # other does NOT give tokens-per-request.
        "counters_aligned": False,
        # vLLM publishes no per-request rows, so percentiles always come from
        # its histograms.  Flagged so a caller never mistakes this for exact.
        "exact": False,
        "requests": total_requests,
        "req_per_s": total_requests / span,
        "errors": errored,
        "error_rate": (errored / total_requests) if total_requests else 0.0,
        "finish_reasons": success,
        "input_tokens": prompt_tok,
        "output_tokens": gen_tok,
        "in_tok_per_s": prompt_tok / span,
        "out_tok_per_s": gen_tok / span,
        "ttft_ms": percentiles(store, source, LATENCY_METRICS["ttft"], start, end),
        "itl_ms": first_with_data(store, source, ITL_CANDIDATES, start, end),
        "e2e_ms": percentiles(store, source, LATENCY_METRICS["e2e"], start, end),
        "queue_ms": percentiles(store, source, LATENCY_METRICS["queue"], start, end),
        "prefill_ms": percentiles(store, source, LATENCY_METRICS["prefill"], start, end),
        "decode_ms": percentiles(store, source, LATENCY_METRICS["decode"], start, end),
        "kv_cache_usage": gauge_stats(store, source, "vllm:kv_cache_usage_perc",
                                      start, end),
        "running": gauge_stats(store, source, "vllm:num_requests_running", start, end),
        "waiting": gauge_stats(store, source, "vllm:num_requests_waiting", start, end),
        "preemptions": preempt,
        "prefix_cache_hit_rate": (pc_h / pc_q) if pc_q else None,
        "prefix_cache_queries": pc_q,
        "cached_tokens": cached_tok,
        # Share of prompt tokens that did not have to be recomputed.  Distinct
        # from prefix_cache_hit_rate, which vLLM reports over queried tokens.
        "prompt_cache_hit_rate": (cached_tok / prompt_tok) if prompt_tok else None,
        "spec_decode": {
            "drafts": drafts, "draft_tokens": draft_tok, "accepted_tokens": accepted,
            # Acceptance rate is what actually explains vLLM throughput swings.
            "acceptance_rate": (accepted / draft_tok) if draft_tok else None,
            "mean_accepted_per_draft": (accepted / drafts) if drafts else None,
        },
    }


def timeseries(store, source: str, start: float, end: float,
               step: int | None = None) -> dict:
    """Bucketed series. Stored rows are per-minute, so step is clamped to 60s."""
    span = max(1.0, end - start)
    step = max(60, int(step or 60))
    nb = int(span // step) + 1
    base = int(start // step) * step
    keys = ("req_per_s", "out_tok_per_s", "in_tok_per_s", "ttft_p50", "ttft_p90",
            "ttft_p99", "queue_p90", "e2e_p90", "kv_cache_usage", "running",
            "waiting", "preemptions", "tpot_p50")
    series = {k: [None] * nb for k in keys}

    def bucket_of(ts):
        i = int((ts - base) // step)
        return i if 0 <= i < nb else None

    # counters -> rates
    for metric, key in (("vllm:request_success_total", "req_per_s"),
                        ("vllm:generation_tokens_total", "out_tok_per_s"),
                        ("vllm:prompt_tokens_total", "in_tok_per_s"),
                        ("vllm:num_preemptions_total", "preemptions")):
        rows = store.query(
            "SELECT ts, SUM(value) v FROM vllm_samples WHERE source=? AND metric=?"
            " AND ts >= ? AND ts < ? GROUP BY ts", (source, metric, start, end))
        for r in rows:
            i = bucket_of(r["ts"])
            if i is None:
                continue
            v = (r["v"] or 0.0)
            add = v if key == "preemptions" else v / step
            series[key][i] = (series[key][i] or 0.0) + add

    # gauges -> mean per bucket
    for metric, key in (("vllm:kv_cache_usage_perc", "kv_cache_usage"),
                        ("vllm:num_requests_running", "running"),
                        ("vllm:num_requests_waiting", "waiting")):
        rows = store.query(
            "SELECT ts, AVG(value) v FROM vllm_samples WHERE source=? AND metric=?"
            " AND ts >= ? AND ts < ? GROUP BY ts", (source, metric, start, end))
        for r in rows:
            i = bucket_of(r["ts"])
            if i is not None:
                series[key][i] = r["v"]

    # histograms -> percentiles per bucket
    for metric, targets in ((LATENCY_METRICS["ttft"],
                             (("ttft_p50", 0.5), ("ttft_p90", 0.9), ("ttft_p99", 0.99))),
                            (LATENCY_METRICS["queue"], (("queue_p90", 0.9),)),
                            (LATENCY_METRICS["e2e"], (("e2e_p90", 0.9),)),
                            (ITL_CANDIDATES[0], (("tpot_p50", 0.5),))):
        rows = store.query(
            "SELECT ts,bounds,counts FROM vllm_hist WHERE source=? AND metric=?"
            " AND ts >= ? AND ts < ? ORDER BY ts", (source, metric, start, end))
        agg: dict[int, tuple] = {}
        for r in rows:
            i = bucket_of(r["ts"])
            if i is None:
                continue
            b = _loads(r["bounds"], None)
            c = _loads(r["counts"], None)
            if not b or not c or len(b) != len(c):
                continue
            if i not in agg:
                agg[i] = (b, list(c))
            elif agg[i][0] == b:
                for j, v in enumerate(c):
                    agg[i][1][j] += v
        scale = 1000.0 if metric in SECONDS_HISTOGRAMS else 1.0
        for i, (b, c) in agg.items():
            for key, q in targets:
                p = p_from_bounds(b, c, q)
                if p is not None:
                    series[key][i] = p * scale

    return {"source": source, "start": base, "step": step, "n": nb, "exact": False,
            "t": [base + i * step for i in range(nb)], "series": series}
