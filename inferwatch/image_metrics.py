"""Query layer for the image-generation tables.

Kept apart from `metrics.py` for the same reason vLLM is: the unit of work is
different.  There are no tokens, no time-to-first-token and no context window,
so nothing here has an analogue there beyond "how long did it take and did it
fail".

There are no rollups for images.  Generations are per-item rows under the raw
retention window, and past it the honest answer is that nothing is stored --
`coverage()` reports the boundary rather than letting an empty range read as a
quiet week.

Durations are exact: they come from ComfyUI's own event timestamps on every
row, never from a histogram.
"""

from __future__ import annotations

import json
import time

from .metrics import _quantiles, pick_step

# A generation this much slower than the window's median is worth surfacing on
# its own; below that a "slowest" list is just noise from normal variation.
SLOW_FACTOR = 2.0


def _loads(raw, default):
    try:
        v = json.loads(raw)
        return v if v is not None else default
    except (TypeError, ValueError):
        return default


def _clause(source: str | None, params: list) -> str:
    if not source:
        return ""
    params.append(source)
    return " AND source = ?"


def coverage(store, start: float) -> dict:
    """Whether stored generations reach back as far as `start`."""
    row = store.query("SELECT MIN(ts) t FROM image_generations")
    first = row[0]["t"] if row and row[0]["t"] is not None else None
    return {"covers_from": first,
            "complete": first is not None and first <= start}


def sources(store) -> list[dict]:
    """Configured image sources that have ever reported, newest sample first."""
    rows = store.query(
        "SELECT source, MAX(ts) last_seen, COUNT(*) samples"
        " FROM image_samples GROUP BY source ORDER BY source")
    out = []
    for r in rows:
        out.append({"source": r["source"], "last_seen": r["last_seen"],
                    "samples": r["samples"],
                    "stale_s": (time.time() - r["last_seen"]) if r["last_seen"] else None})
    return out


def summary(store, source: str | None, start: float, end: float) -> dict:
    """Headline numbers: throughput, failure rate, how long a picture takes."""
    span = max(1e-9, end - start)
    params: list = [start, end]
    clause = _clause(source, params)
    rows = store.query(
        "SELECT status,total_ms,model,error_type FROM image_generations"
        f" WHERE ts >= ? AND ts < ?{clause}", tuple(params))

    done = [r for r in rows if r["status"] in ("success", "error")]
    errors = [r for r in done if r["status"] == "error"]
    durations = [r["total_ms"] for r in done if r["total_ms"] is not None]
    models = {r["model"] for r in done if r["model"]}

    # SwarmUI's prep-vs-gen split, which ComfyUI cannot report. Kept as its own
    # aggregate rather than joined onto a generation: a request for N images
    # produces N finish lines, so a 1:1 pairing would misattribute every batch.
    p2: list = [start, end]
    c2 = _clause(source, p2)
    timing = store.query(
        "SELECT AVG(prep_ms) prep_mean, MAX(prep_ms) prep_max,"
        " AVG(gen_ms) gen_mean, COUNT(*) n FROM image_events"
        f" WHERE kind='generation_finished' AND ts >= ? AND ts < ?{c2}", tuple(p2))[0]

    p3: list = [start, end]
    c3 = _clause(source, p3)
    ev = store.query(
        "SELECT kind, COUNT(*) n FROM image_events"
        f" WHERE ts >= ? AND ts < ?{c3} GROUP BY kind", tuple(p3))
    by_kind = {r["kind"]: r["n"] for r in ev}

    q = _quantiles(durations)
    return {
        "start": start, "end": end, "span_s": span,
        # Every duration is a real measurement, so unlike the token engines
        # there is no bucketed-percentile caveat to carry.
        "exact": True,
        "generations": len(done),
        "running": sum(1 for r in rows if r["status"] == "running"),
        "gen_per_hour": len(done) / (span / 3600.0),
        "errors": len(errors),
        "error_rate": (len(errors) / len(done)) if done else 0.0,
        "models_used": len(models),
        "duration_ms": q,
        "duration_ms_mean": (sum(durations) / len(durations)) if durations else None,
        "duration_ms_max": max(durations) if durations else None,
        # Wall-clock share actually spent generating, which is what a busy box
        # looks like versus an idle one with a few slow jobs.
        "busy_fraction": (sum(durations) / 1000.0 / span) if durations else 0.0,
        "swarm_timing": {
            "samples": timing["n"] or 0,
            "prep_ms_mean": timing["prep_mean"],
            "prep_ms_max": timing["prep_max"],
            "gen_ms_mean": timing["gen_mean"],
        },
        "webapi_errors": by_kind.get("webapi_error", 0),
        "backend_stderr": by_kind.get("backend_stderr", 0),
        "backend_restarts": by_kind.get("backend_up", 0),
    }


def timeseries(store, source: str | None, start: float, end: float,
               step: int | None = None) -> dict:
    """Bucketed series: throughput, duration, failures, and queue depth."""
    span = max(1.0, end - start)
    step = step or pick_step(span)
    nb = int(span // step) + 1
    base = int(start // step) * step
    keys = ("gen_per_hour", "err_per_hour", "dur_p50", "dur_p90",
            "queue_pending", "queue_running", "vram_used_gib")
    series: dict[str, list] = {k: [None] * nb for k in keys}
    counts = [0] * nb

    params: list = [start, end]
    clause = _clause(source, params)
    rows = store.query(
        "SELECT ts,status,total_ms FROM image_generations"
        f" WHERE ts >= ? AND ts < ?{clause}", tuple(params))
    buckets: dict[int, dict] = {}
    for r in rows:
        i = int((r["ts"] - base) // step)
        if not 0 <= i < nb:
            continue
        b = buckets.setdefault(i, {"n": 0, "err": 0, "dur": []})
        if r["status"] in ("success", "error"):
            b["n"] += 1
        if r["status"] == "error":
            b["err"] += 1
        if r["total_ms"] is not None:
            b["dur"].append(r["total_ms"])
    per_hour = 3600.0 / step
    for i, b in buckets.items():
        counts[i] = b["n"]
        series["gen_per_hour"][i] = b["n"] * per_hour
        series["err_per_hour"][i] = b["err"] * per_hour
        if b["dur"]:
            q = _quantiles(b["dur"], (0.5, 0.9))
            series["dur_p50"][i] = q["p50"]
            series["dur_p90"][i] = q["p90"]

    # Gauges: averaged within a bucket, summed across backends first so the
    # queue reads as the instance's depth rather than one backend's.
    p2: list = [start, end]
    c2 = _clause(source, p2)
    grows = store.query(
        "SELECT ts, SUM(queue_pending) qp, SUM(queue_running) qr,"
        " SUM(CASE WHEN vram_total IS NOT NULL AND vram_free IS NOT NULL"
        "          THEN vram_total - vram_free END) vram_used"
        " FROM image_samples"
        f" WHERE backend <> '' AND ts >= ? AND ts < ?{c2} GROUP BY ts", tuple(p2))
    acc: dict[int, dict] = {}
    for r in grows:
        i = int((r["ts"] - base) // step)
        if not 0 <= i < nb:
            continue
        a = acc.setdefault(i, {"qp": [], "qr": [], "vram": []})
        if r["qp"] is not None:
            a["qp"].append(r["qp"])
        if r["qr"] is not None:
            a["qr"].append(r["qr"])
        if r["vram_used"] is not None:
            a["vram"].append(r["vram_used"] / float(1 << 30))
    for i, a in acc.items():
        for key, vals in (("queue_pending", a["qp"]), ("queue_running", a["qr"]),
                          ("vram_used_gib", a["vram"])):
            if vals:
                series[key][i] = sum(vals) / len(vals)

    return {"start": base, "step": step, "n": nb, "exact": True,
            "t": [base + i * step for i in range(nb)],
            "counts": counts, "series": series}


def by_model(store, source: str | None, start: float, end: float) -> list[dict]:
    """Per-model generation counts, durations and failure rate."""
    params: list = [start, end]
    clause = _clause(source, params)
    rows = store.query(
        "SELECT model, COUNT(*) n,"
        " SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) err,"
        " AVG(total_ms) dur_mean, MAX(total_ms) dur_max, MIN(total_ms) dur_min,"
        " MAX(ts) last_used"
        " FROM image_generations"
        f" WHERE ts >= ? AND ts < ? AND status IN ('success','error'){clause}"
        " GROUP BY model ORDER BY n DESC", tuple(params))
    return [{"model": r["model"] or "(unknown)", "generations": r["n"],
             "errors": r["err"] or 0,
             "error_rate": (r["err"] or 0) / r["n"] if r["n"] else 0.0,
             "duration_ms_mean": r["dur_mean"], "duration_ms_max": r["dur_max"],
             "duration_ms_min": r["dur_min"], "last_used": r["last_used"]}
            for r in rows]


def backends(store, source: str | None = None) -> list[dict]:
    """The most recent sample for each backend, with its GPUs."""
    params: list = []
    clause = ""
    if source:
        # Qualified: both sides of the join carry a `source` column.
        clause = " WHERE s.source = ?"
        params.append(source)
    rows = store.query(
        "SELECT s.* FROM image_samples s JOIN ("
        "  SELECT source, backend, MAX(ts) ts FROM image_samples GROUP BY source, backend"
        ") m ON s.source=m.source AND s.backend=m.backend AND s.ts=m.ts"
        f"{clause} ORDER BY s.source, s.backend", tuple(params))
    out = []
    for r in rows:
        d = dict(r)
        raw = d.pop("gpu_indices", None)
        d["gpu_indices"] = _loads(raw, None) if raw else None
        used = None
        if d.get("vram_total") is not None and d.get("vram_free") is not None:
            used = d["vram_total"] - d["vram_free"]
        d["vram_used"] = used
        d["vram_used_pct"] = (used / d["vram_total"]) if used and d.get("vram_total") else None
        d["stale_s"] = (time.time() - d["ts"]) if d.get("ts") else None
        out.append(d)
    return out


def recent_generations(store, source: str | None, start: float, end: float,
                       limit: int = 50) -> list[dict]:
    params: list = [start, end]
    clause = _clause(source, params)
    params.append(limit)
    rows = store.query(
        "SELECT ts,source,prompt_id,status,total_ms,model,models_json,node_count,"
        "cached_nodes,error_node,error_type,error_message FROM image_generations"
        f" WHERE ts >= ? AND ts < ?{clause} ORDER BY ts DESC LIMIT ?", tuple(params))
    out = []
    for r in rows:
        d = dict(r)
        d["models"] = _loads(d.pop("models_json", None), [])
        out.append(d)
    return out


def slowest(store, source: str | None, start: float, end: float,
            limit: int = 10) -> list[dict]:
    params: list = [start, end]
    clause = _clause(source, params)
    params.append(limit)
    rows = store.query(
        "SELECT ts,prompt_id,model,total_ms,status,node_count FROM image_generations"
        f" WHERE ts >= ? AND ts < ? AND total_ms IS NOT NULL{clause}"
        " ORDER BY total_ms DESC LIMIT ?", tuple(params))
    return [dict(r) for r in rows]


def failures(store, source: str | None, start: float, end: float,
             limit: int = 50) -> dict:
    """Everything that went wrong, from both sides.

    A generation that failed inside ComfyUI and an API call that never reached
    a backend are different problems with different fixes, so they are returned
    apart rather than merged into one list.
    """
    params: list = [start, end]
    clause = _clause(source, params)
    params.append(limit)
    gens = store.query(
        "SELECT ts,prompt_id,model,error_node,error_type,error_message,total_ms"
        " FROM image_generations"
        f" WHERE ts >= ? AND ts < ? AND status='error'{clause}"
        " ORDER BY ts DESC LIMIT ?", tuple(params))

    p2: list = [start, end]
    c2 = _clause(source, p2)
    p2.append(limit)
    evs = store.query(
        "SELECT ts,kind,level,backend_index,route,msg FROM image_events"
        f" WHERE ts >= ? AND ts < ? AND kind IN"
        f" ('webapi_error','backend_stderr','problem'){c2}"
        " ORDER BY ts DESC LIMIT ?", tuple(p2))

    # Which node class fails most is the single most actionable number here:
    # it points at the workflow, not the machine.
    p3: list = [start, end]
    c3 = _clause(source, p3)
    top = store.query(
        "SELECT error_type, COUNT(*) n FROM image_generations"
        f" WHERE ts >= ? AND ts < ? AND status='error' AND error_type IS NOT NULL{c3}"
        " GROUP BY error_type ORDER BY n DESC LIMIT 10", tuple(p3))

    return {
        "generation_errors": [dict(r) for r in gens],
        "log_errors": [dict(r) for r in evs],
        "by_node_type": [{"error_type": r["error_type"], "count": r["n"]} for r in top],
    }


def events(store, source: str | None, start: float, end: float,
           kind: str | None = None, limit: int = 100) -> list[dict]:
    params: list = [start, end]
    clause = _clause(source, params)
    if kind:
        clause += " AND kind = ?"
        params.append(kind)
    params.append(limit)
    rows = store.query(
        "SELECT ts,source,kind,level,backend_index,model,user,route,prep_ms,gen_ms,msg"
        f" FROM image_events WHERE ts >= ? AND ts < ?{clause}"
        " ORDER BY ts DESC LIMIT ?", tuple(params))
    return [dict(r) for r in rows]
