"""Query layer for Maya: the Strata queries, plus what only Maya reports.

Maya's rows live in the Strata tables (see maya.py), so summary, timeseries,
recent requests and the rest are strata_metrics' own.  Two things are added:

  * the expert tiers over time -- VRAM hit rate, and per token the experts
    read from RAM, read from disk and promoted into VRAM
  * decode throughput by concurrency: the samples grouped by how many
    requests were running, each group's mean AGGREGATE rate (Maya's live
    tok_s is summed over the running requests) and that divided by the count.
    That is the number a "how fast with three clients" question wants, and a
    per-request row cannot give it: each row only knows its own speed.
"""

from __future__ import annotations

from . import metrics as _m


def tier_timeseries(store, source: str, start: float, end: float,
                    step: int | None = None) -> dict:
    span = max(1.0, end - start)
    step = step or _m.pick_step(span)
    base = int(start // step) * step
    n = max(1, int((end - base) // step) + 1)
    t = [base + i * step for i in range(n)]
    cols = ("vram_hit", "ram_fetch", "disk_fetch", "promo", "writing")
    acc: dict[int, list] = {}
    for r in store.query(
            "SELECT ts, " + ", ".join(cols) +
            "  FROM strata_samples WHERE source=? AND ts >= ? AND ts < ?",
            (source, start, end)):
        acc.setdefault(min(n - 1, max(0, int((r["ts"] - base) // step))), []).append(r)

    def mean(key):
        out = []
        for i in range(n):
            vals = [x[key] for x in acc.get(i, ()) if x[key] is not None]
            out.append((sum(vals) / len(vals)) if vals else None)
        return out

    return {"t": t, "step": step, **{c: mean(c) for c in cols}}


def by_concurrency(store, source: str, start: float, end: float) -> list[dict]:
    """Generating samples grouped by requests running: aggregate and per-request
    decode rate, and how many samples (poll ticks) each figure rests on."""
    rows = store.query(
        "SELECT running, COUNT(*) n, AVG(decode_tps) agg, MAX(decode_tps) peak,"
        "       AVG(vram_hit) vram_hit, AVG(disk_fetch) disk_fetch"
        "  FROM strata_samples WHERE source=? AND ts >= ? AND ts < ?"
        "   AND state='generating' AND decode_tps IS NOT NULL AND running > 0"
        " GROUP BY running ORDER BY running", (source, start, end))
    out = []
    for r in rows:
        d = dict(r)
        d["per_request"] = (d["agg"] / d["running"]) if d["agg"] is not None else None
        out.append(d)
    return out
