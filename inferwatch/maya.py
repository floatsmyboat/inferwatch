"""Collecting Maya: Strata's engine behind Project Maya's server, polled.

Maya (github.com/mw00/project-maya) runs GLM-5.3-Flash and its relatives on
the engine Strata grew from, and its server's `/metrics` is the same JSON
shape -- `engine`, `live`, and a ring of finished `requests` -- so the Strata
poller does almost all of the work and the rows go into the Strata tables
under this source's name.  What differs:

  * identity: `/health` carries no `service`; `/v1/status` says
    `"service": "maya"`
  * `live.active` is the requests being answered at once (`--parallel N`),
    and `live.tok_s` is already SUMMED over them -- the aggregate decode rate
    a several-client test is after, which no per-request row can give
  * a `tiers` block: the engine's latest STAT line -- the share of routed
    experts VRAM held, and per token the experts read from RAM, read from
    disk and promoted into VRAM.  Disk reads matter here: the model lives on
    an NFS share, so a disk read is a network read.
  * each request row carries its own prompt rate (`prefill_tok_s`, reused
    prompt excluded) and no PCIe share or draft counts
"""

from __future__ import annotations

import urllib.error

from .strata import StrataPoller, _num, fetch_json, instance_fields, request_row, sample_row


def maya_request_row(source: str, model: str | None, h: dict) -> dict | None:
    row = request_row(source, model, h)
    if row is not None and _num(h.get("prefill_tok_s")) is not None:
        row["prefill_tps"] = _num(h.get("prefill_tok_s"))
    return row


def maya_sample_row(source: str, ts: float, m: dict) -> dict:
    row = sample_row(source, ts, m)
    live = m.get("live") or {}
    active = _num(live.get("active"))
    if active is not None and row["state"] in ("reading", "generating"):
        row["running"] = active
    row["writing"] = _num(live.get("writing"))
    now = ((m.get("tiers") or {}).get("now")) or {}
    busy = row["state"] in ("reading", "generating")
    row["vram_hit"] = _num(now.get("vram_hit")) if busy else None
    row["ram_fetch"] = _num(now.get("ram_fetch")) if busy else None
    row["disk_fetch"] = _num(now.get("disk")) if busy else None
    row["promo"] = _num(now.get("promo")) if busy else None
    return row


def maya_instance_fields(m: dict, status: dict | None, health: dict | None) -> dict:
    out = instance_fields(m, status, health)
    engine = m.get("engine") or {}
    out["model"] = engine.get("model_label") or out.get("model")
    out["version"] = engine.get("maya_version") or out.get("version")
    if out.get("max_context") is None:
        out["max_context"] = _num((health or {}).get("max_context"))
    return out


class MayaPoller(StrataPoller):
    """StrataPoller with Maya's identity check, its live aggregate and tiers."""

    ENGINE = "maya"
    request_row = staticmethod(maya_request_row)
    sample_row = staticmethod(maya_sample_row)
    instance_fields = staticmethod(maya_instance_fields)

    def probe(self) -> dict:
        out: dict = {"reachable": False, "is_strata": False, "error": None,
                     "health": None, "status": None, "metrics": None}
        try:
            health = fetch_json(f"{self.url}/health", self.api_key)
            out["reachable"] = True
            out["health"] = health if isinstance(health, dict) else {}
        except (urllib.error.URLError, OSError, ValueError) as e:
            out["error"] = f"{type(e).__name__}: {e}"
            return out
        try:
            status = fetch_json(f"{self.url}/v1/status", self.api_key)
        except (urllib.error.URLError, OSError, ValueError):
            status = None   # vLLM and friends have no /v1/status
        service = (status or {}).get("service") if isinstance(status, dict) else None
        if service != "maya":
            out["error"] = f"port answered by another server (service={service!r})"
            return out
        # is_strata is the base class's "the expected engine answered".
        out["is_strata"] = True
        out["status"] = status
        try:
            m = fetch_json(f"{self.url}/metrics", self.api_key)
            if self._ring_may_have_gap(m):
                m = fetch_json(f"{self.url}/metrics?requests=all", self.api_key)
                self.stats["full_reads"] += 1
            out["metrics"] = m
        except (urllib.error.URLError, OSError, ValueError) as e:
            out["error"] = f"/metrics: {type(e).__name__}: {e}"
        return out
