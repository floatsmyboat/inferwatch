"""Collecting Strata: everything is polled, nothing is read from a log.

Strata is the one engine here whose own endpoint already holds per-request
records.  Its `/metrics` answers JSON (the Prometheus text, under vLLM's names,
only when asked for it) with three parts this module uses:

  * `engine`   -- the configuration it runs with: context, KV precision, how
                  much of a context stays in VRAM, the expert cache's size
  * `live`     -- the running request: state, queue, tokens read and made
  * `requests` -- a ring of the requests it finished, newest first, each with
                  its timings, the expert-cache hit rate, and the share of
                  routed experts that had to cross PCIe

So a request row here is the server's record rather than a reconstruction from
two log lines, and the ring is re-read on every poll: identity is the start
time Strata stamps on it (see store.strata_request_key).

The engine also shares its port with whatever else a switcher rotates onto it.
`/health` says `"service": "strata"` when it is Strata; anything else answering
is reported as such and its numbers are not recorded under this source.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request

from . import gpuproc

log = logging.getLogger("inferwatch.strata")

# /metrics returns the last 12 requests unless asked for every one kept (500).
RING_DEFAULT = 12


def fetch_json(url: str, api_key: str = "", timeout: float = 5.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def request_row(source: str, model: str | None, h: dict) -> dict | None:
    """One entry of /metrics `requests` as a stored row; None if it has no start.

    TTFT is the engine's own prompt time.  It leaves out the HTTP and
    tokenisation overhead before the engine starts reading, which on a long
    prompt is a rounding error and on a short one is most of the difference
    from what a client measures.
    """
    started = _num(h.get("time"))
    if started is None:
        return None
    dur = _num(h.get("duration_s"))
    prompt_ms = _num(h.get("prompt_ms"))
    decode_ms = _num(h.get("decode_ms"))
    read = _num(h.get("prompt_read"))
    if read is None:
        read = _num(h.get("prompt_tokens"))
    offered = _num(h.get("drafts_offered"))
    accepted = _num(h.get("drafts_accepted"))
    return {
        "ts": started + (dur or 0.0),
        "started_ts": started,
        "source": source,
        "model": model,
        "finish": h.get("finish"),
        "prompt_tokens": _num(h.get("prompt_tokens")),
        "cached_tokens": _num(h.get("reused")),
        "prompt_tokens_total": _num(h.get("prompt_total")),
        "output_tokens": _num(h.get("output_tokens")),
        "ttft_ms": prompt_ms,
        "latency_ms": dur * 1000.0 if dur is not None else None,
        "prefill_ms": prompt_ms,
        "decode_ms": decode_ms,
        "prefill_tps": (read / (prompt_ms / 1000.0)) if read and prompt_ms else None,
        "decode_tps": _num(h.get("decode_tok_s")),
        "expert_hit_rate": _num(h.get("hit_rate")),
        "pcie_share": _num(h.get("pcie_share")),
        "ram_blobs": _num(h.get("ram_blobs")),
        "file_blobs": _num(h.get("file_blobs")),
        "file_mb": _num(h.get("file_mb")),
        "drafts_offered": offered,
        "drafts_accepted": accepted,
        "draft_accept": (accepted / offered) if offered and accepted is not None else None,
        "reasoning_recoveries": _num(h.get("reasoning_recoveries")),
    }


def sample_row(source: str, ts: float, m: dict) -> dict:
    """The between-requests state of one /metrics reading.

    kv_tokens follows Strata's own definition behind vllm:kv_cache_usage_perc:
    the running request's prompt plus what it has generated, zero when idle.
    """
    engine = m.get("engine") or {}
    live = m.get("live") or {}
    hw = m.get("hardware") or {}
    cc = m.get("conversation_cache") or {}
    state = live.get("state")
    busy = state in ("reading", "generating")
    kv = ((_num(live.get("prompt_tokens")) or 0) + (_num(live.get("generated")) or 0)
          if busy else 0)
    running = _num(live.get("running"))
    if running is None:
        running = 1 if busy else 0
    queued = (_num(live.get("queued")) or 0) + (_num(live.get("waiting")) or 0)
    return {
        "ts": ts, "source": source, "state": state,
        "running": running, "queued": queued,
        "kv_tokens": kv,
        "kv_max": _num(engine.get("max_context")) or _num(engine.get("context")),
        "kv_resident": _num(engine.get("kv_resident")),
        "prompt_read": _num(live.get("prompt_read")),
        "prompt_total": _num(live.get("prompt_total")),
        # tok_s lingers after a request ends; only a generating engine decodes.
        "decode_tps": _num(live.get("tok_s")) if state == "generating" else None,
        "prefill_tps": _num(live.get("prefill_tok_s_mean")) if busy else None,
        "host_pcie_rx_mb": _num(hw.get("gpu_pcie_rx_mb")),
        "host_pcie_tx_mb": _num(hw.get("gpu_pcie_tx_mb")),
        "cpu_pct": _num(hw.get("cpu")),
        "ram_used_bytes": _num(hw.get("ram_used")),
        "ram_total_bytes": _num(hw.get("ram_total")),
        "conv_cache_bytes": _num(cc.get("bytes")),
        "conv_cache_parked": _num(cc.get("parked")),
    }


def instance_fields(m: dict, status: dict | None, health: dict | None) -> dict:
    engine = m.get("engine") or {}
    status = status or {}
    health = health or {}
    conc = status.get("concurrency") or {}
    return {
        "model": engine.get("model") or health.get("model"),
        "version": engine.get("version") or engine.get("engine") or status.get("engine"),
        "started": _num(status.get("started")),
        "max_context": _num(engine.get("max_context")) or _num(engine.get("context")),
        "kv_type": engine.get("kv"),
        "kv_resident": _num(engine.get("kv_resident")),
        "slots": _num(conc.get("serving")),
        "expert_slots": _num(engine.get("expert_slots")),
        "expert_cache_mib": _num(engine.get("expert_cache_mib")),
        "arena_mib": _num(engine.get("arena_mib")),
        "vram_free_mib": _num(engine.get("vram_free_mib")),
        "spec": _num(engine.get("spec")),
        "vision": 1 if health.get("images") else 0,
    }


class StrataPoller:
    """Polls /health, /v1/status and /metrics; stores requests, samples, state."""

    GPU_RECHECK_S = 60.0
    # The engine's name in events and the /metrics mappers; a subclass
    # polling a relative of Strata's server overrides them.
    ENGINE = "strata"
    request_row = staticmethod(request_row)
    sample_row = staticmethod(sample_row)
    instance_fields = staticmethod(instance_fields)

    def __init__(self, store, source: str, cfg: dict, interval_getter=None,
                 on_live=None, on_event=None):
        self.store = store
        self.source = source
        self.url = (cfg.get("url") or "http://127.0.0.1:8001").rstrip("/")
        self.api_key = cfg.get("api_key") or ""
        self.unit = (cfg.get("unit") or "").strip()
        self.interval_getter = interval_getter or (lambda: 10.0)
        self.on_live = on_live or (lambda payload: None)
        self.on_event = on_event or (lambda row: None)
        self.reachable = False
        self.is_strata = False
        self.error: str | None = None
        self.model: str | None = None
        self.started: float | None = None
        self.gpu_indices: list[int] | None = None
        self.gpu_source = "unavailable"
        self.gpu_ts: float | None = None
        self._gpu_checked = 0.0
        # Newest request start already stored, so a poll can tell whether the
        # default 12-entry ring might have skipped some since the last one.
        self.newest_started: float | None = None
        self.stats = {"requests": 0, "samples": 0, "full_reads": 0}
        self.last: dict = {}

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
        service = (out["health"] or {}).get("service")
        if service != "strata":
            # Another engine on the shared port (a switcher variant): say so,
            # and record nothing of it under this source.
            out["error"] = f"port answered by another server (service={service!r})"
            return out
        out["is_strata"] = True
        try:
            out["status"] = fetch_json(f"{self.url}/v1/status", self.api_key)
        except (urllib.error.URLError, OSError, ValueError):
            pass            # optional: slot count and start time only
        try:
            m = fetch_json(f"{self.url}/metrics", self.api_key)
            if self._ring_may_have_gap(m):
                m = fetch_json(f"{self.url}/metrics?requests=all", self.api_key)
                self.stats["full_reads"] += 1
            out["metrics"] = m
        except (urllib.error.URLError, OSError, ValueError) as e:
            out["error"] = f"/metrics: {type(e).__name__}: {e}"
        return out

    def _ring_may_have_gap(self, m: dict) -> bool:
        """True when the short ring is full and its oldest entry is newer than
        anything stored: requests older than it, and newer than the last poll,
        may exist that only the full list holds.  On the first poll after a
        start this also backfills what the server still keeps."""
        ring = m.get("requests") or []
        if len(ring) < RING_DEFAULT:
            return False
        oldest = min((_num(h.get("time")) or 0.0) for h in ring)
        return self.newest_started is None or oldest > self.newest_started

    def resolve_gpus(self, now: float) -> list[int] | None:
        if now - self._gpu_checked < self.GPU_RECHECK_S:
            return self.gpu_indices
        self._gpu_checked = now
        self.gpu_ts = now
        port = gpuproc.port_of(self.url) if gpuproc.is_local(self.url) else None
        try:
            self.gpu_indices, self.gpu_source = gpuproc.resolve(
                unit=self.unit or None, port=port)
        except Exception:
            log.debug("gpu attribution failed for %r", self.source, exc_info=True)
            self.gpu_indices, self.gpu_source = None, "unavailable"
        return self.gpu_indices

    def ingest(self, ts: float, got: dict) -> dict:
        """Store one probe's worth; returns the instance fields written."""
        self.reachable = got["reachable"]
        self.is_strata = got["is_strata"]
        self.error = got["error"]
        m = got.get("metrics")
        fields: dict = {}
        if m:
            fields = self.instance_fields(m, got.get("status"), got.get("health"))
            self.model = fields.get("model") or self.model
            if fields.get("started") and self.started and fields["started"] != self.started:
                self.on_event({"ts": ts, "kind": "start", "source": self.source,
                               "model": self.model,
                               "msg": f"{self.ENGINE} {fields.get('version') or ''} restarted".strip()})
            self.started = fields.get("started") or self.started
            new = 0
            for h in m.get("requests") or []:
                row = self.request_row(self.source, self.model, h)
                if row is None:
                    continue
                if self.store.insert_strata_request(row):
                    new += 1
                    self.on_live({"type": "strata_request", **row})
                if self.newest_started is None or row["started_ts"] > self.newest_started:
                    self.newest_started = row["started_ts"]
            self.stats["requests"] += new
            self.store.insert_strata_sample(self.sample_row(self.source, ts, m))
            self.stats["samples"] += 1
        self.store.upsert_strata_instance(
            self.source, last_seen=ts,
            reachable=1 if (self.reachable and self.is_strata) else 0,
            error=self.error,
            gpu_indices=(json.dumps(self.gpu_indices)
                         if self.gpu_indices is not None else None),
            gpu_source=self.gpu_source, gpu_ts=self.gpu_ts,
            **{k: v for k, v in fields.items()},
            **({} if fields else {"model": self.model}))
        self.store.commit()
        return fields

    def _resume_point(self) -> None:
        if self.newest_started is not None:
            return
        row = self.store.query("SELECT MAX(started_ts) t FROM strata_requests"
                               " WHERE source = ?", (self.source,))
        self.newest_started = row[0]["t"] if row and row[0]["t"] else None

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._resume_point)
        while True:
            try:
                ts = time.time()
                got = await loop.run_in_executor(None, self.probe)
                if got["is_strata"]:
                    await loop.run_in_executor(None, self.resolve_gpus, ts)
                fields = await loop.run_in_executor(None, self.ingest, ts, got)
                m = got.get("metrics") or {}
                self.last = {"ts": ts, "reachable": self.reachable,
                             "is_strata": self.is_strata, "error": self.error,
                             "model": self.model, "gpu_indices": self.gpu_indices,
                             "gpu_source": self.gpu_source,
                             "live": (m.get("live") or {}).get("state"),
                             "version": fields.get("version")}
                self.on_live({"type": "strata_status", **self.last})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("strata poller tick failed")
            await asyncio.sleep(max(1.0, float(self.interval_getter())))
