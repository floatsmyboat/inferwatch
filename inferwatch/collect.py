"""Journal follower, request correlator, and GPU/model pollers.

THE CORRELATION PROBLEM
-----------------------
Ollama's logs split one logical request across three voices that never share an
id.  For a single /v1/chat/completions call this host emits, in order:

    slot ... | task 6763 | new prompt, ... task.n_tokens = 1408
    slot ... | task 6763 | prompt eval time = 1254.52 ms / 55 tokens (...)
    slot ... | task 6763 | eval time = 14591.31 ms / 416 tokens (...)
    slot ... | task 6763 | total time = 15845.83 ms / 471 tokens
    slot ... | task 6763 | stop processing: n_tokens = 131071, truncated = 1
    [GIN] ... | 200 | 16.862061865s | 192.0.2.10 | POST "/v1/chat/completions"
    time=... msg="context for request finished" runner.name=.../llama3.2:3b

The task id is llama.cpp's; the access line has the status/client/endpoint but
no task id; the model NAME only appears afterwards on the scheduler line.  We
join them by arrival order: a task whose `total time` has printed is queued,
and the next inference access line claims the oldest queued task.

That join is exact while one request is in flight.  With OLLAMA_NUM_PARALLEL=2
(this host) two tasks can finish before either access line prints, and nothing
in the log disambiguates them -- so those rows are stored with
attribution='ambiguous' rather than pretending.  Per-request numbers stay
correct in aggregate either way; only the model/client pairing is uncertain.
Set attribution='none' when an inference request had no timings at all (it
failed before reaching the runner, e.g. the 500s in this host's history).

THE PROMPT CACHE, WHICH IS NOT A REQUEST
----------------------------------------
Ollama also logs the state of its prompt cache -- the pool of saved prompt
states that lets a returning conversation skip prefill:

    srv  update:  - cache state: 30 prompts, 8010.969 MiB (limits: 8192.000 MiB, ...)
    srv  get_availabl: prompt cache update took 364.22 ms

That is a GAUGE, and it belongs to the runner rather than to any one request,
so it is emitted through `on_cache` into its own table instead of being forced
into a `requests` row.  It is also sampled on ollama's schedule, not ours: a
line appears when ollama happens to run a cache update, which is bursty.  Rates
derived from it are therefore honest only about the moments it was sampled,
which is why the eviction counters are stored as DELTAS between samples rather
than as monotonic totals -- a runner restart resets llama.cpp's own counters,
and a delta cannot go negative across one.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import re
import shutil
import subprocess
import time
import urllib.request

from . import gpuproc
from .models import ModelIndex
from .parse import parse_line

log = logging.getLogger("inferwatch.collect")

# A finished task waits this long for an access line to claim it before being
# written unattributed.  Generous: the access line follows within milliseconds.
JOIN_TIMEOUT_S = 30.0
# A joined row waits this long for the scheduler line that names the model.
MODEL_TIMEOUT_S = 2.0
# How long a task that printed "total time" may wait for its "release" line
# (which carries the truncation flag) before being retired without it.
TOTAL_GRACE_S = 5.0
# How long a prompt-cache state line waits for the "update took Nms" line that
# follows it, before being stored without the cost figure.
CACHE_GRACE_S = 10.0


class Correlator:
    """Turns a stream of (timestamp, message) into request rows and events.

    Pure and synchronous -- no I/O -- so a captured journal can be replayed
    through it in tests and produce byte-identical rows.
    """

    def __init__(self, on_request=None, on_event=None, on_live=None,
                 on_cache=None, model_index: ModelIndex | None = None):
        self.on_request = on_request or (lambda r: None)
        self.on_event = on_event or (lambda e: None)
        self.on_live = on_live or (lambda x: None)
        self.on_cache = on_cache or (lambda c: None)
        # Resolves the blob digests on load lines to real model names.
        self.model_index = model_index

        self.tasks: dict[int, dict] = {}          # task_id -> accumulating metrics
        self.finished: collections.deque = collections.deque()  # awaiting access line
        self.awaiting_model: collections.deque = collections.deque()  # awaiting model name
        self.blob_to_model: dict[str, str] = {}
        self.loaded_models: set[str] = set()
        self.loaded_count: int = 0
        self.recent_model: str | None = None
        self.pending_load: dict | None = None
        self._last_load: tuple[str | None, float] = (None, 0.0)
        self.last_ts: float = 0.0
        self.stats = collections.Counter()

        # -- prompt cache -------------------------------------------------
        # A state line held until its update-cost line arrives.
        self.pending_cache: dict | None = None
        # Counted between samples, then reset; see the module docstring.
        self.cache_delta = collections.Counter()
        # Highest checkpoint index seen since the last sample, and the cap the
        # runner reports ("2 of 32").  The cap is a constant, so it survives a
        # sample; the high-water mark does not.
        self.ckpt_used: int = 0
        self.ckpt_total: int | None = None
        self.last_cache: dict | None = None
        # Load-time KV sizing, buffered until the load event it belongs to.
        self.pending_kv: dict = {}
        # pid -> when it was last mentioned, for the llama-server processes
        # ollama spawned.  Ollama logs `runner.pid` on its scheduler lines, so
        # this is an exact GPU-attribution key that costs nothing to keep (it
        # was parsed and discarded before).  A dead pid is harmless -- it simply
        # will not appear among nvidia-smi's compute processes -- so entries are
        # pruned by age rather than matched to an unload.
        self.runner_pids: dict[int, float] = {}

    # -- public entry point --------------------------------------------------

    def feed(self, ts: float, msg: str) -> None:
        self.last_ts = ts
        ev = parse_line(msg)
        if ev is None:
            return
        kind = ev["kind"]
        if kind == "gin":
            self._on_gin(ts, ev)
        elif kind == "go":
            self._on_go(ts, ev)
        elif kind in ("cache_state", "cache_update", "cache_save"):
            self._on_srv(ts, ev)
        elif kind in ("kv_size", "kv_buffer"):
            self._on_kv(ev)
        else:
            self._on_slot(ts, ev)
        self._expire(ts)

    def tick(self, now: float | None = None) -> None:
        """Flush rows whose join/model windows have elapsed (call periodically)."""
        self._expire(now if now is not None else time.time(), force_model=True)

    @property
    def inflight(self) -> int:
        return len(self.tasks)

    # -- slot lines ----------------------------------------------------------

    def _on_slot(self, ts: float, ev: dict) -> None:
        kind = ev["kind"]
        # Checkpoint bookkeeping describes the runner's cache, not one request,
        # and llama.cpp sometimes logs it against task -1, so it is handled
        # ahead of the per-task guard below rather than being dropped by it.
        if kind in ("ckpt_create", "ckpt_restore", "ckpt_evict"):
            self._on_ckpt(ev)
            return
        tid = ev["task_id"]
        if tid < 0:  # task -1: slot bookkeeping, not a request
            return

        if kind == "gen_progress":
            # Live decode rate, mid-stream.  Not stored per row; drives the
            # realtime view only.
            self.on_live({"type": "gen", "ts": ts, "task_id": tid,
                          "n_gen": ev["n_gen"], "tps": ev["tg"], "tps_3s": ev["tg_3s"],
                          "model": self.tasks.get(tid, {}).get("model") or self.recent_model})
            return

        t = self.tasks.get(tid)
        if t is None:
            t = self.tasks[tid] = {"task_id": tid, "slot_id": ev.get("slot_id"),
                                   "first_ts": ts, "model": None}
        t["slot_id"] = ev.get("slot_id", t.get("slot_id"))

        if kind == "launch":
            t["launch_ts"] = ts
        elif kind == "new_prompt":
            t["prompt_tokens_total"] = ev["prompt_tokens_total"]
            t["n_ctx_slot"] = ev.get("n_ctx_slot")
        elif kind == "prefill_progress":
            t["prefill_tps"] = ev.get("prefill_tps")
        elif kind == "prompt_eval":
            t["ttft_ms"] = ev["ttft_ms"]
            t["prompt_tokens"] = ev["prompt_tokens"]
            t["prefill_tps"] = ev["prefill_tps"]
        elif kind == "eval":
            t["decode_ms"] = ev["decode_ms"]
            t["output_tokens"] = ev["output_tokens"]
            t["decode_tps"] = ev["decode_tps"]
        elif kind == "draft":
            t["draft_accept"] = ev["draft_accept"]
            t["draft_mean_len"] = ev["draft_mean_len"]
        elif kind == "total":
            t["total_ms"] = ev["total_ms"]
            t["done_ts"] = ts
        elif kind == "release":
            t["context_tokens"] = ev["context_tokens"]
            t["truncated"] = ev["truncated"]
            if ev["truncated"]:
                self.on_event({"ts": ts, "kind": "truncation", "level": "WARN",
                               "model": t.get("model") or self.recent_model,
                               "source": "llama-server",
                               "msg": f"context truncated at {ev['context_tokens']} tokens",
                               "detail": {"task_id": tid}})
            # release is the last slot line for the task: queue it for joining.
            self._retire(tid, ts)
            return

    def _retire(self, tid: int, ts: float) -> None:
        t = self.tasks.pop(tid, None)
        if t is None:
            return
        if "total_ms" not in t and "ttft_ms" not in t:
            return  # nothing measurable; drop it
        t["retire_ts"] = ts
        self.finished.append(t)
        self.stats["tasks_retired"] += 1

    # -- prompt cache --------------------------------------------------------

    def _on_ckpt(self, ev: dict) -> None:
        kind = ev["kind"]
        if kind == "ckpt_create":
            # "created context checkpoint 2 of 32": the index is how many are
            # live, the total is the cap the runner was built with.
            self.ckpt_used = max(self.ckpt_used, ev.get("ckpt_index") or 0)
            self.ckpt_total = ev.get("ckpt_total") or self.ckpt_total
            self.cache_delta["ckpt_created"] += 1
        elif kind == "ckpt_restore":
            self.cache_delta["restores"] += 1
        else:
            self.cache_delta["evictions"] += 1
            self.cache_delta["evict_" + (ev.get("reason") or "other")] += 1

    def _on_srv(self, ts: float, ev: dict) -> None:
        kind = ev["kind"]
        if kind == "cache_state":
            # A state line still pending means its cost line never arrived.
            # Emit that one as it stands rather than overwriting it.
            self._flush_cache()
            self.pending_cache = {
                "ts": ts, "model": self.recent_model, "prompts": ev["prompts"],
                "used_mib": ev["used_mib"], "limit_mib": ev["limit_mib"],
                "usage": ev["usage"], "token_limit": ev["token_limit"],
                "est_tokens": ev["est_tokens"],
            }
        elif kind == "cache_update":
            if self.pending_cache is not None:
                self.pending_cache["update_ms"] = ev["update_ms"]
                self._flush_cache()
            else:
                # An update that reported no state change still cost real time,
                # so the cost is recorded on its own with the gauges left null.
                self._emit_cache({"ts": ts, "model": self.recent_model,
                                  "update_ms": ev["update_ms"]})
        elif kind == "cache_save":
            self.cache_delta["saves"] += 1

    def _take_deltas(self) -> dict:
        """Counters accumulated since the previous sample, then reset."""
        keys = ("evictions", "evict_crowded", "evict_invalidated", "restores",
                "ckpt_created", "saves")
        out = {k: int(self.cache_delta[k]) for k in keys}
        self.cache_delta.clear()
        return out

    def _flush_cache(self) -> None:
        row = self.pending_cache
        if row is None:
            return
        self.pending_cache = None
        row["ckpt_used"] = self.ckpt_used or None
        row["ckpt_total"] = self.ckpt_total
        self.ckpt_used = 0
        self._emit_cache(row)

    def _emit_cache(self, row: dict) -> None:
        row.update(self._take_deltas())
        self.last_cache = row
        self.stats["cache_samples"] += 1
        self.on_cache(row)
        self.on_live({"type": "cache", **row})

    # -- load-time KV sizing -------------------------------------------------

    def _on_kv(self, ev: dict) -> None:
        """Buffer the `llama_kv_cache:` lines a load prints.

        They arrive between "starting llama-server" and "llama-server started
        in Ns", so they are held here and attached to the load event, which is
        the only row that describes a load.
        """
        if ev["kind"] == "kv_buffer":
            self.pending_kv.setdefault("buffers_mib", {})[ev["device"]] = ev["kv_mib"]
        else:
            self.pending_kv.update({k: v for k, v in ev.items() if k != "kind"})

    def _kv_detail(self) -> dict:
        """The buffered KV sizing, with the GPU/CPU split stated outright.

        A KV cache that did not fit in VRAM is the loudest single explanation
        for a model that decodes slowly, so the split is computed here rather
        than left for a reader to add up from device names.
        """
        kv = dict(self.pending_kv)
        buffers = kv.get("buffers_mib") or {}
        if buffers:
            total = sum(buffers.values())
            cpu = sum(v for dev, v in buffers.items() if dev.upper() == "CPU")
            kv["cpu_mib"] = cpu
            kv["gpu_mib"] = total - cpu
            kv["cpu_fraction"] = (cpu / total) if total else None
        return kv

    # -- access lines --------------------------------------------------------

    def _on_gin(self, ts: float, ev: dict) -> None:
        cls = ev["class"]
        row = {
            "ts": ts,
            "endpoint": ev["endpoint"],
            "method": ev["method"],
            "class": cls,
            "status": ev["status"],
            "client_ip": ev["client_ip"],
            "latency_ms": ev["latency_ms"],
        }
        if ev["latency_ms"] is not None:
            row["started_ts"] = ts - ev["latency_ms"] / 1000.0

        if cls in ("inference", "embed"):
            claimed = self.finished.popleft() if self.finished else None
            if claimed is not None:
                ambiguous = len(self.finished) > 0
                row.update({k: v for k, v in claimed.items() if k in (
                    "ttft_ms", "decode_ms", "total_ms", "prompt_tokens",
                    "prompt_tokens_total", "output_tokens", "prefill_tps", "decode_tps",
                    "draft_accept", "draft_mean_len", "truncated", "context_tokens",
                    "n_ctx_slot", "slot_id", "task_id")})
                if row.get("prompt_tokens_total") is not None and row.get("prompt_tokens") is not None:
                    row["cached_tokens"] = max(0, row["prompt_tokens_total"] - row["prompt_tokens"])
                if row.get("latency_ms") is not None and row.get("total_ms") is not None:
                    # Wall clock minus runner time = scheduler + queue + HTTP.
                    row["queue_ms"] = max(0.0, row["latency_ms"] - row["total_ms"])
                row["attribution"] = "ambiguous" if ambiguous else "exact"
                row["model"] = claimed.get("model")
                self.stats["joined_" + row["attribution"]] += 1
            else:
                row["attribution"] = "none"
                self.stats["joined_none"] += 1
            # Hold briefly: the model name arrives on the NEXT line.
            self.awaiting_model.append((ts, row))
        else:
            self._emit(row)

    # -- go lines ------------------------------------------------------------

    RUNNER_PID_TTL_S = 3600.0

    def _on_go(self, ts: float, ev: dict) -> None:
        model = ev.get("model")
        blob = ev.get("blob")
        pid = ev.get("runner_pid")
        if pid:
            self.runner_pids[pid] = ts
            if len(self.runner_pids) > 8:
                cutoff = ts - self.RUNNER_PID_TTL_S
                self.runner_pids = {p: t for p, t in self.runner_pids.items()
                                    if t >= cutoff} or {pid: ts}
        if model:
            self.recent_model = model
            if blob:
                self.blob_to_model[blob] = model
        elif blob:
            model = self.blob_to_model.get(blob)
            if not model and self.model_index is not None:
                model = self.model_index.resolve(blob)
                if model:
                    self.blob_to_model[blob] = model

        event = ev.get("event")

        if event == "request_finished" and model:
            # Names the model for the access line we just held.  A row with
            # attribution 'none' never reached a runner (e.g. an unknown-model
            # 404), so this line does not belong to it -- skip past it rather
            # than labelling it with whatever ran most recently.
            for idx, (_ts, row) in enumerate(self.awaiting_model):
                if row.get("attribution") == "none":
                    continue
                if not row.get("model"):
                    row["model"] = model
                del self.awaiting_model[idx]
                self._emit(row)
                break
            return

        if event == "load_start":
            self.pending_load = {"ts": ts, "blob": blob, "num_ctx": ev.get("num_ctx"),
                                 "parallel": ev.get("parallel"), "spec_type": ev.get("spec_type"),
                                 "port": ev.get("port")}
            # The KV sizing lines for THIS load have not been printed yet;
            # anything still buffered belongs to a previous one.
            self.pending_kv = {}
        elif event == "model_loaded":
            detail = dict(self.pending_load or {})
            blob_ref = detail.get("blob")
            name = model or self.blob_to_model.get(blob_ref or "")
            if not name and self.model_index is not None:
                name = self.model_index.resolve(blob_ref)
            if name:
                self.loaded_models.add(name)
            # Ollama prints "llama-server started in Ns" once per llama-server
            # handle (main model and projector), so the same load surfaces
            # twice a few ms apart.  Keep the first.
            # Keep pending_load intact: the second line arrives with no
            # load_start of its own, and clearing it would strip the blob the
            # dedup key needs.  load_start overwrites it on the next real load.
            prev_blob, prev_ts = self._last_load
            if ts - prev_ts < 2.0 and (blob_ref is None or prev_blob == blob_ref):
                return
            self._last_load = (blob_ref or prev_blob, ts)
            kv = self._kv_detail() if self.pending_kv else None
            if kv:
                detail["kv_cache"] = kv
            self.pending_kv = {}
            self.on_event({"ts": ts, "kind": "model_loaded", "level": "INFO", "model": name,
                           "source": ev.get("source"), "msg": ev["msg"],
                           "duration_ms": ev.get("load_ms"), "detail": detail or None})
            # A KV cache partly on host RAM caps decode throughput for the
            # whole life of the load, so it is called out rather than left
            # buried in the load event's detail blob.
            if kv and (kv.get("cpu_fraction") or 0) > 0:
                self.on_event({
                    "ts": ts, "kind": "kv_offload", "level": "WARN", "model": name,
                    "source": ev.get("source"),
                    "msg": f"{kv['cpu_fraction']:.0%} of the KV cache "
                           f"({kv['cpu_mib']:.0f} of {kv['cpu_mib'] + kv['gpu_mib']:.0f} MiB) "
                           f"is on host RAM, not VRAM",
                    "detail": kv})
        elif event == "unload":
            if model:
                self.loaded_models.discard(model)
            self.on_event({"ts": ts, "kind": "unload", "level": "INFO", "model": model,
                           "source": ev.get("source"), "msg": ev["msg"], "detail": None})
        elif event == "runner_count":
            self.loaded_count = ev.get("count") or 0
        elif event == "idle_timer":
            self.on_event({"ts": ts, "kind": "idle_timer", "level": "INFO", "model": model,
                           "source": ev.get("source"), "msg": ev["msg"],
                           "detail": {"keep_alive": ev.get("keep_alive")}})
        elif event == "problem":
            self.on_event({"ts": ts, "kind": "problem", "level": ev["level"], "model": model,
                           "source": ev.get("source"), "msg": ev["msg"],
                           "detail": ev.get("kv") or None})

    # -- flush ---------------------------------------------------------------

    def _expire(self, now: float, force_model: bool = False) -> None:
        # Rows still waiting for a model name.
        while self.awaiting_model:
            ts, row = self.awaiting_model[0]
            if not force_model and now - ts < MODEL_TIMEOUT_S:
                break
            self.awaiting_model.popleft()
            if not row.get("model") and row.get("attribution") != "none":
                # Unambiguous only when exactly one model is loaded.
                if len(self.loaded_models) == 1:
                    row["model"] = next(iter(self.loaded_models))
                else:
                    row["model"] = self.recent_model
                    if row.get("attribution") == "exact":
                        row["attribution"] = "ambiguous"
            self._emit(row)

        # Tasks nobody claimed (client disconnected before the access line).
        while self.finished and now - self.finished[0]["retire_ts"] > JOIN_TIMEOUT_S:
            t = self.finished.popleft()
            row = {"ts": t["retire_ts"], "endpoint": "(unclaimed)", "method": None,
                   "class": "inference", "status": None, "client_ip": None,
                   "attribution": "orphan", "model": t.get("model") or self.recent_model}
            for k in ("ttft_ms", "decode_ms", "total_ms", "prompt_tokens", "prompt_tokens_total",
                      "output_tokens", "prefill_tps", "decode_tps", "draft_accept",
                      "draft_mean_len", "truncated", "context_tokens", "n_ctx_slot",
                      "slot_id", "task_id"):
                if k in t:
                    row[k] = t[k]
            self.stats["orphans"] += 1
            self._emit(row)

        # A task that printed its timings but never released (client aborted,
        # runner died) is retired after a short grace so its metrics are not
        # lost -- 'release' normally follows 'total time' within milliseconds.
        for tid, t in [(k, v) for k, v in self.tasks.items()
                       if v.get("done_ts") and now - v["done_ts"] > TOTAL_GRACE_S]:
            self._retire(tid, t["done_ts"])

        # A cache state line whose cost line never arrived.
        if self.pending_cache is not None and (
                force_model or now - self.pending_cache["ts"] > CACHE_GRACE_S):
            self._flush_cache()

        # Stale in-progress tasks (runner died mid-request).
        for tid in [k for k, v in self.tasks.items() if now - v["first_ts"] > 3600]:
            self.tasks.pop(tid, None)

    def _emit(self, row: dict) -> None:
        self.stats["requests"] += 1
        self.on_request(row)
        if row["class"] in ("inference", "embed"):
            self.on_live({"type": "request", **{k: row.get(k) for k in (
                "ts", "model", "endpoint", "status", "ttft_ms", "decode_tps",
                "output_tokens", "prompt_tokens", "latency_ms", "queue_ms",
                "attribution", "client_ip")}})


# --------------------------------------------------------------------------
# journal following
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# pollers
# --------------------------------------------------------------------------

_NVIDIA_QUERY = ("index,name,utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw")


def sample_gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        gpus.append({
            "index": _num(parts[0], int), "name": parts[1],
            "util_pct": _num(parts[2], float), "mem_used": _num(parts[3], int),
            "mem_total": _num(parts[4], int), "temp_c": _num(parts[5], float),
            "power_w": _num(parts[6], float),
        })
    return gpus


def _num(s, cast):
    try:
        return cast(float(s))
    except (TypeError, ValueError):
        return None


def sample_ps(base_url: str) -> list[dict]:
    """GET /api/ps -- currently loaded models and their VRAM footprint."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/ps", timeout=3) as r:
            data = json.loads(r.read())
    except Exception:
        return []
    out = []
    for m in data.get("models", []):
        out.append({"name": m.get("name") or m.get("model"),
                    "size": m.get("size"), "size_vram": m.get("size_vram"),
                    "expires_at": m.get("expires_at"),
                    "context_length": m.get("context_length")})
    return out


class GpuPoller:
    """Samples nvidia-smi.  GPUs belong to the host, not to a source, so this
    runs once regardless of how many engines are being monitored."""

    def __init__(self, store, interval_getter, on_live=None):
        self.store = store
        self.interval_getter = interval_getter
        self.on_live = on_live or (lambda x: None)
        self.last: dict = {}

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                ts = time.time()
                gpus = await loop.run_in_executor(None, sample_gpus)
                if gpus:
                    self.store.insert_gpu_samples(ts, gpus)
                    self.store.commit()
                self.last = {"ts": ts, "gpus": gpus}
                self.on_live({"type": "gpu", **self.last})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("gpu poller tick failed")
            await asyncio.sleep(max(1.0, float(self.interval_getter())))


class PsPoller:
    """Polls one ollama instance's /api/ps, and attributes its GPUs.

    Ollama had no GPU attribution at all: every card on the host was plotted on
    its pane, and its tiles summed VRAM and watts across all of them -- so
    another engine's memory and power were reported as ollama's.  That is the
    exact thing `gpuproc` exists to prevent, and the vLLM pane already avoided
    it, so the two panes disagreed about what their identical tiles meant.
    """

    GPU_RECHECK_S = 30.0

    def __init__(self, store, corr: Correlator, base_url: str, interval_getter,
                 on_live=None, unit: str = ""):
        self.store = store
        self.corr = corr
        self.base_url = base_url
        self.interval_getter = interval_getter
        self.on_live = on_live or (lambda x: None)
        self.unit = (unit or "").strip()
        self.last: dict = {}
        self.gpu_indices: list[int] | None = None
        self.gpu_source: str = "unavailable"
        self.gpu_ts: float | None = None
        self._gpu_checked = 0.0

    def resolve_gpus(self, now: float) -> list[int] | None:
        """Attribute ollama's GPUs, re-checked on an interval.

        Re-resolved rather than cached once, because ollama's runners come and
        go with keep-alive: a model unloading genuinely changes the answer to
        "holds nothing", and that has to be able to propagate.
        """
        if now - self._gpu_checked < self.GPU_RECHECK_S:
            return self.gpu_indices
        self._gpu_checked = now
        self.gpu_ts = now
        # runner.pid from the log is exact where available; the unit's cgroup
        # covers runners it never mentioned.
        pids = sorted(self.corr.runner_pids) if self.corr else []
        port = (gpuproc.port_of(self.base_url)
                if gpuproc.is_local(self.base_url) else None)
        try:
            self.gpu_indices, self.gpu_source = gpuproc.resolve(
                unit=self.unit or None, pids=pids, port=port)
        except Exception:
            log.debug("gpu attribution failed for ollama", exc_info=True)
            self.gpu_indices, self.gpu_source = None, "unavailable"
        return self.gpu_indices

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                ts = time.time()
                ps = await loop.run_in_executor(None, sample_ps, self.base_url)
                inflight = self.corr.inflight if self.corr else 0
                self.store.insert_ps_sample(ts, len(ps), ps, inflight)
                self.store.commit()
                await loop.run_in_executor(None, self.resolve_gpus, ts)
                self.last = {"ts": ts, "models": ps, "inflight": inflight,
                             "gpu_indices": self.gpu_indices,
                             "gpu_source": self.gpu_source}
                self.on_live({"type": "sample", **self.last})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ps poller tick failed")
            await asyncio.sleep(max(1.0, float(self.interval_getter())))


class Maintainer:
    """Periodic rollup rebuild + retention prune.

    Reads its intervals and retention through getters so a change in the
    settings screen takes effect on the next tick, with no restart.
    """

    def __init__(self, store, raw_retention_getter, sample_retention_getter,
                 interval_getter):
        self.store = store
        self.raw_retention_getter = raw_retention_getter
        self.sample_retention_getter = sample_retention_getter
        self.interval_getter = interval_getter

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        # On startup, roll up everything already in the table (a first run has a
        # full backfill to fold in).
        await loop.run_in_executor(None, self._initial)
        last_prune = 0.0
        while True:
            await asyncio.sleep(max(5.0, float(self.interval_getter())))
            now = time.time()
            try:
                # Re-roll the trailing window so late-joined rows are picked up.
                await loop.run_in_executor(
                    None, self.store.rebuild_rollups, now - 900, now)
                if now - last_prune > 3600:
                    dropped = self.store.prune(float(self.raw_retention_getter()),
                                               float(self.sample_retention_getter()))
                    if any(dropped.values()):
                        log.info("pruned %s", dropped)
                    last_prune = now
            except Exception:
                log.exception("maintenance tick failed")

    def _initial(self) -> None:
        row = self.store.query("SELECT MIN(ts) a, MAX(ts) b FROM requests")
        if row and row[0]["a"]:
            self.store.rebuild_rollups(row[0]["a"], row[0]["b"] + 60)
