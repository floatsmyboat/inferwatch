"""Builds and rebuilds the collector set from the `sources` table.

Editing sources or intervals in the settings screen calls `request_reload()`,
which tears the collector tasks down and rebuilds them without restarting the
process, so the dashboard and its open SSE streams stay up.  Settings marked
`restart=True` in the config spec (the bind address and port) genuinely need a
new process, and the API reports that back to the UI rather than pretending.

SCOPE NOTE -- one Ollama source, many vLLM sources
--------------------------------------------------
vLLM rows are keyed by source throughout (`vllm_samples.source` and friends),
so any number of vLLM instances can be monitored side by side.

The Ollama tables (`requests`, `events`, `ps_samples`) are NOT source-
partitioned, so exactly one Ollama source may be enabled at a time.  Enabling a
second is rejected with an explanation instead of silently blending two
instances' requests into one set of numbers.  Partitioning those tables is a
schema change worth doing deliberately, not as a side effect.
"""

from __future__ import annotations

import asyncio
import logging

from .collect import Correlator, GpuPoller, Maintainer, PsPoller
from .models import ModelIndex
from .readers import build_reader
from .vllm import VllmCollector

log = logging.getLogger("inferwatch.supervisor")


class SourceRuntime:
    """A running source: its collectors and whatever state the UI wants to see."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.tasks: list[asyncio.Task] = []
        self.reader = None
        self.corr: Correlator | None = None
        self.vllm: VllmCollector | None = None
        self.ps: PsPoller | None = None

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def kind(self) -> str:
        return self.spec["kind"]

    def status(self) -> dict:
        d = {"name": self.name, "kind": self.kind, "enabled": self.spec.get("enabled", True)}
        if self.reader is not None:
            d["reader"] = self.reader.describe()
            d["lines_read"] = self.reader.lines
        if self.corr is not None:
            d["inflight"] = self.corr.inflight
            d["stats"] = dict(self.corr.stats)
            # The most recent prompt-cache reading, so /api/status can say
            # whether the gauge is arriving at all.
            d["cache"] = self.corr.last_cache
        if self.ps is not None:
            # Held on the runtime rather than left a local, so /api/status can
            # say when /api/ps was last read and whether it is answering --
            # otherwise this was the one collector with no visibility.
            last = self.ps.last or {}
            d["ps"] = {"ts": last.get("ts"), "url": self.ps.base_url,
                       "loaded_count": len(last.get("models") or []),
                       "models": [m.get("name") for m in (last.get("models") or [])]}
        if self.vllm is not None:
            d["url"] = self.vllm.url
            d["scrapes"] = self.vllm.scrapes
            d["consecutive_errors"] = self.vllm.errors
            d["last"] = {k: v for k, v in (self.vllm.last or {}).items()
                         if k in ("ts", "model")}
        return d


class Supervisor:
    def __init__(self, store, config, hub):
        self.store = store
        self.config = config
        self.hub = hub
        self.runtimes: dict[str, SourceRuntime] = {}
        self.gpu: GpuPoller | None = None
        self.maintainer: Maintainer | None = None
        self._shared: list[asyncio.Task] = []
        self._reload = asyncio.Event()
        self._stopping = False

    # -- lifecycle -------------------------------------------------------

    def request_reload(self) -> None:
        """Ask the loop to rebuild collectors (safe from any thread/context)."""
        self._reload.set()

    async def run(self) -> None:
        await self._start_shared()
        await self._reconcile()
        try:
            while not self._stopping:
                await self._reload.wait()
                self._reload.clear()
                if self._stopping:
                    break
                log.info("configuration changed; rebuilding collectors")
                await self._reconcile()
        except asyncio.CancelledError:
            await self.stop()
            raise

    async def stop(self) -> None:
        self._stopping = True
        for rt in list(self.runtimes.values()):
            await self._stop_runtime(rt)
        for t in self._shared:
            t.cancel()
        if self._shared:
            await asyncio.gather(*self._shared, return_exceptions=True)
        self._shared.clear()

    # -- shared collectors ----------------------------------------------

    async def _start_shared(self) -> None:
        # nvidia-smi is host-wide, not per-source, so it runs once.
        self.gpu = GpuPoller(self.store,
                             interval_getter=lambda: self.config.get(
                                 "collection.poll_interval_s"),
                             on_live=self.hub.publish)
        self.maintainer = Maintainer(
            self.store,
            raw_retention_getter=lambda: self.config.get("retention.raw_days"),
            sample_retention_getter=lambda: self.config.get("retention.sample_days"),
            interval_getter=lambda: self.config.get("collection.rollup_interval_s"))
        self._shared = [
            asyncio.create_task(self.gpu.run(), name="gpu-poller"),
            asyncio.create_task(self.maintainer.run(), name="maintainer"),
        ]

    # -- reconciliation --------------------------------------------------

    def desired(self) -> list[dict]:
        """Enabled sources, with the one-Ollama rule applied."""
        out, seen_ollama = [], False
        for s in self.store.list_sources():
            if not s.get("enabled"):
                continue
            if s["kind"] == "ollama":
                if seen_ollama:
                    log.warning("source %r ignored: only one Ollama source may be "
                                "enabled at a time (its tables are not "
                                "source-partitioned)", s["name"])
                    continue
                seen_ollama = True
            out.append(s)
        return out

    async def _reconcile(self) -> None:
        desired = {s["name"]: s for s in self.desired()}

        for name in list(self.runtimes):
            rt = self.runtimes[name]
            want = desired.get(name)
            # Restart on any definition change; comparing the whole spec is
            # cheaper than reasoning about which fields matter.
            if want is None or want.get("config") != rt.spec.get("config") \
                    or want.get("kind") != rt.kind:
                await self._stop_runtime(rt)
                self.runtimes.pop(name, None)

        for name, spec in desired.items():
            if name not in self.runtimes:
                try:
                    self.runtimes[name] = await self._start_runtime(spec)
                except Exception:
                    log.exception("failed to start source %r", name)

    async def _start_runtime(self, spec: dict) -> SourceRuntime:
        rt = SourceRuntime(spec)
        cfg = spec.get("config") or {}
        if spec["kind"] == "ollama":
            models_dir = cfg.get("models_dir") or None
            index = ModelIndex(models_dir)
            rt.corr = Correlator(on_request=self.store.insert_request,
                                 on_event=self.store.insert_event,
                                 on_live=self.hub.publish,
                                 on_cache=self.store.insert_cache_sample,
                                 model_index=index)
            rt.reader = build_reader(self.store, spec["name"], cfg,
                                     backfill=self.config.get("collection.backfill"))
            corr, store = rt.corr, self.store

            def on_line(ts, msg, corr=corr):
                corr.feed(ts, msg)

            def on_flush(now, corr=corr, store=store):
                corr.tick(now)
                store.commit()

            rt.ps = PsPoller(self.store, rt.corr,
                             cfg.get("url") or "http://127.0.0.1:11434",
                             interval_getter=lambda: self.config.get(
                                 "collection.poll_interval_s"),
                             on_live=self.hub.publish)
            rt.tasks = [
                asyncio.create_task(rt.reader.run(on_line, on_flush),
                                    name=f"reader:{spec['name']}"),
                asyncio.create_task(rt.ps.run(), name=f"ps:{spec['name']}"),
            ]
            log.info("source %r started (%s)", spec["name"], rt.reader.describe())
        elif spec["kind"] == "vllm":
            rt.vllm = VllmCollector(
                self.store, spec["name"], cfg,
                interval_getter=lambda: self.config.get("collection.scrape_interval_s"),
                on_live=self.hub.publish)
            rt.tasks = [asyncio.create_task(rt.vllm.run(), name=f"vllm:{spec['name']}")]
            log.info("source %r started (vllm %s)", spec["name"], rt.vllm.url)
        else:
            raise ValueError(f"unknown source kind {spec['kind']!r}")
        return rt

    async def _stop_runtime(self, rt: SourceRuntime) -> None:
        for t in rt.tasks:
            t.cancel()
        if rt.tasks:
            await asyncio.gather(*rt.tasks, return_exceptions=True)
        rt.tasks.clear()
        log.info("source %r stopped", rt.name)

    # -- introspection ---------------------------------------------------

    def status(self) -> dict:
        return {
            "sources": [rt.status() for rt in self.runtimes.values()],
            "gpu_devices": len(self.gpu.last.get("gpus", [])) if self.gpu else 0,
            "ollama_source": next((rt.name for rt in self.runtimes.values()
                                   if rt.kind == "ollama"), None),
            "vllm_sources": [rt.name for rt in self.runtimes.values()
                             if rt.kind == "vllm"],
        }

    def ollama_runtime(self) -> SourceRuntime | None:
        return next((rt for rt in self.runtimes.values() if rt.kind == "ollama"), None)
