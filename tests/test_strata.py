"""Strata: polled, not log-read.

/metrics carries the finished requests (a ring the server keeps) as well as
the gauges, so the collector is one poller.  What is particular to Strata and
tested here: the ring is re-read every poll and must not double count; a
switcher can put another engine on the same port; and the expert-cache numbers
are weighted by the work done, not by the request count.
"""

import os
import shutil
import tempfile
import unittest

from inferwatch import strata_metrics
from inferwatch.store import Store
from inferwatch.strata import (RING_DEFAULT, StrataPoller, instance_fields,
                               request_row, sample_row)

T0 = 1_792_100_000.0

# Shapes copied from a live Strata 0.1.44.1 /metrics and /v1/status.
REQ = {"projection": None, "time": T0 + 10.2223165, "duration_s": 5.8, "finish": "stop",
       "prompt_tokens": 85, "reused": 0, "output_tokens": 540, "prompt_total": 85,
       "prompt_read": 85, "reasoning_recoveries": 0, "reasoning_repeat_coverage": 0.0,
       "engine_generated": 540, "prompt_ms": 586.0, "decode_ms": 5171.2,
       "decode_tok_s": 104.4, "hit_rate": 0.975, "pcie_share": 0.004, "ram_blobs": 0,
       "file_blobs": 0, "file_mb": 0, "drafts_offered": 469, "drafts_accepted": 374}
ENGINE = {"model": "qwen3.8-flash-next-iq2_xs", "max_context": 131072, "images": True,
          "version": "0.1.44.1", "context": 131072, "kv": "int8", "kv_resident": 32768,
          "expert_slots": 14331, "expert_cache_mib": 19693, "spec": 6,
          "vram_free_mib": 343, "arena_mib": 33812, "engine": "0.1.44.1"}
LIVE = {"state": "generating", "queued": 1, "phase": "answering", "prompt_tokens": 40000,
        "prompt_read": None, "prompt_total": None, "generated": 969, "max_tokens": 16384,
        "elapsed_s": 11.8, "tok_s": 102.9, "tok_s_mean": 90.8, "prefill_tok_s_mean": 126.9}
HW = {"gpu_pcie_rx_mb": 810.4, "gpu_pcie_tx_mb": 62.8, "cpu": 39.7,
      "ram_used": 54_313_787_392, "ram_total": 65_166_417_920}
STATUS = {"service": "strata", "engine": "0.1.44.1", "started": T0 - 6000,
          "concurrency": {"serving": 1, "requested": 1}}
HEALTH = {"status": "ok", "max_context": 131072, "model": "qwen3.8-flash-next-iq2_xs",
          "images": True, "service": "strata", "loaded": True}


def metrics(requests, live=None):
    return {"engine": ENGINE, "live": live or LIVE, "requests": requests,
            "hardware": HW, "conversation_cache": {"bytes": 0, "parked": 0}}


def req(i, **kw):
    """The i-th request, i seconds after T0, newest-first ordering left to the caller."""
    return {**REQ, "time": T0 + i, **kw}


class FakePoller(StrataPoller):
    """A poller whose HTTP answers are canned, so the ring logic is testable."""

    def __init__(self, store, answers):
        super().__init__(store, "strata", {"url": "http://127.0.0.1:8001"})
        self.answers = answers
        self.urls = []

    def probe(self):
        import inferwatch.strata as mod
        real = mod.fetch_json

        def fake(url, api_key="", timeout=5.0):
            self.urls.append(url)
            path = url.split("8001", 1)[1]
            return self.answers[path]
        mod.fetch_json = fake
        try:
            return super().probe()
        finally:
            mod.fetch_json = real


class TestMapping(unittest.TestCase):
    def test_a_ring_entry_becomes_a_complete_row(self):
        r = request_row("strata", "m", REQ)
        self.assertEqual(r["started_ts"], REQ["time"])
        self.assertAlmostEqual(r["ts"], REQ["time"] + 5.8)
        self.assertEqual(r["latency_ms"], 5800.0)
        self.assertEqual(r["ttft_ms"], 586.0)
        self.assertEqual(r["decode_tps"], 104.4)
        self.assertAlmostEqual(r["prefill_tps"], 85 / 0.586)
        self.assertEqual(r["expert_hit_rate"], 0.975)
        self.assertEqual(r["pcie_share"], 0.004)
        self.assertAlmostEqual(r["draft_accept"], 374 / 469)
        self.assertEqual(r["cached_tokens"], 0)
        self.assertEqual(r["prompt_tokens_total"], 85)

    def test_an_entry_with_no_start_is_dropped_not_invented(self):
        self.assertIsNone(request_row("strata", "m", {**REQ, "time": None}))

    def test_missing_numbers_stay_unknown(self):
        """An older engine leaves pcie_share and drafts out; None, not zero."""
        r = request_row("strata", "m", {**REQ, "pcie_share": None,
                                         "drafts_offered": None, "drafts_accepted": None})
        self.assertIsNone(r["pcie_share"])
        self.assertIsNone(r["draft_accept"])

    def test_context_in_use_is_prompt_plus_generated_while_busy(self):
        s = sample_row("strata", T0, metrics([]))
        self.assertEqual(s["kv_tokens"], 40969)
        self.assertEqual(s["kv_max"], 131072)
        self.assertEqual(s["kv_resident"], 32768)
        self.assertEqual(s["running"], 1)
        self.assertEqual(s["queued"], 1)
        self.assertEqual(s["decode_tps"], 102.9)

    def test_an_idle_engine_holds_no_context_and_decodes_nothing(self):
        """tok_s lingers after a request ends; reporting it would show a
        phantom decode on an idle engine."""
        s = sample_row("strata", T0, metrics([], live={**LIVE, "state": "idle"}))
        self.assertEqual(s["kv_tokens"], 0)
        self.assertEqual(s["running"], 0)
        self.assertIsNone(s["decode_tps"])
        self.assertIsNone(s["prefill_tps"])

    def test_instance_fields_come_from_engine_status_and_health(self):
        f = instance_fields(metrics([]), STATUS, HEALTH)
        self.assertEqual(f["version"], "0.1.44.1")
        self.assertEqual(f["slots"], 1)
        self.assertEqual(f["kv_type"], "int8")
        self.assertEqual(f["expert_cache_mib"], 19693)
        self.assertEqual(f["vision"], 1)
        self.assertEqual(f["started"], T0 - 6000)


class TestPoller(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))
        self.st.add_source("strata", "strata", {"url": "http://127.0.0.1:8001"})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def poll(self, p, ts=T0 + 100):
        return p.ingest(ts, p.probe())

    def count(self):
        return self.st.query("SELECT COUNT(*) n FROM strata_requests")[0]["n"]

    def test_re_reading_the_ring_does_not_double_count(self):
        ring = [req(3), req(2), req(1)]
        p = FakePoller(self.st, {"/health": HEALTH, "/v1/status": STATUS,
                                 "/metrics": metrics(ring)})
        self.poll(p)
        self.poll(p, T0 + 110)
        self.assertEqual(self.count(), 3)
        self.assertEqual(p.stats["requests"], 3)

    def test_a_full_ring_newer_than_anything_stored_fetches_the_whole_list(self):
        """Twelve requests can finish between two polls (fast tool calls); the
        short ring would then silently drop the older ones."""
        short = [req(i) for i in range(100, 100 - RING_DEFAULT, -1)]
        full = [req(i) for i in range(100, 50, -1)]
        p = FakePoller(self.st, {"/health": HEALTH, "/v1/status": STATUS,
                                 "/metrics": metrics(short),
                                 "/metrics?requests=all": metrics(full)})
        self.poll(p)
        self.assertEqual(self.count(), 50)
        self.assertEqual(p.stats["full_reads"], 1)

    def test_a_full_ring_that_overlaps_what_is_stored_is_enough(self):
        ring = [req(i) for i in range(20, 20 - RING_DEFAULT, -1)]
        p = FakePoller(self.st, {"/health": HEALTH, "/v1/status": STATUS,
                                 "/metrics": metrics(ring)})
        p.newest_started = T0 + 15         # already stored up to here
        self.poll(p)
        self.assertNotIn("http://127.0.0.1:8001/metrics?requests=all", p.urls)

    def test_another_engine_on_the_port_records_nothing_and_says_so(self):
        """A switcher rotates engines onto one port.  Its numbers must not be
        filed under Strata, and the configuration last seen must survive."""
        p = FakePoller(self.st, {"/health": HEALTH, "/v1/status": STATUS,
                                 "/metrics": metrics([req(1)])})
        self.poll(p)
        p.answers = {"/health": {"status": "healthy", "issues": []}}
        self.poll(p, T0 + 200)
        self.assertEqual(self.count(), 1)
        self.assertNotIn("http://127.0.0.1:8001/metrics", p.urls[-1:])
        inst = strata_metrics.instances(self.st)[0]
        self.assertFalse(inst["reachable"])
        self.assertIn("another server", inst["error"])
        self.assertEqual(inst["max_context"], 131072, "configuration was wiped")
        self.assertEqual(inst["version"], "0.1.44.1")
        # And when Strata is back, the error clears.
        p.answers = {"/health": HEALTH, "/v1/status": STATUS,
                     "/metrics": metrics([req(1)])}
        self.poll(p, T0 + 300)
        inst = strata_metrics.instances(self.st)[0]
        self.assertTrue(inst["reachable"])
        self.assertIsNone(inst["error"])

    def test_a_new_start_time_is_recorded_as_a_restart(self):
        events = []
        p = FakePoller(self.st, {"/health": HEALTH, "/v1/status": STATUS,
                                 "/metrics": metrics([])})
        p.on_event = events.append
        self.poll(p)
        p.answers["/v1/status"] = {**STATUS, "started": T0 + 50}
        self.poll(p, T0 + 120)
        self.assertEqual([e["kind"] for e in events], ["start"])
        self.assertEqual(events[0]["source"], "strata")

    def test_the_resume_point_comes_from_the_stored_rows(self):
        self.st.insert_strata_request(request_row("strata", "m", req(42)))
        p = FakePoller(self.st, {})
        p._resume_point()
        self.assertEqual(p.newest_started, T0 + 42)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))
        self.st.add_source("strata", "strata", {"url": "http://127.0.0.1:8001"})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def add(self, i, **kw):
        self.st.insert_strata_request(request_row("strata", "m", req(i, **kw)))

    def test_expert_hit_rate_is_weighted_by_tokens_generated(self):
        """A 4-token reply with a cold cache must not count as much as a
        4,000-token answer with a warm one."""
        self.add(1, hit_rate=0.5, output_tokens=4)
        self.add(2, hit_rate=1.0, output_tokens=3996)
        s = strata_metrics.summary(self.st, "strata", T0, T0 + 1000)
        self.assertAlmostEqual(s["expert_hit_rate"], (0.5 * 4 + 3996) / 4000)
        self.assertEqual(s["expert_hit_min"], 0.5)

    def test_draft_acceptance_is_totalled_not_averaged(self):
        self.add(1, drafts_offered=4, drafts_accepted=4)
        self.add(2, drafts_offered=1000, drafts_accepted=500)
        s = strata_metrics.summary(self.st, "strata", T0, T0 + 1000)
        self.assertAlmostEqual(s["draft_accept_mean"], 504 / 1004)

    def test_percentiles_are_exact(self):
        for i, ms in enumerate((100, 200, 300, 400, 5000)):
            self.add(i, prompt_ms=float(ms))
        s = strata_metrics.summary(self.st, "strata", T0, T0 + 1000)
        self.assertEqual(s["ttft_ms"]["p50"], 300.0)
        self.assertTrue(s["exact"])

    def test_disk_reads_are_counted(self):
        """Low-RAM mode reads experts from a file; worth seeing when it happens."""
        self.add(1, file_blobs=12)
        self.add(2)
        s = strata_metrics.summary(self.st, "strata", T0, T0 + 1000)
        self.assertEqual(s["disk_reads"], 1)

    def test_context_past_the_vram_window_is_reported(self):
        self.st.insert_strata_sample(sample_row("strata", T0 + 5, metrics([])))
        ts = strata_metrics.sample_timeseries(self.st, "strata", T0, T0 + 60, 60)
        self.assertEqual(ts["kv_streamed"][0], 40969 - 32768)
        self.assertAlmostEqual(ts["kv_frac"][0], 40969 / 131072)
        lv = strata_metrics.live(self.st, "strata", T0, T0 + 60)
        self.assertEqual(lv["kv_peak"], 40969)
        self.assertEqual(lv["queued"], 1)

    def test_instances_are_filtered_to_configured_sources(self):
        self.st.upsert_strata_instance("strata", last_seen=T0, reachable=1)
        self.st.upsert_strata_instance("ghost", last_seen=T0, reachable=1)
        names = [i["source"] for i in strata_metrics.instances(self.st)]
        self.assertEqual(names, ["strata"])

    def test_deleting_a_source_knows_its_tables(self):
        from inferwatch.store import SOURCE_DATA_TABLES
        tables = {t for t, _ in SOURCE_DATA_TABLES["strata"]["history"]}
        self.assertEqual(tables, {"strata_requests", "strata_samples"})

    def test_prune_drops_old_rows(self):
        self.add(1)
        self.st.insert_strata_sample(sample_row("strata", T0, metrics([])))
        self.st.commit()
        got = self.st.prune(raw_retention_days=0, sample_retention_days=0)
        self.assertEqual(got["strata_requests"], 1)
        self.assertEqual(got["strata_samples"], 1)


if __name__ == "__main__":
    unittest.main()
