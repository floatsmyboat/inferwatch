"""Maya: Strata's poller with Maya's identity, live aggregate and tiers.

What is particular to Maya and tested here: it is recognised by /v1/status
rather than /health; its live decode rate is the sum over the requests it
answers at once, and the concurrency split groups samples by that count; the
expert tiers are recorded only while it is busy; and its rows share the
Strata tables without showing up as Strata instances.
"""

import os
import shutil
import tempfile
import unittest

from inferwatch import maya_metrics, strata_metrics
from inferwatch.maya import MayaPoller, maya_request_row, maya_sample_row
from inferwatch.store import Store

T0 = 1_792_100_000.0

# Shapes from Maya v1.1.38's serve/server.py: metrics(), v1_status(), /health.
REQ = {"projection": None, "time": T0 + 10.25, "duration_s": 12.0, "finish": "stop",
       "api": "chat", "prefill_tok_s": 950.0, "prompt_tokens": 2100, "reused": 0,
       "output_tokens": 512, "engine_generated": 512, "prompt_ms": 2210.0,
       "decode_ms": 9600.0, "decode_tok_s": 53.3, "hit_rate": 0.91, "loop_guard": None}
ENGINE = {"model": "GLM-5.3-Flash-Maya-S24", "model_label": "GLM-5.3 · Maya-S24",
          "max_context": 131072, "images": False, "maya_version": "1.1.38"}
LIVE = {"state": "generating", "queued": 0, "active": 3, "writing": 3, "phase": "answering",
        "prompt_tokens": 2100, "prompt_read": None, "prompt_total": None, "generated": 300,
        "max_tokens": 1024, "elapsed_s": 6.0, "tok_s": 96.0, "tok_s_mean": 33.0,
        "prefill_tok_s_mean": None, "tok_s_window_s": 3}
TIERS = {"now": {"tok_s": 32.0, "vram_hit": 0.88, "ram_fetch": 4.1, "disk": 0.0, "promo": 0.6},
         "history": {}}
HW = {"cpu": 45.0, "ram_used": 80 * 2**30, "ram_total": 94 * 2**30}
STATUS = {"service": "maya", "model": "GLM-5.3-Flash-Maya-S24", "started": T0 - 600,
          "concurrency": {"serving": 3, "requested": 3}}
HEALTH = {"status": "ok", "max_context": 131072, "model": "GLM-5.3-Flash-Maya-S24",
          "model_label": "GLM-5.3 · Maya-S24", "images": False, "version": "1.1.38"}


def metrics(requests, live=None, tiers=TIERS):
    return {"engine": ENGINE, "live": live or LIVE, "tiers": tiers, "requests": requests,
            "hardware": HW, "time": T0 + 30}


class FakePoller(MayaPoller):
    def __init__(self, store, answers, source="maya"):
        super().__init__(store, source, {"url": "http://127.0.0.1:8001"})
        self.answers = answers

    def probe(self):
        import inferwatch.maya as mod
        real = mod.fetch_json

        def fake(url, api_key="", timeout=5.0):
            path = url.split("8001", 1)[1]
            if path not in self.answers:
                raise OSError("404")
            return self.answers[path]
        mod.fetch_json = fake
        try:
            return super().probe()
        finally:
            mod.fetch_json = real


class TestMapping(unittest.TestCase):
    def test_the_prompt_rate_is_maya_own_reused_excluded(self):
        r = maya_request_row("maya", "m", REQ)
        self.assertEqual(r["prefill_tps"], 950.0)
        self.assertEqual(r["decode_tps"], 53.3)
        self.assertEqual(r["expert_hit_rate"], 0.91)
        self.assertIsNone(r["pcie_share"])

    def test_running_is_the_requests_answered_at_once(self):
        s = maya_sample_row("maya", T0, metrics([]))
        self.assertEqual(s["running"], 3)
        self.assertEqual(s["writing"], 3)
        self.assertEqual(s["decode_tps"], 96.0)          # already the sum over the three
        self.assertEqual(s["vram_hit"], 0.88)
        self.assertEqual(s["disk_fetch"], 0.0)

    def test_an_idle_engine_records_no_tiers(self):
        """The STAT line lingers after a request; an idle engine fetches nothing."""
        s = maya_sample_row("maya", T0, metrics([], live={**LIVE, "state": "idle", "active": 0}))
        self.assertEqual(s["running"], 0)
        self.assertIsNone(s["vram_hit"])
        self.assertIsNone(s["ram_fetch"])
        self.assertIsNone(s["decode_tps"])


class TestPolling(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = Store(os.path.join(self.dir, "t.db"))
        self.store.add_source("maya", "maya", {"url": "http://127.0.0.1:8001"})
        self.store.add_source("strata", "strata", {"url": "http://127.0.0.1:8001"})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def answers(self, m):
        return {"/health": HEALTH, "/v1/status": STATUS, "/metrics": m}

    def test_maya_is_recognised_by_v1_status(self):
        p = FakePoller(self.store, self.answers(metrics([REQ])))
        got = p.probe()
        self.assertTrue(got["is_strata"])
        p.ingest(T0 + 30, got)
        inst = strata_metrics.instances(self.store, "maya")
        self.assertEqual([i["source"] for i in inst], ["maya"])
        self.assertEqual(inst[0]["model"], "GLM-5.3 · Maya-S24")
        self.assertEqual(inst[0]["version"], "1.1.38")
        self.assertEqual(inst[0]["slots"], 3)
        self.assertEqual(inst[0]["max_context"], 131072)

    def test_another_engine_on_the_port_records_nothing(self):
        """A vLLM flavour on :8001 answers /health but has no /v1/status."""
        p = FakePoller(self.store, {"/health": {}, "/metrics": metrics([REQ])})
        got = p.probe()
        self.assertFalse(got["is_strata"])
        self.assertIn("another server", got["error"])
        p.ingest(T0 + 30, got)
        self.assertEqual(self.store.query("SELECT COUNT(*) n FROM strata_requests")[0]["n"], 0)

    def test_the_ring_is_not_double_counted(self):
        p = FakePoller(self.store, self.answers(metrics([REQ])))
        p.ingest(T0 + 30, p.probe())
        p.ingest(T0 + 40, p.probe())
        self.assertEqual(self.store.query("SELECT COUNT(*) n FROM strata_requests")[0]["n"], 1)

    def test_maya_sources_are_not_strata_instances(self):
        p = FakePoller(self.store, self.answers(metrics([REQ])))
        p.ingest(T0 + 30, p.probe())
        self.assertEqual(strata_metrics.instances(self.store, "strata"), [])

    def test_concurrency_split_groups_aggregate_rate_by_running(self):
        one = {**LIVE, "active": 1, "writing": 1, "tok_s": 40.0}
        p = FakePoller(self.store, self.answers(metrics([], live=one)))
        p.ingest(T0 + 1, p.probe())
        p.ingest(T0 + 2, p.probe())
        p.answers = self.answers(metrics([]))      # three at once, 96 tok/s in all
        p.ingest(T0 + 3, p.probe())
        rows = maya_metrics.by_concurrency(self.store, "maya", T0, T0 + 10)
        self.assertEqual([r["running"] for r in rows], [1, 3])
        self.assertEqual(rows[0]["agg"], 40.0)
        self.assertEqual(rows[0]["n"], 2)
        self.assertEqual(rows[1]["agg"], 96.0)
        self.assertAlmostEqual(rows[1]["per_request"], 32.0)

    def test_tier_series_are_bucketed_means(self):
        p = FakePoller(self.store, self.answers(metrics([])))
        p.ingest(T0 + 1, p.probe())
        t = maya_metrics.tier_timeseries(self.store, "maya", T0, T0 + 60, step=60)
        self.assertEqual(t["vram_hit"][0], 0.88)
        self.assertEqual(t["ram_fetch"][0], 4.1)


if __name__ == "__main__":
    unittest.main()
