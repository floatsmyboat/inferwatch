"""SourceRuntime.status(): what the UI can see about a running collector.

Every collector a source starts should be reachable from its runtime, because
`status()` is the only window /api/status has into whether collection is
actually working.
"""

import unittest

from inferwatch.collect import Correlator, PsPoller
from inferwatch.supervisor import SourceRuntime


class FakeReader:
    kind = "file"
    lines = 42

    def describe(self):
        return "file /var/log/ollama.log"


class TestRuntimeStatus(unittest.TestCase):
    def runtime(self):
        return SourceRuntime({"name": "ollama", "kind": "ollama", "enabled": True})

    def test_bare_runtime_reports_only_identity(self):
        d = self.runtime().status()
        self.assertEqual(d["name"], "ollama")
        self.assertEqual(d["kind"], "ollama")
        self.assertTrue(d["enabled"])
        for absent in ("reader", "inflight", "ps", "url"):
            self.assertNotIn(absent, d)

    def test_reader_and_correlator_are_surfaced(self):
        rt = self.runtime()
        rt.reader = FakeReader()
        rt.corr = Correlator()
        d = rt.status()
        self.assertEqual(d["lines_read"], 42)
        self.assertIn("ollama.log", d["reader"])
        self.assertEqual(d["inflight"], 0)
        self.assertEqual(d["stats"], {})

    def test_the_ps_poller_is_visible(self):
        """Regression: the PsPoller was a local in _start_runtime, so its last
        sample was unreachable and /api/status could not say whether /api/ps was
        answering -- the one collector with no visibility."""
        rt = self.runtime()
        rt.corr = Correlator()
        rt.ps = PsPoller(None, rt.corr, "http://127.0.0.1:11434",
                         interval_getter=lambda: 5.0)
        d = rt.status()
        self.assertIn("ps", d)
        self.assertEqual(d["ps"]["url"], "http://127.0.0.1:11434")
        # Nothing polled yet, so there is no timestamp to report.
        self.assertIsNone(d["ps"]["ts"])
        self.assertEqual(d["ps"]["loaded_count"], 0)
        self.assertEqual(d["ps"]["models"], [])

    def test_ps_reports_the_models_from_its_last_sample(self):
        rt = self.runtime()
        rt.ps = PsPoller(None, None, "http://127.0.0.1:11434",
                         interval_getter=lambda: 5.0)
        rt.ps.last = {"ts": 1_700_000_000.0,
                      "models": [{"name": "llama3.2:3b"}, {"name": "qwen3:8b"}],
                      "inflight": 1}
        d = rt.status()
        self.assertEqual(d["ps"]["ts"], 1_700_000_000.0)
        self.assertEqual(d["ps"]["loaded_count"], 2)
        self.assertEqual(d["ps"]["models"], ["llama3.2:3b", "qwen3:8b"])

    def test_the_prompt_cache_reading_is_surfaced(self):
        rt = self.runtime()
        rt.corr = Correlator()
        self.assertIsNone(rt.status()["cache"])
        rt.corr.last_cache = {"ts": 1.0, "usage": 0.5}
        self.assertEqual(rt.status()["cache"]["usage"], 0.5)


if __name__ == "__main__":
    unittest.main()
