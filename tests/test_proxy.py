"""The proxy in front of vLLM is the only place a client address exists."""

import os
import shutil
import tempfile
import unittest

from inferwatch import vllm_metrics
from inferwatch.parse_proxy import parse_req
from inferwatch.store import Store

REQ = ("2026-09-17 16:16:39,533 REQ v1/chat/completions from=10.0.0.96 "
       "model=qwen3.8 msgs=29 prompt=19,172 limit=131072 max_tokens=16384 "
       "max_completion_tokens=None stream=True")


class TestParseReq(unittest.TestCase):
    def test_a_request_line_yields_the_client_and_the_exact_prompt(self):
        got = parse_req(REQ)
        self.assertEqual(got["client"], "10.0.0.96")
        self.assertEqual(got["endpoint"], "v1/chat/completions")
        self.assertEqual(got["model"], "qwen3.8")
        self.assertEqual(got["messages"], 29)
        # Rendered with thousands separators by the proxy.
        self.assertEqual(got["prompt_tokens"], 19172)
        self.assertEqual(got["context_limit"], 131072)
        self.assertEqual(got["max_tokens"], 16384)
        self.assertEqual(got["stream"], 1)

    def test_none_and_unknown_stay_none_rather_than_zero(self):
        """`prompt=unknown` means the proxy could not tokenise the body, which
        is not a request that carried no prompt; averaging zeros in would drag
        every per-client token figure down."""
        got = parse_req(REQ.replace("prompt=19,172", "prompt=unknown")
                           .replace("max_tokens=16384", "max_tokens=None"))
        self.assertIsNone(got["prompt_tokens"])
        self.assertIsNone(got["max_tokens"])

    def test_lines_that_are_not_requests_are_ignored(self):
        for line in ("2026-09-17 16:16:39,609 RESP v1/chat/completions -> 200 in 0.12s",
                     "2026-09-17 16:16:49,304 DONE metrics in 0.01s",
                     "INFO:     127.0.0.1:55734 - \"GET /metrics HTTP/1.1\" 200 OK",
                     "", "REQ"):
            self.assertIsNone(parse_req(line), line)

    def test_a_missing_stream_flag_is_unknown_not_false(self):
        got = parse_req(REQ[:REQ.index(" stream=")])
        self.assertIsNone(got["stream"])


class TestClientMetrics(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def add(self, ts, client, prompt=1000, model="qwen3.8", stream=1):
        self.st.insert_client_request(
            {"ts": ts, "source": "v", "client": client, "endpoint": "v1/chat/completions",
             "model": model, "messages": 4, "prompt_tokens": prompt,
             "context_limit": 131072, "max_tokens": None,
             "max_completion_tokens": None, "stream": stream})
        self.st.commit()

    def test_clients_are_ranked_with_exact_prompt_totals(self):
        self.add(100, "10.0.0.96", 1000)
        self.add(101, "10.0.0.96", 3000)
        self.add(102, "10.0.0.122", 500)
        got = vllm_metrics.clients(self.st, "v", 0, 1000)
        self.assertEqual([c["client"] for c in got], ["10.0.0.96", "10.0.0.122"])
        self.assertEqual(got[0]["requests"], 2)
        self.assertEqual(got[0]["prompt_tokens"], 4000)
        self.assertEqual(got[0]["prompt_max"], 3000)
        self.assertEqual(got[0]["top_model"], "qwen3.8")

    def test_context_peak_is_the_largest_prompt_against_the_window(self):
        self.add(100, "a", 131072 // 2)
        self.assertAlmostEqual(
            vllm_metrics.clients(self.st, "v", 0, 1000)[0]["context_peak_pct"], 0.5)

    def test_a_replayed_journal_does_not_double_count(self):
        for _ in range(3):
            self.add(100, "10.0.0.96", 1000)
        self.assertEqual(vllm_metrics.clients(self.st, "v", 0, 1000)[0]["requests"], 1)

    def test_unsized_requests_are_counted_not_hidden(self):
        self.add(100, "a", None)
        self.add(101, "a", 2000)
        got = vllm_metrics.clients(self.st, "v", 0, 1000)[0]
        self.assertEqual(got["requests"], 2)
        self.assertEqual(got["unsized"], 1)
        self.assertEqual(got["prompt_tokens"], 2000)

    def test_only_the_window_and_the_named_source_count(self):
        self.add(100, "a")
        self.add(900, "a")
        self.assertEqual(vllm_metrics.clients(self.st, "v", 500, 1000)[0]["requests"], 1)
        self.assertEqual(vllm_metrics.clients(self.st, "other", 0, 1000), [])

    def test_availability_separates_no_proxy_from_no_traffic(self):
        self.assertFalse(vllm_metrics.clients_available(self.st, "v"))
        self.add(100, "a")
        self.assertTrue(vllm_metrics.clients_available(self.st, "v"))

    def test_deleting_the_source_takes_the_client_rows(self):
        sid = self.st.add_source("vllm", "v", {"url": "http://x:8000"})
        self.add(100, "a")
        self.st.delete_source(sid, purge=True)
        self.assertEqual(
            self.st.query("SELECT COUNT(*) n FROM vllm_client_requests")[0]["n"], 0)
