"""NInfer: the one engine whose log carries both halves.

Per-request timings like ollama's, and an evenly sampled throughput line like
vLLM's gauges -- so unlike either, nothing has to be reconstructed or given up.
"""

import os
import shutil
import tempfile
import unittest

from inferwatch import ninfer_metrics
from inferwatch.ninfer import NinferCorrelator
from inferwatch.parse_ninfer import parse
from inferwatch.store import Store

DONE = ("[req 2] done finish=tool_calls tool_calls=1 prompt=41854 gen=2177 "
        "cache=512 reuse=full_reset ttft=19893ms prefill=2113.1tok/s "
        "decode=70.8tok/s wall=50.67s speculative=mtp 2.69tok/round (56.3%)")
# A request too short to have a decode rate prints a BARE `n/a` with no unit.
SHORT = ("[req 1] done finish=output_limit prompt=48004 gen=1 cache=0 "
         "reuse=full_reset ttft=25425ms prefill=1905.9tok/s decode=n/a "
         "wall=25.42s speculative=mtp n/a")
SUBMIT = ("[req 2] openai_chat_completions stream msgs=29 max_tokens=16384 "
          "(client) tools=11 tool_choice=auto tool_history=yes thinking=on "
          "preserve_thinking=off sampler=[temp=1.00] → submitted")
THRU = ("throughput interval=5.000s prefill=0.0tok/s decode=65.2tok/s running=2 "
        "prefilling=1 decode_ready=1 waiting=3 avg_decode_batch=1.69")
IDLE = ("throughput interval=5.000s prefill=2252.8tok/s decode=0.0tok/s running=1 "
        "prefilling=1 decode_ready=0 waiting=0 avg_decode_batch=n/a")


class TestParse(unittest.TestCase):
    def test_a_completion_carries_the_whole_request(self):
        r = parse(DONE)
        self.assertEqual(r["kind"], "done")
        self.assertEqual(r["finish"], "tool_calls")
        self.assertEqual(r["tool_calls"], 1)
        self.assertEqual(r["prompt_tokens"], 41854)
        self.assertEqual(r["cached_tokens"], 512)
        # What the client sent is what was evaluated plus what reuse saved.
        self.assertEqual(r["prompt_tokens_total"], 42366)
        self.assertEqual(r["output_tokens"], 2177)
        self.assertEqual(r["ttft_ms"], 19893.0)
        self.assertEqual(r["latency_ms"], 50670.0)
        self.assertAlmostEqual(r["decode_ms"], 30777.0)
        self.assertEqual(r["decode_tps"], 70.8)
        self.assertAlmostEqual(r["draft_accept"], 0.563)
        self.assertEqual(r["draft_mean_len"], 2.69)

    def test_a_bare_n_a_rate_still_parses(self):
        """The unit is only printed when there IS a rate. Requiring `tok/s`
        silently dropped 7 of 16 completions in a day's log -- every
        single-token one, which is exactly the set worth seeing."""
        r = parse(SHORT)
        self.assertIsNotNone(r, "a completion with no decode rate must not vanish")
        self.assertEqual(r["kind"], "done")
        self.assertEqual(r["output_tokens"], 1)
        self.assertIsNone(r["decode_tps"], "no rate is unknown, not zero")
        self.assertEqual(r["prefill_tps"], 1905.9)
        self.assertIsNone(r["draft_accept"])

    def test_a_submission_describes_the_shape_of_the_ask(self):
        r = parse(SUBMIT)
        self.assertEqual(r["kind"], "submit")
        self.assertEqual(r["stream"], 1)
        self.assertEqual(r["messages"], 29)
        self.assertEqual(r["max_tokens"], 16384)
        self.assertEqual(r["tools"], 11)
        self.assertEqual(r["thinking"], 1)

    def test_the_throughput_line_gives_queue_depth(self):
        r = parse(THRU)
        self.assertEqual(r["kind"], "throughput")
        self.assertEqual((r["running"], r["waiting"]), (2, 3))
        self.assertEqual(r["avg_decode_batch"], 1.69)

    def test_an_idle_batch_is_unknown_not_zero(self):
        self.assertIsNone(parse(IDLE)["avg_decode_batch"])

    def test_noise_is_ignored(self):
        for line in ("loading model...", "0, 14 MiB", "", "Started ninfer.service",
                     "load  weights  50.19%  10.50 GiB / 20.92 GiB  10.101 s"):
            self.assertIsNone(parse(line), line)

    def test_startup_lines_are_recognised(self):
        self.assertEqual(parse("model loaded in 25.8391 s")["load_ms"], 25839.1)
        kv = parse("KV capacity explicit resolved=217600 tokens pages=3400/8192 "
                   "runtime=4.76 GiB")
        self.assertEqual(kv["kv_tokens"], 217600)
        lis = parse("listening on http://0.0.0.0:8011 (model id: qwen3.8-27b, "
                    "auth: bearer)")
        self.assertEqual(lis["model"], "qwen3.8-27b")


class TestCorrelator(unittest.TestCase):
    def setUp(self):
        self.rows, self.samples = [], []
        self.c = NinferCorrelator("n", on_request=self.rows.append,
                                  on_sample=self.samples.append)

    def test_submission_and_completion_join_by_id(self):
        self.c.feed(100.0, SUBMIT)
        self.c.feed(150.0, DONE)
        self.assertEqual(len(self.rows), 1)
        r = self.rows[0]
        self.assertEqual(r["paired"], 1)
        self.assertEqual(r["messages"], 29)        # from the submission
        self.assertEqual(r["output_tokens"], 2177)  # from the completion
        self.assertEqual(self.c.stats["joined"], 1)

    def test_a_completion_with_no_submission_is_still_stored(self):
        """The reader can start mid-request; dropping the completion would lose
        every field it states for want of the other half."""
        self.c.feed(150.0, DONE)
        self.assertEqual(len(self.rows), 1)
        self.assertEqual(self.rows[0]["paired"], 0)
        self.assertEqual(self.rows[0]["output_tokens"], 2177)
        self.assertEqual(self.c.stats["unpaired"], 1)

    def test_a_restart_retires_pending_rather_than_letting_ids_collide(self):
        """The [req N] counter restarts at 1 with the process, so a fresh
        request would otherwise be completed by the previous run's line."""
        self.c.feed(100.0, SUBMIT)
        self.assertEqual(self.c.inflight, 1)
        self.c.feed(110.0, "listening on http://0.0.0.0:8011 (model id: m, auth: bearer)")
        self.assertEqual(self.c.inflight, 0)
        self.assertEqual(self.c.stats["abandoned"], 1)
        self.assertEqual(self.rows, [], "an abandoned request must not be invented")

    def test_an_error_is_a_request_with_a_finish_of_error(self):
        self.c.feed(100.0, SUBMIT)
        self.c.feed(120.0, "[req 2] error inference request expired while waiting")
        self.assertEqual(self.rows[0]["finish"], "error")
        self.assertIn("expired", self.rows[0]["error"])
        self.assertEqual(self.c.stats["errors"], 1)

    def test_the_epoch_advances_with_each_run(self):
        self.c.feed(10.0, "listening on http://x:8011 (model id: m, auth: bearer)")
        self.c.feed(20.0, DONE)
        self.c.feed(30.0, "listening on http://x:8011 (model id: m, auth: bearer)")
        self.c.feed(40.0, DONE)
        self.assertEqual([r["epoch"] for r in self.rows], [1, 2])

    def test_throughput_lines_become_samples(self):
        self.c.feed(100.0, THRU)
        self.assertEqual(len(self.samples), 1)
        self.assertEqual(self.samples[0]["waiting"], 3)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))
        self.c = NinferCorrelator("n",
                                  on_request=self.st.insert_ninfer_request,
                                  on_sample=self.st.insert_ninfer_sample)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def feed(self, ts, line):
        self.c.feed(ts, line)
        self.st.commit()

    def test_percentiles_are_exact_not_bucketed(self):
        """The distinction the vLLM tab has to make and this one does not: the
        raw values are stored, so the percentile is a real order statistic."""
        for i, ttft in enumerate((100, 200, 300, 400, 500)):
            self.feed(100.0 + i, DONE.replace("ttft=19893ms", f"ttft={ttft}ms")
                                     .replace("[req 2]", f"[req {i + 10}]"))
        s = ninfer_metrics.summary(self.st, "n", 0, 1000)
        self.assertEqual(s["requests"], 5)
        self.assertEqual(s["ttft_ms"]["p50"], 300.0)
        self.assertTrue(s["exact"])

    def test_cache_hit_rate_is_against_what_clients_sent(self):
        self.feed(100.0, DONE)
        s = ninfer_metrics.summary(self.st, "n", 0, 1000)
        self.assertAlmostEqual(s["cache_hit_rate"], 512 / 42366)

    def test_errors_are_counted_from_the_finish_reason(self):
        self.feed(100.0, DONE)
        self.feed(101.0, "[req 9] error upstream exploded")
        s = ninfer_metrics.summary(self.st, "n", 0, 1000)
        self.assertEqual((s["requests"], s["errors"]), (2, 1))
        self.assertAlmostEqual(s["error_rate"], 0.5)

    def test_gauges_come_from_the_engine_not_the_request_rows(self):
        self.feed(100.0, THRU)
        ts = ninfer_metrics.timeseries(self.st, "n", 0, 1000, step=1000)
        self.assertEqual(ts["waiting"][0], 3.0)
        self.assertEqual(ts["running"][0], 2.0)

    def test_a_replayed_log_does_not_double_count(self):
        for _ in range(3):
            self.feed(100.0, DONE)
        self.assertEqual(ninfer_metrics.summary(self.st, "n", 0, 1000)["requests"], 1)

    def test_finish_reasons_stand_in_for_status(self):
        self.feed(100.0, DONE)
        self.feed(101.0, SHORT)
        got = {r["finish"]: r["count"] for r in
               ninfer_metrics.finish_reasons(self.st, "n", 0, 1000)}
        self.assertEqual(got, {"tool_calls": 1, "output_limit": 1})

    def test_instances_are_filtered_to_configured_sources(self):
        self.st.upsert_ninfer_instance("ghost", last_seen=1.0, reachable=1)
        self.st.commit()
        self.assertEqual(ninfer_metrics.instances(self.st), [])
        self.st.add_source("ninfer", "ghost", {"unit": "x"})
        self.assertEqual([i["source"] for i in ninfer_metrics.instances(self.st)],
                         ["ghost"])


class TestClients(unittest.TestCase):
    """NInfer logs no client address, so identity comes from the socket table.
    Those are connections, not requests, and the two are kept apart."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))
        self.c = NinferCorrelator("n", on_request=self.st.insert_ninfer_request)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def conns(self, ts, counts):
        self.st.insert_ninfer_client_samples(ts, "n", counts)
        self.st.commit()

    def request(self, ts, ttft=1000, wall=2.0):
        """A completion at `ts` whose wall time places its start before it."""
        self.c.feed(ts, DONE.replace("ttft=19893ms", f"ttft={ttft}ms")
                            .replace("wall=50.67s", f"wall={wall}s")
                            .replace("[req 2]", f"[req {int(ts)}]"))
        self.st.commit()

    def test_a_lone_connected_client_gets_the_request(self):
        """Not a guess: with one client connected for the whole request, it was
        that client's by elimination."""
        self.conns(98.0, {"10.0.0.96": 1})
        self.conns(100.0, {"10.0.0.96": 1})
        self.request(101.0, wall=3.0)
        got = ninfer_metrics.clients(self.st, "n", 0, 1000)
        self.assertEqual(got["attributed"], 1)
        self.assertEqual(got["ambiguous"], 0)
        self.assertEqual(got["clients"][0]["client"], "10.0.0.96")
        self.assertEqual(got["clients"][0]["requests"], 1)

    def test_two_connected_clients_make_it_unattributable(self):
        """Nothing in the log says which of them asked, so neither is charged."""
        self.conns(100.0, {"10.0.0.96": 1, "10.0.0.122": 1})
        self.request(101.0, wall=3.0)
        got = ninfer_metrics.clients(self.st, "n", 0, 1000)
        self.assertEqual((got["attributed"], got["ambiguous"]), (0, 1))
        self.assertTrue(all(c["requests"] == 0 for c in got["clients"]))

    def test_a_second_client_appearing_mid_request_spoils_it(self):
        """Attribution needs EVERY sample while it ran to show one client; a
        request that began alone and finished alongside another did not."""
        self.conns(99.0, {"a": 1})
        self.conns(100.0, {"a": 1, "b": 1})
        self.request(101.0, wall=3.0)
        self.assertEqual(ninfer_metrics.clients(self.st, "n", 0, 1000)["ambiguous"], 1)

    def test_a_request_no_sample_covers_is_not_attributed(self):
        self.conns(10.0, {"a": 1})
        self.request(500.0, wall=1.0)
        got = ninfer_metrics.clients(self.st, "n", 0, 1000)
        self.assertEqual((got["attributed"], got["ambiguous"]), (0, 1))
        self.assertEqual(got["clients"][0]["requests"], 0)

    def test_a_connected_client_that_never_asked_still_appears(self):
        """Being connected is worth showing on its own -- it is how an idle
        keep-alive holder is told apart from nobody being there."""
        self.conns(100.0, {"idle": 2})
        got = ninfer_metrics.clients(self.st, "n", 0, 1000)
        self.assertEqual(got["clients"][0]["client"], "idle")
        self.assertEqual(got["clients"][0]["requests"], 0)
        self.assertEqual(got["clients"][0]["peak_conns"], 2)

    def test_presence_is_the_share_of_samples_it_was_seen_in(self):
        self.conns(100.0, {"a": 1})
        self.conns(200.0, {"a": 1, "b": 1})
        got = {c["client"]: c for c in
               ninfer_metrics.clients(self.st, "n", 0, 1000)["clients"]}
        self.assertEqual(got["a"]["presence"], 1.0)
        self.assertEqual(got["b"]["presence"], 0.5)

    def test_connections_are_counted_per_address_not_per_socket(self):
        from inferwatch.ninfer import sample_connections
        self.assertIsInstance(sample_connections(9999), dict)


class TestConnectionParsing(unittest.TestCase):
    def test_peer_addresses_are_counted_by_host(self):
        from unittest import mock

        import inferwatch.ninfer as n
        out = ("0 0 10.0.0.98:8011 10.0.0.96:52311\n"
               "0 0 10.0.0.98:8011 10.0.0.96:52468\n"
               "0 0 10.0.0.98:8011 10.0.0.122:41000\n"
               "0 0 [::1]:8011 [2001:db8::5]:4000\n")
        with mock.patch.object(n.shutil, "which", return_value="/usr/bin/ss"), \
             mock.patch.object(n.subprocess, "run",
                               return_value=mock.Mock(stdout=out)):
            got = n.sample_connections(8011)
        # Two sockets from one machine are one client holding two connections.
        self.assertEqual(got["10.0.0.96"], 2)
        self.assertEqual(got["10.0.0.122"], 1)
        # An IPv6 peer is bracketed; splitting on ":" would cut the address.
        self.assertEqual(got["2001:db8::5"], 1)

    def test_no_ss_is_unknown_not_empty(self):
        from unittest import mock

        import inferwatch.ninfer as n
        with mock.patch.object(n.shutil, "which", return_value=None):
            self.assertIsNone(n.sample_connections(8011))
