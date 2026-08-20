"""vLLM collection tests.

The fixture below is trimmed from a real vLLM /metrics response, keeping the
shapes that matter: labelled gauges, labelled counters, a cumulative histogram
with an +Inf bucket, and the process start time used for restart detection.
"""

import json
import os
import shutil
import tempfile
import unittest

from inferwatch import vllm_metrics
from inferwatch.store import Store
from inferwatch.vllm import (MinuteAccumulator, Snapshot, VllmCollector,
                             json_bounds, label_key, parse_prometheus)

SCRAPE_1 = """
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.5
vllm:prompt_tokens_total{engine="0",model_name="m"} 1000.0
vllm:generation_tokens_total{engine="0",model_name="m"} 500.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 10.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 2.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="m"} 4.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0",model_name="m"} 9.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m"} 12.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 12.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 6.0
process_start_time_seconds 1700000000.0
"""

SCRAPE_2 = """
vllm:num_requests_running{engine="0",model_name="m"} 4.0
vllm:num_requests_waiting{engine="0",model_name="m"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.9
vllm:prompt_tokens_total{engine="0",model_name="m"} 1300.0
vllm:generation_tokens_total{engine="0",model_name="m"} 700.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 13.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 2.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="m"} 5.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0",model_name="m"} 11.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m"} 15.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 15.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 9.0
process_start_time_seconds 1700000000.0
"""

# Same as scrape 2 but the engine restarted: counters begin again from a low
# value and the start time changed.
SCRAPE_RESTARTED = """
vllm:num_requests_running{engine="0",model_name="m"} 1.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 12.0
vllm:generation_tokens_total{engine="0",model_name="m"} 5.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="m"} 1.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0",model_name="m"} 1.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m"} 1.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 1.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 0.05
process_start_time_seconds 1700009999.0
"""

TTFT = "vllm:time_to_first_token_seconds"


class TestParse(unittest.TestCase):
    def test_labels_and_values(self):
        got = parse_prometheus(SCRAPE_1)
        names = {n for n, _l, _v in got}
        self.assertIn("vllm:num_requests_running", names)
        running = [(l, v) for n, l, v in got if n == "vllm:num_requests_running"][0]
        self.assertEqual(running[0]["model_name"], "m")
        self.assertEqual(running[1], 2.0)

    def test_inf_bucket_parsed(self):
        buckets = [(l["le"], v) for n, l, v in parse_prometheus(SCRAPE_1)
                   if n == TTFT + "_bucket"]
        self.assertIn(("+Inf", 12.0), buckets)

    def test_comments_and_garbage_skipped(self):
        got = parse_prometheus("# HELP x y\n\nnot a metric line at all !!\nvalid_metric 1.0\n")
        self.assertEqual(got, [("valid_metric", {}, 1.0)])

    def test_nan_is_dropped(self):
        self.assertEqual(parse_prometheus("some_metric NaN\n"), [])

    def test_label_key_is_stable_and_drops_le(self):
        a = label_key({"model_name": "m", "engine": "0", "le": "0.1"})
        b = label_key({"engine": "0", "model_name": "m", "le": "9"})
        self.assertEqual(a, b)
        self.assertEqual(a, "engine=0,model_name=m")


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        self.s = Snapshot(1000.0, parse_prometheus(SCRAPE_1))

    def test_classification(self):
        self.assertIn(("vllm:num_requests_running", "engine=0,model_name=m"), self.s.gauges)
        self.assertIn(("vllm:prompt_tokens_total", "engine=0,model_name=m"), self.s.counters)
        self.assertIn(TTFT, self.s.hist)

    def test_model_and_engine_start(self):
        self.assertEqual(self.s.model, "m")
        self.assertEqual(self.s.engine_start, 1700000000.0)

    def test_histogram_count_and_sum(self):
        self.assertEqual(self.s.hist_count[TTFT], 12.0)
        self.assertEqual(self.s.hist_sum[TTFT], 6.0)

    def test_bounds_sorted_with_inf_last(self):
        b = self.s.bounds(TTFT)
        self.assertEqual(b[:2], [0.1, 1.0])
        self.assertEqual(b[-1], float("inf"))


class TestJsonBounds(unittest.TestCase):
    def test_inf_becomes_null_so_browsers_can_parse_it(self):
        """json.dumps would emit the bare token Infinity, which JSON.parse
        rejects -- the dashboard would throw on it."""
        raw = json_bounds([0.1, 1.0, float("inf")])
        self.assertEqual(raw, "[0.1, 1.0, null]")

        def reject_constants(c):
            raise AssertionError(f"non-strict JSON token: {c}")

        self.assertEqual(json.loads(raw, parse_constant=reject_constants),
                         [0.1, 1.0, None])


class TestDeltas(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))
        self.c = VllmCollector(self.st, "v1", {"url": "http://x"}, lambda: 10.0)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def ingest(self, text, ts):
        snap = Snapshot(ts, parse_prometheus(text))
        return self.c.ingest(snap)

    def test_first_scrape_emits_no_counter_deltas(self):
        """With nothing to compare against, a cumulative counter says nothing
        about the interval -- emitting its absolute value would be a spike."""
        self.ingest(SCRAPE_1, 1000.0)
        self.assertEqual(self.c.acc.counter, {})

    def test_counter_deltas(self):
        self.ingest(SCRAPE_1, 1000.0)
        self.ingest(SCRAPE_2, 1010.0)
        got = {m: v for (m, _l), v in self.c.acc.counter.items()}
        self.assertEqual(got["vllm:prompt_tokens_total"], 300.0)
        self.assertEqual(got["vllm:generation_tokens_total"], 200.0)

    def test_histogram_delta_is_per_bucket_not_cumulative(self):
        self.ingest(SCRAPE_1, 1000.0)
        self.ingest(SCRAPE_2, 1010.0)
        h = self.c.acc.hist[TTFT]
        # cumulative 4/9/12 -> 5/11/15 means per-bucket 4/5/3 -> 5/6/4
        self.assertEqual(h["counts"], [1.0, 1.0, 1.0])
        self.assertEqual(h["n"], 3.0)
        self.assertAlmostEqual(h["sum"], 3.0)

    def test_restart_is_detected_and_the_interval_dropped(self):
        self.ingest(SCRAPE_1, 1000.0)
        self.ingest(SCRAPE_2, 1010.0)
        before = dict(self.c.acc.counter)          # keyed by (metric, labels)
        # Stay inside the same minute bucket: crossing one flushes and clears
        # the accumulator, which would mask what this test is checking.
        self.ingest(SCRAPE_RESTARTED, 1015.0)
        # Nothing is added for the interval spanning the restart: the counters
        # went backwards, so a delta would be meaningless (and negative).
        self.assertEqual(self.c.acc.counter, before)

    def test_restart_still_records_gauges(self):
        """A restart invalidates counter deltas, not the current gauge reading."""
        self.ingest(SCRAPE_1, 1000.0)
        self.ingest(SCRAPE_RESTARTED, 1005.0)
        key = ("vllm:num_requests_running", "engine=0,model_name=m")
        self.assertEqual(self.c.acc.gauge[key][1], 2)   # sampled twice

    def test_gauges_recorded_every_scrape(self):
        self.ingest(SCRAPE_1, 1000.0)
        self.ingest(SCRAPE_2, 1010.0)
        key = ("vllm:kv_cache_usage_perc", "engine=0,model_name=m")
        total, n, lo, hi = self.c.acc.gauge[key]
        self.assertEqual(n, 2)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.9)

    def test_minute_rollover_flushes(self):
        self.ingest(SCRAPE_1, 1000.0)      # minute bucket 960
        self.ingest(SCRAPE_2, 1010.0)
        self.ingest(SCRAPE_2, 1100.0)      # minute bucket 1080 -> flush
        rows = self.st.query("SELECT COUNT(*) n FROM vllm_samples")[0]["n"]
        self.assertGreater(rows, 0)


class TestAccumulator(unittest.TestCase):
    def test_bounds_change_restarts_the_row(self):
        """If vLLM restarts with a different bucket layout, adding the arrays
        would misalign counts into the wrong buckets."""
        acc = MinuteAccumulator()
        acc.start(60)
        acc.add_hist("m", [1.0, 2.0], [1, 1], 2, 3.0)
        acc.add_hist("m", [1.0, 5.0], [7, 7], 14, 20.0)
        self.assertEqual(acc.hist["m"]["bounds"], [1.0, 5.0])
        self.assertEqual(acc.hist["m"]["counts"], [7, 7])


class TestQueries(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def add_hist(self, ts, counts, n, total, metric=TTFT):
        self.st.insert_vllm_hist([(ts, "v1", metric, "m",
                                  json.dumps([0.1, 1.0, None]), json.dumps(counts),
                                  n, total)])
        self.st.commit()

    def test_percentile_is_bucket_upper_bound_in_ms(self):
        self.add_hist(600, [8, 1, 1], 10, 1.0)
        got = vllm_metrics.percentiles(self.st, "v1", TTFT, 0, 10_000)
        self.assertEqual(got["p50"], 100.0)      # 0.1s -> 100ms
        self.assertTrue(got["bucketed"])
        self.assertEqual(got["observations"], 10)

    def test_mean_uses_exact_sum_and_count(self):
        self.add_hist(600, [8, 1, 1], 10, 2.0)   # 2s over 10 observations
        got = vllm_metrics.percentiles(self.st, "v1", TTFT, 0, 10_000)
        self.assertAlmostEqual(got["mean"], 200.0)   # 0.2s -> 200ms

    def test_overflow_bucket_reports_last_finite_bound(self):
        self.add_hist(600, [0, 0, 5], 5, 100.0)
        got = vllm_metrics.percentiles(self.st, "v1", TTFT, 0, 10_000)
        self.assertEqual(got["p50"], 1000.0)     # last finite bound, 1.0s

    def test_histograms_add_across_rows(self):
        self.add_hist(600, [1, 0, 0], 1, 0.05)
        self.add_hist(660, [1, 0, 0], 1, 0.05)
        h = vllm_metrics.hist_range(self.st, "v1", TTFT, 0, 10_000)
        self.assertEqual(h["counts"], [2, 0, 0])
        self.assertEqual(h["observations"], 2)

    def test_mismatched_bounds_are_skipped_not_blended(self):
        self.add_hist(600, [1, 0, 0], 1, 0.05)
        self.st.insert_vllm_hist([(660, "v1", TTFT, "m", json.dumps([0.5, 9.0, None]),
                                  json.dumps([3, 3, 3]), 9, 5.0)])
        self.st.commit()
        h = vllm_metrics.hist_range(self.st, "v1", TTFT, 0, 10_000)
        self.assertEqual(h["counts"], [1, 0, 0])
        self.assertEqual(h["rows_skipped"], 1)

    def test_itl_falls_back_across_vllm_naming(self):
        """vLLM has spelled inter-token latency three ways; whichever has data
        should be used."""
        self.add_hist(600, [2, 1, 0], 3, 0.3,
                      metric="vllm:inter_token_latency_seconds")
        got = vllm_metrics.first_with_data(
            self.st, "v1", vllm_metrics.ITL_CANDIDATES, 0, 10_000)
        self.assertEqual(got["metric"], "vllm:inter_token_latency_seconds")
        self.assertEqual(got["observations"], 3)

    def test_no_data_returns_nulls_not_zeros(self):
        got = vllm_metrics.percentiles(self.st, "v1", TTFT, 0, 10_000)
        self.assertIsNone(got["p50"])
        self.assertIsNone(got["mean"])


if __name__ == "__main__":
    unittest.main()
