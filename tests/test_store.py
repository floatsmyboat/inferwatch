"""Store tests: histogram percentiles, rollup correctness, retention."""

import os
import shutil
import tempfile
import time
import unittest

from inferwatch import metrics
from inferwatch.store import (HIST_BOUNDS_MS, NBUCKETS, Store, hist_add,
                           p_from_hist, sum_hists)


class TestHistogram(unittest.TestCase):
    def test_percentile_reports_bucket_upper_bound(self):
        h = [0] * NBUCKETS
        for v in [80, 90, 120]:
            hist_add(h, v)
        # 80 and 90 land in the <=100ms bucket, 120 in the <=200ms bucket.
        # The median (90ms) therefore reports as 100ms -- the bucket's upper
        # bound, an over-estimate by design, never an interpolated guess.
        self.assertEqual(p_from_hist(h, 0.5), 100.0)
        self.assertEqual(p_from_hist(h, 0.99), 200.0)

    def test_empty_histogram_is_none(self):
        self.assertIsNone(p_from_hist([0] * NBUCKETS, 0.5))

    def test_histograms_are_additive(self):
        import json
        a = [0] * NBUCKETS; hist_add(a, 80)
        b = [0] * NBUCKETS; hist_add(b, 25000)
        merged = sum_hists([json.dumps(a), json.dumps(b)])
        self.assertEqual(sum(merged), 2)
        # p99 of the merged pair must reach the slow bucket
        self.assertGreaterEqual(p_from_hist(merged, 0.99), 25000)

    def test_overflow_goes_to_last_bucket(self):
        h = [0] * NBUCKETS
        hist_add(h, 10_000_000)
        self.assertEqual(h[-1], 1)
        self.assertEqual(p_from_hist(h, 0.5), float(HIST_BOUNDS_MS[-1]))


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "t.db")
        self.st = Store(self.path)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def add(self, ts, **kw):
        row = {"ts": ts, "endpoint": "/api/chat", "method": "POST", "class": "inference",
               "status": 200, "model": "m:latest", "attribution": "exact"}
        row.update(kw)
        self.st.insert_request(row)


class TestRollups(StoreCase):
    def test_rollup_totals_match_raw(self):
        base = 1_700_000_000
        for i in range(10):
            self.add(base + i, ttft_ms=100.0 + i, latency_ms=500.0, decode_ms=1000.0,
                     output_tokens=10, prompt_tokens=5, cached_tokens=2, queue_ms=50.0)
        self.st.commit()
        self.st.rebuild_rollups(base, base + 60)
        rows = self.st.query("SELECT * FROM rollup_1m")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["req_count"], 10)
        self.assertEqual(r["out_tokens"], 100)
        self.assertEqual(r["in_tokens"], 50)
        self.assertEqual(r["cached_tokens"], 20)
        self.assertEqual(r["ttft_n"], 10)

    def test_rollup_is_idempotent(self):
        base = 1_700_000_000
        for i in range(5):
            self.add(base + i, ttft_ms=100.0, latency_ms=200.0, output_tokens=1)
        self.st.commit()
        self.st.rebuild_rollups(base, base + 60)
        first = [dict(r) for r in self.st.query("SELECT * FROM rollup_1m ORDER BY bucket")]
        self.st.rebuild_rollups(base, base + 60)
        self.st.rebuild_rollups(base, base + 60)
        again = [dict(r) for r in self.st.query("SELECT * FROM rollup_1m ORDER BY bucket")]
        self.assertEqual(first, again)

    def test_errors_counted_separately(self):
        base = 1_700_000_000
        self.add(base, status=200, latency_ms=10.0)
        self.add(base + 1, status=500, latency_ms=10.0)
        self.add(base + 2, status=404, latency_ms=10.0)
        self.st.commit()
        self.st.rebuild_rollups(base, base + 60)
        r = self.st.query("SELECT * FROM rollup_1m")[0]
        self.assertEqual(r["req_count"], 3)
        self.assertEqual(r["err_count"], 2)


class TestRetention(StoreCase):
    def test_prune_drops_raw_but_keeps_rollups(self):
        now = time.time()
        old = now - 30 * 86400
        self.add(old, ttft_ms=100.0, latency_ms=200.0, output_tokens=7)
        self.add(now - 60, ttft_ms=100.0, latency_ms=200.0, output_tokens=7)
        self.st.commit()
        self.st.rebuild_rollups(old, now + 60)
        rollups_before = self.st.query("SELECT COUNT(*) n FROM rollup_1h")[0]["n"]

        dropped = self.st.prune(raw_retention_days=7)
        self.assertEqual(dropped["requests"], 1)
        self.assertEqual(self.st.query("SELECT COUNT(*) n FROM requests")[0]["n"], 1)
        # the aggregate for the pruned day survives -- that is the whole point
        self.assertEqual(
            self.st.query("SELECT COUNT(*) n FROM rollup_1h")[0]["n"], rollups_before)


class CacheStoreCase(StoreCase):
    def add_cache(self, ts, **kw):
        row = {"ts": ts, "model": "m:latest", "prompts": 10, "used_mib": 4096.0,
               "limit_mib": 8192.0, "usage": 0.5, "token_limit": 32768,
               "ckpt_total": 32, "evictions": 0, "evict_crowded": 0,
               "evict_invalidated": 0, "restores": 0, "ckpt_created": 0, "saves": 0}
        row.update(kw)
        self.st.insert_cache_sample(row)


class TestPromptCacheStore(CacheStoreCase):
    def test_round_trip(self):
        self.add_cache(1_700_000_000, prompts=30, used_mib=8010.969, usage=0.9779,
                       update_ms=364.22, evictions=3, evict_crowded=2,
                       evict_invalidated=1)
        self.st.commit()
        r = self.st.query("SELECT * FROM ollama_cache_samples")[0]
        self.assertEqual(r["prompts"], 30)
        self.assertAlmostEqual(r["used_mib"], 8010.969)
        self.assertAlmostEqual(r["update_ms"], 364.22)
        self.assertEqual(r["evict_crowded"], 2)

    def test_replaying_the_same_log_is_a_no_op(self):
        """Re-reading a journal must not double-count; the timestamp is the
        identity, and a second read reproduces an identical row."""
        for _ in range(3):
            self.add_cache(1_700_000_000, prompts=30)
        self.st.commit()
        self.assertEqual(
            self.st.query("SELECT COUNT(*) n FROM ollama_cache_samples")[0]["n"], 1)

    def test_prune_uses_the_sample_retention(self):
        now = time.time()
        self.add_cache(now - 60 * 86400)
        self.add_cache(now - 60)
        self.st.commit()
        dropped = self.st.prune(raw_retention_days=7, sample_retention_days=30)
        self.assertEqual(dropped["ollama_cache_samples"], 1)
        self.assertEqual(
            self.st.query("SELECT COUNT(*) n FROM ollama_cache_samples")[0]["n"], 1)


class TestPromptCacheMetrics(CacheStoreCase):
    def test_summary_aggregates_gauges_and_totals_counters(self):
        base = 1_700_000_000
        self.add_cache(base, usage=0.2, used_mib=1638.4, prompts=5, update_ms=10.0,
                       evictions=1, ckpt_used=3)
        self.add_cache(base + 60, usage=0.9, used_mib=7372.8, prompts=28,
                       update_ms=500.0, evictions=4, ckpt_used=9)
        self.st.commit()
        s = metrics.cache_summary(self.st, base - 1, base + 120)

        self.assertEqual(s["samples"], 2)
        self.assertAlmostEqual(s["usage"]["mean"], 0.55)
        self.assertAlmostEqual(s["usage"]["max"], 0.9)
        self.assertAlmostEqual(s["usage"]["last"], 0.9)
        self.assertEqual(s["prompts"]["max"], 28)
        self.assertEqual(s["limit_mib"], 8192.0)
        self.assertAlmostEqual(s["update_ms"]["max"], 500.0)
        # Counters are summed, never averaged: they are already deltas.
        self.assertEqual(s["evictions"], 5)
        self.assertEqual(s["checkpoints"]["used"]["max"], 9)
        self.assertEqual(s["checkpoints"]["total"], 32)

    def test_pressure_is_flagged_on_the_peak_not_the_mean(self):
        """A cache that spent one minute full evicted during that minute; a
        comfortable average does not undo it."""
        base = 1_700_000_000
        self.add_cache(base, usage=0.05)
        self.add_cache(base + 60, usage=0.98)
        self.st.commit()
        s = metrics.cache_summary(self.st, base - 1, base + 120)
        self.assertLess(s["usage"]["mean"], metrics.CACHE_PRESSURE)
        self.assertTrue(s["under_pressure"])

    def test_an_empty_window_says_why(self):
        """No samples is not the same as an empty cache, and the difference is
        usually OLLAMA_DEBUG being off."""
        s = metrics.cache_summary(self.st, 1_700_000_000, 1_700_003_600)
        self.assertEqual(s["samples"], 0)
        self.assertIsNone(s["usage"]["mean"])
        self.assertFalse(s["under_pressure"])
        self.assertIn("OLLAMA_DEBUG", s["note"])

    def test_series_averages_gauges_and_sums_counters_per_bucket(self):
        base = 1_700_000_000
        for i, usage in enumerate((0.2, 0.4)):
            self.add_cache(base + i, usage=usage, evictions=2)
        self.st.commit()
        ser = metrics.cache_series(self.st, base, base + 300, step=300)
        self.assertAlmostEqual(ser["series"]["usage"][0], 0.3)
        self.assertEqual(ser["series"]["evictions"][0], 4)
        self.assertEqual(ser["counts"][0], 2)

    def test_series_leaves_unsampled_buckets_null(self):
        """An idle stretch must not draw a line down to zero: nothing was
        measured there, which is different from a cache that emptied."""
        base = 1_700_000_000
        self.add_cache(base, usage=0.5)
        self.st.commit()
        ser = metrics.cache_series(self.st, base, base + 1200, step=300)
        self.assertAlmostEqual(ser["series"]["usage"][0], 0.5)
        self.assertIsNone(ser["series"]["usage"][1])
        self.assertIsNone(ser["series"]["evictions"][1])


class TestLiveKvOccupancy(StoreCase):
    def test_occupancy_comes_from_the_token_and_capacity_pair(self):
        base = time.time() - 60
        self.add(base, latency_ms=100.0, context_tokens=16384, n_ctx_slot=32768)
        self.add(base + 1, latency_ms=100.0, context_tokens=32700, n_ctx_slot=32768)
        self.st.commit()
        s = metrics.summary(self.st, base - 1, base + 60)
        self.assertTrue(s["exact"])
        self.assertEqual(s["ctx_usage"]["n"], 2)
        self.assertAlmostEqual(s["ctx_usage"]["max"], 32700 / 32768)

    def test_rows_without_a_capacity_are_skipped_not_guessed(self):
        """Rows written before the column existed keep NULL; back-computing
        occupancy from an assumed context size would invent the number."""
        base = time.time() - 60
        self.add(base, latency_ms=100.0, context_tokens=16384)
        self.st.commit()
        self.assertIsNone(metrics.summary(self.st, base - 1, base + 60)["ctx_usage"])

    def test_long_windows_report_null_rather_than_a_rollup_guess(self):
        """The rollups aggregate per model and class, not per slot, so the
        capacity is not in them."""
        now = time.time()
        self.add(now - 86400, latency_ms=100.0, context_tokens=16384, n_ctx_slot=32768)
        self.st.commit()
        s = metrics.summary(self.st, now - 7 * 86400, now)
        self.assertFalse(s["exact"])
        self.assertIsNone(s["ctx_usage"])


class TestMetricsSources(StoreCase):
    def test_short_window_is_exact_long_window_is_not(self):
        now = time.time()
        for i in range(20):
            self.add(now - i * 10, ttft_ms=1000.0, latency_ms=2000.0, output_tokens=5,
                     prompt_tokens=3, decode_ms=500.0)
        self.st.commit()
        self.st.rebuild_rollups(now - 3600, now + 60)

        fresh = metrics.summary(self.st, now - 900, now + 1)
        self.assertTrue(fresh["exact"])
        self.assertEqual(fresh["requests"], 20)
        self.assertAlmostEqual(fresh["ttft_ms"]["p50"], 1000.0)

        wide = metrics.summary(self.st, now - 30 * 86400, now + 1)
        self.assertFalse(wide["exact"])
        self.assertEqual(wide["requests"], 20)

    def test_health_traffic_excluded_from_inference_rates(self):
        now = time.time()
        self.add(now - 5, output_tokens=10, ttft_ms=50.0, latency_ms=100.0)
        for i in range(50):
            self.add(now - i, **{"class": "health", "endpoint": "/api/ps",
                                 "method": "GET", "latency_ms": 0.03})
        self.st.commit()
        s = metrics.summary(self.st, now - 900, now + 1)
        self.assertEqual(s["requests"], 1)          # inference only
        self.assertEqual(s["requests_all"], 51)     # everything seen

    def test_timeseries_gaps_stay_null(self):
        now = time.time()
        self.add(now - 5, output_tokens=10, ttft_ms=50.0, latency_ms=100.0, decode_ms=10.0)
        self.st.commit()
        ts = metrics.timeseries(self.st, now - 900, now + 1, step=60)
        vals = ts["series"]["req_per_s"]
        self.assertTrue(any(v is not None for v in vals))
        self.assertTrue(any(v is None for v in vals),
                        "idle buckets must be null, not zero")

    def test_bad_window_rejected(self):
        with self.assertRaises(ValueError):
            metrics.parse_window("yesterday")


if __name__ == "__main__":
    unittest.main()
