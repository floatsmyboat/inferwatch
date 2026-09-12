"""Store tests: histogram percentiles, rollup correctness, retention."""

import os
import shutil
import tempfile
import time
import unittest

from inferwatch import metrics
from inferwatch.store import (HIST_BOUNDS_MS, NBUCKETS, Store, hist_add,
                           p_from_hist, request_key, sum_hists)


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
        """No samples is not the same as an empty cache: a caller must not read
        the absence of a gauge as a cache that sat at zero."""
        s = metrics.cache_summary(self.st, 1_700_000_000, 1_700_003_600)
        self.assertEqual(s["samples"], 0)
        self.assertIsNone(s["usage"]["mean"])
        self.assertIsNone(s["usage"]["max"])
        self.assertFalse(s["under_pressure"])
        self.assertIn("cache update", s["note"])

    def test_an_empty_window_reports_no_counter_totals(self):
        """Counters must read 0 rather than null: nothing was observed, and a
        sum over nothing is genuinely zero."""
        s = metrics.cache_summary(self.st, 1_700_000_000, 1_700_003_600)
        self.assertEqual(s["evictions"], 0)
        self.assertEqual(s["restores"], 0)

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


class TestPerClient(StoreCase):
    """by_client: who is calling, for which models, at what context size."""

    def setUp(self):
        super().setUp()
        self.base = time.time() - 300

    def test_aggregates_per_client(self):
        self.add(self.base, client_ip="10.0.0.1", latency_ms=100.0, output_tokens=10)
        self.add(self.base + 1, client_ip="10.0.0.1", latency_ms=100.0, output_tokens=5,
                 status=500)
        self.add(self.base + 2, client_ip="10.0.0.2", latency_ms=100.0, output_tokens=7)
        self.st.commit()
        rows = metrics.by_client(self.st, self.base - 1, self.base + 60)
        by_ip = {r["client_ip"]: r for r in rows}
        self.assertEqual(by_ip["10.0.0.1"]["requests"], 2)
        self.assertEqual(by_ip["10.0.0.1"]["errors"], 1)
        self.assertEqual(by_ip["10.0.0.1"]["output_tokens"], 15)
        self.assertEqual(by_ip["10.0.0.2"]["requests"], 1)
        # Busiest first, so the noisiest caller is the first thing read.
        self.assertEqual(rows[0]["client_ip"], "10.0.0.1")

    def test_lists_which_models_a_client_requested(self):
        for i in range(3):
            self.add(self.base + i, client_ip="10.0.0.1", model="llama3.2:3b",
                     latency_ms=10.0)
        self.add(self.base + 9, client_ip="10.0.0.1", model="qwen3:8b", latency_ms=10.0)
        self.st.commit()
        c = metrics.by_client(self.st, self.base - 1, self.base + 60)[0]
        self.assertEqual([(m["model"], m["requests"]) for m in c["models"]],
                         [("llama3.2:3b", 3), ("qwen3:8b", 1)])

    def test_requests_with_no_model_are_counted_not_dropped(self):
        """Ollama names the model on a per-request scheduler line; when that
        line is missing the request still happened, so it must not vanish from
        the client's totals or make its model list look complete."""
        self.add(self.base, client_ip="10.0.0.1", model="llama3.2:3b", latency_ms=10.0)
        self.add(self.base + 1, client_ip="10.0.0.1", model=None, latency_ms=10.0)
        self.add(self.base + 2, client_ip="10.0.0.1", model=None, latency_ms=10.0)
        self.st.commit()
        c = metrics.by_client(self.st, self.base - 1, self.base + 60)[0]
        self.assertEqual(c["requests"], 3)
        self.assertEqual(c["unattributed"], 2)
        self.assertEqual([m["model"] for m in c["models"]], ["llama3.2:3b"])

    def test_context_usage_is_computed_per_request_before_aggregating(self):
        """The whole point of the column. Pairing the peak prompt with the peak
        capacity would report 20000/262144 = 7.6%; the real worst case is the
        small-capacity request at 12000/32768 = 36.6%."""
        self.add(self.base, client_ip="10.0.0.1", latency_ms=10.0,
                 context_tokens=12000, n_ctx_slot=32768)
        self.add(self.base + 1, client_ip="10.0.0.1", latency_ms=10.0,
                 context_tokens=20000, n_ctx_slot=262144)
        self.st.commit()
        c = metrics.by_client(self.st, self.base - 1, self.base + 60)[0]
        self.assertAlmostEqual(c["ctx_usage_max"], 12000 / 32768, places=6)
        self.assertEqual(c["context_tokens_max"], 20000)
        self.assertEqual(c["n_ctx_max"], 262144)

    def test_prompt_size_reports_mean_and_peak(self):
        self.add(self.base, client_ip="10.0.0.1", latency_ms=10.0,
                 prompt_tokens_total=1000)
        self.add(self.base + 1, client_ip="10.0.0.1", latency_ms=10.0,
                 prompt_tokens_total=3000)
        self.st.commit()
        c = metrics.by_client(self.st, self.base - 1, self.base + 60)[0]
        self.assertAlmostEqual(c["prompt_tokens_mean"], 2000.0)
        self.assertEqual(c["prompt_tokens_max"], 3000)

    def test_rows_without_a_capacity_leave_usage_null(self):
        self.add(self.base, client_ip="10.0.0.1", latency_ms=10.0, context_tokens=500)
        self.st.commit()
        c = metrics.by_client(self.st, self.base - 1, self.base + 60)[0]
        self.assertIsNone(c["ctx_usage_max"])

    def test_health_traffic_is_excluded(self):
        """A client that only polls /api/ps is not an inference client."""
        self.add(self.base, client_ip="10.0.0.9", **{"class": "health"},
                 endpoint="/api/ps", latency_ms=1.0)
        self.st.commit()
        self.assertEqual(metrics.by_client(self.st, self.base - 1, self.base + 60), [])

    def test_limit_is_respected(self):
        for i in range(5):
            self.add(self.base + i, client_ip=f"10.0.0.{i}", latency_ms=10.0)
        self.st.commit()
        self.assertEqual(len(metrics.by_client(self.st, self.base - 1, self.base + 60,
                                              limit=3)), 3)


class TestConcurrency(StoreCase):
    """Concurrency is derived from request intervals, not read off the poll."""

    def setUp(self):
        super().setUp()
        # Aligned to an hour so bucket boundaries land where the arithmetic in
        # these tests expects; series buckets are floored to the step, so an
        # unaligned base silently shifts the first one.
        self.base = 1_699_999_200.0

    def req(self, start, dur, **kw):
        """One request occupying [start, start+dur]."""
        self.add(self.base + start + dur, started_ts=self.base + start,
                 latency_ms=dur * 1000.0, total_ms=dur * 1000.0, **kw)

    def window(self):
        return self.base, self.base + 600

    def test_sequential_requests_never_overlap(self):
        self.req(0, 10); self.req(20, 10); self.req(40, 10)
        self.st.commit()
        c = metrics.concurrency(self.st, *self.window())
        self.assertEqual(c["peak"], 1)
        self.assertEqual(c["requests"], 3)

    def test_two_overlapping_requests_peak_at_two(self):
        self.req(0, 30); self.req(10, 30)
        self.st.commit()
        c = metrics.concurrency(self.st, *self.window())
        self.assertEqual(c["peak"], 2)
        # 0-10 one, 10-30 two, 30-40 one: 20s at level two.
        self.assertAlmostEqual(c["seconds_at_level"]["2"], 20.0, places=3)
        self.assertAlmostEqual(c["seconds_at_level"]["1"], 20.0, places=3)

    def test_a_burst_between_polls_is_still_seen(self):
        """The whole reason for deriving this rather than sampling: three
        requests inside one second would be invisible to a 5-second gauge."""
        self.req(0, 0.4); self.req(0.1, 0.4); self.req(0.2, 0.4)
        self.st.commit()
        self.assertEqual(metrics.concurrency(self.st, *self.window())["peak"], 3)

    def test_the_mean_is_time_weighted_over_the_whole_window(self):
        """So an idle stretch pulls it down -- 'how busy was it', not 'how busy
        when busy'."""
        self.req(0, 60)            # 60 request-seconds in a 600s window
        self.st.commit()
        c = metrics.concurrency(self.st, *self.window())
        self.assertAlmostEqual(c["mean"], 0.1, places=4)

    def test_an_interval_is_clipped_to_the_window(self):
        """A request that began before the window still contributes the part
        inside it."""
        self.req(-100, 150)        # runs from -100 to +50
        self.st.commit()
        c = metrics.concurrency(self.st, *self.window())
        self.assertEqual(c["peak"], 1)
        self.assertAlmostEqual(c["seconds_at_level"]["1"], 50.0, places=3)

    def test_slot_capacity_and_saturation(self):
        self.req(0, 30); self.req(10, 30)
        self.st.insert_ps_sample(self.base + 1, 1, [], 2, slots=2)
        self.st.commit()
        c = metrics.concurrency(self.st, *self.window())
        self.assertEqual(c["slots"], 2)
        self.assertAlmostEqual(c["saturated_s"], 20.0, places=3)
        self.assertIsNotNone(c["utilisation"])

    def test_unknown_capacity_is_none_not_zero(self):
        """No load observed means capacity is unknown; zero would make
        utilisation look infinite and saturation look total."""
        self.req(0, 10)
        self.st.insert_ps_sample(self.base + 1, 1, [], 1)      # slots omitted
        self.st.commit()
        c = metrics.concurrency(self.st, *self.window())
        self.assertIsNone(c["slots"])
        self.assertIsNone(c["utilisation"])
        self.assertIsNone(c["saturated_fraction"])

    def test_rows_without_a_start_are_skipped(self):
        """started_ts comes from the access log's latency; without it the
        interval is unknown and must not be invented."""
        self.add(self.base + 5, latency_ms=None)
        self.st.commit()
        self.assertEqual(metrics.concurrency(self.st, *self.window())["requests"], 0)

    def test_health_checks_do_not_count_as_traffic(self):
        self.req(0, 30, **{"class": "health"})
        self.req(1, 30, **{"class": "health"})
        self.st.commit()
        self.assertEqual(metrics.concurrency(self.st, *self.window())["peak"], 0)

    def test_the_series_splits_a_segment_across_buckets(self):
        """A request spanning a bucket boundary is charged to both, not to
        whichever one it started in."""
        self.req(0, 120)
        self.st.commit()
        ts = metrics.concurrency_series(self.st, self.base, self.base + 180, step=60)
        self.assertEqual(ts["series"]["peak"][0], 1)
        self.assertEqual(ts["series"]["peak"][1], 1)
        self.assertAlmostEqual(ts["series"]["mean"][0], 1.0, places=3)

    def test_a_reader_predating_the_slots_column_degrades(self):
        """Migrations only run read-write, so the MCP server can be new code on
        an old database; a missing column must not raise."""
        self.st.db.execute("ALTER TABLE ps_samples RENAME TO ps_old")
        self.st.db.execute("CREATE TABLE ps_samples (ts REAL PRIMARY KEY,"
                           " loaded_count INTEGER, models_json TEXT, inflight INTEGER)")
        self.st.db.commit()
        self.assertFalse(self.st.has_column("ps_samples", "slots"))
        c = metrics.concurrency(self.st, *self.window())
        self.assertIsNone(c["slots"])
        metrics.concurrency_series(self.st, *self.window())   # must not raise
        metrics.loaded_models(self.st)


class TestRawCoverage(StoreCase):
    """Raw-only breakdowns must distinguish "not stored" from "nothing happened"."""

    def test_coverage_is_complete_when_rows_reach_back_far_enough(self):
        now = time.time()
        self.add(now - 3600, latency_ms=10.0)
        self.st.commit()
        cov = metrics.coverage(self.st, now - 1800)
        self.assertTrue(cov["complete"])
        self.assertAlmostEqual(cov["covers_from"], now - 3600)

    def test_coverage_is_incomplete_when_the_window_predates_the_oldest_row(self):
        now = time.time()
        self.add(now - 3600, latency_ms=10.0)
        self.st.commit()
        self.assertFalse(metrics.coverage(self.st, now - 30 * 86400)["complete"])

    def test_an_empty_table_is_never_reported_as_complete(self):
        """With nothing stored, no window is covered -- returning True here
        would let an empty result read as a genuinely idle window."""
        cov = metrics.coverage(self.st, time.time() - 60)
        self.assertIsNone(cov["covers_from"])
        self.assertFalse(cov["complete"])

    def test_coverage_tracks_retention_not_the_rollup_threshold(self):
        """The bound is how long raw rows are KEPT, not the 6h point where other
        queries switch to rollups. A 24h window over 7 days of rows is complete
        even though summary() reports exact=False for it."""
        now = time.time()
        self.add(now - 7 * 86400, latency_ms=10.0)
        self.add(now - 60, latency_ms=10.0)
        self.st.commit()
        self.assertTrue(metrics.coverage(self.st, now - 86400)["complete"])
        self.assertFalse(metrics.summary(self.st, now - 86400, now)["exact"])

    def test_the_raw_only_breakdowns_still_answer_a_long_window(self):
        """They read raw rows directly, so a 24h range works as long as the rows
        are there -- the finding was silent emptiness, not a hard limit."""
        now = time.time()
        self.add(now - 20 * 3600, endpoint="/api/chat", status=500,
                 client_ip="10.0.0.1", latency_ms=10.0, ttft_ms=5.0)
        self.st.commit()
        start = now - 86400
        self.assertEqual(len(metrics.by_endpoint(self.st, start, now)), 1)
        self.assertEqual(len(metrics.status_breakdown(self.st, start, now)), 1)
        self.assertEqual(len(metrics.recent_errors(self.st, start, now)), 1)
        self.assertEqual(len(metrics.slowest(self.st, start, now, "ttft_ms")), 1)
        self.assertEqual(len(metrics.recent_requests(self.st, start, now)), 1)
        self.assertEqual(len(metrics.by_client(self.st, start, now)), 1)


class TestDedupeMigration(unittest.TestCase):
    """The one-time dedupe backfill must be one-time, and bounded."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "t.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _legacy_db(self, rows=12, dup_ts=None):
        """Rewind a real database to the pre-schema-2 state.

        Built from the live DDL rather than a hand-written subset, so the
        migration under test meets the same columns and indexes it would on a
        real upgrade -- a trimmed fixture silently diverges from it.
        """
        import sqlite3
        st = Store(self.path)
        st.db.close()
        db = sqlite3.connect(self.path)
        db.execute("DROP INDEX IF EXISTS idx_req_dedupe")
        db.execute("DROP INDEX IF EXISTS idx_ev_dedupe")
        for i in range(rows):
            db.execute("INSERT INTO requests (ts,endpoint,class,method,client_ip,"
                       "status,latency_ms,task_id) VALUES (?,?,?,?,?,?,?,?)",
                       (1_700_000_000 + i, "/api/chat", "inference", "POST",
                        "10.0.0.1", 200, 10.0, i))
        for _ in range(dup_ts or 0):
            db.execute("INSERT INTO requests (ts,endpoint,class,method,client_ip,"
                       "status,latency_ms,task_id) VALUES (?,?,?,?,?,?,?,?)",
                       (1_700_000_000.5, "/api/chat", "inference", "POST",
                        "10.0.0.1", 200, 10.0, 7))
        db.execute("UPDATE requests SET dedupe_key = NULL")
        db.execute("UPDATE events SET dedupe_key = NULL")
        db.commit(); db.close()

    def test_backfill_fills_every_row_across_batch_boundaries(self):
        self._legacy_db(rows=12)
        st = Store(self.path)
        try:
            n = st.query("SELECT COUNT(*) n FROM requests WHERE dedupe_key IS NULL")[0]["n"]
            self.assertEqual(n, 0)
            self.assertEqual(st.query("SELECT COUNT(*) n FROM requests")[0]["n"], 12)
        finally:
            st.db.close()

    def test_batching_walks_a_set_larger_than_one_batch(self):
        """Each pass must shrink the candidate set, or the loop never ends."""
        self._legacy_db(rows=12)
        import sqlite3
        db = sqlite3.connect(self.path)
        db.execute("UPDATE requests SET dedupe_key = NULL")
        db.commit(); db.close()
        st = Store(self.path, read_only=False)
        try:
            # __init__ already keyed them; blank them again to drive the loop
            # directly with a batch smaller than the row count.
            st.db.execute("UPDATE requests SET dedupe_key = NULL")
            st.db.execute("DROP INDEX IF EXISTS idx_req_dedupe")
            done = st._backfill_dedupe("requests", request_key, batch=5)
            self.assertEqual(done, 12)
            self.assertEqual(
                st.query("SELECT COUNT(*) n FROM requests WHERE dedupe_key IS NULL")[0]["n"], 0)
        finally:
            st.db.close()

    def test_the_expensive_steps_are_skipped_once_the_index_exists(self):
        """They ran on every startup: a full scan for NULLs plus a whole-table
        GROUP BY. With the unique index present neither can find anything."""
        self._legacy_db(rows=5)
        st = Store(self.path)
        st.db.close()
        st2 = Store(self.path)
        try:
            self.assertTrue(st2._has_index("idx_req_dedupe"))
            sqls = []
            st2.db.set_trace_callback(sqls.append)
            st2._migrate()
            st2.db.set_trace_callback(None)
            joined = " | ".join(" ".join(q.split()) for q in sqls)
            self.assertTrue(sqls, "trace callback saw nothing -- test is not looking")
            self.assertNotIn("dedupe_key IS NULL", joined)
            self.assertNotIn("GROUP BY dedupe_key", joined)
        finally:
            st2.db.close()

    def test_duplicates_present_before_the_index_are_collapsed(self):
        """Three byte-identical rows (a replayed journal) must become one, or
        CREATE UNIQUE INDEX would fail and the migration would abort."""
        self._legacy_db(rows=0, dup_ts=3)
        st = Store(self.path)
        try:
            self.assertEqual(st.query("SELECT COUNT(*) n FROM requests")[0]["n"], 1)
            self.assertTrue(st._has_index("idx_req_dedupe"))
        finally:
            st.db.close()


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
