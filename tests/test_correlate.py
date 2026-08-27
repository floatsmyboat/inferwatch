"""Correlator tests: joining llama.cpp task timings to HTTP access lines."""

import unittest

from inferwatch.collect import Correlator


def task_lines(task, slot=0, prompt_total=1408, ttft=1254.52, prompt_tok=55,
               decode=14591.31, out_tok=416, truncated=0):
    """The slot-line sequence ollama emits for one completed request."""
    return [
        f"slot launch_slot_: id  {slot} | task {task} | processing task, is_child = 0",
        f"slot   operator(): id  {slot} | task {task} | new prompt, n_ctx_slot = 131072,"
        f" n_keep = 4, task.n_tokens = {prompt_total}",
        f"slot print_timing: id  {slot} | task {task} | prompt eval time =    {ttft} ms /"
        f"    {prompt_tok} tokens (   22.81 ms per token,    43.84 tokens per second)",
        f"slot print_timing: id  {slot} | task {task} |        eval time =   {decode} ms /"
        f"   {out_tok} tokens (   35.16 ms per token,    28.44 tokens per second)",
        f"slot print_timing: id  {slot} | task {task} |       total time =   {ttft + decode} ms /"
        f"   {prompt_tok + out_tok} tokens",
        f"slot      release: id  {slot} | task {task} | stop processing: n_tokens = 5000,"
        f" truncated = {truncated}",
    ]


GIN = ('[GIN] 2026/08/19 - 11:17:18 | {status} | {lat} |'
       '       192.0.2.10 | POST     "{path}"')
SCHED = ('time=2026-08-19T11:17:18.443-05:00 level=DEBUG source=sched.go:489'
         ' msg="context for request finished"'
         ' runner.name=registry.ollama.ai/library/{model}'
         ' runner.model=/models/blobs/sha256-abc runner.num_ctx=131072')


class Harness:
    def __init__(self):
        self.requests = []
        self.events = []
        self.live = []
        self.cache = []
        self.corr = Correlator(on_request=self.requests.append,
                               on_event=self.events.append,
                               on_live=self.live.append,
                               on_cache=self.cache.append)
        self.t = 1000.0

    def feed(self, *lines):
        for line in lines:
            self.t += 0.01
            self.corr.feed(self.t, line)
        return self

    def finish(self):
        self.corr.tick(self.t + 3600)
        return self


class TestSingleRequest(unittest.TestCase):
    def test_exact_join_and_derived_fields(self):
        h = Harness()
        h.feed(*task_lines(6763),
               GIN.format(status=200, lat="16.862061865s", path="/v1/chat/completions"),
               SCHED.format(model="llama3.2:3b")).finish()

        self.assertEqual(len(h.requests), 1)
        r = h.requests[0]
        self.assertEqual(r["attribution"], "exact")
        self.assertEqual(r["model"], "llama3.2:3b")
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["client_ip"], "192.0.2.10")
        self.assertEqual(r["output_tokens"], 416)
        self.assertEqual(r["prompt_tokens"], 55)
        self.assertAlmostEqual(r["ttft_ms"], 1254.52)
        # cached = full prompt - what was actually evaluated
        self.assertEqual(r["cached_tokens"], 1408 - 55)
        # queue = wall latency - runner time
        self.assertAlmostEqual(r["queue_ms"], 16862.061865 - (1254.52 + 14591.31), places=4)
        self.assertEqual(r["task_id"], 6763)

    def test_queue_never_negative(self):
        """A runner time slightly exceeding measured wall latency must clamp,
        not produce a negative wait."""
        h = Harness()
        h.feed(*task_lines(1, ttft=1000.0, decode=9000.0),
               GIN.format(status=200, lat="5s", path="/api/chat"),
               SCHED.format(model="m:latest")).finish()
        self.assertEqual(h.requests[0]["queue_ms"], 0.0)

    def test_truncation_raises_an_event(self):
        h = Harness()
        h.feed(*task_lines(2, truncated=1),
               GIN.format(status=200, lat="1s", path="/api/chat"),
               SCHED.format(model="m:latest")).finish()
        kinds = [e["kind"] for e in h.events]
        self.assertIn("truncation", kinds)
        self.assertEqual(h.requests[0]["truncated"], 1)


class TestAttribution(unittest.TestCase):
    def test_parallel_completion_is_marked_ambiguous(self):
        """Two tasks finishing before either access line prints cannot be told
        apart from the log alone -- say so rather than guess silently."""
        h = Harness()
        h.feed(*task_lines(10, out_tok=100))
        h.feed(*task_lines(11, out_tok=200))
        h.feed(GIN.format(status=200, lat="5s", path="/api/chat"),
               SCHED.format(model="m:latest"))
        h.feed(GIN.format(status=200, lat="6s", path="/api/chat"),
               SCHED.format(model="m:latest"))
        h.finish()
        self.assertEqual(len(h.requests), 2)
        self.assertEqual(h.requests[0]["attribution"], "ambiguous")
        # the second join has nothing left queued behind it, so it is exact
        self.assertEqual(h.requests[1]["attribution"], "exact")

    def test_failed_request_gets_no_timings_and_no_guessed_model(self):
        """A 404 for an unknown model never reaches a runner.  It must not
        inherit the model that happened to run most recently."""
        h = Harness()
        h.feed(*task_lines(20),
               GIN.format(status=200, lat="5s", path="/api/chat"),
               SCHED.format(model="llama3.2:3b"))
        h.feed(GIN.format(status=404, lat="4.4ms", path="/api/chat")).finish()

        bad = [r for r in h.requests if r["status"] == 404][0]
        self.assertEqual(bad["attribution"], "none")
        self.assertIsNone(bad.get("model"))
        self.assertIsNone(bad.get("ttft_ms"))

    def test_sched_line_does_not_label_a_failed_row(self):
        """The scheduler line naming a model must skip past a 'none' row and
        land on the request that actually ran."""
        h = Harness()
        h.feed(GIN.format(status=404, lat="4ms", path="/api/chat"))
        h.feed(*task_lines(30),
               GIN.format(status=200, lat="5s", path="/api/chat"),
               SCHED.format(model="llama3.2:3b")).finish()
        ok = [r for r in h.requests if r["status"] == 200][0]
        bad = [r for r in h.requests if r["status"] == 404][0]
        self.assertEqual(ok["model"], "llama3.2:3b")
        self.assertIsNone(bad.get("model"))

    def test_orphan_when_client_disconnects(self):
        """Timings with no access line still carry token counts, flagged."""
        h = Harness()
        h.feed(*task_lines(40)).finish()
        self.assertEqual(len(h.requests), 1)
        self.assertEqual(h.requests[0]["attribution"], "orphan")
        self.assertEqual(h.requests[0]["output_tokens"], 416)

    def test_health_checks_are_recorded_but_classified(self):
        h = Harness()
        h.feed('[GIN] 2026/08/19 - 11:17:18 | 200 |      33.313µs |'
               '       127.0.0.1 | GET      "/api/ps"').finish()
        self.assertEqual(h.requests[0]["class"], "health")
        self.assertIsNone(h.requests[0].get("attribution"))


class TestLifecycle(unittest.TestCase):
    LOAD_START = ('time=2026-08-19T08:08:08.444-05:00 level=INFO source=llama_server.go:433'
                  ' msg="starting llama-server" cmd="/x/llama-server'
                  ' --model /models/blobs/sha256-abc --port 35259 -c 4096 -np 1"')
    LOADED = ('time=2026-08-19T08:08:36.523-05:00 level=INFO source=llama_server.go:1362'
              ' msg="llama-server started in 28.08 seconds"')

    def test_duplicate_load_line_is_collapsed(self):
        """Ollama prints the started-in line once per llama-server handle
        (model and projector), so one load surfaces twice."""
        h = Harness()
        h.corr.blob_to_model["/models/blobs/sha256-abc"] = "llama3.2:3b"
        h.feed(self.LOAD_START, self.LOADED, self.LOADED).finish()
        loads = [e for e in h.events if e["kind"] == "model_loaded"]
        self.assertEqual(len(loads), 1)
        self.assertAlmostEqual(loads[0]["duration_ms"], 28080.0)

    def test_load_duration_is_recorded(self):
        h = Harness()
        h.feed(self.LOAD_START, self.LOADED).finish()
        loads = [e for e in h.events if e["kind"] == "model_loaded"]
        self.assertEqual(len(loads), 1)
        self.assertAlmostEqual(loads[0]["duration_ms"], 28080.0)

    def test_inflight_tracks_running_tasks(self):
        h = Harness()
        lines = task_lines(50)
        h.feed(*lines[:3])
        self.assertEqual(h.corr.inflight, 1)
        h.feed(*lines[3:])
        self.assertEqual(h.corr.inflight, 0)

    def test_live_events_emitted_for_streaming_progress(self):
        h = Harness()
        h.feed("slot print_timing: id  0 | task 60 | n_gen =    103, tg =  28.41 t/s,"
               " tg_3s =  28.68 t/s")
        gens = [e for e in h.live if e["type"] == "gen"]
        self.assertEqual(len(gens), 1)
        self.assertAlmostEqual(gens[0]["tps_3s"], 28.68)


# --------------------------------------------------------------------------
# prompt cache
# --------------------------------------------------------------------------

def cache_lines(prompts=30, used=8010.969, limit=8192.000, took=364.22):
    """The srv-line burst ollama emits when it runs a prompt-cache update."""
    return [
        "srv  get_availabl: updating prompt cache",
        f"srv        update:  - cache state: {prompts} prompts, {used:.3f} MiB "
        f"(limits: {limit:.3f} MiB, 32768 tokens, 68068 est)",
        f"srv  get_availabl: prompt cache update took {took} ms",
    ]


CKPT_CREATE = ("slot create_check: id  0 | task {task} | created context checkpoint "
               "{n} of 32 (pos_min = 1, pos_max = 1, n_tokens = 1008, size = 50.251 MiB)")
CKPT_CROWDED = ("slot create_check: id  0 | task {task} | erasing context checkpoint too "
                "close to an earlier one (pos_min = 1, pos_max = 1, n_tokens = 500, "
                "size = 50.251 MiB)")
CKPT_INVALID = ("slot   operator(): id  0 | task {task} | erased invalidated context "
                "checkpoint (pos_min = 1, pos_max = 1, n_tokens = 2, n_swa = 3, "
                "pos_next = 4, size = 50.251 MiB)")
CKPT_RESTORE = ("slot   operator(): id  0 | task {task} | restored context checkpoint "
                "(pos_min = 1, pos_max = 1, n_tokens = 1008, n_past = 1008, "
                "size = 50.251 MiB)")


class TestPromptCacheSampling(unittest.TestCase):
    def test_a_state_line_plus_its_cost_line_make_one_sample(self):
        h = Harness()
        h.feed(*cache_lines()).finish()
        self.assertEqual(len(h.cache), 1)
        c = h.cache[0]
        self.assertEqual(c["prompts"], 30)
        self.assertAlmostEqual(c["used_mib"], 8010.969)
        self.assertAlmostEqual(c["usage"], 8010.969 / 8192.0)
        # The cost line follows the state line, so it must land on the same row.
        self.assertAlmostEqual(c["update_ms"], 364.22)

    def test_a_state_line_with_no_cost_line_is_still_stored(self):
        """The cost line can be missing; losing the occupancy gauge with it
        would be the worse failure."""
        h = Harness()
        h.feed("srv        update:  - cache state: 5 prompts, 100.000 MiB "
               "(limits: 8192.000 MiB, 32768 tokens, 0 est)").finish()
        self.assertEqual(len(h.cache), 1)
        self.assertEqual(h.cache[0]["prompts"], 5)
        self.assertIsNone(h.cache[0].get("update_ms"))

    def test_a_second_state_line_flushes_the_first(self):
        h = Harness()
        h.feed("srv        update:  - cache state: 1 prompts, 10.000 MiB "
               "(limits: 8192.000 MiB, 32768 tokens, 0 est)",
               "srv        update:  - cache state: 2 prompts, 20.000 MiB "
               "(limits: 8192.000 MiB, 32768 tokens, 0 est)").finish()
        self.assertEqual([c["prompts"] for c in h.cache], [1, 2])

    def test_a_cost_line_alone_records_the_cost(self):
        """An update that changed nothing still cost real time inside a
        request, so it is kept with the gauges left null rather than dropped."""
        h = Harness()
        h.feed("srv  get_availabl: prompt cache update took 9.88 ms").finish()
        self.assertEqual(len(h.cache), 1)
        self.assertAlmostEqual(h.cache[0]["update_ms"], 9.88)
        self.assertIsNone(h.cache[0].get("usage"))

    def test_the_model_is_carried_from_the_scheduler_lines(self):
        """The cache-state line names no model; the last active one is the
        only honest label available."""
        h = Harness()
        h.feed(*task_lines(1),
               GIN.format(status=200, lat="1s", path="/api/chat"),
               SCHED.format(model="llama3.2:3b"),
               *cache_lines()).finish()
        self.assertEqual(h.cache[0]["model"], "llama3.2:3b")

    def test_a_sample_is_pushed_to_the_live_feed(self):
        h = Harness()
        h.feed(*cache_lines()).finish()
        self.assertEqual([e["type"] for e in h.live if e["type"] == "cache"], ["cache"])


class TestCheckpointCounters(unittest.TestCase):
    def test_counters_are_deltas_between_samples(self):
        """Stored as deltas so a runner restart cannot make a rate negative."""
        h = Harness()
        h.feed(CKPT_CROWDED.format(task=1), CKPT_CROWDED.format(task=1),
               CKPT_INVALID.format(task=1), CKPT_RESTORE.format(task=1),
               *cache_lines(prompts=1))
        h.feed(CKPT_CROWDED.format(task=2), *cache_lines(prompts=2)).finish()

        first, second = h.cache
        self.assertEqual(first["evictions"], 3)
        self.assertEqual(first["evict_crowded"], 2)
        self.assertEqual(first["evict_invalidated"], 1)
        self.assertEqual(first["restores"], 1)
        # The second sample counts only what happened after the first.
        self.assertEqual(second["evictions"], 1)
        self.assertEqual(second["evict_crowded"], 1)
        self.assertEqual(second["restores"], 0)

    def test_checkpoint_high_water_and_cap(self):
        h = Harness()
        h.feed(CKPT_CREATE.format(task=1, n=2), CKPT_CREATE.format(task=1, n=5),
               CKPT_CREATE.format(task=1, n=3), *cache_lines()).finish()
        c = h.cache[0]
        self.assertEqual(c["ckpt_used"], 5)   # high-water, not the last seen
        self.assertEqual(c["ckpt_total"], 32)
        self.assertEqual(c["ckpt_created"], 3)

    def test_the_cap_survives_a_sample_but_the_high_water_resets(self):
        h = Harness()
        h.feed(CKPT_CREATE.format(task=1, n=7), *cache_lines(prompts=1))
        h.feed(*cache_lines(prompts=2)).finish()
        self.assertEqual(h.cache[0]["ckpt_used"], 7)
        self.assertIsNone(h.cache[1]["ckpt_used"])
        # The cap is a property of the runner, not of the window.
        self.assertEqual(h.cache[1]["ckpt_total"], 32)

    def test_bookkeeping_against_task_minus_one_still_counts(self):
        """These lines describe the runner's cache, not a request, and
        llama.cpp sometimes logs them with no task attached."""
        h = Harness()
        h.feed(CKPT_CROWDED.format(task=-1), *cache_lines()).finish()
        self.assertEqual(h.cache[0]["evictions"], 1)


class TestLiveKvSizing(unittest.TestCase):
    LOAD_START = ('time=2026-08-19T08:08:08.444-05:00 level=INFO'
                  ' source=llama_server.go:433 msg="starting llama-server"'
                  ' cmd="/usr/local/lib/ollama/llama-server --model'
                  ' /models/blobs/sha256-abc --port 35259 -c 32768 -np 1"')
    LOADED = ('time=2026-08-19T08:08:20.100-05:00 level=INFO source=server.go:1'
              ' msg="llama-server started in 11.7 seconds"'
              ' runner.name=registry.ollama.ai/library/qwen3:8b')

    def kv_lines(self, cpu=0.0, gpu=4608.0):
        out = []
        if gpu:
            out.append(f"llama_kv_cache:      CUDA0 KV buffer size =  {gpu:.2f} MiB")
        if cpu:
            out.append(f"llama_kv_cache:        CPU KV buffer size =  {cpu:.2f} MiB")
        out.append(f"llama_kv_cache: size = {cpu + gpu:.2f} MiB ( 32768 cells,  "
                   "36 layers,  1/1 seqs), K (f16): 2304.00 MiB, V (f16): 2304.00 MiB")
        return out

    def loads(self, **kw):
        h = Harness()
        h.feed(self.LOAD_START, *self.kv_lines(**kw), self.LOADED).finish()
        return h

    def test_sizing_is_attached_to_the_load_event(self):
        h = self.loads()
        loaded = [e for e in h.events if e["kind"] == "model_loaded"]
        self.assertEqual(len(loaded), 1)
        kv = loaded[0]["detail"]["kv_cache"]
        self.assertAlmostEqual(kv["kv_mib"], 4608.0)
        self.assertEqual(kv["cells"], 32768)
        self.assertEqual(kv["gpu_mib"], 4608.0)
        self.assertEqual(kv["cpu_mib"], 0)
        self.assertEqual(kv["cpu_fraction"], 0.0)

    def test_a_cache_that_fits_in_vram_raises_no_warning(self):
        self.assertEqual([e for e in self.loads().events if e["kind"] == "kv_offload"], [])

    def test_a_cache_spilled_to_host_ram_is_called_out(self):
        """It caps decode throughput for the whole life of the load, which is
        too important to leave buried in a detail blob."""
        h = self.loads(cpu=4096.0, gpu=512.0)
        ev = [e for e in h.events if e["kind"] == "kv_offload"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["level"], "WARN")
        self.assertAlmostEqual(ev[0]["detail"]["cpu_fraction"], 4096.0 / 4608.0)
        self.assertIn("89%", ev[0]["msg"])

    def test_sizing_does_not_leak_into_the_next_load(self):
        """A load whose KV lines were not captured must report none, not the
        previous model's figures."""
        h = Harness()
        h.feed(self.LOAD_START, *self.kv_lines(cpu=4096.0, gpu=512.0), self.LOADED)
        # Past the window that collapses the duplicate "started in Ns" line
        # ollama prints per llama-server handle, so this is a genuine reload.
        h.t += 60
        h.feed(self.LOAD_START, self.LOADED).finish()
        loaded = [e for e in h.events if e["kind"] == "model_loaded"]
        self.assertEqual(len(loaded), 2)
        self.assertIn("kv_cache", loaded[0]["detail"])
        self.assertNotIn("kv_cache", loaded[1]["detail"] or {})


class TestSlotCapacityOnRequests(unittest.TestCase):
    def test_the_slot_capacity_reaches_the_request_row(self):
        """context_tokens alone is a token count; only the pair is an
        occupancy, so both must survive the join."""
        h = Harness()
        h.feed(*task_lines(9),
               GIN.format(status=200, lat="1s", path="/api/chat"),
               SCHED.format(model="m:latest")).finish()
        r = h.requests[0]
        self.assertEqual(r["n_ctx_slot"], 131072)
        self.assertEqual(r["context_tokens"], 5000)


if __name__ == "__main__":
    unittest.main()
