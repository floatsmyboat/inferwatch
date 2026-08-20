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
        self.corr = Correlator(on_request=self.requests.append,
                               on_event=self.events.append,
                               on_live=self.live.append)
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


if __name__ == "__main__":
    unittest.main()
