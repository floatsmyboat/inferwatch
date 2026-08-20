"""Parser tests.  Every fixture line is verbatim real output from ollama
0.32.14 with OLLAMA_DEBUG=1, copied out of this host's journal."""

import unittest

from inferwatch.parse import (classify_endpoint, parse_duration_ms, parse_gin,
                           parse_go, parse_line, parse_slot)


class TestDuration(unittest.TestCase):
    def test_go_duration_forms(self):
        cases = {
            "16.862061865s": 16862.061865,
            "33.313µs": 0.033313,
            "20.969ms": 20.969,
            "922ns": 0.000922,
            "1m30s": 90_000.0,
            "5m0s": 300_000.0,
            "14m24s": 864_000.0,
            "999.754867ms": 999.754867,
        }
        for text, want in cases.items():
            self.assertAlmostEqual(parse_duration_ms(text), want, places=6, msg=text)

    def test_rejects_garbage(self):
        for bad in ("", "garbage", "12", "abc1s", None):
            self.assertIsNone(parse_duration_ms(bad))


class TestGin(unittest.TestCase):
    LINE = ('[GIN] 2026/08/19 - 11:17:18 | 200 | 16.862061865s |'
            '       192.0.2.10 | POST     "/v1/chat/completions"')

    def test_full_line(self):
        ev = parse_gin(self.LINE)
        self.assertEqual(ev["status"], 200)
        self.assertEqual(ev["client_ip"], "192.0.2.10")
        self.assertEqual(ev["method"], "POST")
        self.assertEqual(ev["endpoint"], "/v1/chat/completions")
        self.assertEqual(ev["class"], "inference")
        self.assertAlmostEqual(ev["latency_ms"], 16862.061865, places=5)

    def test_error_status(self):
        line = ('[GIN] 2026/08/19 - 12:00:00 | 500 | 1.5s |'
                '       ::1 | POST     "/v1/chat/completions"')
        self.assertEqual(parse_gin(line)["status"], 500)

    def test_health_traffic_is_classified_apart(self):
        # These two dominate request counts on any polled instance; counting
        # them as traffic would bury real inference in the rates.
        self.assertEqual(classify_endpoint("HEAD", "/"), "health")
        self.assertEqual(classify_endpoint("GET", "/api/ps"), "health")
        self.assertEqual(classify_endpoint("GET", "/api/version"), "health")

    def test_classification(self):
        self.assertEqual(classify_endpoint("POST", "/api/chat"), "inference")
        self.assertEqual(classify_endpoint("POST", "/api/generate"), "inference")
        self.assertEqual(classify_endpoint("POST", "/v1/chat/completions"), "inference")
        self.assertEqual(classify_endpoint("POST", "/api/embed"), "embed")
        self.assertEqual(classify_endpoint("POST", "/api/show"), "admin")
        self.assertEqual(classify_endpoint("GET", "/api/tags"), "admin")


class TestSlot(unittest.TestCase):
    def test_prompt_eval_is_ttft(self):
        line = ("slot print_timing: id  0 | task 6763 | prompt eval time =    1254.52 ms /"
                "    55 tokens (   22.81 ms per token,    43.84 tokens per second)")
        ev = parse_slot(line)
        self.assertEqual(ev["kind"], "prompt_eval")
        self.assertEqual(ev["task_id"], 6763)
        self.assertEqual(ev["slot_id"], 0)
        self.assertAlmostEqual(ev["ttft_ms"], 1254.52)
        self.assertEqual(ev["prompt_tokens"], 55)
        self.assertAlmostEqual(ev["prefill_tps"], 43.84)

    def test_eval_not_confused_with_prompt_eval(self):
        line = ("slot print_timing: id  0 | task 6763 |        eval time =   14591.31 ms /"
                "   416 tokens (   35.16 ms per token,    28.44 tokens per second)")
        ev = parse_slot(line)
        self.assertEqual(ev["kind"], "eval")
        self.assertEqual(ev["output_tokens"], 416)
        self.assertAlmostEqual(ev["decode_ms"], 14591.31)
        self.assertAlmostEqual(ev["decode_tps"], 28.44)

    def test_total(self):
        line = ("slot print_timing: id  0 | task 6763 |       total time =   15845.83 ms /"
                "   471 tokens")
        ev = parse_slot(line)
        self.assertEqual(ev["kind"], "total")
        self.assertAlmostEqual(ev["total_ms"], 15845.83)

    def test_gen_progress(self):
        line = "slot print_timing: id  0 | task 5202 | n_gen =    103, tg =  28.41 t/s, tg_3s =  28.68 t/s"
        ev = parse_slot(line)
        self.assertEqual(ev["kind"], "gen_progress")
        self.assertEqual(ev["n_gen"], 103)
        self.assertAlmostEqual(ev["tg"], 28.41)
        self.assertAlmostEqual(ev["tg_3s"], 28.68)

    def test_new_prompt_carries_full_prompt_length(self):
        line = ("slot   operator(): id  0 | task 5202 | new prompt, n_ctx_slot = 131072,"
                " n_keep = 4, task.n_tokens = 126097")
        ev = parse_slot(line)
        self.assertEqual(ev["kind"], "new_prompt")
        self.assertEqual(ev["prompt_tokens_total"], 126097)
        self.assertEqual(ev["n_ctx_slot"], 131072)

    def test_release_with_truncation(self):
        line = ("slot      release: id  0 | task 6763 | stop processing: n_tokens = 131071,"
                " truncated = 1")
        ev = parse_slot(line)
        self.assertEqual(ev["kind"], "release")
        self.assertEqual(ev["truncated"], 1)
        self.assertEqual(ev["context_tokens"], 131071)

    def test_draft_acceptance(self):
        line = ("slot print_timing: id  0 | task 6763 | draft acceptance = 0.47150"
                " (  273 accepted /   579 generated), mean len =  2.88")
        ev = parse_slot(line)
        self.assertEqual(ev["kind"], "draft")
        self.assertAlmostEqual(ev["draft_accept"], 0.4715)
        self.assertAlmostEqual(ev["draft_mean_len"], 2.88)

    def test_bookkeeping_lines_are_ignored(self):
        for line in (
            "slot launch_slot_: id  0 | task -1 | sampler params:",
            "slot get_availabl: id  0 | task -1 |  - skipping, slot is empty",
            "slot   operator(): id  0 | task 0 | cached n_tokens = 0, memory_seq_rm [0, end)",
        ):
            ev = parse_slot(line)
            self.assertTrue(ev is None or ev.get("task_id", 0) < 0, line)


class TestGo(unittest.TestCase):
    def test_load_duration(self):
        line = ('time=2026-08-19T08:08:36.523-05:00 level=INFO source=llama_server.go:1362'
                ' msg="llama-server started in 28.08 seconds"')
        ev = parse_go(line)
        self.assertEqual(ev["event"], "model_loaded")
        self.assertAlmostEqual(ev["load_ms"], 28080.0)

    def test_model_name_is_stripped_of_registry(self):
        line = ('time=2026-08-19T11:17:18.443-05:00 level=DEBUG source=sched.go:489'
                ' msg="context for request finished"'
                ' runner.name=registry.ollama.ai/library/llama3.2:3b'
                ' runner.size="24.0 GiB" runner.vram="24.0 GiB" runner.parallel=1'
                ' runner.pid=34572'
                ' runner.model=/var/lib/ollama/models/blobs/sha256-f5f1dd89'
                ' runner.num_ctx=131072')
        ev = parse_go(line)
        self.assertEqual(ev["event"], "request_finished")
        self.assertEqual(ev["model"], "llama3.2:3b")
        self.assertEqual(ev["num_ctx"], 131072)
        self.assertEqual(ev["parallel"], 1)

    def test_quoted_values_do_not_break_kv_split(self):
        line = ('time=2026-08-19T08:08:08.444-05:00 level=INFO source=llama_server.go:433'
                ' msg="starting llama-server" cmd="/usr/local/lib/ollama/llama-server'
                ' --model /var/lib/ollama/models/blobs/sha256-abc --port 35259'
                ' -c 262144 -np 2 --spec-type draft-mtp"')
        ev = parse_go(line)
        self.assertEqual(ev["event"], "load_start")
        self.assertEqual(ev["port"], 35259)
        self.assertEqual(ev["num_ctx"], 262144)
        self.assertEqual(ev["parallel"], 2)
        self.assertEqual(ev["spec_type"], "draft-mtp")

    def test_warnings_become_problems(self):
        line = ('time=2026-08-19T08:46:42.196-05:00 level=WARN source=sched.go:509'
                ' msg="model architecture does not currently support parallel requests"'
                ' architecture=qwen35')
        ev = parse_go(line)
        self.assertEqual(ev["event"], "problem")
        self.assertEqual(ev["level"], "WARN")
        self.assertEqual(ev["kv"]["architecture"], "qwen35")


class TestDispatch(unittest.TestCase):
    def test_unowned_lines_return_none(self):
        for line in ("", "srv  update_slots: all slots are idle",
                     "spec common_specu: statistics draft-mtp: ..."):
            self.assertIsNone(parse_line(line))


if __name__ == "__main__":
    unittest.main()
