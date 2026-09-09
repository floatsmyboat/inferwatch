"""SwarmUI / ComfyUI collection.

Fixtures are verbatim from this host: real journal lines from SwarmUI 0.9.8.3
and a real /history entry shape from ComfyUI 0.34.0.
"""

import json
import os
import shutil
import tempfile
import time
import unittest

from inferwatch import image_metrics as im
from inferwatch.images import (ImagesPoller, SwarmLogCollector, extract_models,
                               parse_history_entry, primary_model)
from inferwatch.parse_swarm import parse_banner_date, parse_line, strip_ansi
from inferwatch.store import Store


class TestSwarmLogParsing(unittest.TestCase):
    def test_generation_request(self):
        ev = parse_line("14:39:51.139 [Info] User local requested 1 image with "
                        "model 'OfficialStableDiffusion/sd3.5_large_fp8_scaled.safetensors'...")
        self.assertEqual(ev["kind"], "generation_requested")
        self.assertEqual(ev["user"], "local")
        self.assertEqual(ev["images"], 1)
        self.assertTrue(ev["model"].endswith("sd3.5_large_fp8_scaled.safetensors"))

    def test_a_batch_records_its_size(self):
        """A request for N images produces N finish lines, which is why the two
        are never paired 1:1."""
        ev = parse_line("14:39:51.139 [Info] User bob requested 4 images with model 'x.safetensors'...")
        self.assertEqual(ev["images"], 4)

    def test_generation_finished_splits_prep_from_gen(self):
        ev = parse_line("14:41:39.623 [Info] Generated an image in 4.03 sec (prep) "
                        "and 94.66 sec (gen)")
        self.assertEqual(ev["kind"], "generation_finished")
        # Seconds in the log, milliseconds everywhere in the store.
        self.assertAlmostEqual(ev["prep_ms"], 4030.0, places=3)
        self.assertAlmostEqual(ev["gen_ms"], 94660.0, places=3)

    def test_backend_port_is_the_only_place_ports_are_published(self):
        ev = parse_line("06:01:55.678 [Init] Self-Start ComfyUI-0 on port 7821 started.")
        self.assertEqual(ev["kind"], "backend_up")
        self.assertEqual(ev["backend_index"], 0)
        self.assertEqual(ev["port"], 7821)
        loading = parse_line("06:01:42.369 [Init] Self-Start ComfyUI-1 on port 7822 is loading...")
        self.assertEqual(loading["kind"], "backend_loading")
        self.assertEqual(loading["port"], 7822)

    def test_backend_stop_carries_the_pid(self):
        ev = parse_line("23:58:32.086 [Info] Shutting down self-start ComfyUI "
                        "(port=7822) process #5882...")
        self.assertEqual(ev["kind"], "backend_stopping")
        self.assertEqual((ev["port"], ev["pid"]), (7822, 5882))

    def test_webapi_error_with_and_without_a_user(self):
        a = parse_line("14:50:53.765 [Error] [WebAPI] Error handling API request "
                       "'/API/ListModels' for user 'local': Missing required parameter 'depth'")
        self.assertEqual(a["kind"], "webapi_error")
        self.assertEqual(a["route"], "/API/ListModels")
        self.assertEqual(a["user"], "local")
        self.assertIn("depth", a["reason"])
        b = parse_line("23:58:20.001 [Error] [WebAPI] Error handling API request "
                       "'/api/doc': Invalid request method: GET")
        self.assertEqual(b["route"], "/api/doc")
        self.assertIsNone(b["user"])

    def test_backend_stderr_is_tagged_with_its_backend(self):
        ev = parse_line("15:19:48.374 [Warning] [ComfyUI-0/STDERR] "
                        "RuntimeError: ERROR: clip input is invalid: None")
        self.assertEqual(ev["kind"], "backend_stderr")
        self.assertEqual(ev["backend_index"], 0)
        self.assertTrue(ev["text"].startswith("RuntimeError"))

    def test_ansi_colour_is_stripped(self):
        """ComfyUI colours its output; stored raw it renders as mojibake and
        defeats any grouping of similar errors."""
        ev = parse_line("15:19:48.373 [Warning] [ComfyUI-0/STDERR] "
                        "\x1b[1m\x1b[31m[ERROR]\x1b[0m !!! Exception during processing !!!")
        self.assertEqual(ev["text"], "[ERROR] !!! Exception during processing !!!")
        self.assertEqual(strip_ansi("\x1b[32mx\x1b[0m"), "x")

    def test_startup_stderr_arrives_without_the_prefix(self):
        ev = parse_line("[ComfyUI-1/STDERR] \x1b[32m[INFO]\x1b[0m loading nodes")
        self.assertEqual(ev["kind"], "backend_stderr")
        self.assertEqual(ev["backend_index"], 1)
        self.assertEqual(ev["text"], "[INFO] loading nodes")

    def test_blank_stderr_is_dropped(self):
        self.assertIsNone(parse_line("15:19:48.373 [Warning] [ComfyUI-0/STDERR]"))
        self.assertIsNone(parse_line("[ComfyUI-0/STDERR] "))

    def test_unrecognised_errors_are_still_kept(self):
        """An unrecognised failure is the one you most want to see."""
        ev = parse_line("10:00:00.000 [Error] Something nobody wrote a rule for")
        self.assertEqual(ev["kind"], "problem")
        self.assertEqual(ev["level"], "Error")

    def test_routine_info_is_not_claimed(self):
        for line in ("14:39:00.000 [Info] Creating new session 'local' for 10.0.0.5",
                     "06:01:42.000 [Init] Scan for web extensions...",
                     "", "not a swarm line at all"):
            self.assertIsNone(parse_line(line), line)

    def test_the_banner_is_the_only_line_with_a_date(self):
        """Every other line carries a time but no date, which is why journald
        is the supported reader."""
        self.assertEqual(parse_banner_date("== SwarmUI logs 2026-09-09 15:19 =="),
                         "2026-09-09")
        self.assertIsNone(parse_banner_date("14:39:51.139 [Info] hello"))


GRAPH = {
    "3": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "Flux/flux1-dev-fp8.safetensors"}},
    "4": {"class_type": "LoraLoader",
          "inputs": {"lora_name": "detail.safetensors", "strength_model": 0.8}},
    "5": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.sft"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat, highly detailed"}},
}


class TestModelExtraction(unittest.TestCase):
    def test_every_loaded_model_is_found(self):
        got = {(m["role"], m["name"]) for m in extract_models(GRAPH)}
        self.assertEqual(got, {
            ("ckpt_name", "Flux/flux1-dev-fp8.safetensors"),
            ("lora_name", "detail.safetensors"),
            ("vae_name", "ae.sft"),
        })

    def test_prompt_text_is_not_mistaken_for_a_model(self):
        names = [m["name"] for m in extract_models(GRAPH)]
        self.assertNotIn("a cat, highly detailed", names)

    def test_a_custom_node_is_still_matched_by_its_value(self):
        """Detection is by shape, not by a list of loader classes -- this
        ecosystem is mostly custom nodes and a class list would miss them."""
        graph = {"9": {"class_type": "SomeThirdPartyLoader",
                       "inputs": {"weights": "custom/thing.gguf"}}}
        self.assertEqual([m["name"] for m in extract_models(graph)],
                         ["custom/thing.gguf"])

    def test_the_checkpoint_is_the_primary_model(self):
        self.assertEqual(primary_model(extract_models(GRAPH)),
                         "Flux/flux1-dev-fp8.safetensors")

    def test_a_unet_workflow_has_no_checkpoint(self):
        models = extract_models({"1": {"class_type": "UNETLoader",
                                       "inputs": {"unet_name": "flux.safetensors"}}})
        self.assertEqual(primary_model(models), "flux.safetensors")

    def test_nothing_loaded_is_none_not_a_crash(self):
        self.assertIsNone(primary_model([]))
        self.assertEqual(extract_models({}), [])
        self.assertEqual(extract_models(None), [])


def history_entry(status="success", start=1_788_985_100_000, end=1_788_985_180_000,
                  error=None, cached=("5",)):
    messages = [["execution_start", {"timestamp": start}]]
    if cached:
        messages.append(["execution_cached", {"nodes": list(cached), "timestamp": start}])
    if error:
        messages.append(["execution_error", {**error, "timestamp": end}])
    elif end:
        messages.append(["execution_success", {"timestamp": end}])
    return {"status": {"status_str": status, "completed": status == "success",
                       "messages": messages},
            "prompt": [3, "pid", GRAPH, {"create_time": start}, ["9"]],
            "outputs": {}, "meta": {}}


class TestHistoryEntry(unittest.TestCase):
    def test_a_successful_generation(self):
        r = parse_history_entry("abc", history_entry())
        self.assertEqual(r["prompt_id"], "abc")
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["total_ms"], 80_000)
        self.assertEqual(r["model"], "Flux/flux1-dev-fp8.safetensors")
        self.assertEqual(r["node_count"], 4)
        self.assertEqual(r["cached_nodes"], 1)
        self.assertIsNone(r["error_type"])

    def test_timestamps_are_milliseconds(self):
        r = parse_history_entry("abc", history_entry())
        # 1_788_985_100_000 ms is a 2026 date, not the year 58000.
        self.assertAlmostEqual(r["started_ts"], 1_788_985_100.0)
        self.assertLess(r["ts"], 2_000_000_000)

    def test_a_failed_generation_names_the_node(self):
        r = parse_history_entry("abc", history_entry(
            status="error", error={"node_id": "7", "node_type": "CLIPTextEncode",
                                   "exception_message": "clip input is invalid: None"}))
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["error_node"], "7")
        self.assertEqual(r["error_type"], "CLIPTextEncode")
        self.assertIn("clip input", r["error_message"])

    def test_an_in_flight_generation_gets_no_invented_end(self):
        r = parse_history_entry("abc", history_entry(end=None))
        self.assertEqual(r["status"], "running")
        self.assertIsNone(r["total_ms"])
        self.assertIsNotNone(r["started_ts"])

    def test_an_enormous_traceback_is_truncated(self):
        r = parse_history_entry("abc", history_entry(
            status="error", error={"node_id": "1", "node_type": "X",
                                   "exception_message": "y" * 9000}))
        self.assertEqual(len(r["error_message"]), 2000)

    def test_an_entry_with_no_timestamps_is_skipped(self):
        self.assertIsNone(parse_history_entry("abc", {"status": {"messages": []}}))
        self.assertIsNone(parse_history_entry("abc", "not a dict"))


class TestLogCollector(unittest.TestCase):
    def collect(self, *lines):
        events = []
        lc = SwarmLogCollector("swarm", on_event=events.append)
        t = 1_700_000_000.0
        for line in lines:
            t += 1.0
            lc.feed(t, line)
        return lc, events

    def test_backend_ports_are_learned(self):
        lc, _ = self.collect(
            "06:01:55.678 [Init] Self-Start ComfyUI-0 on port 7821 started.",
            "06:02:02.004 [Init] Self-Start ComfyUI-1 on port 7822 started.")
        self.assertEqual(lc.backend_ports, {0: 7821, 1: 7822})
        self.assertEqual(lc.backend_status, {0: "running", 1: "running"})

    def test_a_backend_going_down_is_tracked(self):
        lc, _ = self.collect(
            "06:01:55.678 [Init] Self-Start ComfyUI-0 on port 7821 started.",
            "23:58:31.000 [Info] ComfyUI backend 0 shutting down...")
        self.assertEqual(lc.backend_status[0], "down")
        # The port is remembered: it is where the backend will come back.
        self.assertEqual(lc.backend_ports, {0: 7821})

    def test_requests_and_finishes_are_not_paired(self):
        """Pairing them 1:1 would misattribute every batch, and /history is
        already the authority for per-generation records."""
        _, events = self.collect(
            "14:39:51.139 [Info] User local requested 4 images with model 'a.safetensors'...",
            "14:41:39.623 [Info] Generated an image in 1.00 sec (prep) and 2.00 sec (gen)")
        kinds = [e["kind"] for e in events]
        self.assertEqual(kinds, ["generation_requested", "generation_finished"])
        self.assertEqual(events[0]["model"], "a.safetensors")
        self.assertIsNone(events[0]["prep_ms"])
        self.assertIsNone(events[1]["model"])

    def test_events_carry_a_readable_message(self):
        _, events = self.collect(
            "14:50:53.765 [Error] [WebAPI] Error handling API request '/API/X' "
            "for user 'local': nope")
        self.assertEqual(events[0]["msg"], "/API/X: nope")
        self.assertEqual(events[0]["level"], "Error")

    def test_the_version_is_learned(self):
        lc, _ = self.collect("06:01:55.700 [Init] SwarmUI v0.9.8.3 - Local is now running.")
        self.assertEqual(lc.version, "0.9.8.3")


class PollerCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))

    def tearDown(self):
        self.st.db.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def poller(self, cfg=None, log=None):
        return ImagesPoller(self.st, "swarm",
                            {"url": "http://127.0.0.1:7801", **(cfg or {})},
                            interval_getter=lambda: 10.0, log_collector=log)


class TestBackendDiscovery(PollerCase):
    def test_configured_urls_win(self):
        lc = SwarmLogCollector("swarm")
        lc.backend_ports = {0: 7821}
        p = self.poller({"backends": "http://a:1, http://b:2"}, log=lc)
        self.assertEqual(p.backend_urls(),
                         {"backend-0": "http://a:1", "backend-1": "http://b:2"})

    def test_ports_come_from_the_log(self):
        lc = SwarmLogCollector("swarm")
        lc.backend_ports = {0: 7821, 1: 7822}
        self.assertEqual(self.poller(log=lc).backend_urls(), {
            "ComfyUI-0": "http://127.0.0.1:7821",
            "ComfyUI-1": "http://127.0.0.1:7822"})

    def test_ports_survive_a_collector_restart(self):
        """They are announced only when SwarmUI starts its backends, so a
        restart resumes the journal past them; without remembering, the source
        would silently collect no generations at all."""
        lc = SwarmLogCollector("swarm")
        lc.backend_ports = {0: 7821}
        self.poller(log=lc).backend_urls()          # learns and persists
        fresh = self.poller(log=SwarmLogCollector("swarm"))   # nothing in the log
        self.assertEqual(fresh.backend_urls(), {"ComfyUI-0": "http://127.0.0.1:7821"})

    def test_the_host_follows_the_swarm_url(self):
        lc = SwarmLogCollector("swarm")
        lc.backend_ports = {0: 7821}
        p = ImagesPoller(self.st, "swarm", {"url": "http://10.0.0.9:7801"},
                         interval_getter=lambda: 10.0, log_collector=lc)
        self.assertEqual(p.backend_urls(), {"ComfyUI-0": "http://10.0.0.9:7821"})


class MetricsCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))
        self.base = time.time() - 600

    def tearDown(self):
        self.st.db.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def gen(self, offset, status="success", ms=10_000.0, model="a.safetensors", **kw):
        row = {"source": "swarm", "prompt_id": f"p{offset}", "ts": self.base + offset,
               "status": status, "total_ms": ms, "model": model}
        row.update(kw)
        self.st.insert_image_generation(row)

    def window(self):
        return self.base - 1, self.base + 3600


class TestImageMetrics(MetricsCase):
    def test_summary_counts_and_rates(self):
        self.gen(1); self.gen(2); self.gen(3, status="error", ms=500.0)
        self.st.commit()
        s = im.summary(self.st, "swarm", *self.window())
        self.assertEqual(s["generations"], 3)
        self.assertEqual(s["errors"], 1)
        self.assertAlmostEqual(s["error_rate"], 1 / 3)
        self.assertTrue(s["exact"])

    def test_running_generations_are_counted_apart(self):
        """An in-flight generation is not a completed one, and folding it in
        would deflate the success rate for as long as it runs."""
        self.gen(1)
        self.gen(2, status="running", ms=None)
        self.st.commit()
        s = im.summary(self.st, "swarm", *self.window())
        self.assertEqual(s["generations"], 1)
        self.assertEqual(s["running"], 1)

    def test_durations_are_exact_not_bucketed(self):
        for i, ms in enumerate((1000.0, 5000.0, 9000.0)):
            self.gen(i, ms=ms)
        self.st.commit()
        s = im.summary(self.st, "swarm", *self.window())
        self.assertEqual(s["duration_ms"]["p50"], 5000.0)
        self.assertEqual(s["duration_ms_max"], 9000.0)

    def test_per_model_breakdown(self):
        self.gen(1, model="flux.safetensors")
        self.gen(2, model="flux.safetensors", status="error")
        self.gen(3, model="sdxl.safetensors")
        self.st.commit()
        rows = {m["model"]: m for m in im.by_model(self.st, "swarm", *self.window())}
        self.assertEqual(rows["flux.safetensors"]["generations"], 2)
        self.assertAlmostEqual(rows["flux.safetensors"]["error_rate"], 0.5)
        self.assertEqual(rows["sdxl.safetensors"]["errors"], 0)

    def test_failures_keep_the_two_kinds_apart(self):
        """A generation that threw inside ComfyUI and a call that never reached
        a backend are different problems with different fixes."""
        self.gen(1, status="error", error_type="CLIPTextEncode", error_node="7",
                 error_message="clip is None")
        self.st.insert_image_event({"ts": self.base + 2, "source": "swarm",
                                    "kind": "webapi_error", "level": "Error",
                                    "route": "/API/X", "msg": "/API/X: nope"})
        self.st.commit()
        f = im.failures(self.st, "swarm", *self.window())
        self.assertEqual(len(f["generation_errors"]), 1)
        self.assertEqual(len(f["log_errors"]), 1)
        self.assertEqual(f["by_node_type"], [{"error_type": "CLIPTextEncode", "count": 1}])

    def test_swarm_timings_are_reported_separately(self):
        self.st.insert_image_event({"ts": self.base + 1, "source": "swarm",
                                    "kind": "generation_finished",
                                    "prep_ms": 1000.0, "gen_ms": 9000.0})
        self.st.commit()
        s = im.summary(self.st, "swarm", *self.window())
        self.assertEqual(s["swarm_timing"]["samples"], 1)
        self.assertAlmostEqual(s["swarm_timing"]["prep_ms_mean"], 1000.0)

    def test_sources_are_isolated(self):
        self.gen(1)
        self.st.insert_image_generation({"source": "other", "prompt_id": "x",
                                         "ts": self.base + 1, "status": "success",
                                         "total_ms": 1.0})
        self.st.commit()
        self.assertEqual(im.summary(self.st, "swarm", *self.window())["generations"], 1)
        self.assertEqual(im.summary(self.st, None, *self.window())["generations"], 2)

    def test_coverage_marks_a_window_older_than_the_data(self):
        """Complete means the window does not start before the oldest stored
        generation. Deliberately conservative: a window reaching back further
        than anything stored might be missing history, and an empty stretch
        must not read as a quiet one."""
        self.gen(1)
        self.st.commit()
        oldest = self.base + 1
        self.assertTrue(im.coverage(self.st, oldest)["complete"])
        self.assertTrue(im.coverage(self.st, oldest + 60)["complete"])
        self.assertFalse(im.coverage(self.st, oldest - 1)["complete"])
        self.assertFalse(im.coverage(self.st, self.base - 30 * 86400)["complete"])

    def test_empty_coverage_is_never_complete(self):
        self.assertFalse(im.coverage(self.st, time.time() - 60)["complete"])

    def test_timeseries_buckets_rate_and_duration(self):
        for i in range(3):
            self.gen(i, ms=2000.0 * (i + 1))
        self.st.commit()
        ts = im.timeseries(self.st, "swarm", self.base - 1, self.base + 900, step=900)
        self.assertEqual(ts["counts"][0], 3)
        self.assertIsNotNone(ts["series"]["gen_per_hour"][0])
        self.assertEqual(ts["series"]["dur_p50"][0], 4000.0)

    def test_backends_report_their_latest_sample(self):
        for i, free in enumerate((10, 20)):
            self.st.insert_image_sample({"ts": self.base + i, "source": "swarm",
                                         "backend": "ComfyUI-0", "status": "running",
                                         "vram_total": 100, "vram_free": free,
                                         "gpu_indices": json.dumps([0])})
        self.st.commit()
        b = im.backends(self.st, "swarm")[0]
        self.assertEqual(b["vram_free"], 20)      # the newest, not the first
        self.assertEqual(b["vram_used"], 80)
        self.assertEqual(b["gpu_indices"], [0])


if __name__ == "__main__":
    unittest.main()
