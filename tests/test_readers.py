"""Log reader tests: timestamp derivation and file following."""

import asyncio
import os
import shutil
import tempfile
import time
import unittest

from inferwatch.readers import (DockerReader, FileReader, build_reader,
                                derive_timestamp, parse_iso)
from inferwatch.store import Store

GO_LINE = ('time=2026-08-19T11:17:18.443-05:00 level=DEBUG source=sched.go:489'
           ' msg="context for request finished"')
GIN_LINE = ('[GIN] 2026/08/19 - 11:17:18 | 200 | 16.8s |'
            '       192.0.2.10 | POST     "/api/chat"')
SLOT_LINE = ("slot print_timing: id  0 | task 1 | prompt eval time =  100.00 ms /"
             "  10 tokens (   10.00 ms per token,   100.00 tokens per second)")


class TestTimestamps(unittest.TestCase):
    def test_iso_with_offset(self):
        self.assertIsNotNone(parse_iso("2026-08-19T11:17:18.443-05:00"))

    def test_iso_with_excess_fractional_digits(self):
        """Docker emits nanoseconds; datetime accepts at most microseconds."""
        self.assertIsNotNone(parse_iso("2026-08-19T11:17:18.123456789Z"))

    def test_go_line_supplies_its_own_time(self):
        ts = derive_timestamp(GO_LINE, None)
        self.assertAlmostEqual(ts, parse_iso("2026-08-19T11:17:18.443-05:00"), places=3)

    def test_gin_line_supplies_second_resolution(self):
        ts = derive_timestamp(GIN_LINE, None)
        self.assertIsNotNone(ts)
        self.assertEqual(int(ts), int(parse_iso("2026-08-19T11:17:18")))

    def test_slot_line_inherits_the_previous_timestamp(self):
        """llama.cpp's timing lines carry no clock of their own; they must not
        jump to wall-clock time in the middle of a replayed file."""
        prev = 1_700_000_000.0
        self.assertEqual(derive_timestamp(SLOT_LINE, prev), prev)

    def test_second_resolution_line_does_not_move_time_backwards(self):
        """A gin line parses to whole seconds, so following a Go line stamped
        .443 it would otherwise appear 443ms earlier."""
        go_ts = derive_timestamp(GO_LINE, None)
        gin_ts = derive_timestamp(GIN_LINE, go_ts)
        self.assertGreaterEqual(gin_ts, go_ts)

    def test_a_genuine_jump_backwards_is_still_trusted(self):
        """Rotating in an older file is a real regression, not an artefact."""
        much_later = derive_timestamp(GO_LINE, None) + 86400
        ts = derive_timestamp(GO_LINE, much_later)
        self.assertLess(ts, much_later)

    def test_falls_back_to_now_when_nothing_is_known(self):
        ts = derive_timestamp(SLOT_LINE, None)
        self.assertAlmostEqual(ts, time.time(), delta=5)


class FileCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "server.log")
        self.st = Store(os.path.join(self.dir, "t.db"))
        self.seen = []

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, *lines, mode="a"):
        with open(self.path, mode) as fh:
            for line in lines:
                fh.write(line + "\n")

    async def follow(self, seconds=0.6, reader=None):
        rd = reader or FileReader(self.st, "src", {"reader": "file", "path": self.path},
                                  poll=0.02)
        task = asyncio.create_task(
            rd.run(lambda ts, msg: self.seen.append((ts, msg)), lambda now: None))
        await asyncio.sleep(seconds)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return rd


class TestFileReader(FileCase):
    def test_reads_existing_content_then_follows(self):
        self.write("first line", mode="w")

        async def go():
            rd = await self.follow(0.3)
            self.write("second line")
            await self.follow(0.3, reader=rd)
            return rd

        asyncio.run(go())
        bodies = [m for _t, m in self.seen]
        self.assertIn("first line", bodies)
        self.assertIn("second line", bodies)

    def test_offset_is_persisted_so_a_restart_does_not_replay(self):
        self.write("a", "b", "c", mode="w")
        asyncio.run(self.follow(0.3))
        state = self.st.get_meta("reader_state:src")
        self.assertIsNotNone(state)
        first = len(self.seen)
        self.assertGreaterEqual(first, 3)

        # A fresh reader over the same file must resume, not re-read.
        self.seen.clear()
        asyncio.run(self.follow(0.3))
        self.assertEqual(self.seen, [])

    def test_truncation_in_place_restarts_from_the_top(self):
        self.write("old one", "old two", mode="w")
        asyncio.run(self.follow(0.3))
        self.seen.clear()
        self.write("fresh", mode="w")        # truncate + rewrite
        asyncio.run(self.follow(0.4))
        self.assertIn("fresh", [m for _t, m in self.seen])

    def test_rotation_to_a_new_inode_is_followed(self):
        self.write("before rotate", mode="w")
        asyncio.run(self.follow(0.3))
        self.seen.clear()
        os.rename(self.path, self.path + ".1")
        self.write("after rotate", mode="w")
        asyncio.run(self.follow(0.5))
        self.assertIn("after rotate", [m for _t, m in self.seen])

    def test_missing_file_is_waited_for_not_fatal(self):
        async def go():
            rd = FileReader(self.st, "src", {"reader": "file", "path": self.path},
                            poll=0.02)
            task = asyncio.create_task(rd.run(lambda ts, m: self.seen.append(m),
                                              lambda now: None))
            await asyncio.sleep(0.2)
            self.write("appeared later", mode="w")
            await asyncio.sleep(2.4)          # the retry sleep is 2s
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(go())
        self.assertIn("appeared later", self.seen)

    def test_timestamps_are_monotonic_over_a_mixed_log(self):
        """A real ollama log interleaves timestamped Go lines with bare slot
        lines; ordering must hold because the joins depend on it."""
        self.write(GO_LINE, SLOT_LINE, SLOT_LINE, GIN_LINE, mode="w")
        asyncio.run(self.follow(0.3))
        stamps = [t for t, _m in self.seen]
        self.assertEqual(stamps, sorted(stamps))


class TestBuildReader(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_selects_by_reader_field(self):
        from inferwatch.readers import JournaldReader
        self.assertIsInstance(build_reader(self.st, "s", {"reader": "journald",
                                                          "unit": "ollama"}),
                              JournaldReader)
        self.assertIsInstance(build_reader(self.st, "s", {"reader": "file",
                                                          "path": "/x"}), FileReader)
        self.assertIsInstance(build_reader(self.st, "s", {"reader": "docker",
                                                          "container": "o"}),
                              DockerReader)

    def test_defaults_to_journald(self):
        from inferwatch.readers import JournaldReader
        self.assertIsInstance(build_reader(self.st, "s", {}), JournaldReader)

    def test_unknown_reader_is_an_error(self):
        with self.assertRaises(ValueError):
            build_reader(self.st, "s", {"reader": "smoke-signals"})

    def test_readers_describe_themselves_for_the_ui(self):
        self.assertIn("ollama", build_reader(self.st, "s", {"reader": "journald",
                                                            "unit": "ollama"}).describe())
        self.assertIn("/x", build_reader(self.st, "s", {"reader": "file",
                                                        "path": "/x"}).describe())


if __name__ == "__main__":
    unittest.main()
