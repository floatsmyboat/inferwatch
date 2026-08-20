"""Restart/resume behaviour: idempotent writes and cursor handling.

These cover the failure modes that only show up across a restart -- which is
what happens on every reboot now that the service starts at boot.
"""

import os
import shutil
import tempfile
import time
import unittest

from inferwatch.readers import JournaldReader, cursor_timestamp_us
from inferwatch.store import SCHEMA_VERSION, Store, event_key, request_key


def a_request(ts, **kw):
    row = {"ts": ts, "endpoint": "/api/chat", "method": "POST", "class": "inference",
           "status": 200, "client_ip": "192.0.2.10", "latency_ms": 1234.5,
           "model": "m:latest", "output_tokens": 42, "task_id": 7,
           "attribution": "exact"}
    row.update(kw)
    return row


class TestIdempotentWrites(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def count(self, table="requests"):
        return self.st.query(f"SELECT COUNT(*) n FROM {table}")[0]["n"]

    def test_replaying_the_same_row_is_a_noop(self):
        """Re-reading journal lines already ingested must not double metrics."""
        row = a_request(1_700_000_000.123456)
        for _ in range(5):
            self.st.insert_request(row)
        self.st.commit()
        self.assertEqual(self.count(), 1)
        self.assertEqual(
            self.st.query("SELECT SUM(output_tokens) t FROM requests")[0]["t"], 42)

    def test_distinct_requests_are_kept(self):
        base = 1_700_000_000.0
        self.st.insert_request(a_request(base))
        self.st.insert_request(a_request(base + 0.000001))     # 1us apart
        self.st.insert_request(a_request(base, task_id=8))     # different task
        self.st.insert_request(a_request(base, status=500))    # different status
        self.st.insert_request(a_request(base, client_ip="192.0.2.11"))
        self.st.commit()
        self.assertEqual(self.count(), 5)

    def test_events_are_idempotent_too(self):
        ev = {"ts": 1_700_000_000.5, "kind": "model_loaded", "level": "INFO",
              "model": "m:latest", "source": "llama_server.go:1362",
              "msg": "llama-server started in 28.08 seconds", "duration_ms": 28080.0}
        for _ in range(3):
            self.st.insert_event(ev)
        self.st.commit()
        self.assertEqual(self.count("events"), 1)

    def test_keys_do_not_collide_on_null_fields(self):
        """SQLite treats NULLs in a UNIQUE index as distinct, so the key must
        render them as text rather than relying on column NULLs."""
        k1 = request_key({"ts": 1.0, "endpoint": "/x"})
        k2 = request_key({"ts": 1.0, "endpoint": "/x", "status": None})
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, request_key({"ts": 1.0, "endpoint": "/y"}))
        self.assertNotEqual(event_key({"ts": 1.0, "kind": "a"}),
                            event_key({"ts": 1.0, "kind": "b"}))

    def test_commit_nowait_succeeds_when_lock_is_free(self):
        self.st.insert_request(a_request(1_700_000_001.0))
        self.assertTrue(self.st.commit_nowait())

    def test_commit_nowait_skips_rather_than_blocking(self):
        """Called from a signal handler, so it must never wait on the lock."""
        self.st.lock.acquire()
        try:
            self.assertFalse(self.st.commit_nowait())
        finally:
            self.st.lock.release()


class TestMigration(unittest.TestCase):
    """A database written by schema 1 must upgrade in place without loss."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "old.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_upgrade_preserves_rows_and_collapses_duplicates(self):
        import sqlite3
        # a schema-1 shaped table, with one duplicated row
        db = sqlite3.connect(self.path)
        db.executescript("""
            CREATE TABLE requests (
                id INTEGER PRIMARY KEY, ts REAL NOT NULL, started_ts REAL,
                model TEXT, endpoint TEXT NOT NULL, method TEXT, class TEXT NOT NULL,
                status INTEGER, client_ip TEXT, latency_ms REAL, ttft_ms REAL,
                decode_ms REAL, total_ms REAL, queue_ms REAL, prompt_tokens INTEGER,
                prompt_tokens_total INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
                prefill_tps REAL, decode_tps REAL, draft_accept REAL,
                draft_mean_len REAL, truncated INTEGER, context_tokens INTEGER,
                slot_id INTEGER, task_id INTEGER, attribution TEXT);
            CREATE TABLE events (
                id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL,
                level TEXT, model TEXT, source TEXT, msg TEXT, duration_ms REAL,
                detail_json TEXT);
        """)
        for _ in range(2):     # identical rows: a replayed journal
            db.execute("INSERT INTO requests (ts,endpoint,class,status,output_tokens)"
                       " VALUES (1700000000.5,'/api/chat','inference',200,10)")
        db.execute("INSERT INTO requests (ts,endpoint,class,status,output_tokens)"
                   " VALUES (1700000001.5,'/api/chat','inference',200,20)")
        db.commit()
        db.close()

        st = Store(self.path)              # runs the migration
        self.assertEqual(st.get_meta("schema_version"), str(SCHEMA_VERSION))
        self.assertEqual(st.query("SELECT COUNT(*) n FROM requests")[0]["n"], 2)
        self.assertEqual(st.query("SELECT COUNT(*) n FROM requests"
                                 " WHERE dedupe_key IS NULL")[0]["n"], 0)
        # a further replay is now rejected
        st.insert_request({"ts": 1700000001.5, "endpoint": "/api/chat",
                           "class": "inference", "status": 200, "output_tokens": 20})
        st.commit()
        self.assertEqual(st.query("SELECT COUNT(*) n FROM requests")[0]["n"], 2)


class TestCursor(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_timestamp_extracted_from_cursor(self):
        cur = ("s=258a59cc4f2c43f889894b5d99d7a767;i=d1d0;"
               "b=d4c47474d23c43148949946fc336d395;m=739516457;t=6596d4ec80341;"
               "x=ccca4740a190ae86")
        self.assertEqual(cursor_timestamp_us(cur), 0x6596d4ec80341)
        self.assertIsNone(cursor_timestamp_us(None))
        self.assertIsNone(cursor_timestamp_us("no-timestamp-here"))

    def test_m_field_is_not_mistaken_for_t(self):
        """The cursor also carries ';m=' (monotonic); only ';t=' is realtime."""
        cur = "s=abc;i=1;b=def;m=739516457;t=100;x=1"
        self.assertEqual(cursor_timestamp_us(cur), 0x100)

    def reader(self, backfill="-2 days"):
        return JournaldReader(self.st, "ollama", {"unit": "ollama"}, backfill=backfill)

    def set_cursor(self, cursor):
        import json
        self.st.set_meta("reader_state:ollama", json.dumps({"cursor": cursor}))

    def test_normal_cursor_is_used(self):
        good = "s=abc;i=1;b=def;m=1;t=%x;x=1" % int((time.time() - 300) * 1e6)
        self.set_cursor(good)
        argv = self.reader()._argv()
        self.assertIn(f"--after-cursor={good}", argv)

    def test_future_stamped_cursor_is_rejected(self):
        """A cursor stamped ahead of now makes journalctl wait forever, so the
        collector would silently stall; fall back to a timestamp resume."""
        future = "s=abc;i=1;b=def;m=1;t=%x;x=1" % int((time.time() + 86400) * 1e6)
        self.set_cursor(future)
        argv = self.reader()._argv()
        self.assertFalse(any(a.startswith("--after-cursor") for a in argv))
        self.assertIn("--since", argv)

    def test_resume_uses_newest_stored_row_with_overlap(self):
        newest = 1_700_000_000.0
        self.st.insert_request(a_request(newest))
        self.st.commit()
        since = self.reader()._resume_since()
        self.assertTrue(since.startswith("@"))
        # a little overlap, absorbed by the dedupe key
        self.assertLess(int(since[1:]), newest)
        self.assertGreaterEqual(int(since[1:]), newest - 120)

    def test_resume_falls_back_to_backfill_on_empty_db(self):
        self.assertEqual(self.reader(backfill="-3 days")._resume_since(), "-3 days")

    def test_reader_state_is_per_source(self):
        """Two sources must not share resume state."""
        a = JournaldReader(self.st, "box-a", {"unit": "ollama"})
        b = JournaldReader(self.st, "box-b", {"unit": "ollama"})
        a.save_state({"cursor": "A"})
        b.save_state({"cursor": "B"})
        self.assertEqual(a.load_state()["cursor"], "A")
        self.assertEqual(b.load_state()["cursor"], "B")


if __name__ == "__main__":
    unittest.main()
