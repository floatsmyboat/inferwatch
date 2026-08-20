"""Configuration tests: coercion, precedence, locking, and source validation."""

import os
import shutil
import tempfile
import unittest

from inferwatch.config import (BY_KEY, SOURCE_KINDS, Config, ConfigError,
                               Setting, validate_source)
from inferwatch.store import Store


class TestCoercion(unittest.TestCase):
    def s(self, **kw):
        base = {"key": "k", "label": "L", "type_": "int", "default": 1, "group": "G"}
        base.update(kw)
        return Setting(**base)

    def test_numbers_and_bounds(self):
        st = self.s(type_="float", minimum=1.0, maximum=10.0)
        self.assertEqual(st.coerce("2.5"), 2.5)
        with self.assertRaises(ValueError):
            st.coerce(0.5)
        with self.assertRaises(ValueError):
            st.coerce(11)
        with self.assertRaises(ValueError):
            st.coerce("not a number")

    def test_bool_accepts_the_forms_a_form_post_sends(self):
        st = self.s(type_="bool", default=False)
        for truthy in (True, "true", "True", "1", "yes", "on"):
            self.assertTrue(st.coerce(truthy), truthy)
        for falsy in (False, "false", "0", "no", "off"):
            self.assertFalse(st.coerce(falsy), falsy)
        with self.assertRaises(ValueError):
            st.coerce("maybe")

    def test_enum(self):
        st = self.s(type_="enum", choices=["a", "b"], default="a")
        self.assertEqual(st.coerce("b"), "b")
        with self.assertRaises(ValueError):
            st.coerce("c")

    def test_duration(self):
        st = self.s(type_="duration", default="1h")
        for good in ("30s", "15m", "6h", "7d", "2w"):
            self.assertEqual(st.coerce(good), good)
        for bad in ("yesterday", "10", "1y", ""):
            with self.assertRaises(ValueError):
                st.coerce(bad)

    def test_every_shipped_default_validates(self):
        """A default that its own coercion rejects would be a latent bug."""
        for st in BY_KEY.values():
            self.assertEqual(st.coerce(st.default), st.default, st.key)


class ConfigCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def cfg(self, environ=None, overrides=None):
        return Config(self.st, environ=environ or {}, overrides=overrides)


class TestPrecedence(ConfigCase):
    KEY = "retention.raw_days"

    def test_default_when_nothing_set(self):
        c = self.cfg()
        self.assertEqual(c.get(self.KEY), 7.0)
        self.assertEqual(c.origin(self.KEY), "default")
        self.assertFalse(c.locked(self.KEY))

    def test_database_beats_default(self):
        c = self.cfg()
        c.set_many({self.KEY: 14})
        self.assertEqual(c.get(self.KEY), 14.0)
        self.assertEqual(c.origin(self.KEY), "database")

    def test_env_beats_database(self):
        c = self.cfg()
        c.set_many({self.KEY: 14})
        c2 = self.cfg(environ={"INFERWATCH_RETENTION_RAW_DAYS": "21"})
        self.assertEqual(c2.get(self.KEY), 21.0)
        self.assertEqual(c2.origin(self.KEY), "env")
        self.assertTrue(c2.locked(self.KEY))

    def test_flag_beats_env(self):
        c = self.cfg(environ={"INFERWATCH_RETENTION_RAW_DAYS": "21"},
                     overrides={self.KEY: 30})
        self.assertEqual(c.get(self.KEY), 30.0)
        self.assertEqual(c.origin(self.KEY), "flag")

    def test_legacy_env_prefix_still_honoured(self):
        c = self.cfg(environ={"OLLAMON_RETENTION_RAW_DAYS": "9"})
        self.assertEqual(c.get(self.KEY), 9.0)

    def test_unparseable_env_falls_through(self):
        """A typo in the environment must not crash startup."""
        c = self.cfg(environ={"INFERWATCH_RETENTION_RAW_DAYS": "banana"})
        self.assertEqual(c.get(self.KEY), 7.0)
        self.assertEqual(c.origin(self.KEY), "default")

    def test_corrupt_database_value_falls_back_to_default(self):
        self.st.set_config(self.KEY, "not json")
        self.st.commit()
        self.assertEqual(self.cfg().get(self.KEY), 7.0)


class TestWrites(ConfigCase):
    def test_save_is_all_or_nothing(self):
        c = self.cfg()
        with self.assertRaises(ConfigError) as ctx:
            c.set_many({"retention.raw_days": 14, "collection.poll_interval_s": 0.0})
        self.assertIn("collection.poll_interval_s", ctx.exception.errors)
        # the valid key in the same batch must NOT have been applied
        self.assertEqual(c.get("retention.raw_days"), 7.0)

    def test_unknown_key_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            self.cfg().set_many({"nope.nope": 1})
        self.assertIn("nope.nope", ctx.exception.errors)

    def test_locked_key_cannot_be_written(self):
        c = self.cfg(overrides={"server.port": 7070})
        with self.assertRaises(ConfigError) as ctx:
            c.set_many({"server.port": 9999})
        self.assertIn("pinned by flag", ctx.exception.errors["server.port"])

    def test_restart_required_is_reported(self):
        c = self.cfg()
        out = c.set_many({"server.host": "0.0.0.0", "retention.raw_days": 8})
        self.assertEqual(out["restart_required"], ["server.host"])

    def test_reset_returns_to_default(self):
        c = self.cfg()
        c.set_many({"retention.raw_days": 14})
        c.reset("retention.raw_days")
        self.assertEqual(c.get("retention.raw_days"), 7.0)
        self.assertEqual(c.origin("retention.raw_days"), "default")

    def test_listeners_fire_on_change(self):
        c = self.cfg()
        hits = []
        c.on_change(lambda: hits.append(1))
        c.set_many({"retention.raw_days": 8})
        self.assertEqual(len(hits), 1)

    def test_a_broken_listener_does_not_break_saving(self):
        c = self.cfg()
        c.on_change(lambda: 1 / 0)
        c.set_many({"retention.raw_days": 8})     # must not raise
        self.assertEqual(c.get("retention.raw_days"), 8.0)


class TestSourceValidation(unittest.TestCase):
    def test_journald_requires_a_unit(self):
        with self.assertRaises(ValueError):
            validate_source("ollama", "a", {"reader": "journald", "unit": ""})
        ok = validate_source("ollama", "a", {"reader": "journald", "unit": "ollama"})
        self.assertEqual(ok["unit"], "ollama")

    def test_file_reader_requires_a_path(self):
        with self.assertRaises(ValueError) as ctx:
            validate_source("ollama", "a", {"reader": "file"})
        self.assertIn("path", str(ctx.exception))
        ok = validate_source("ollama", "a", {"reader": "file", "path": "/var/log/o.log"})
        self.assertEqual(ok["path"], "/var/log/o.log")

    def test_docker_reader_requires_a_container(self):
        with self.assertRaises(ValueError):
            validate_source("ollama", "a", {"reader": "docker", "container": ""})

    def test_unknown_reader_rejected(self):
        with self.assertRaises(ValueError):
            validate_source("ollama", "a", {"reader": "carrier-pigeon"})

    def test_vllm_requires_a_url(self):
        with self.assertRaises(ValueError):
            validate_source("vllm", "v", {"url": ""})
        ok = validate_source("vllm", "v", {"url": "http://127.0.0.1:8000"})
        self.assertEqual(ok["url"], "http://127.0.0.1:8000")

    def test_name_is_required(self):
        with self.assertRaises(ValueError):
            validate_source("vllm", "  ", {"url": "http://x"})

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            validate_source("tensorrt", "t", {})

    def test_defaults_filled_in(self):
        out = validate_source("ollama", "a", {"reader": "journald", "unit": "ollama"})
        for field in SOURCE_KINDS["ollama"]["fields"]:
            self.assertIn(field["key"], out)


class TestDescribe(ConfigCase):
    def test_describe_carries_everything_the_ui_needs(self):
        self.st.add_source("vllm", "v1", {"url": "http://x"})
        d = self.cfg(overrides={"server.port": 1234}).describe()
        self.assertEqual(set(d), {"groups", "settings", "source_kinds", "sources"})
        port = next(s for s in d["settings"] if s["key"] == "server.port")
        self.assertTrue(port["locked"])
        self.assertEqual(port["origin"], "flag")
        self.assertEqual(port["value"], 1234)
        self.assertEqual(len(d["sources"]), 1)
        for s in d["settings"]:
            self.assertIn(s["group"], d["groups"], s["key"])


if __name__ == "__main__":
    unittest.main()
