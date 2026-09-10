"""Secret handling for source configuration.

A vLLM api_key has to be stored somewhere the collector can read it, which
means plain text in the `sources` table.  What these tests pin down is that it
never travels back out: the HTTP API has no authentication, so anything served
there is readable by whatever can reach the port.
"""

import json
import os
import shutil
import tempfile
import unittest

from inferwatch.config import (REDACTED, SOURCE_KINDS, is_secret_ref,
                               merge_secrets, redact_config, redact_sources,
                               resolve_secret, secret_fields, validate_source)
from inferwatch.store import Store

KEY = "sk-not-a-real-key-0123456789"


class TestSecretFields(unittest.TestCase):
    def test_the_vllm_api_key_is_marked_secret(self):
        self.assertEqual(secret_fields("vllm"), {"api_key"})

    def test_engines_with_no_secrets_have_none(self):
        self.assertEqual(secret_fields("ollama"), set())
        self.assertEqual(secret_fields("swarmui"), set())

    def test_an_unknown_kind_does_not_explode(self):
        self.assertEqual(secret_fields("nope"), set())

    def test_every_field_named_like_a_secret_is_marked(self):
        """A future field called api_key/token/password must not be added
        without the flag, or it would be served in the clear."""
        for kind, spec in SOURCE_KINDS.items():
            for f in spec["fields"]:
                looks_secret = any(w in f["key"] for w in
                                   ("key", "token", "secret", "password", "passwd"))
                if looks_secret:
                    self.assertTrue(f.get("secret"),
                                    f"{kind}.{f['key']} looks like a secret but "
                                    f"is not marked secret=True")


class TestRedaction(unittest.TestCase):
    def test_a_literal_is_masked(self):
        got = redact_config("vllm", {"url": "http://x:8000", "api_key": KEY})
        self.assertEqual(got["api_key"], REDACTED)
        self.assertEqual(got["url"], "http://x:8000")

    def test_an_empty_value_stays_empty(self):
        """So a caller can tell 'nothing is set' from 'set, and not readable'."""
        self.assertEqual(redact_config("vllm", {"api_key": ""})["api_key"], "")

    def test_an_environment_reference_is_shown_not_masked(self):
        """The reference is not the secret, and knowing which variable is
        referenced is the useful part."""
        for ref in ("${VLLM_API_KEY}", "$VLLM_API_KEY"):
            self.assertEqual(redact_config("vllm", {"api_key": ref})["api_key"], ref)
            self.assertTrue(is_secret_ref(ref))

    def test_a_literal_is_not_mistaken_for_a_reference(self):
        for literal in (KEY, "sk-${weird}tail", "", "  "):
            self.assertFalse(is_secret_ref(literal), literal)

    def test_the_original_is_not_mutated(self):
        cfg = {"api_key": KEY}
        redact_config("vllm", cfg)
        self.assertEqual(cfg["api_key"], KEY)

    def test_redact_sources_covers_a_whole_listing(self):
        rows = [{"kind": "vllm", "name": "a", "config": {"api_key": KEY}},
                {"kind": "ollama", "name": "b", "config": {"unit": "ollama"}}]
        got = redact_sources(rows)
        self.assertEqual(got[0]["config"]["api_key"], REDACTED)
        self.assertEqual(got[1]["config"]["unit"], "ollama")


class TestRoundTrip(unittest.TestCase):
    """The classic way redaction turns into data loss."""

    def test_sending_the_mask_back_keeps_the_stored_key(self):
        got = merge_secrets("vllm", {"url": "u", "api_key": REDACTED},
                            {"url": "u", "api_key": KEY})
        self.assertEqual(got["api_key"], KEY)

    def test_an_empty_string_still_clears_it(self):
        """Masking must not make a key impossible to remove."""
        got = merge_secrets("vllm", {"api_key": ""}, {"api_key": KEY})
        self.assertEqual(got["api_key"], "")

    def test_a_new_value_replaces_the_old(self):
        got = merge_secrets("vllm", {"api_key": "sk-new"}, {"api_key": "sk-old"})
        self.assertEqual(got["api_key"], "sk-new")

    def test_the_mask_on_a_brand_new_source_is_dropped(self):
        """Nothing is stored yet, so a mask here is a mistake, not 'unchanged'
        -- storing the literal string would make it the key."""
        got = merge_secrets("vllm", {"url": "u", "api_key": REDACTED}, None)
        self.assertNotIn("api_key", got)
        clean = validate_source("vllm", "v", got)
        self.assertEqual(clean["api_key"], "")


class TestProbe(unittest.TestCase):
    def test_a_mask_is_never_dialled_out_as_a_token(self):
        """A probe describes unsaved config, so there is nothing to merge from;
        sending "***redacted***" as a bearer token would fail confusingly."""
        cfg = merge_secrets("vllm", {"url": "http://x:8000", "api_key": REDACTED}, None)
        self.assertEqual(resolve_secret(cfg.get("api_key")), "")


class TestResolution(unittest.TestCase):
    def test_a_reference_is_expanded_at_use(self):
        self.assertEqual(resolve_secret("${K}", {"K": KEY}), KEY)
        self.assertEqual(resolve_secret("$K", {"K": KEY}), KEY)

    def test_a_literal_passes_through(self):
        self.assertEqual(resolve_secret(KEY, {}), KEY)

    def test_an_unset_variable_resolves_to_empty_not_to_the_literal(self):
        """Sending "${VLLM_API_KEY}" as a bearer token would fail confusingly;
        sending no header at all fails as a clean 401."""
        self.assertEqual(resolve_secret("${MISSING_VAR}", {}), "")

    def test_empty_and_none_are_empty(self):
        self.assertEqual(resolve_secret("", {}), "")
        self.assertEqual(resolve_secret(None, {}), "")


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.st = Store(os.path.join(self.dir, "t.db"))
        self.st.add_source("vllm", "v1", {"url": "http://127.0.0.1:8000",
                                          "api_key": KEY})
        self.st.commit()

    def tearDown(self):
        self.st.db.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class TestStoreStillHasTheRealValue(StoreCase):
    def test_the_collector_can_read_it(self):
        """Redaction is a serving concern; the collector needs the real value
        to send as a bearer token."""
        cfg = self.st.list_sources()[0]["config"]
        self.assertEqual(cfg["api_key"], KEY)

    def test_but_a_listing_served_from_it_is_masked(self):
        served = redact_sources(self.st.list_sources())
        self.assertEqual(served[0]["config"]["api_key"], REDACTED)


class TestRunSqlEscapeHatch(StoreCase):
    def test_selecting_the_config_column_is_masked(self):
        """run_sql is a documented read-only escape hatch, and a plain SELECT
        was enough to read the key straight back out."""
        from inferwatch import mcp_server
        rows = self.st.query("SELECT config_json FROM sources")
        got = mcp_server._redact_row(dict(rows[0]))
        self.assertNotIn(KEY, got["config_json"])
        self.assertEqual(json.loads(got["config_json"])["api_key"], REDACTED)

    def test_selecting_the_whole_row_is_masked_too(self):
        from inferwatch import mcp_server
        rows = self.st.query("SELECT * FROM sources")
        got = mcp_server._redact_row(dict(rows[0]))
        self.assertNotIn(KEY, json.dumps(got))

    def test_other_columns_are_untouched(self):
        from inferwatch import mcp_server
        rows = self.st.query("SELECT name, kind, config_json FROM sources")
        got = mcp_server._redact_row(dict(rows[0]))
        self.assertEqual((got["name"], got["kind"]), ("v1", "vllm"))

    def test_a_reference_survives_the_escape_hatch(self):
        from inferwatch import mcp_server
        got = mcp_server._redact_row(
            {"config_json": json.dumps({"api_key": "${VLLM_API_KEY}"})})
        self.assertEqual(json.loads(got["config_json"])["api_key"], "${VLLM_API_KEY}")

    def test_a_non_json_column_is_left_alone(self):
        from inferwatch import mcp_server
        self.assertEqual(mcp_server._redact_row({"config_json": "not json"}),
                         {"config_json": "not json"})


class TestCliDoesNotPrintIt(StoreCase):
    def test_sources_list_masks_the_key(self):
        """It used to print `api_key=<the key>` straight to the terminal, which
        also lands in shell history and any captured output."""
        served = redact_sources(self.st.list_sources())
        line = " ".join(f"{k}={v}" for k, v in served[0]["config"].items() if v)
        self.assertNotIn(KEY, line)
        self.assertIn(f"api_key={REDACTED}", line)


if __name__ == "__main__":
    unittest.main()
