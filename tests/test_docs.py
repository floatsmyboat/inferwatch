"""Documentation tests.

A README that quietly falls behind the code is worse than no README, so the
configuration reference is generated from the spec and checked here.
"""

import os
import subprocess
import sys
import unittest

from inferwatch.config import SPEC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
GENERATOR = os.path.join(ROOT, "scripts", "gen-config-docs.py")


def readme() -> str:
    with open(README, encoding="utf-8") as fh:
        return fh.read()


class TestConfigReference(unittest.TestCase):
    def test_generated_block_is_current(self):
        """Fails when the spec changed but the README was not regenerated."""
        proc = subprocess.run([sys.executable, GENERATOR, "--check"],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0,
                         f"{proc.stdout}{proc.stderr}\n"
                         f"run: python scripts/gen-config-docs.py")

    def test_every_setting_is_documented(self):
        text = readme()
        for s in SPEC:
            self.assertIn(s.key, text, f"{s.key} is not in the README")
            self.assertIn(s.env, text, f"{s.env} is not in the README")

    def test_every_setting_has_help_text(self):
        """The Settings tab renders this under each field; blank looks broken."""
        for s in SPEC:
            self.assertTrue(s.help.strip(), f"{s.key} has no help text")

    def test_regenerating_is_idempotent(self):
        before = readme()
        subprocess.run([sys.executable, GENERATOR], capture_output=True, cwd=ROOT)
        self.assertEqual(readme(), before)


class TestReadmeCoversTheBasics(unittest.TestCase):
    def test_documents_both_engines(self):
        text = readme().lower()
        for term in ("ollama", "vllm", "ollama_debug", "/metrics"):
            self.assertIn(term, text)

    def test_documents_the_precedence_chain(self):
        text = readme()
        for term in ("default", "database", "environment", "command line"):
            self.assertIn(term, text)

    def test_documents_each_log_reader(self):
        text = readme()
        for reader in ("journald", "file", "docker"):
            self.assertIn(reader, text)

    def test_states_that_there_is_no_authentication(self):
        """Anyone binding this to a network needs to know that up front."""
        self.assertIn("no authentication", readme().lower())

    def test_has_a_limitations_section(self):
        self.assertIn("limitations", readme().lower())


if __name__ == "__main__":
    unittest.main()
