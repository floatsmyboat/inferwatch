"""API-key auth for the MCP server's HTTP transports.

Over stdio there is nothing to protect -- the client owns the subprocess.  Over
HTTP the server answers questions about the whole metrics database, client
addresses included, and `run_sql` is a general query tool over all of it.
"""

import asyncio
import os
import stat
import tempfile
import unittest

from inferwatch.mcp_auth import (DEFAULT_KEY_FILE, ENV_KEY, ENV_KEY_FILE,
                                 BearerAuth, KeyAbsent, KeyError_,
                                 check_key_strength, load_api_key,
                                 presented_key, read_key_file)

KEY = "0123456789abcdef0123456789abcdef"


class KeyFileCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def write(self, text, mode=0o640):
        path = os.path.join(self.dir, "key")
        with open(path, "w") as fh:
            fh.write(text)
        os.chmod(path, mode)
        return path


class TestKeyFile(KeyFileCase):
    def test_a_bare_key_is_read(self):
        self.assertEqual(read_key_file(self.write(KEY + "\n")), KEY)

    def test_an_environmentfile_line_is_tolerated(self):
        """A bare key file and an EnvironmentFile look alike to a tired
        operator, and getting it wrong should not mean a broken service."""
        self.assertEqual(read_key_file(self.write(f"{ENV_KEY}={KEY}\n")), KEY)
        self.assertEqual(read_key_file(self.write(f'{ENV_KEY}="{KEY}"\n')), KEY)

    def test_a_world_readable_file_is_refused_not_warned(self):
        """A secret every local user can read is not a secret, and refusing to
        start is the only response that cannot be ignored."""
        path = self.write(KEY, mode=0o644)
        with self.assertRaises(KeyError_) as cm:
            read_key_file(path)
        self.assertIn("world-readable", str(cm.exception))

    def test_group_readable_is_allowed(self):
        """Sharing with a service group is the normal way to hand it to a unit."""
        self.assertEqual(read_key_file(self.write(KEY, mode=0o640)), KEY)

    def test_an_empty_file_is_refused(self):
        with self.assertRaises(KeyError_):
            read_key_file(self.write("   \n"))

    def test_a_missing_file_raises_absent_specifically(self):
        """Absent and present-but-unreadable need opposite advice -- create
        one, versus fix who can read the one you have."""
        with self.assertRaises(KeyAbsent):
            read_key_file(os.path.join(self.dir, "nope"))

    def test_an_unreadable_file_is_not_reported_as_absent(self):
        """Regression: /etc/inferwatch is 0750, so an unprivileged process
        cannot even stat the key inside it. Probing with os.path.exists first
        reported the key as missing and told the operator to create a file that
        already existed."""
        path = self.write(KEY, mode=0o600)
        if os.geteuid() == 0:
            self.skipTest("root can read anything")
        os.chmod(path, 0o000)
        with self.assertRaises(KeyError_) as cm:
            read_key_file(path)
        self.assertNotIsInstance(cm.exception, KeyAbsent)
        self.assertIn("permission denied", str(cm.exception))
        # and it says how to fix it, not just what failed
        self.assertIn("--group", str(cm.exception))

    def test_load_reports_an_unreadable_default_as_an_error_not_absence(self):
        path = self.write(KEY, mode=0o000)
        if os.geteuid() == 0:
            self.skipTest("root can read anything")
        with self.assertRaises(KeyError_) as cm:
            load_api_key(path, {})
        self.assertNotIsInstance(cm.exception, KeyAbsent)

    def test_a_short_key_is_refused(self):
        with self.assertRaises(KeyError_):
            check_key_strength("tooshort")
        check_key_strength(KEY)          # no raise


class TestKeyResolution(KeyFileCase):
    def test_an_explicit_file_wins(self):
        path = self.write(KEY)
        key, src = load_api_key(path, {ENV_KEY: "from-env"})
        self.assertEqual(key, KEY)
        self.assertIn(path, src)

    def test_the_env_value_beats_the_env_file(self):
        path = self.write("from-file")
        key, src = load_api_key(None, {ENV_KEY: KEY, ENV_KEY_FILE: path})
        self.assertEqual(key, KEY)
        self.assertIn(ENV_KEY, src)

    def test_the_env_file_is_used_when_no_value_is_set(self):
        path = self.write(KEY)
        key, _ = load_api_key(None, {ENV_KEY_FILE: path})
        self.assertEqual(key, KEY)

    def test_nothing_configured_returns_none_with_a_reason(self):
        """None rather than an exception, so the caller decides whether it is
        fatal -- and for an HTTP transport it is."""
        key, why = load_api_key(None, {})
        if not os.path.exists(DEFAULT_KEY_FILE):
            self.assertIsNone(key)
            self.assertIn(DEFAULT_KEY_FILE, why)

    def test_the_key_never_comes_from_the_repo_or_the_database(self):
        """The resolution order is env, env-file, /etc -- nothing relative."""
        _, why = load_api_key(None, {})
        self.assertTrue(DEFAULT_KEY_FILE.startswith("/etc/"), DEFAULT_KEY_FILE)
        self.assertNotIn(".db", why)


class TestPresentedKey(unittest.TestCase):
    def hdr(self, name, value):
        return [(name.encode(), value.encode())]

    def test_bearer_is_read(self):
        self.assertEqual(presented_key(self.hdr("authorization", "Bearer " + KEY)), KEY)

    def test_the_scheme_is_case_insensitive(self):
        self.assertEqual(presented_key(self.hdr("Authorization", "bearer " + KEY)), KEY)

    def test_x_api_key_is_accepted(self):
        """Some clients can only set an arbitrary header."""
        self.assertEqual(presented_key(self.hdr("x-api-key", KEY)), KEY)

    def test_other_schemes_are_ignored(self):
        self.assertIsNone(presented_key(self.hdr("authorization", "Basic " + KEY)))

    def test_absent_and_empty_are_none(self):
        self.assertIsNone(presented_key([]))
        self.assertIsNone(presented_key(self.hdr("authorization", "Bearer ")))
        self.assertIsNone(presented_key(self.hdr("x-api-key", "  ")))


class Recorder:
    """A minimal ASGI app that records whether it was reached."""

    def __init__(self):
        self.calls = []

    async def __call__(self, scope, receive, send):
        self.calls.append(scope.get("type"))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


class TestBearerAuth(unittest.TestCase):
    def run_request(self, headers, scope_type="http"):
        app = Recorder()
        mw = BearerAuth(app, KEY)
        sent = []

        async def send(msg):
            sent.append(msg)

        async def receive():
            return {"type": "http.request"}

        asyncio.run(mw({"type": scope_type, "headers": headers,
                        "client": ("10.0.0.9", 1234)}, receive, send))
        return app, sent

    def test_a_correct_key_reaches_the_app(self):
        app, sent = self.run_request([(b"authorization", b"Bearer " + KEY.encode())])
        self.assertEqual(app.calls, ["http"])
        self.assertEqual(sent[0]["status"], 200)

    def test_no_credentials_is_401_and_never_reaches_the_app(self):
        app, sent = self.run_request([])
        self.assertEqual(app.calls, [])
        self.assertEqual(sent[0]["status"], 401)

    def test_a_wrong_key_is_401(self):
        app, sent = self.run_request([(b"authorization", b"Bearer wrong" + b"0" * 30)])
        self.assertEqual(app.calls, [])
        self.assertEqual(sent[0]["status"], 401)

    def test_the_401_carries_a_challenge_and_no_data(self):
        _, sent = self.run_request([])
        headers = dict(sent[0]["headers"])
        self.assertIn(b"www-authenticate", headers)
        self.assertNotIn(KEY.encode(), sent[1]["body"])

    def test_lifespan_passes_through_untouched(self):
        """Checking it would stop the app ever starting."""
        app = Recorder()
        mw = BearerAuth(app, KEY)
        seen = []

        async def send(msg):
            seen.append(msg)

        async def receive():
            return {"type": "lifespan.startup"}

        async def drive():
            await mw({"type": "lifespan"}, receive, send)

        asyncio.run(drive())
        self.assertEqual(app.calls, ["lifespan"])

    def test_a_prefix_of_the_key_is_rejected(self):
        app, sent = self.run_request(
            [(b"authorization", b"Bearer " + KEY[:-1].encode())])
        self.assertEqual(sent[0]["status"], 401)
        self.assertEqual(app.calls, [])


if __name__ == "__main__":
    unittest.main()
