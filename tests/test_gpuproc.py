"""GPU-to-engine attribution.

The rule everywhere here is that attribution degrades to None rather than to a
guess: a wrong GPU on an instance's pane is worse than an unlabelled one.
"""

import unittest
from unittest import mock

from inferwatch import gpuproc


class TestDescendants(unittest.TestCase):
    def tree(self, parents):
        """Run descendants() against a fake /proc described by {pid: ppid}."""
        pids = [str(p) for p in parents]
        with mock.patch.object(gpuproc.os, "listdir", return_value=pids), \
             mock.patch.object(gpuproc, "_ppid", side_effect=lambda p: parents.get(p)):
            return gpuproc.descendants(1000)

    def test_includes_the_root_and_its_children(self):
        got = self.tree({1000: 1, 1001: 1000, 1002: 1000, 2000: 1})
        self.assertEqual(got, {1000, 1001, 1002})

    def test_follows_grandchildren(self):
        got = self.tree({1000: 1, 1001: 1000, 1002: 1001, 1003: 1002})
        self.assertEqual(got, {1000, 1001, 1002, 1003})

    def test_unrelated_processes_are_excluded(self):
        got = self.tree({1000: 1, 2000: 1, 2001: 2000})
        self.assertEqual(got, {1000})

    def test_a_parent_cycle_does_not_hang(self):
        """/proc is read pid by pid, so a torn read can imply a loop."""
        got = self.tree({1000: 1, 3000: 3001, 3001: 3000})
        self.assertEqual(got, {1000})

    def test_a_missing_parent_is_survived(self):
        """A process can exit between listdir() and reading its stat."""
        got = self.tree({1000: 1, 1001: 1000, 1002: None})
        self.assertEqual(got, {1000, 1001})

    def test_a_descendant_is_found_however_proc_happens_to_be_ordered(self):
        """Regression: the scan was capped at 4096 entries, and
        os.listdir("/proc") has no meaningful order -- so on a busy host the
        parent map dropped arbitrary processes and the tree came back short,
        which reads downstream as an engine holding fewer GPUs than it does."""
        parents = {1000: 1}
        for pid in range(5000, 10000):        # 5000 unrelated processes
            parents[pid] = 1
        parents[9999] = 1000                  # the real child, listed last
        got = self.tree(parents)
        self.assertIn(9999, got, "a descendant was dropped by the /proc scan")
        self.assertEqual(got, {1000, 9999})

    def test_an_unreadable_proc_yields_just_the_root(self):
        with mock.patch.object(gpuproc.os, "listdir", side_effect=OSError):
            self.assertEqual(gpuproc.descendants(42), {42})


class TestUrlHelpers(unittest.TestCase):
    def test_port_is_extracted(self):
        self.assertEqual(gpuproc.port_of("http://127.0.0.1:8000"), 8000)
        self.assertEqual(gpuproc.port_of("http://localhost:11434/"), 11434)

    def test_no_port_is_none(self):
        self.assertIsNone(gpuproc.port_of("http://example.invalid"))
        self.assertIsNone(gpuproc.port_of(""))

    def test_local_hosts_are_recognised(self):
        for url in ("http://127.0.0.1:8000", "http://localhost:8000",
                    "http://[::1]:8000", "http://0.0.0.0:8000"):
            self.assertTrue(gpuproc.is_local(url), url)

    def test_a_bracketed_ipv6_loopback_is_local(self):
        """Regression: the authority was split on ":" before the brackets were
        stripped, so the host came out as "[" and an instance bound to
        http://[::1]:8000 lost GPU attribution entirely."""
        self.assertTrue(gpuproc.is_local("http://[::1]:8000"))
        self.assertEqual(gpuproc.port_of("http://[::1]:8000"), 8000)

    def test_a_remote_ipv6_host_is_not_local(self):
        self.assertFalse(gpuproc.is_local("http://[2001:db8::1]:8000"))

    def test_a_remote_host_is_not_local(self):
        """Attribution walks /proc, which only exists for this machine."""
        self.assertFalse(gpuproc.is_local("http://10.0.0.5:8000"))

    def test_attribution_is_none_without_visible_compute_processes(self):
        """Common inside a container: nvidia-smi lists no PIDs. That is
        'unknown', which the UI must not render as 'holds no GPU'."""
        with mock.patch.object(gpuproc, "compute_apps", return_value=[]):
            self.assertIsNone(gpuproc.gpus_for_port(8000))

    def test_attribution_is_none_when_the_port_has_no_listener(self):
        with mock.patch.object(gpuproc, "compute_apps",
                               return_value=[{"pid": 1, "gpu_index": 0}]), \
             mock.patch.object(gpuproc, "listener_pid", return_value=None):
            self.assertIsNone(gpuproc.gpus_for_port(8000))

    def test_gpus_held_by_the_process_tree_are_reported(self):
        with mock.patch.object(gpuproc, "compute_apps", return_value=[
                {"pid": 500, "gpu_index": 1}, {"pid": 501, "gpu_index": 2},
                {"pid": 900, "gpu_index": 0}]), \
             mock.patch.object(gpuproc, "listener_pid", return_value=500), \
             mock.patch.object(gpuproc, "descendants", return_value={500, 501}):
            self.assertEqual(gpuproc.gpus_for_port(8000), [1, 2])


if __name__ == "__main__":
    unittest.main()
