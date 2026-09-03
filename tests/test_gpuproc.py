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

    def test_attribution_is_none_when_nvidia_smi_cannot_be_asked(self):
        """No nvidia-smi at all is 'unknown', which the UI must not render as
        'holds no GPU'."""
        with mock.patch.object(gpuproc, "compute_apps", return_value=None), \
             mock.patch.object(gpuproc, "listener_pid", return_value=1):
            self.assertIsNone(gpuproc.gpus_for_port(8000))

    def test_an_engine_holding_no_gpu_is_an_empty_list_not_none(self):
        """nvidia-smi answered and this engine is on none of the cards -- a real
        result for a CPU-only instance, and a different statement from 'could
        not tell', which the two renderings depend on distinguishing."""
        with mock.patch.object(gpuproc, "compute_apps", return_value=[]), \
             mock.patch.object(gpuproc, "listener_pid", return_value=1), \
             mock.patch.object(gpuproc, "descendants", return_value={1}):
            self.assertEqual(gpuproc.gpus_for_port(8000), [])

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


class TestUnitAttribution(unittest.TestCase):
    """Attribution by cgroup: the only method that survives reparenting."""

    APPS = [{"pid": 100, "gpu_index": 0}, {"pid": 200, "gpu_index": 2},
            {"pid": 201, "gpu_index": 3}]

    def resolve_unit(self, unit, units, apps=None):
        """Run gpus_for_unit against a fake /proc described by {pid: unit}."""
        pids = [str(p) for p in units]
        with mock.patch.object(gpuproc.os, "listdir", return_value=pids), \
             mock.patch.object(gpuproc, "unit_of", side_effect=lambda p: units.get(p)), \
             mock.patch.object(gpuproc, "compute_apps",
                               return_value=self.APPS if apps is None else apps):
            return gpuproc.gpus_for_unit(unit)

    def test_a_unit_gets_only_its_own_cards(self):
        """Regression for the real failure: the engine's workers were in
        vllm-qwen38.service while the URL's listener was in vllm-proxy.service,
        so the descendant walk found nothing and the UI showed stale indices."""
        units = {100: "swarmui.service", 200: "vllm-qwen38.service",
                 201: "vllm-qwen38.service", 300: "vllm-proxy.service"}
        self.assertEqual(self.resolve_unit("vllm-qwen38", units), [2, 3])
        self.assertEqual(self.resolve_unit("swarmui", units), [0])

    def test_the_proxy_unit_holds_nothing(self):
        units = {100: "swarmui.service", 300: "vllm-proxy.service"}
        self.assertEqual(self.resolve_unit("vllm-proxy", units), [])

    def test_a_bare_unit_name_is_suffixed(self):
        units = {200: "ollama.service"}
        self.assertEqual(self.resolve_unit("ollama", units),
                         self.resolve_unit("ollama.service", units))

    def test_a_unit_with_no_processes_is_unknown_not_empty(self):
        """A running unit always has a process in its cgroup, so nothing there
        means it is not running or the name is wrong -- not that it holds no
        GPU. Returning [] would render as a confident 'no cards'."""
        self.assertIsNone(self.resolve_unit("typo", {200: "ollama.service"}))

    def test_an_unreadable_proc_is_unknown(self):
        with mock.patch.object(gpuproc.os, "listdir", side_effect=OSError):
            self.assertIsNone(gpuproc.gpus_for_unit("ollama"))

    def test_normalise_leaves_scopes_and_slices_alone(self):
        self.assertEqual(gpuproc.normalise_unit("a.scope"), "a.scope")
        self.assertEqual(gpuproc.normalise_unit("a.slice"), "a.slice")
        self.assertEqual(gpuproc.normalise_unit("a"), "a.service")
        self.assertEqual(gpuproc.normalise_unit(""), "")


class TestUnitOf(unittest.TestCase):
    def cgroup(self, text):
        m = mock.mock_open(read_data=text)
        with mock.patch("builtins.open", m):
            return gpuproc.unit_of(1234)

    def test_reads_a_service_unit(self):
        self.assertEqual(self.cgroup("0::/system.slice/vllm-qwen38.service"),
                         "vllm-qwen38.service")

    def test_reads_the_innermost_unit_of_a_user_session(self):
        """`ollama serve` run by hand lives in a scope, not a service. Slices
        are deliberately not matched: they group units rather than being one."""
        self.assertEqual(
            self.cgroup("0::/user.slice/user-1000.slice/session-3.scope"),
            "session-3.scope")

    def test_a_cgroup_with_no_unit_is_none(self):
        self.assertIsNone(self.cgroup("0::/"))

    def test_an_exited_process_is_none(self):
        with mock.patch("builtins.open", side_effect=OSError):
            self.assertIsNone(gpuproc.unit_of(1234))


class TestResolveOrder(unittest.TestCase):
    """resolve() prefers the most trustworthy method that can answer at all."""

    def test_a_unit_answer_wins_over_the_port_walk(self):
        with mock.patch.object(gpuproc, "gpus_for_unit", return_value=[2, 3]), \
             mock.patch.object(gpuproc, "gpus_for_port", return_value=[1]):
            got, how = gpuproc.resolve(unit="vllm-qwen38", port=8000)
        self.assertEqual(got, [2, 3])
        self.assertEqual(how, "cgroup:vllm-qwen38.service")

    def test_an_empty_unit_answer_still_wins(self):
        """A unit that demonstrably holds no GPU is a real result. Falling
        through to the port walk could match somebody else's processes."""
        with mock.patch.object(gpuproc, "gpus_for_unit", return_value=[]), \
             mock.patch.object(gpuproc, "gpus_for_port", return_value=[1]):
            got, how = gpuproc.resolve(unit="cpu-only", port=8000)
        self.assertEqual(got, [])
        self.assertTrue(how.startswith("cgroup:"))

    def test_falls_back_to_pids_then_port(self):
        with mock.patch.object(gpuproc, "gpus_for_unit", return_value=None), \
             mock.patch.object(gpuproc, "gpus_for_pids", return_value=[1]), \
             mock.patch.object(gpuproc, "gpus_for_port", return_value=[0]):
            self.assertEqual(gpuproc.resolve(unit="x", pids=[9], port=8000),
                             ([1], "pids"))
        with mock.patch.object(gpuproc, "gpus_for_unit", return_value=None), \
             mock.patch.object(gpuproc, "gpus_for_pids", return_value=None), \
             mock.patch.object(gpuproc, "gpus_for_port", return_value=[0]):
            self.assertEqual(gpuproc.resolve(unit="x", pids=[9], port=8000),
                             ([0], "port:8000"))

    def test_an_empty_port_answer_degrades_to_unknown(self):
        """Nothing on a GPU behind that port almost always means the wrong
        process tree -- a proxy in front, or reparented workers -- not a
        CPU-only engine. Asserting "holds no GPU" from the weakest method would
        be a confident wrong answer, which is what this whole change is about."""
        with mock.patch.object(gpuproc, "gpus_for_port", return_value=[]):
            self.assertEqual(gpuproc.resolve(port=8000), (None, "unavailable"))

    def test_a_non_empty_port_answer_is_still_used(self):
        with mock.patch.object(gpuproc, "gpus_for_port", return_value=[1]):
            self.assertEqual(gpuproc.resolve(port=8000), ([1], "port:8000"))

    def test_an_empty_pids_answer_is_trusted(self):
        """A pid set is precise: if those exact processes hold no GPU, they
        hold no GPU."""
        with mock.patch.object(gpuproc, "gpus_for_pids", return_value=[]), \
             mock.patch.object(gpuproc, "gpus_for_port", return_value=[9]):
            self.assertEqual(gpuproc.resolve(pids=[42], port=8000), ([], "pids"))

    def test_nothing_available_reports_unavailable(self):
        self.assertEqual(gpuproc.resolve(), (None, "unavailable"))

    def test_the_method_travels_with_the_answer(self):
        """So a surprising attribution is diagnosable without re-running it."""
        with mock.patch.object(gpuproc, "gpus_for_pids", return_value=[3]):
            self.assertEqual(gpuproc.resolve(pids=[42])[1], "pids")


if __name__ == "__main__":
    unittest.main()
