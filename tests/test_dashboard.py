"""Dashboard checks.

The dashboard is a single hand-written HTML file, so nothing type-checks it.
These tests catch the two failures that would otherwise ship silently:
a JavaScript syntax error (blank page) and a reference to an element id that
does not exist in the markup (silently dead panel).

esprima is a pure-Python JS parser; quickjs is a real engine used to execute
the pure formatting helpers.  Both are optional -- the suite skips rather than
fails when they are absent.
"""

import os
import re
import unittest

HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "inferwatch", "web", "index.html")


def read():
    with open(HTML, encoding="utf-8") as fh:
        return fh.read()


def script(html):
    return re.search(r"<script>(.*)</script>", html, re.S).group(1)


def extract_function(js, name):
    """Return the source of one top-level `function name(...) {...}`."""
    m = re.search(r"^function " + name + r"\(", js, re.M)
    if not m:
        raise AssertionError(f"function {name} not found")
    start = js.index("{", m.start())
    depth = 0
    for i in range(start, len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[m.start():i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


class TestMarkup(unittest.TestCase):
    def setUp(self):
        self.html = read()

    def test_single_script_block_and_closed_document(self):
        self.assertEqual(self.html.count("<script"), 1)
        self.assertEqual(self.html.count("</script>"), 1)
        self.assertTrue(self.html.rstrip().endswith("</html>"))

    def test_every_referenced_element_id_exists(self):
        ids = set(re.findall(r'\bid="([^"]+)"', self.html))
        refs = set(re.findall(r'\$\("([^"]+)"\)', script(self.html)))
        self.assertTrue(refs, "no element lookups found -- did the script change?")
        self.assertFalse(refs - ids, f"referenced but missing from markup: {refs - ids}")

    def test_every_chart_card_is_actually_drawn(self):
        """A card with no draw() call renders as a blank panel and no other test
        notices -- exactly how the vLLM temperature and power charts shipped
        empty once."""
        js = script(self.html)
        cards = re.findall(r'class="card" id="c-([a-z]+)"', self.html)
        drawn = set(re.findall(r'draw\("([a-z]+)"', js))
        undrawn = sorted(c for c in cards if c not in drawn)
        self.assertFalse(undrawn, f"chart cards with no draw() call: {undrawn}")

    def test_no_draw_call_targets_a_missing_card(self):
        """The mirror image: a draw() for a card that does not exist."""
        js = script(self.html)
        cards = set(re.findall(r'class="card" id="c-([a-z]+)"', self.html))
        drawn = set(re.findall(r'draw\("([a-z]+)"', js))
        orphans = sorted(d for d in drawn if d not in cards)
        self.assertFalse(orphans, f"draw() calls with no card: {orphans}")

    def test_chart_cards_have_plot_legend_and_table_targets(self):
        cards = re.findall(r'class="card" id="c-([a-z]+)"', self.html)
        # 8 ollama panels + 8 vllm panels
        self.assertGreaterEqual(len(cards), 16)
        for c in cards:
            for prefix in ("p", "l", "t"):
                self.assertIn(f'id="{prefix}-{c}"', self.html,
                              f"card c-{c} is missing its {prefix}- target")

    def test_no_external_resources(self):
        """Must work offline on a box with no CDN access."""
        for pattern in (r'src="https?://', r'href="https?://[^"]*\.css',
                        r'@import\s+url\(https?://'):
            self.assertIsNone(re.search(pattern, self.html),
                              f"external resource matched {pattern}")

    def test_all_three_tabs_exist_and_are_wired(self):
        for tab in ("ollama", "vllm", "settings"):
            self.assertIn(f'id="tab-{tab}"', self.html)
            self.assertIn(f'data-tab="{tab}"', self.html)

    def test_only_the_first_tab_is_visible_initially(self):
        """The other sections must ship hidden, or every panel renders at once."""
        self.assertIn('<section id="tab-vllm" hidden>', self.html)
        self.assertIn('<section id="tab-settings" hidden>', self.html)

    def test_vllm_panels_do_not_promise_per_request_detail(self):
        """vLLM publishes no per-request rows, so offering a request table or a
        client breakdown there would be a lie."""
        vllm = self.html[self.html.index('id="tab-vllm"'):
                         self.html.index('id="tab-settings"')]
        for forbidden in ("Recent requests", "Slowest by TTFT", "client"):
            self.assertNotIn(forbidden, vllm)

    def test_vllm_tab_has_gpu_and_vram_panels(self):
        vllm = self.html[self.html.index('id="tab-vllm"'):
                         self.html.index('id="tab-settings"')]
        self.assertIn('id="c-vgpu"', vllm)
        self.assertIn('id="c-vvram"', vllm)
        self.assertIn("GPU utilisation", vllm)
        self.assertIn("VRAM in use", vllm)

    def test_both_tabs_show_gpu_temperature_and_power(self):
        for card in ("c-gtemp", "c-vtemp", "c-gpower", "c-vpower"):
            self.assertIn(f'id="{card}"', self.html)
        # one card heading per tab; the KPI tiles reuse the same label
        self.assertEqual(self.html.count("<h2>GPU temperature</h2>"), 2)
        self.assertEqual(self.html.count("<h2>GPU power draw</h2>"), 2)

    def test_temperature_and_power_are_not_on_one_axis(self):
        """Different units on one plot invents a correlation; they get their
        own charts."""
        js = script(self.html)
        self.assertIn("tempChartSpec", js)
        self.assertIn("powerChartSpec", js)

    def test_gpu_attribution_is_expressed_as_emphasis(self):
        """A host's other GPUs still appear, in the de-emphasis grey, rather
        than being hidden or drawn as if the instance owned them."""
        js = script(self.html)
        self.assertIn("--deemph", js)
        self.assertIn("other engine", js)
        self.assertIn("gpu_indices", js)

    def test_vllm_tab_states_its_provenance(self):
        vllm = self.html[self.html.index('id="tab-vllm"'):
                         self.html.index('id="tab-settings"')]
        self.assertIn("pre-aggregated", vllm)
        self.assertIn("means are exact", vllm)

    def test_dark_mode_is_defined_in_both_scopes(self):
        """A media query alone loses to an explicit theme toggle, and vice versa."""
        self.assertIn("@media (prefers-color-scheme: dark)", self.html)
        self.assertIn('[data-theme="dark"]', self.html)


class TestSyntax(unittest.TestCase):
    def test_javascript_parses(self):
        try:
            import esprima
        except ImportError:
            self.skipTest("esprima not installed")
        try:
            esprima.parseScript(script(read()), {"tolerant": False})
        except Exception as e:  # noqa: BLE001 -- esprima raises its own type
            self.fail(f"dashboard JavaScript does not parse: {e}")


class TestFormatters(unittest.TestCase):
    """Execute the pure helpers in a real JS engine."""

    @classmethod
    def setUpClass(cls):
        try:
            import quickjs
        except ImportError:
            raise unittest.SkipTest("quickjs not installed")
        js = script(read())
        src = "\n".join(extract_function(js, n) for n in
                        ("fmtCompact", "fmtMs", "fmtPct", "fmtBytes", "niceTicks"))
        cls.ctx = quickjs.Context()
        cls.ctx.eval(src)

    def js(self, expr):
        return self.ctx.eval(expr)

    def test_duration_labels(self):
        self.assertEqual(self.js("fmtMs(823)"), "823ms")
        self.assertEqual(self.js("fmtMs(4.48)"), "4.5ms")
        self.assertEqual(self.js("fmtMs(2545.91)"), "2.55s")
        self.assertEqual(self.js("fmtMs(201180.36)"), "3m21s")
        self.assertEqual(self.js("fmtMs(851954.71)"), "14m12s")

    def test_compact_counts(self):
        self.assertEqual(self.js("fmtCompact(790)"), "790")
        self.assertEqual(self.js("fmtCompact(12900)"), "12.9K")
        self.assertEqual(self.js("fmtCompact(4560319)"), "4.6M")
        self.assertEqual(self.js("fmtCompact(0.0189, 3)"), "0.019")

    def test_nulls_render_as_dash_not_nan(self):
        """Idle buckets arrive as null; NaN or 'undefined' on screen is a bug."""
        for expr in ("fmtMs(null)", "fmtCompact(null)", "fmtPct(null)",
                     "fmtBytes(null)", "fmtCompact(undefined)", "fmtMs(NaN)"):
            self.assertEqual(self.js(expr), "—", expr)

    def test_axis_ticks_are_round_numbers(self):
        import json
        for lo, hi, want in [(0, 100, [0, 50, 100]),
                             (0, 1, [0, 0.5, 1]),
                             (0, 790, [0, 200, 400, 600, 800])]:
            got = json.loads(self.js(f"JSON.stringify(niceTicks({lo},{hi},4))"))
            self.assertEqual(got, want)

    def test_ticks_cover_the_domain(self):
        import json
        for hi in (0.0189, 33.83, 201180.36):
            got = json.loads(self.js(f"JSON.stringify(niceTicks(0,{hi},4))"))
            self.assertGreaterEqual(got[-1], hi, f"ticks stop below the max for {hi}")
            self.assertEqual(got[0], 0)


if __name__ == "__main__":
    unittest.main()


class TestTemperature(unittest.TestCase):
    """Temperature is the one series that should not be plotted from zero."""

    @classmethod
    def setUpClass(cls):
        try:
            import quickjs
        except ImportError:
            raise unittest.SkipTest("quickjs not installed")
        js = script(read())
        src = "\n".join(extract_function(js, n) for n in
                        ("fmtCompact", "fmtTemp", "tempFloor", "hottest",
                         "fmtWatts", "totalPower"))
        cls.ctx = quickjs.Context()
        cls.ctx.eval(src)

    def js(self, expr):
        return self.ctx.eval(expr)

    def test_formats_with_a_degree_sign(self):
        self.assertEqual(self.js("fmtTemp(68)"), "68°C")
        self.assertEqual(self.js("fmtTemp(33.4)"), "33°C")

    def test_null_reads_as_a_dash_not_zero_degrees(self):
        for expr in ("fmtTemp(null)", "fmtTemp(undefined)", "fmtTemp(NaN)"):
            self.assertEqual(self.js(expr), "—", expr)

    def test_floor_sits_below_the_coldest_reading(self):
        gpus = '[{"temp_c":[33,68]},{"temp_c":[36,63]}]'
        self.assertEqual(self.js(f"tempFloor({gpus})"), 20)

    def test_floor_ignores_nulls_and_never_goes_negative(self):
        self.assertEqual(self.js('tempFloor([{"temp_c":[null,2]}])'), 0)
        self.assertEqual(self.js("tempFloor([])"), 0)

    def test_hottest_picks_the_latest_reading_per_card(self):
        gpus = ('[{"index":0,"temp_c":[80,34]},'
                '{"index":1,"temp_c":[35,68]},'
                '{"index":2,"temp_c":[36,null]}]')
        got = self.js(f"JSON.stringify(hottest({gpus}))")
        import json
        # GPU0's 80 is stale; its latest is 34, so GPU1 at 68 is hottest now.
        # GPU2 falls back to its last non-null reading, 36.
        self.assertEqual(json.loads(got), {"temp": 68, "index": 1})

    def test_hottest_of_nothing_is_null(self):
        self.assertEqual(self.js("JSON.stringify(hottest([]))"), "null")

    def test_power_formats_in_watts(self):
        self.assertEqual(self.js("fmtWatts(107)"), "107 W")
        self.assertEqual(self.js("fmtWatts(null)"), "—")

    def test_power_sums_across_cards(self):
        """Watts add; utilisation would average and temperature would peak."""
        gpus = '[{"power_w":[4,107]},{"power_w":[5,103]}]'
        self.assertEqual(self.js(f"totalPower({gpus})"), 210)

    def test_power_of_nothing_is_null_not_zero(self):
        self.assertIsNone(self.js("totalPower([])"))
        self.assertIsNone(self.js('totalPower([{"power_w":[null]}])'))
