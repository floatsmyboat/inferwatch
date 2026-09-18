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


def spark_specs(js):
    """Every object literal containing `spark: true`, brace-matched.

    A regex cannot do this: the spec nests `series: [{...}]`.
    """
    out = []
    for m in re.finditer(r"spark:\s*true", js):
        depth, start = 0, None
        for i in range(m.start(), -1, -1):      # nearest unclosed '{' to the left
            if js[i] == "}":
                depth += 1
            elif js[i] == "{":
                if depth == 0:
                    start = i
                    break
                depth -= 1
        assert start is not None, "no enclosing brace for a spark spec"
        depth = 0
        for j in range(start, len(js)):
            if js[j] == "{":
                depth += 1
            elif js[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(js[start:j + 1])
                    break
    return out


def extract_function_body(js, name):
    """Return the source of a CLASS METHOD `name(...) {...}`.

    extract_function() only matches top-level `function name(`, which the Chart
    class's methods are not.
    """
    m = re.search(r"^\s{2,}" + name + r"\s*\(", js, re.M)
    if not m:
        raise AssertionError(f"method {name} not found")
    start = js.index("{", m.end() - 1)
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
        """vLLM publishes no per-request rows, so a request table or a latency
        ranking there would be a lie. The client panel is the one exception and
        it earns it by naming its source -- see the test below."""
        vllm = self.html[self.html.index('id="tab-vllm"'):
                         self.html.index('id="tab-settings"')]
        for forbidden in ("Recent requests", "Slowest by TTFT"):
            self.assertNotIn(forbidden, vllm)

    def test_vllm_tab_has_gpu_and_vram_panels(self):
        vllm = self.html[self.html.index('id="tab-vllm"'):
                         self.html.index('id="tab-settings"')]
        self.assertIn('id="c-vgpu"', vllm)
        self.assertIn('id="c-vvram"', vllm)
        self.assertIn("GPU utilisation", vllm)
        self.assertIn("VRAM in use", vllm)

    def test_every_engine_tab_shows_gpu_temperature_and_power(self):
        """Each engine pane carries its own pair, scoped to the cards that
        engine holds. The count is derived from the cards present rather than
        hardcoded, so adding an engine does not mean editing a literal here --
        which is what this test did when NInfer arrived."""
        for card in ("c-gtemp", "c-vtemp", "c-ntemp",
                     "c-gpower", "c-vpower", "c-npower"):
            self.assertIn(f'id="{card}"', self.html)
        temp = len(re.findall(r'class="card" id="c-\w*temp"', self.html))
        power = len(re.findall(r'class="card" id="c-\w*power"', self.html))
        self.assertGreaterEqual(temp, 3)
        # One card heading each; the KPI tiles reuse the same label.
        self.assertEqual(self.html.count("<h2>GPU temperature</h2>"), temp)
        self.assertEqual(self.html.count("<h2>GPU power draw</h2>"), power)

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


class TestSparkHover(unittest.TestCase):
    """The hero sparkline is a chart like any other; it must answer the mouse."""

    def setUp(self):
        self.html = read()
        self.js = script(self.html)

    def test_bind_does_not_opt_sparks_out(self):
        """A `return` for sparks at the top of _bind() silently removes the
        pointer listener, which is what made the hero chart inert."""
        bind = extract_function_body(self.js, "_bind")
        self.assertNotIn("spec.spark) return", bind.replace(" ", ""))

    def test_pointer_and_keyboard_handlers_are_bound(self):
        bind = extract_function_body(self.js, "_bind")
        for ev in ("pointermove", "pointerleave", "keydown"):
            self.assertIn(ev, bind)

    def test_spark_tooltip_opens_downward(self):
        """A spark sits at the top of the page with ~46px of height, so an
        upward tooltip would be clipped by the viewport."""
        self.assertIn(".tip.below", self.html)
        self.assertIn('classList.toggle("below"', self.js)

    def test_every_spark_spec_carries_a_unit_formatter(self):
        """Without fmt the tooltip shows a bare number, and 'output 12.3' does
        not say tokens per second."""
        specs = spark_specs(self.js)
        self.assertTrue(specs, "no spark specs found -- did the hero change?")
        for spec in specs:
            self.assertIn("fmt:", spec, f"spark spec without fmt: {spec[:90]}")


class TestHoverIndexMath(unittest.TestCase):
    """_indexAt maps a pointer x to a bucket. Exercised directly because there
    is no DOM here, and this is the part of hover that can be silently wrong."""

    @classmethod
    def setUpClass(cls):
        try:
            import quickjs
        except ImportError:
            raise unittest.SkipTest("quickjs not installed")
        body = extract_function_body(script(read()), "_indexAt")
        # Rebind the method as a plain function over an injected `self`.
        src = "function indexAt(self, px) { const fn = function " + body.strip() + \
              "; return fn.call(self, px); }"
        cls.ctx = quickjs.Context()
        cls.ctx.eval(src)

    def at(self, px, n=10, x0=3, x1=403):
        t = ",".join(str(i) for i in range(n))
        self_js = f'{{spec:{{t:[{t}]}}, geom:{{x0:{x0},x1:{x1}}}}}'
        return self.ctx.eval(f"indexAt({self_js}, {px})")

    def test_left_edge_is_the_first_bucket(self):
        self.assertEqual(self.at(3), 0)

    def test_right_edge_is_the_last_bucket(self):
        self.assertEqual(self.at(403, n=10), 9)

    def test_midpoint_lands_mid_series(self):
        # 10 buckets across 400px: the centre is bucket 4 or 5, not off the end.
        self.assertIn(self.at(203, n=10), (4, 5))

    def test_just_outside_the_plot_still_snaps(self):
        """A small overshoot is slack for the pointer, not a miss -- otherwise
        the tooltip flickers off at the very edge of the trace."""
        self.assertEqual(self.at(-5), 0)
        self.assertEqual(self.at(410, n=10), 9)

    def test_far_outside_clears_the_hover(self):
        self.assertIsNone(self.at(-40))
        self.assertIsNone(self.at(500))

    def test_an_empty_series_never_returns_an_index(self):
        self.assertIsNone(self.ctx.eval(
            "indexAt({spec:{t:[]}, geom:{x0:3,x1:403}}, 200)"))


class TestClientTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import quickjs
        except ImportError:
            raise unittest.SkipTest("quickjs not installed")
        js = script(read())
        src = "\n".join(extract_function(js, n) for n in ("fmtCompact", "modelMix"))
        cls.ctx = quickjs.Context()
        cls.ctx.eval(src)

    def js(self, expr):
        return self.ctx.eval(expr)

    def test_lists_models_busiest_first(self):
        c = '{"models":[{"model":"llama3.2:3b","requests":42},' \
            '{"model":"qwen3:8b","requests":3}],"unattributed":0}'
        self.assertEqual(self.js(f"modelMix({c})"), "llama3.2:3b \u00d742, qwen3:8b \u00d73")

    def test_long_lists_are_summarised_not_truncated_silently(self):
        c = ('{"models":[{"model":"a","requests":9},{"model":"b","requests":8},'
             '{"model":"c","requests":7},{"model":"d","requests":6}],"unattributed":0}')
        self.assertEqual(self.js(f"modelMix({c})"), "a \u00d79, b \u00d78, +2 more")

    def test_unattributed_requests_are_stated(self):
        """Omitting them would make a client look like it used only the models
        that happened to be logged."""
        c = '{"models":[{"model":"a","requests":2}],"unattributed":121}'
        self.assertEqual(self.js(f"modelMix({c})"), "a \u00d72, 121 unattributed")

    def test_a_client_with_nothing_named_reads_as_a_dash(self):
        self.assertEqual(self.js('modelMix({"models":[],"unattributed":0})'), "\u2014")

    def test_only_unattributed_still_says_so(self):
        self.assertEqual(self.js('modelMix({"models":[],"unattributed":5})'),
                         "5 unattributed")


class TestClientMarkup(unittest.TestCase):
    def test_the_client_table_exists_and_is_rendered(self):
        """by_client() was in the API payload but rendered nowhere, so the data
        was computed and thrown away."""
        html = read()
        self.assertIn('id="t-clients"', html)
        self.assertIn('table("t-clients"', script(html))

    def test_a_vllm_client_table_must_name_the_proxy_as_its_source(self):
        """vLLM itself publishes no client addresses, so this panel is only
        honest while it says where the rows actually come from. Without that
        label it reads as per-request detail from /metrics, which does not
        exist -- which is why the panel was forbidden outright before a real
        source for it existed."""
        html = read()
        vllm = html[html.index('id="tab-vllm"'):html.index('id="tab-settings"')]
        if 'id="t-vclients"' not in vllm:
            return                      # no panel is still a valid answer
        note = vllm[vllm.index('id="vclient-note"'):]
        note = note[:note.index("</div>")]
        self.assertIn("proxy in front of vLLM", note)
        self.assertIn("/metrics", note)

    def test_the_client_table_offers_no_status_or_latency_column(self):
        """The proxy logs those on separate lines with no request id, and this
        workload runs requests concurrently, so a per-client latency or status
        would be a guess presented as a measurement."""
        js = script(read())
        block = js[js.index('table("t-vclients"'):]
        block = block[:block.index('text($("n-vclients")')]
        for forbidden in ("Latency", "TTFT", "Status", "Errors", "Duration"):
            self.assertNotIn(forbidden, block)


class TestGpuView(unittest.TestCase):
    """Both engine panes render GPU ownership through one helper."""

    @classmethod
    def setUpClass(cls):
        try:
            import quickjs
        except ImportError:
            raise unittest.SkipTest("quickjs not installed")
        js = script(read())
        src = "\n".join(extract_function(js, n) for n in
                         ("fmtCompact", "gpuView", "fmtAgo"))
        # sc()/css() reach for the DOM; stub them with stable markers.
        src = ("function sc(i) { return 'series-' + i; }\n"
               "function css(v) { return v; }\n") + src
        cls.ctx = quickjs.Context()
        cls.ctx.eval(src)

    def view(self, indices, n=4, idle=None):
        gpus = ", ".join('{"index":%d,"name":"NVIDIA RTX","mem_total":16311}' % i
                         for i in range(n))
        owned = "null" if indices is None else str(list(indices))
        extra = "" if idle is None else f", {idle!r}"
        return f"gpuView([{gpus}], {owned}{extra})"

    def js(self, expr):
        return self.ctx.eval(expr)

    def test_only_owned_cards_are_aggregated(self):
        """The Ollama pane summed VRAM and watts across every card on the host,
        so another engine's memory was reported as ollama's."""
        self.assertEqual(self.js(self.view([2, 3]) + ".cards.length"), 2)
        self.assertEqual(
            self.js("JSON.stringify(" + self.view([2, 3]) + ".cards.map(c=>c.index))"),
            "[2,3]")

    def test_unknown_attribution_keeps_every_card(self):
        """Null means could-not-tell. De-emphasising an arbitrary subset would
        be a guess, so nothing is claimed and every card stays coloured."""
        v = self.view(None)
        self.assertEqual(self.js(v + ".cards.length"), 4)
        self.assertFalse(self.js(v + ".known"))
        self.assertEqual(self.js(v + ".scope()"), "all devices (unattributed)")

    def test_an_engine_holding_nothing_is_distinct_from_unknown(self):
        v = self.view([])
        self.assertTrue(self.js(v + ".known"))
        self.assertEqual(self.js(v + ".cards.length"), 0)
        self.assertEqual(self.js(v + ".scope()"), "holds no GPU")

    def test_foreign_cards_are_labelled_and_de_emphasised(self):
        v = self.view([2, 3])
        self.assertIn("(other engine)", self.js(v + ".label({index:0,name:'NVIDIA RTX'})"))
        self.assertEqual(self.js(v + ".colour({index:0})"), "--deemph")
        self.assertNotIn("(other engine)", self.js(v + ".label({index:2,name:'NVIDIA RTX'})"))

    def test_an_idle_engine_does_not_claim_the_cards_were_taken(self):
        """Holding nothing is not evidence that anyone else holds them. The
        pane used to label all four "(other engine)" when ollama was simply
        between runners, which reads as "these cards were taken from you"."""
        v = self.view([], idle="no runner resident")
        self.assertFalse(self.js(v + ".held"))
        for i in range(4):
            self.assertNotIn("(other engine)",
                             self.js(v + ".label({index:%d,name:'NVIDIA RTX'})" % i))

    def test_the_idle_reason_is_named_in_scope_and_note(self):
        v = self.view([], idle="no runner resident")
        self.assertEqual(self.js(v + ".scope()"), "no runner resident")
        self.assertEqual(self.js(v + '.note("cgroup:ollama.service")'),
                         "no runner resident \u2014 holds none of the 4 devices "
                         "\u00b7 via cgroup:ollama.service")

    def test_a_pane_that_gives_no_reason_keeps_the_plain_wording(self):
        self.assertEqual(self.js(self.view([]) + ".scope()"), "holds no GPU")

    def test_a_card_keeps_one_colour_however_often_it_is_asked(self):
        """Colours used to come from a counter incremented during rendering and
        reset by hand before each draw, so one missed reset would recolour a
        card between charts."""
        v = self.view([1, 3])
        first = self.js(v + ".colour({index:3})")
        expr = ("(function(){const v=" + self.view([1, 3]) + ";"
                "for(let i=0;i<5;i++){v.colour({index:1});v.colour({index:3});}"
                "return v.colour({index:3});})()")
        self.assertEqual(self.js(expr), first)

    def test_owned_cards_take_the_series_palette_in_index_order(self):
        v = self.view([1, 3])
        self.assertEqual(self.js(v + ".colour({index:1})"), "series-0")
        self.assertEqual(self.js(v + ".colour({index:3})"), "series-1")

    def test_the_note_states_the_denominator_and_the_method(self):
        self.assertEqual(self.js(self.view([2, 3]) + '.note("cgroup:vllm-qwen38.service")'),
                         "holds GPU 2, 3 of 4 \u00b7 via cgroup:vllm-qwen38.service")
        self.assertEqual(self.js(self.view([]) + '.note("cgroup:x")'),
                         "holds none of the 4 devices \u00b7 via cgroup:x")
        self.assertEqual(self.js(self.view(None) + '.note("unavailable")'),
                         "4 devices; could not attribute")

    def test_no_samples_says_so_rather_than_claiming_zero_cards(self):
        self.assertEqual(self.js("gpuView([], [1]).note('cgroup:x')"), "no samples yet")

    def test_age_is_rendered_compactly(self):
        for secs, want in ((5, "5s"), (600, "10m"), (7200, "2h"), (200000, "2d")):
            self.assertEqual(self.js(f"fmtAgo({secs})"), want)
        self.assertEqual(self.js("fmtAgo(null)"), "\u2014")


class TestGpuMarkupSymmetry(unittest.TestCase):
    def test_both_panes_go_through_the_shared_view(self):
        js = script(read())
        self.assertGreaterEqual(js.count("gpuView("), 3,
                                "expected the helper plus one call per pane")

    def test_neither_pane_aggregates_the_raw_host_list(self):
        """Regression: hottest()/totalPower()/latestGpu() over d.gpu.gpus
        attributed every card on the box to whichever engine was on screen."""
        js = script(read())
        for banned in ("hottest(d.gpu && d.gpu.gpus)", "totalPower(d.gpu && d.gpu.gpus)",
                       "latestGpu()"):
            self.assertNotIn(banned, js, banned)

    def test_the_hand_reset_colour_counter_is_gone(self):
        self.assertNotIn("slot = 0;\n  draw(", script(read()))


class TestImagesTab(unittest.TestCase):
    def setUp(self):
        self.html = read()
        self.js = script(self.html)

    def test_the_tab_exists_and_is_reachable(self):
        self.assertIn('data-tab="images"', self.html)
        self.assertIn('id="tab-images"', self.html)
        self.assertIn('"images"', self.js)

    def test_the_tab_ships_hidden(self):
        """Only the first tab may be visible initially, or every panel renders
        at once on load."""
        section = self.html[self.html.index('id="tab-images"'):]
        self.assertTrue(section[:120].strip().startswith('id="tab-images" hidden'))

    def test_it_is_switchable_and_refreshable(self):
        self.assertIn('"ollama", "vllm", "images", "settings"', self.js)
        self.assertIn("refreshImages()", self.js)

    def test_no_token_panels_leaked_in(self):
        """Image generation has no tokens, TTFT or context; a tile borrowed
        from the token engines would read as permanently blank."""
        section = self.html[self.html.index('id="tab-images"'):
                            self.html.index('id="tab-vllm"')]
        for wrong in ("Time to first token", "tok/s", "Prompt cache", "TTFT"):
            self.assertNotIn(wrong, section, wrong)

    def test_it_says_where_the_numbers_come_from(self):
        section = self.html[self.html.index('id="tab-images"'):
                            self.html.index('id="tab-vllm"')]
        self.assertIn("ComfyUI's history", section)
        self.assertIn("never a second count", section)

    def test_failures_use_the_status_colour_not_a_series_hue(self):
        """Status colours are reserved: they mean a state, not an identity."""
        i = self.js.index('draw("ifail"')
        self.assertIn('css("--critical")', self.js[i:i + 400])

    def test_queue_and_vram_are_not_on_one_axis(self):
        """Two measures of different scale get two charts, never a second
        y-axis."""
        self.assertIn('draw("iqueue"', self.js)
        self.assertIn('draw("ivram"', self.js)
        i = self.js.index('draw("iqueue"')
        spec = self.js[i:self.js.index('draw(', i + 5)]     # this call only
        self.assertNotIn("vram", spec.lower())
        self.assertIn("queue_pending", spec)


class TestImagesHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import quickjs
        except ImportError:
            raise unittest.SkipTest("quickjs not installed")
        js = script(read())
        src = "\n".join(extract_function(js, n) for n in
                         ("fmtCompact", "shortModel", "firstLine", "extraModels"))
        cls.ctx = quickjs.Context()
        cls.ctx.eval(src)

    def js(self, expr):
        return self.ctx.eval(expr)

    def test_a_checkpoint_is_shown_without_its_folder_or_suffix(self):
        self.assertEqual(
            self.js("shortModel('OfficialStableDiffusion/sd3.5_large_fp8_scaled.safetensors')"),
            "sd3.5_large_fp8_scaled")
        self.assertEqual(self.js("shortModel('flux.gguf')"), "flux")

    def test_a_missing_model_is_a_dash(self):
        self.assertEqual(self.js("shortModel(null)"), "\u2014")
        self.assertEqual(self.js("shortModel('')"), "\u2014")

    def test_an_exception_shows_its_first_line(self):
        self.assertEqual(
            self.js(r"firstLine('RuntimeError: bad clip\n\nIf the clip is from a checkpoint')"),
            "RuntimeError: bad clip")

    def test_a_very_long_line_is_clipped_with_an_ellipsis(self):
        got = self.js("firstLine('%s')" % ("z" * 400))
        self.assertLessEqual(len(got), 160)
        self.assertGreater(len(got), 100, "clipped so hard it says nothing")
        self.assertTrue(got.endswith("\u2026"))

    def test_extra_models_exclude_the_primary(self):
        models = ('[{"name":"Flux/flux.safetensors"},{"name":"lora/detail.safetensors"},'
                  '{"name":"vae/ae.sft"}]')
        self.assertEqual(self.js(f"extraModels({models}, 'Flux/flux.safetensors')"),
                         "detail, ae")

    def test_extra_models_are_summarised_not_truncated_silently(self):
        models = "[" + ",".join('{"name":"m%d.safetensors"}' % i for i in range(6)) + "]"
        got = self.js(f"extraModels({models}, null)")
        self.assertTrue(got.endswith("+3"), got)

    def test_a_workflow_with_only_a_checkpoint_shows_a_dash(self):
        self.assertEqual(
            self.js("extraModels([{\"name\":\"a.safetensors\"}], 'a.safetensors')"),
            "\u2014")


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# Undefined-identifier analysis, used by TestUndefinedIdentifiers below.
# --------------------------------------------------------------------------

BROWSER = {
    "window","document","console","Math","JSON","Date","Object","Array","String",
    "Number","Boolean","Set","Map","WeakMap","Promise","fetch","URLSearchParams",
    "setTimeout","clearTimeout","setInterval","clearInterval","requestAnimationFrame",
    "history","location","navigator","performance","Error","TypeError","RangeError",
    "isNaN","parseFloat","parseInt","undefined","NaN","Infinity","EventSource",
    "WebSocket","localStorage","sessionStorage","CustomEvent","Event","Intl",
    "SVGElement","HTMLElement","Node","NodeList","DOMParser","ResizeObserver",
    "IntersectionObserver","AbortController","structuredClone","globalThis","matchMedia",
    "getComputedStyle","alert","confirm","prompt","encodeURIComponent","decodeURIComponent",
    "Symbol","Proxy","Reflect","BigInt","queueMicrotask","btoa","atob","crypto","self",
}

def walk(node, fn):
    if isinstance(node, list):
        for n in node: walk(n, fn)
        return
    if not hasattr(node, "type"): return
    fn(node)
    for key in dir(node):
        if key.startswith("_") or key == "type": continue
        try: val = getattr(node, key)
        except Exception: continue
        if isinstance(val, list) or hasattr(val, "type"): walk(val, fn)

def pattern_names(node, out):
    if node is None or not hasattr(node, "type"): return
    t = node.type
    if t == "Identifier": out.add(node.name)
    elif t == "ObjectPattern":
        for p in node.properties or []:
            pattern_names(getattr(p, "value", None) or getattr(p, "argument", None), out)
    elif t == "ArrayPattern":
        for e in node.elements or []: pattern_names(e, out)
    elif t == "AssignmentPattern": pattern_names(node.left, out)
    elif t == "RestElement": pattern_names(node.argument, out)

def declared_in(node):
    out = set()
    def visit(n):
        t = n.type
        if t == "VariableDeclarator": pattern_names(n.id, out)
        if t in ("FunctionDeclaration","ClassDeclaration") and getattr(n,"id",None):
            out.add(n.id.name)
        # Independent of the branch above: a named function declaration has
        # BOTH an id and params, and an elif here silently dropped every
        # parameter in the file.
        if t in ("FunctionDeclaration","FunctionExpression","ArrowFunctionExpression"):
            for p in n.params or []: pattern_names(p, out)
        if t == "CatchClause" and getattr(n,"param",None): pattern_names(n.param, out)
    walk(node, visit)
    return out

def referenced_in(node):
    out = set()
    skip = set()
    def visit(n):
        t = n.type
        if t == "MemberExpression" and not n.computed and getattr(n.property,"type","")=="Identifier":
            skip.add(id(n.property))
        elif t == "Property" and not getattr(n,"computed",False) and getattr(n.key,"type","")=="Identifier":
            skip.add(id(n.key))
        elif t in ("BreakStatement","ContinueStatement","LabeledStatement") and getattr(n,"label",None):
            skip.add(id(n.label))
        elif t == "MethodDefinition" and not getattr(n,"computed",False):
            skip.add(id(n.key))
    walk(node, visit)
    def collect(n):
        if n.type == "Identifier" and id(n) not in skip: out.add(n.name)
    walk(node, collect)
    return out

def walk_shallow(node, fn, _root=True):
    """Like walk, but does not descend into function bodies.

    The top-level scope must be collected this way: descending meant a `const`
    inside one render function counted as a global for every other, which is
    exactly the bug this check exists to catch and is why the first version of
    it reported a clean file.
    """
    if isinstance(node, list):
        for n in node: walk_shallow(n, fn, False)
        return
    if not hasattr(node, "type"): return
    if not _root and node.type in ("FunctionDeclaration", "FunctionExpression",
                                   "ArrowFunctionExpression"):
        fn(node)                      # its own name is declared out here
        return
    fn(node)
    for key in dir(node):
        if key.startswith("_") or key == "type": continue
        try: val = getattr(node, key)
        except Exception: continue
        if isinstance(val, list) or hasattr(val, "type"):
            walk_shallow(val, fn, False)


def declared_top(tree):
    out = set()
    def visit(n):
        t = n.type
        if t == "VariableDeclarator": pattern_names(n.id, out)
        if t in ("FunctionDeclaration", "ClassDeclaration") and getattr(n, "id", None):
            out.add(n.id.name)
    walk_shallow(tree, visit)
    return out


class TestUndefinedIdentifiers(unittest.TestCase):
    """A third silent failure, alongside the two in this file's docstring.

    A reference to a variable that does not exist is valid JavaScript: it
    parses, so the syntax test passes, and it names no element id, so that test
    passes too.  It throws only when the line runs -- and every render function
    here is called inside a try/catch that logs to the console, so the tab
    renders whatever came before the throw and nothing after it.

    That shipped.  renderNinfer used `fmtTick`, which every OTHER render
    function declares as its own local, and the NInfer tab showed its hero and
    the static notes in its markup with not one chart under them.

    Locals are over-approximated on purpose -- every declaration anywhere in a
    function counts as visible throughout it -- so this cannot flag a real
    binding.  The top-level scope is collected without descending into function
    bodies, which is the part that matters: descending is what made the first
    version of this check report a clean file, since a `const` inside one
    render function then counted as a global for every other.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import esprima
        except ImportError:
            raise unittest.SkipTest("esprima not installed")
        cls.esprima = esprima

    def flagged(self, js):
        tree = self.esprima.parseScript(js, {"tolerant": True})
        top = declared_top(tree)
        out = {}
        for node in tree.body:
            if node.type != "FunctionDeclaration" or not node.id:
                continue
            local = declared_in(node) | {node.id.name}
            unknown = sorted(referenced_in(node) - local - top - BROWSER)
            if unknown:
                out[node.id.name] = unknown
        return out

    def test_no_function_references_an_undefined_name(self):
        self.assertEqual(
            self.flagged(script(read())), {},
            "these functions reference names declared nowhere they can see; "
            "each throws at runtime and the panel below it silently stops")

    def test_the_check_catches_a_local_borrowed_from_another_function(self):
        """The shape of the real bug: a name that exists, but only as someone
        else's local.  Without this the check could pass by being vacuous."""
        js = ("function a() { const helper = 1; return helper; }\n"
              "function b() { return helper; }\n")
        self.assertEqual(self.flagged(js), {"b": ["helper"]})

    def test_parameters_and_nested_declarations_are_not_flagged(self):
        js = ("function a(x, {y}, [z], w = 2) {\n"
              "  const q = 1; let r = 2; var s = 3;\n"
              "  function inner(p) { return p + q; }\n"
              "  try { inner(x); } catch (e) { console.log(e); }\n"
              "  return [y, z, w, r, s, inner];\n"
              "}\n")
        self.assertEqual(self.flagged(js), {})

    def test_property_names_are_not_mistaken_for_variables(self):
        js = ("function a(o) { return o.fmtTick + o['x'] + {fmtTick: 1}.fmtTick; }\n")
        self.assertEqual(self.flagged(js), {})
