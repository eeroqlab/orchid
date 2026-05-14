"""ControlPanel — standalone Dash/DAQ instrument control UI."""

from __future__ import annotations

import concurrent.futures
import math
import pathlib
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bench import Bench
    from .controller import Controller

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"

# One color per instrument group, cycling if more than 8
_GROUP_COLORS = [
    "#5bb4ff", "#5bd7a9", "#ffd23a", "#ff5d3b",
    "#c87cff", "#ff7dbf", "#ff9f43", "#00d2d3",
]


def _step_options(lo: float, hi: float) -> list[float]:
    """Return up to 4 logarithmically-spaced step options for [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [1.0]
    mag = 10 ** math.floor(math.log10(span))
    raw = [mag / 100, mag / 10, mag, mag * 10]
    opts = [o for o in raw if 0 < o < span]
    return opts[-4:] or [span / 100]


def _fmt_step(v: float) -> str:
    """Format a step value compactly: 0.001 → '1m', 1000 → '1k'."""
    if v <= 0:
        return f"{v:g}"
    if v >= 1e6:
        return f"{v/1e6:g}M"
    if v >= 1e3:
        return f"{v/1e3:g}k"
    if v >= 1:
        return f"{v:g}"
    if v >= 1e-3:
        m = v * 1e3
        return f"{m:g}m"
    if v >= 1e-6:
        u = v * 1e6
        return f"{u:g}μ"
    return f"{v:.2g}"


def _nice_marks(lo: float, hi: float) -> dict:
    """Labeled tick marks at nice intervals, skeleton style (+X for positive values).

    Targets ~5 ticks; rounds the interval to 1 / 2 / 5 × a power of 10.
    Returns a Dash marks dict: ``{value: {"label": str}}``.
    """
    span = hi - lo
    if span <= 0:
        return {lo: {"label": f"{lo:g}"}}

    raw = span / 5
    mag = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        if raw <= factor * mag:
            tick_step = factor * mag
            break
    else:
        tick_step = mag

    first = math.ceil(lo / tick_step - 1e-9) * tick_step
    ticks: list[float] = []
    v = first
    while v <= hi + tick_step * 1e-9:
        tv = round(v, 10)
        if lo - tick_step * 0.01 <= tv <= hi + tick_step * 0.01:
            ticks.append(tv)
        v += tick_step

    def _fmt(val: float) -> str:
        s = f"{val:g}"
        return f"+{s}" if val > 0 else s

    return {v: {"label": _fmt(v)} for v in ticks}


def _is_readable(ctrl: Controller) -> bool:
    """Return True if the controller can read back its current value.

    A controller is *set-only* when it has no instrument binding **and** no
    explicit ``get_func`` — i.e. a virtual binding that only writes.
    """
    return not (ctrl.instrument is None and ctrl.get_func is None)


def _fmt_sp(v: float, unit: str = "", prec: int = 4) -> str:
    """Format a setpoint value: sign prefix + fixed decimals + optional unit."""
    sign = "−" if v < 0 else "+"
    return f"{sign}{abs(v):.{prec}f}" + (f" {unit}" if unit else "")


# ══════════════════════════════════════════════════════════════════════
#  Thread-safe last-value-wins queue
# ══════════════════════════════════════════════════════════════════════

class _SetterQueue:
    """Dict-based pending-set store: putting the same key twice keeps only the last value.

    The setter thread blocks in ``drain()`` until at least one value is
    available, then atomically pops and returns everything.  Calling
    ``stop()`` unblocks ``drain()`` and signals the thread to exit.
    """

    def __init__(self) -> None:
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stopped = False

    # ── Producer side ─────────────────────────────────────────────────

    def put(self, name: str, value: float) -> None:
        """Queue a set; overwrites any previous pending value for *name*."""
        with self._lock:
            self._pending[name] = value
        self._event.set()

    def stop(self) -> None:
        """Signal the consumer thread to exit after draining."""
        with self._lock:
            self._stopped = True
        self._event.set()

    # ── Consumer side ─────────────────────────────────────────────────

    def drain(self) -> dict[str, float] | None:
        """Block until items are available or stopped.

        Returns a snapshot dict or ``None`` when stop() was called.
        """
        while True:
            self._event.wait()
            self._event.clear()
            with self._lock:
                if self._pending:
                    items, self._pending = self._pending, {}
                    return items
                if self._stopped:
                    return None

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._pending)


# ══════════════════════════════════════════════════════════════════════
#  ControlPanel
# ══════════════════════════════════════════════════════════════════════

class ControlPanel:
    """Standalone Dash/DAQ control panel for bench controllers.

    Displays one vertical strip per controller, grouped by instrument
    with tab switching.  Controller writes are queued through a dedicated
    setter thread so blocking instrument I/O never stalls the UI.

    Parameters
    ----------
    bench : Bench
        The lab bench whose controllers this panel controls.
    port : int
        TCP port for the Dash server (default ``8051``).
    controllers : list of str, optional
        Names of controllers to show.  ``None`` shows all.
    open_browser : bool
        Open a browser tab automatically when the panel starts.
    readback : bool
        If ``True`` (default), poll each controller's current value
        periodically and display it on the LCD.
    readback_interval : int
        Readback poll period in milliseconds (default ``2000``).
    steps : dict[str, float], optional
        Override the default selected step size per controller.
        Keys are controller names; values are step sizes.

    Examples
    --------
    >>> panel = ControlPanel(bench, port=8051)
    >>> panel.start()
    >>> panel.set("Vgt", 0.5)
    >>> panel.stop()
    """

    def __init__(
        self,
        bench: Bench,
        port: int = 8051,
        controllers: list[str] | None = None,
        open_browser: bool = False,
        readback: bool = True,
        readback_interval: int = 2000,
        steps: dict[str, float] | None = None,
    ) -> None:
        self.bench = bench
        self.port = port
        self._ctrl_names: list[str] = controllers or list(bench.controllers.keys())
        self.open_browser = open_browser
        self.readback = readback
        self.readback_interval = readback_interval
        self._steps: dict[str, float] = steps or {}

        self._queue = _SetterQueue()
        self._setter_thread: threading.Thread | None = None
        self._server_thread: threading.Thread | None = None
        self._wsgi_server = None
        self._last_set: tuple[str, float, float] | None = None
        self._last_error: str | None = None   # set by setter thread on exception

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the setter thread and Dash server."""
        self._start_setter_thread()
        self._start_dash()

    def stop(self) -> None:
        """Stop the setter thread and Dash server."""
        self._queue.stop()
        if self._wsgi_server is not None:
            self._wsgi_server.shutdown()
            self._wsgi_server.server_close()
            self._wsgi_server = None
        self._server_thread = None
        print("Control panel stopped.")

    def set(self, name: str, value: float) -> None:
        """Queue a controller write.  Safe to call from any thread."""
        if name not in self._ctrl_names:
            raise KeyError(
                f"Controller {name!r} is not in this panel. "
                f"Available: {self._ctrl_names}"
            )
        self._queue.put(name, value)

    @property
    def is_running(self) -> bool:
        """``True`` while the Dash server thread is alive."""
        return self._server_thread is not None and self._server_thread.is_alive()

    # ── Setter thread ─────────────────────────────────────────────────

    def _start_setter_thread(self) -> None:
        def _loop() -> None:
            while True:
                items = self._queue.drain()
                if items is None:
                    break
                for name, val in items.items():
                    try:
                        self.bench[name] = val
                        self._last_set = (name, val, time.time())
                        self._last_error = None
                    except Exception as exc:
                        self._last_error = f"{name}: {exc}"
                        print(f"ControlPanel: error setting {name} = {val}: {exc}")

        self._setter_thread = threading.Thread(target=_loop, daemon=True)
        self._setter_thread.start()

    # ── Dash server ───────────────────────────────────────────────────

    def _start_dash(self) -> None:
        import logging

        from dash import Dash, dcc, html, no_update
        from dash import Input, Output, State, ALL, ctx

        for lg in ("werkzeug", "dash", "dash.dash", "flask", "flask.app"):
            logging.getLogger(lg).setLevel(logging.ERROR)

        app = Dash(
            __name__,
            assets_folder=str(_ASSETS_DIR),
            update_title=None,
            suppress_callback_exceptions=True,
        )
        app.title = "Orchid Control Panel"
        app.logger.setLevel(logging.ERROR)


        panel = self
        bench = self.bench
        ctrl_names = self._ctrl_names

        # ── Group controllers by instrument ───────────────────────────
        groups: dict[str, list[str]] = {}      # {instr_name: [ctrl_name, ...]}
        for name in ctrl_names:
            ctrl = bench.controllers[name]
            instr_name = ctrl.instrument.name if ctrl.instrument else "Custom"
            groups.setdefault(instr_name, []).append(name)

        group_color: dict[str, str] = {
            instr: _GROUP_COLORS[i % len(_GROUP_COLORS)]
            for i, instr in enumerate(groups)
        }

        tabs = ["ALL"] + list(groups.keys())

        # ── Read initial values (best-effort) ─────────────────────────
        def _safe_get(name: str) -> float:
            try:
                return float(bench.controllers[name].get())
            except Exception:
                return 0.0

        initial: dict[str, float] = {n: _safe_get(n) for n in ctrl_names}

        # ── Layout helpers ─────────────────────────────────────────────

        def _strip(name: str, color: str) -> html.Div:
            ctrl = bench.controllers[name]
            val = initial[name]
            unit = ctrl.unit or ""
            has_limits = ctrl.limits is not None
            readable = _is_readable(ctrl)
            # show LCD if readback is on (readable) OR controller is set-only
            show_lcd = panel.readback or not readable
            lcd_role = "lcd" if readable else "lcd-sp"

            if has_limits:
                lo, hi = ctrl.limits
                opts = _step_options(lo, hi)
                default_step = panel._steps.get(name, opts[1] if len(opts) > 1 else opts[0])
                slider_val = max(lo, min(hi, val))
                marks = _nice_marks(lo, hi)
            else:
                opts = []
                default_step = panel._steps.get(name, 1.0)

            children: list = [
                # ── Header ───────────────────────────────────────────
                html.Div(className="strip-head", children=[
                    html.Div(className="strip-label-row", children=[
                        html.Span(
                            className="strip-color-dot",
                            style={"background": color},
                        ),
                        html.Span(name, className="strip-label"),
                    ]),
                    html.Span(unit, className="strip-unit"),
                ]),

                # ── LCD: readback (readable) or setpoint mirror (set-only) ──
                *([ html.Div(className="lcd-frame", children=[
                    html.Span(
                        id={"role": lcd_role, "ch": name},
                        className="lcd-value",
                        children=f"{val:.4g}",
                    ),
                    html.Span(unit, className="lcd-unit"),
                ])] if show_lcd else []),

                # ── Setpoint row (readable) / no-readback badge (set-only) ──
                html.Div(className="sp-row", children=[
                    *([
                        html.Span("SP", className="sp-tag"),
                        html.Span(
                            id={"role": "sp-text", "ch": name},
                            className="sp-text",
                            children=_fmt_sp(val, unit),
                        ),
                    ] if readable else [
                        html.Span("NO RDBACK", className="no-rdback-badge"),
                        # hidden sink keeps the callback output valid
                        html.Span(
                            id={"role": "sp-text", "ch": name},
                            style={"display": "none"},
                        ),
                    ]),
                ]),
            ]

            # ── Vertical slider ───────────────────────────────────────
            if has_limits:
                children.append(
                    dcc.Slider(
                        id={"role": "slider", "ch": name},
                        min=lo, max=hi,
                        value=slider_val,
                        step=default_step,
                        marks=marks,
                        vertical=True,
                        verticalHeight=320,
                        updatemode="mouseup",
                        className="vslider",
                        tooltip={"placement": "right", "always_visible": False},
                    )
                )
                children.append(
                    html.Div(id={"role": "limit", "ch": name}, className="limit-warn")
                )

            # ── Numeric input ─────────────────────────────────────────
            input_kwargs: dict = dict(
                id={"role": "input", "ch": name},
                type="number",
                value=val,
                step=default_step,
                className="sci-input",
                debounce=True,
            )
            if has_limits:
                input_kwargs["min"] = lo
                input_kwargs["max"] = hi
            children.append(dcc.Input(**input_kwargs))

            # ── Step chips + nudge ────────────────────────────────────
            if opts:
                children.append(
                    html.Div(className="step-chips", children=[
                        html.Button(
                            _fmt_step(s),
                            id={"role": "step-chip", "ch": name, "step": s},
                            className=(
                                "step-chip step-chip-on"
                                if abs(s - default_step) < 1e-12
                                else "step-chip"
                            ),
                            n_clicks=0,
                        )
                        for s in opts
                    ]),
                )
                children.append(
                    html.Div(className="nudge-row", children=[
                        html.Button(
                            "−",
                            id={"role": "nudge", "ch": name, "dir": -1},
                            className="nudge-btn",
                            n_clicks=0,
                        ),
                        html.Button(
                            "+",
                            id={"role": "nudge", "ch": name, "dir": 1},
                            className="nudge-btn",
                            n_clicks=0,
                        ),
                    ]),
                )

            # Stores and hidden sinks
            children.append(dcc.Store(id={"role": "step-store", "ch": name}, data=default_step))
            children.append(html.Span(id={"role": "sink", "ch": name}, style={"display": "none"}))

            return html.Div(className="strip", children=children)

        ACCENTS = [
            ("Blue",    "#4f8ef7"),
            ("Red",     "#ff3838"),
            ("Amber",   "#ffb000"),
            ("Green",   "#39ff14"),
            ("Cyan",    "#00e7ff"),
            ("Magenta", "#c87cff"),
        ]

        def _appearance_menu() -> html.Div:
            return html.Div(
                id="appearance",
                className="appearance",
                children=[
                    html.Button(
                        id="appearance-toggle",
                        className="appearance-btn",
                        n_clicks=0,
                        children=[
                            html.Span(className="appearance-icon"),
                            html.Span("APPEARANCE", className="appearance-label"),
                            html.Span(id="appearance-accent-dot", className="appearance-accent-dot"),
                            html.Span("▾", className="appearance-chev"),
                        ],
                    ),
                    html.Div(
                        id="appearance-panel",
                        className="appearance-panel appearance-hidden",
                        children=[
                            html.Div("Theme", className="appearance-section"),
                            html.Div(className="theme-row", children=[
                                html.Button("◐  Dark",  id={"role": "theme", "v": "dark"},
                                            className="theme-btn", n_clicks=0),
                                html.Button("◑  Light", id={"role": "theme", "v": "light"},
                                            className="theme-btn", n_clicks=0),
                            ]),
                            html.Div("LCD accent", className="appearance-section"),
                            html.Div(className="accent-row", children=[
                                html.Button(
                                    "8.8",
                                    id={"role": "accent", "v": hexc},
                                    className="accent-swatch",
                                    style={"color": hexc, "textShadow": f"0 0 6px {hexc}99"},
                                    n_clicks=0,
                                )
                                for _, hexc in ACCENTS
                            ]),
                            html.Div(id="appearance-accent-name", className="appearance-accent-name",
                                     children=ACCENTS[0][0]),
                        ],
                    ),
                ],
            )

        def _strip_group(instr_name: str) -> html.Div:
            color = group_color[instr_name]
            return html.Div(
                id={"role": "strip-group", "instr": instr_name},
                className="strip-group",
                children=[_strip(n, color) for n in groups[instr_name]],
            )

        # ── Full layout ────────────────────────────────────────────────
        tab_buttons = [
            html.Button(
                t,
                id={"role": "tab", "tab": t},
                className="tab" + (" tab-on" if t == "ALL" else ""),
                n_clicks=0,
            )
            for t in tabs
        ]

        has_readable = any(_is_readable(bench.controllers[n]) for n in ctrl_names)

        intervals: list = [
            dcc.Interval(id="status-interval", interval=500, n_intervals=0),
        ]
        if panel.readback and has_readable:
            intervals.append(
                dcc.Interval(
                    id="readback-interval",
                    interval=panel.readback_interval,
                    n_intervals=0,
                )
            )

        app.layout = html.Div(
            id="app-root",
            className="app",
            children=[
                html.Div(className="chassis-head", children=[
                    html.Div(className="chassis-screw"),
                    html.Div(className="brand", children=[
                        html.Div("ORCHID · CONTROL", className="brand-name"),
                        html.Div(
                            f"{len(ctrl_names)}-CH · {len(groups)} instrument"
                            + ("s" if len(groups) != 1 else ""),
                            className="brand-sub",
                        ),
                    ]),
                    html.Div(className="tabs", children=tab_buttons),
                    html.Div(className="spacer"),
                    html.Div(className="indicators", children=[
                        html.Div(className="indicator", children=[
                            html.Span(className="led led-green"),
                            html.Span("PWR", className="led-label"),
                        ]),
                        html.Div(className="indicator", children=[
                            html.Span(id="led-run", className="led led-green pulse"),
                            html.Span("RUN", className="led-label"),
                        ]),
                        html.Div(className="indicator", children=[
                            html.Span(id="led-fault", className="led led-off"),
                            html.Span("FAULT", className="led-label"),
                        ]),
                    ]),
                    _appearance_menu(),
                    html.Div(className="chassis-screw"),
                ]),

                html.Div(
                    className="strip-rack",
                    children=[_strip_group(instr) for instr in groups],
                ),


                dcc.Store(id="active-tab", data="ALL"),
                dcc.Store(
                    id="appearance-state",
                    storage_type="local",
                    data={"theme": "dark", "accent": "#4f8ef7", "open": False},
                ),
                *intervals,
            ],
        )

        # ── Callbacks ──────────────────────────────────────────────────

        # Tab click → update active-tab store (clientside, instant)
        app.clientside_callback(
            """
            function(clicks, ids) {
                const t = window.dash_clientside.callback_context.triggered;
                if (!t || !t.length) return window.dash_clientside.no_update;
                const id = JSON.parse(t[0].prop_id.split('.')[0]);
                return id.tab;
            }
            """,
            Output("active-tab", "data"),
            Input({"role": "tab", "tab": ALL}, "n_clicks"),
            prevent_initial_call=True,
        )

        # RUN + FAULT LEDs — wired to setter queue depth and last error
        @app.callback(
            Output("led-run",   "className"),
            Output("led-fault", "className"),
            Input("status-interval", "n_intervals"),
        )
        def _update_leds(_n):
            pending = panel._queue.depth
            run_cls   = "led led-amber pulse" if pending else "led led-green pulse"
            fault_cls = "led led-red pulse"   if panel._last_error else "led led-off"
            return run_cls, fault_cls

        # Active tab → update tab button classes (clientside)
        app.clientside_callback(
            """
            function(active, tab_ids) {
                return tab_ids.map(id =>
                    id.tab === active ? 'tab tab-on' : 'tab'
                );
            }
            """,
            Output({"role": "tab", "tab": ALL}, "className"),
            Input("active-tab", "data"),
            State({"role": "tab", "tab": ALL}, "id"),
        )

        # Appearance: toggle open/closed
        app.clientside_callback(
            """
            function(n, state) {
                state = state || {theme:'dark', accent:'#4f8ef7', open:false};
                return Object.assign({}, state, {open: !state.open});
            }
            """,
            Output("appearance-state", "data", allow_duplicate=True),
            Input("appearance-toggle", "n_clicks"),
            State("appearance-state", "data"),
            prevent_initial_call=True,
        )

        # Appearance: theme buttons
        app.clientside_callback(
            """
            function(clicks, state) {
                const t = window.dash_clientside.callback_context.triggered;
                if (!t || !t.length || !t[0].value) return window.dash_clientside.no_update;
                const id = JSON.parse(t[0].prop_id.split('.')[0]);
                return Object.assign({}, state, {theme: id.v});
            }
            """,
            Output("appearance-state", "data", allow_duplicate=True),
            Input({"role": "theme", "v": ALL}, "n_clicks"),
            State("appearance-state", "data"),
            prevent_initial_call=True,
        )

        # Appearance: accent swatches
        app.clientside_callback(
            """
            function(clicks, state) {
                const t = window.dash_clientside.callback_context.triggered;
                if (!t || !t.length || !t[0].value) return window.dash_clientside.no_update;
                const id = JSON.parse(t[0].prop_id.split('.')[0]);
                return Object.assign({}, state, {accent: id.v});
            }
            """,
            Output("appearance-state", "data", allow_duplicate=True),
            Input({"role": "accent", "v": ALL}, "n_clicks"),
            State("appearance-state", "data"),
            prevent_initial_call=True,
        )

        # Appearance: apply state → DOM (theme attr + --accent var + panel visibility)
        accent_names = {hexc: label for label, hexc in ACCENTS}
        app.clientside_callback(
            f"""
            function(state) {{
                state = state || {{theme:'dark', accent:'#4f8ef7', open:false}};
                document.documentElement.setAttribute('data-theme', state.theme);
                document.documentElement.style.setProperty('--accent', state.accent);
                const names = {accent_names};
                return [
                    state.open ? 'appearance-panel' : 'appearance-panel appearance-hidden',
                    {{background: state.accent, boxShadow: '0 0 6px ' + state.accent + 'aa'}},
                    names[state.accent] || state.accent,
                ];
            }}
            """,
            Output("appearance-panel",       "className"),
            Output("appearance-accent-dot",  "style"),
            Output("appearance-accent-name", "children"),
            Input("appearance-state", "data"),
        )

        # Appearance: close panel on click outside
        app.clientside_callback(
            """
            function(_n) {
                if (window.__orchidAppearanceBound) return window.dash_clientside.no_update;
                window.__orchidAppearanceBound = true;
                document.addEventListener('pointerdown', function(e) {
                    const root = document.getElementById('appearance');
                    if (!root || root.contains(e.target)) return;
                    const panel = document.getElementById('appearance-panel');
                    if (panel && !panel.classList.contains('appearance-hidden')) {
                        document.getElementById('appearance-toggle').click();
                    }
                }, true);
                return window.dash_clientside.no_update;
            }
            """,
            Output("appearance-state", "data", allow_duplicate=True),
            Input("status-interval", "n_intervals"),
            prevent_initial_call="initial_duplicate",
        )

        # Active tab → show/hide strip groups (clientside)
        app.clientside_callback(
            """
            function(active, group_ids) {
                return group_ids.map(id =>
                    (active === 'ALL' || id.instr === active)
                        ? {display: 'flex', gap: '10px'}
                        : {display: 'none'}
                );
            }
            """,
            Output({"role": "strip-group", "instr": ALL}, "style"),
            Input("active-tab", "data"),
            State({"role": "strip-group", "instr": ALL}, "id"),
        )

        # Per-controller callbacks
        for name in ctrl_names:
            ctrl = bench.controllers[name]
            has_limits = ctrl.limits is not None
            readable = _is_readable(ctrl)
            show_lcd = panel.readback or not readable

            if has_limits:
                lo, hi = ctrl.limits
                soft_lo = lo + (hi - lo) * 0.05
                soft_hi = hi - (hi - lo) * 0.05
                unit = ctrl.unit or ""

                # Slider ↔ input sync + queue set + limit warning
                # For set-only controllers also update the lcd-sp element
                _sync_outputs = [
                    Output({"role": "input",   "ch": name}, "value"),
                    Output({"role": "slider",  "ch": name}, "value"),
                    Output({"role": "sp-text", "ch": name}, "children"),
                    Output({"role": "limit",   "ch": name}, "children"),
                    Output({"role": "limit",   "ch": name}, "className"),
                ]
                if show_lcd and not readable:
                    _sync_outputs.append(Output({"role": "lcd-sp", "ch": name}, "children"))

                @app.callback(
                    *_sync_outputs,
                    Input({"role": "slider",    "ch": name}, "value"),
                    Input({"role": "input",     "ch": name}, "value"),
                    prevent_initial_call=True,
                )
                def _sync(slider_val, input_val,
                          _name=name, _lo=lo, _hi=hi,
                          _soft_lo=soft_lo, _soft_hi=soft_hi, _unit=unit,
                          _readable=readable, _show_lcd=show_lcd):
                    triggered = ctx.triggered_id
                    val = (
                        slider_val
                        if triggered and triggered.get("role") == "slider"
                        else input_val
                    )
                    if val is not None:
                        panel.set(_name, float(val))
                    sp = _fmt_sp(float(val), _unit) if val is not None else "—"
                    # Limit warning
                    v = float(val) if val is not None else _lo
                    if v < _lo or v > _hi:
                        lim_txt, lim_cls = "⚠ OUT OF LIMIT", "limit-warn limit-over"
                    elif v < _soft_lo or v > _soft_hi:
                        lim_txt, lim_cls = "⚠ NEAR LIMIT",   "limit-warn limit-near"
                    else:
                        lim_txt, lim_cls = "",                 "limit-warn"
                    result = [val, val, sp, lim_txt, lim_cls]
                    if _show_lcd and not _readable:
                        result.append(f"{float(val):.4g}" if val is not None else "---")
                    return result

                # Step chip → update step store + highlight + slider step
                @app.callback(
                    Output({"role": "step-store", "ch": name}, "data"),
                    Output({"role": "step-chip",  "ch": name, "step": ALL}, "className"),
                    Output({"role": "slider",     "ch": name}, "step"),
                    Output({"role": "input",      "ch": name}, "step"),
                    Input({"role": "step-chip",   "ch": name, "step": ALL}, "n_clicks"),
                    State({"role": "step-chip",   "ch": name, "step": ALL}, "id"),
                    prevent_initial_call=True,
                )
                def _on_step_chip(_clicks, chip_ids, _name=name):
                    trig = ctx.triggered_id
                    if trig is None:
                        return no_update, [no_update] * len(chip_ids), no_update, no_update
                    step = float(trig["step"])
                    classes = [
                        "step-chip step-chip-on"
                        if abs(float(cid["step"]) - step) < 1e-12
                        else "step-chip"
                        for cid in chip_ids
                    ]
                    return step, classes, step, step

                # Nudge − / + → bump value by current step
                @app.callback(
                    Output({"role": "slider", "ch": name}, "value", allow_duplicate=True),
                    Output({"role": "input",  "ch": name}, "value", allow_duplicate=True),
                    Input({"role": "nudge",   "ch": name, "dir": -1}, "n_clicks"),
                    Input({"role": "nudge",   "ch": name, "dir":  1}, "n_clicks"),
                    State({"role": "slider",  "ch": name}, "value"),
                    State({"role": "step-store", "ch": name}, "data"),
                    prevent_initial_call=True,
                )
                def _on_nudge(_m, _p, current_val, step,
                              _name=name, _lo=lo, _hi=hi):
                    trig = ctx.triggered_id
                    if trig is None or current_val is None:
                        return no_update, no_update
                    direction = float(trig["dir"])
                    new_val = round(
                        max(_lo, min(_hi, current_val + direction * step)), 10
                    )
                    panel.set(_name, new_val)
                    return new_val, new_val

            else:
                # No limits: input only → queue set
                unit = ctrl.unit or ""

                _no_lim_outputs = [
                    Output({"role": "sink",    "ch": name}, "children"),
                    Output({"role": "sp-text", "ch": name}, "children"),
                ]
                if show_lcd and not readable:
                    _no_lim_outputs.append(Output({"role": "lcd-sp", "ch": name}, "children"))

                @app.callback(
                    *_no_lim_outputs,
                    Input({"role": "input", "ch": name}, "value"),
                    prevent_initial_call=True,
                )
                def _set_no_limits(input_val, _name=name, _unit=unit,
                                   _readable=readable, _show_lcd=show_lcd):
                    if input_val is not None:
                        panel.set(_name, float(input_val))
                    sp = _fmt_sp(float(input_val), _unit) if input_val is not None else "—"
                    result = [no_update, sp]
                    if _show_lcd and not _readable:
                        result.append(f"{float(input_val):.4g}" if input_val is not None else "---")
                    return result

        # Readback — parallel reads via thread pool (readable controllers only)
        if panel.readback and has_readable:
            @app.callback(
                Output({"role": "lcd", "ch": ALL}, "children"),
                Input("readback-interval", "n_intervals"),
                State({"role": "lcd", "ch": ALL}, "id"),
            )
            def _update_readbacks(_n, ids):
                names = [i["ch"] for i in ids]

                def _read(nm: str) -> str:
                    try:
                        return f"{float(bench.controllers[nm].get()):.4g}"
                    except Exception:
                        return "Err"

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return list(pool.map(_read, names))


        # ── Start WSGI server ──────────────────────────────────────────
        try:
            import flask.cli
            flask.cli.show_server_banner = lambda *a, **kw: None
        except (ImportError, AttributeError):
            pass

        from werkzeug.serving import make_server

        srv = make_server("127.0.0.1", self.port, app.server)
        self._wsgi_server = srv
        self._server_thread = threading.Thread(target=srv.serve_forever, daemon=True)
        self._server_thread.start()

        time.sleep(0.5)

        if self.open_browser:
            import webbrowser
            webbrowser.open(f"http://localhost:{self.port}")

        print(f"Control panel started at http://localhost:{self.port}")
