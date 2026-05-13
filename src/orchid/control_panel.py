"""ControlPanel — standalone Dash/DAQ instrument control UI."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bench import Bench


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

    # ── Producer side (UI callbacks / notebook cells) ─────────────────

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

    # ── Consumer side (setter thread) ─────────────────────────────────

    def drain(self) -> dict[str, float] | None:
        """Block until items are available or stopped.

        Returns a snapshot dict of ``{name: value}`` pairs, or ``None``
        when ``stop()`` has been called and the queue is empty.
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
                # spurious wakeup — loop

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def depth(self) -> int:
        """Number of distinct controllers with a pending set."""
        with self._lock:
            return len(self._pending)


# ══════════════════════════════════════════════════════════════════════
#  ControlPanel
# ══════════════════════════════════════════════════════════════════════

class ControlPanel:
    """Standalone Dash/DAQ control panel for bench controllers.

    Opens a browser window (or tab) with one row per controller.
    Controller writes are queued through a dedicated setter thread so
    blocking instrument I/O never stalls the UI.  If a value is changed
    several times before the setter thread drains the queue, only the
    latest value is applied (last-value-wins semantics).

    Calling ``bench[name] = val`` inside the panel fires the bench's
    event log automatically, so parameter-change markers appear on any
    running :class:`DashPlotter` live plot without extra wiring.

    Parameters
    ----------
    bench : Bench
        The lab bench whose controllers this panel controls.
    port : int
        TCP port for the Dash server (default ``8051``).
    controllers : list of str, optional
        Names of controllers to show.  ``None`` (default) shows all
        controllers registered in *bench*.
    open_browser : bool
        Open a browser tab automatically when the panel starts.
    readback : bool
        If ``True`` (default), poll each controller's current value
        periodically and display it next to the input.  Disable if
        instrument reads are slow or unavailable.
    readback_interval : int
        Readback poll period in milliseconds (default ``2000``).
        Reads are parallelised across controllers via a thread pool.

    Examples
    --------
    >>> panel = ControlPanel(bench, port=8051)
    >>> panel.start()
    >>> # adjust controllers from browser, or programmatically:
    >>> panel.set("Vgt", 0.5)
    >>> panel.stop()
    """

    def __init__(
        self,
        bench: Bench,
        port: int = 8051,
        controllers: list[str] | None = None,
        open_browser: bool = True,
        readback: bool = True,
        readback_interval: int = 2000,
        steps: dict[str, float] | None = None,
        precision: int | dict[str, int] | None = None,
    ) -> None:
        self.bench = bench
        self.port = port
        self._ctrl_names: list[str] = controllers or list(bench.controllers.keys())
        self.open_browser = open_browser
        self.readback = readback
        self.readback_interval = readback_interval
        self._steps: dict[str, float] = steps or {}
        self._precision: int | dict[str, int] = precision if precision is not None else 4

        self._queue = _SetterQueue()
        self._setter_thread: threading.Thread | None = None
        self._server_thread: threading.Thread | None = None
        self._wsgi_server = None

        # (controller_name, value, unix_timestamp) — updated by setter thread
        self._last_set: tuple[str, float, float] | None = None

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
        """Queue a controller write.  Safe to call from any thread.

        Parameters
        ----------
        name : str
            Controller name (must be in this panel's controller list).
        value : float
            Value to set.
        """
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
                if items is None:          # stop() was called
                    break
                for name, val in items.items():
                    try:
                        self.bench[name] = val   # fires event log + limit clamping
                        self._last_set = (name, val, time.time())
                    except Exception as exc:
                        print(f"ControlPanel: error setting {name} = {val}: {exc}")

        self._setter_thread = threading.Thread(target=_loop, daemon=True)
        self._setter_thread.start()

    # ── Dash server ───────────────────────────────────────────────────

    def _start_dash(self) -> None:
        import logging

        from dash import Dash, ctx, dcc, html, no_update
        from dash.dependencies import Input, Output
        import dash_daq as daq

        for logger_name in ("werkzeug", "dash", "dash.dash", "flask", "flask.app"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        app = Dash(
            __name__,
            update_title=None,
            suppress_callback_exceptions=True,
            external_stylesheets=[
                "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
            ],
        )
        app.title = "Orchid Control Panel"
        app.logger.setLevel(logging.ERROR)

        # Remove native spinner arrows from number inputs (all browsers)
        app.index_string = app.index_string.replace(
            "</head>",
            "<style>"
            "input[type=number]::-webkit-inner-spin-button,"
            "input[type=number]::-webkit-outer-spin-button{"
            "-webkit-appearance:none;margin:0}"
            "input[type=number]{-moz-appearance:textfield}"
            "</style></head>",
        )

        # ── Design tokens ──────────────────────────────────────────────
        FONT_UI    = "'Inter', system-ui, -apple-system, sans-serif"
        FONT_MONO  = "'JetBrains Mono', 'Fira Mono', monospace"
        C_BG       = "#f5f6fa"
        C_CARD     = "#ffffff"
        C_BORDER   = "#e8eaed"
        C_HEADER   = "#1e1e2e"
        C_HEADER_FG = "#e0e0f0"
        C_LABEL    = "#374151"
        C_READBACK = "#6b7280"
        C_ACCENT   = "#4f8ef7"
        C_PENDING  = "#f59e0b"
        C_IDLE     = "#9ca3af"
        C_LED      = "#4ade80" # "#FF5E5E"   # LED display digit colour
        C_LED_BG   = "#0f172a"  # "#fdfdfd" # LED display background

        panel = self
        bench = self.bench
        ctrl_names = self._ctrl_names

        # ── Read initial values (best-effort) ─────────────────────────
        def _safe_get(name: str) -> float:
            try:
                return float(bench.controllers[name].get())
            except Exception:
                return 0.0

        initial: dict[str, float] = {n: _safe_get(n) for n in ctrl_names}

        # ── Layout helpers ─────────────────────────────────────────────

        def _ctrl_row(name: str) -> html.Div:
            ctrl = bench.controllers[name]
            val = initial[name]
            unit = ctrl.unit or ""
            has_limits = ctrl.limits is not None

            name_pill = html.Span(
                name,
                style={
                    "fontFamily": FONT_UI,
                    "fontWeight": "500",
                    "fontSize": "16px",
                    "color": C_LABEL,
                    "background": "#f1f5f9",
                    "border": "1px solid #e2e8f0",
                    "borderRadius": "6px",
                    "padding": "2px 8px",
                    "whiteSpace": "nowrap",
                },
            )
            unit_pill = html.Span(
                unit,
                style={
                    "fontFamily": FONT_UI,
                    "fontWeight": "400",
                    "fontSize": "11px",
                    "color": "#94a3b8",
                    "background": "#f8fafc",
                    "border": "1px solid #e2e8f0",
                    "borderRadius": "6px",
                    "padding": "2px 6px",
                    "whiteSpace": "nowrap",
                },
            ) if unit else None

            label_div = html.Div(
                [name_pill, unit_pill] if unit_pill else [name_pill],
                style={
                    "flex": "1 1 50px",
                    "minWidth": "40px",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "6px",
                    "overflow": "hidden",
                },
            )

            children: list = [label_div]

            if has_limits:
                lo, hi = ctrl.limits
                step = panel._steps.get(name, (hi - lo) / 100)
                slider_val = max(lo, min(hi, val))   # clamp: bench value may exceed limits
                slider_kwargs: dict = dict(
                    id=f"slider-{name}",
                    min=lo,
                    max=hi,
                    value=slider_val,
                    step=step,
                    marks={
                        lo: f"{lo:g}",
                        **{lo + i * (hi - lo) / 4: "" for i in range(1, 4)},
                        hi: f"{hi:g}",
                    },
                    updatemode="mouseup",
                    tooltip={"always_visible": False, "placement": "bottom"},
                )
                children.append(
                    html.Div(
                        dcc.Slider(**slider_kwargs),
                        style={
                            "width": "300px",    # fixed — does not scale with browser
                            "flexShrink": "0",
                            "margin": "0 18px",
                            "alignSelf": "center",
                        },
                    )
                )

            prec = (
                panel._precision.get(name, 4)
                if isinstance(panel._precision, dict)
                else panel._precision
            )
            precision_kwargs: dict = {
                "id": f"input-{name}",
                "value": val,
                "precision": prec,
                "size": 150,
                "style": {"flexShrink": "0"},
            }
            if has_limits:
                precision_kwargs["min"] = ctrl.limits[0]
                precision_kwargs["max"] = ctrl.limits[1]

            children.append(daq.PrecisionInput(**precision_kwargs))

            if panel.readback:
                children.append(
                    html.Div(
                        daq.LEDDisplay(
                            id=f"readback-{name}",
                            value=f"{val:.5g}",
                            size=24,
                            color=C_LED,
                            backgroundColor=C_LED_BG,
                        ),
                        style={
                            "marginLeft": "14px",
                            "width": "150px",      # fixed — prevents row shift on digit count change
                            "flexShrink": "0",
                            "overflow": "hidden",
                        },
                    )
                )

            # Hidden sink for no-limits controllers (need at least one Output)
            children.append(
                html.Span(id=f"sink-{name}", style={"display": "none"})
            )

            return html.Div(
                children,
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "padding": "10px 20px",
                    "borderBottom": f"1px solid {C_BORDER}",
                    "transition": "background 0.15s",
                },
            )

        # ── Intervals ──────────────────────────────────────────────────
        intervals: list = [
            dcc.Interval(id="status-interval", interval=500, n_intervals=0),
        ]
        if panel.readback:
            intervals.append(
                dcc.Interval(
                    id="readback-interval",
                    interval=panel.readback_interval,
                    n_intervals=0,
                )
            )

        # ── Full layout ────────────────────────────────────────────────
        app.layout = html.Div(
            [
                # Page background
                html.Div(
                    [
                        # Card
                        html.Div(
                            [
                                # Header bar
                                html.Div(
                                    [
                                        html.Span(
                                            "⬡",
                                            style={
                                                "color": C_ACCENT,
                                                "marginRight": "8px",
                                                "fontSize": "16px",
                                            },
                                        ),
                                        html.Span("Control Panel"),
                                    ],
                                    style={
                                        "fontFamily": FONT_UI,
                                        "fontWeight": "600",
                                        "fontSize": "14px",
                                        "letterSpacing": "0.04em",
                                        "textTransform": "uppercase",
                                        "color": C_HEADER_FG,
                                        "background": C_HEADER,
                                        "padding": "14px 20px",
                                        "borderRadius": "10px 10px 0 0",
                                        "display": "flex",
                                        "alignItems": "center",
                                    },
                                ),
                                # Controller rows
                                html.Div(
                                    [_ctrl_row(n) for n in ctrl_names],
                                    style={"background": C_CARD},
                                ),
                                # Status bar
                                html.Div(
                                    id="status-bar",
                                    style={
                                        "fontFamily": FONT_MONO,
                                        "fontSize": "11.5px",
                                        "color": C_IDLE,
                                        "padding": "8px 20px",
                                        "background": C_CARD,
                                        "borderTop": f"1px solid {C_BORDER}",
                                        "borderRadius": "0 0 10px 10px",
                                        "minHeight": "30px",
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "12px",
                                    },
                                ),
                            ],
                            style={
                                "background": C_CARD,
                                "borderRadius": "10px",
                                "boxShadow": "0 2px 16px rgba(0,0,0,0.08)",
                                "overflow": "hidden",
                                "border": f"1px solid {C_BORDER}",
                            },
                        ),
                    ],
                    style={
                        "maxWidth": "760px",
                        "margin": "40px auto",
                        "padding": "0 16px",
                        "fontFamily": FONT_UI,
                    },
                ),
                *intervals,
            ],
            style={"background": C_BG, "minHeight": "100vh"},
        )

        # ── Callbacks ──────────────────────────────────────────────────

        for name in ctrl_names:
            ctrl = bench.controllers[name]
            has_limits = ctrl.limits is not None

            if has_limits:
                # Slider OR NumericInput → sync both, queue set
                @app.callback(
                    Output(f"input-{name}", "value"),
                    Output(f"slider-{name}", "value"),
                    Input(f"slider-{name}", "value"),
                    Input(f"input-{name}", "value"),
                    prevent_initial_call=True,
                )
                def _sync(slider_val, input_val, _name=name):
                    triggered = ctx.triggered_id
                    val = slider_val if triggered == f"slider-{_name}" else input_val
                    if val is not None:
                        panel.set(_name, float(val))
                    return val, val

            else:
                # NumericInput only → queue set, sink output discarded
                @app.callback(
                    Output(f"sink-{name}", "children"),
                    Input(f"input-{name}", "value"),
                    prevent_initial_call=True,
                )
                def _set(input_val, _name=name):
                    if input_val is not None:
                        panel.set(_name, float(input_val))
                    return no_update

        # Readback — parallel instrument reads via thread pool
        if panel.readback:
            @app.callback(
                [Output(f"readback-{n}", "value") for n in ctrl_names],
                Input("readback-interval", "n_intervals"),
            )
            def _update_readbacks(n):
                def _read(name: str) -> str:
                    try:
                        val = bench.controllers[name].get()
                        return f"{val:.5g}"
                    except Exception:
                        return "Err"

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return list(pool.map(_read, ctrl_names))

        # Status bar — pill badges for last-set and pending
        @app.callback(
            Output("status-bar", "children"),
            Input("status-interval", "n_intervals"),
        )
        def _update_status(n):
            children = []
            last = panel._last_set
            if last is not None:
                lname, lval, ltime = last
                elapsed = time.time() - ltime
                children.append(
                    html.Span(
                        f"{lname} = {lval:.5g}  ·  {elapsed:.1f} s ago",
                        style={
                            "background": "#eff6ff",
                            "color": C_ACCENT,
                            "border": f"1px solid #bfdbfe",
                            "borderRadius": "999px",
                            "padding": "2px 10px",
                            "fontSize": "11.5px",
                            "fontFamily": FONT_MONO,
                        },
                    )
                )
            depth = panel._queue.depth
            if depth:
                children.append(
                    html.Span(
                        f"{depth} pending",
                        style={
                            "background": "#fffbeb",
                            "color": C_PENDING,
                            "border": "1px solid #fde68a",
                            "borderRadius": "999px",
                            "padding": "2px 10px",
                            "fontSize": "11.5px",
                            "fontFamily": FONT_MONO,
                        },
                    )
                )
            if not children:
                children = [html.Span("idle", style={"color": C_IDLE})]
            return children

        # ── Start WSGI server ──────────────────────────────────────────
        try:
            import flask.cli
            flask.cli.show_server_banner = lambda *a, **kw: None
        except (ImportError, AttributeError):
            pass

        from werkzeug.serving import make_server

        srv = make_server("127.0.0.1", self.port, app.server)
        self._wsgi_server = srv

        self._server_thread = threading.Thread(
            target=srv.serve_forever, daemon=True
        )
        self._server_thread.start()

        time.sleep(0.5)

        if self.open_browser:
            import webbrowser
            webbrowser.open(f"http://localhost:{self.port}")

        print(f"Control panel started at http://localhost:{self.port}")
