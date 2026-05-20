"""BrowseApp — standalone Dash experiment browser (DashPlotter.browse).

Layout
------
  header
  ┌─────────────────┬──────────────────────────────────────────┐
  │  selector rail  │  plot panel (lp-plots + lp-graph)        │
  │  (folder names) ├──────────────────────────────── lp-rail  │
  │  ─────────────  │  (reconstructed from plotter_config)     │
  │  info panel     │                                          │
  └─────────────────┴──────────────────────────────────────────┘

Clicking an entry loads figure.json.gz directly into the right-side
dcc.Graph and rebuilds the rail from plotter_config.yaml — identical
to how DashPlotter.load() would look, but embedded inline.

All Dash component IDs use the ``br-`` prefix, disjoint from ``lp-``.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

from ._registry import _browse_registry
# _lp_has_rail / _lp_rail_children are imported at call time to avoid
# any potential import-order issues (they're in _dash.py which imports us
# only inside a function body, so there is no cycle).


# ══════════════════════════════════════════════════════════════════════
#  Directory scanner
# ══════════════════════════════════════════════════════════════════════

def _scan_experiments(root_dir: Path) -> list[dict]:
    """Walk *root_dir* for plotter_config.yaml files.

    Returns a list of dicts sorted newest-first::

        {
            "folder_name": str,   # directory basename  ← shown in selector
            "name":        str,   # proc_name from meta ← shown in info panel
            "elapsed_str": str,
            "date_str":    str,
            "timestamp":   float,
            "data_dir":    str,   # absolute path (unique key)
            "has_figure":  bool,
        }

    Silently skips corrupted or partially-saved directories.
    """
    import yaml

    results = []
    for config_path in sorted(root_dir.rglob("plotter_config.yaml")):
        try:
            config   = yaml.safe_load(config_path.read_text())
            meta     = config.get("meta", {})
            ts       = meta.get("timestamp") or os.path.getmtime(config_path)
            date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d  %H:%M")
            results.append({
                "folder_name": config_path.parent.name,
                "name":        meta.get("proc_name") or config_path.parent.name,
                "elapsed_str": meta.get("elapsed") or "--:--",
                "date_str":    date_str,
                "timestamp":   float(ts),
                "data_dir":    str(config_path.parent),
                "has_figure":  (config_path.parent / "figure.json.gz").exists(),
            })
        except Exception:
            continue

    results.sort(key=lambda e: e["timestamp"], reverse=True)
    return results


def _port_registry_dirs() -> set[str]:
    """Return data_dirs of currently live DashPlotter instances."""
    from ._registry import _port_registry
    return {
        str(p._data_dir)
        for p in _port_registry.values()
        if getattr(p, "_data_dir", None)
    }


# ══════════════════════════════════════════════════════════════════════
#  Frozen plotter reconstruction
# ══════════════════════════════════════════════════════════════════════

def _reconstruct_frozen_plotter(data_dir: str | Path):
    """Load a saved experiment into a frozen DashPlotter without starting a server.

    Returns ``(plotter, fig_dict)`` where *plotter* is a fully initialised
    DashPlotter instance (``_stopped=True``, no server) and *fig_dict* is
    the raw Plotly figure dict from ``figure.json.gz``.
    """
    import gzip, json, yaml
    from ._spec import EventLineConfig, PlotSpec
    from ._dash import DashPlotter

    data_dir = Path(data_dir)
    config   = yaml.safe_load((data_dir / "plotter_config.yaml").read_text())

    fig_dict = None
    fig_path = data_dir / "figure.json.gz"
    if fig_path.exists():
        with gzip.open(fig_path, "rb") as fh:
            fig_dict = json.loads(fh.read())

    pa = dict(config["plotter"])
    pa["open_browser"] = False          # no browser, no server
    if "event_line" in pa and pa["event_line"] is not None:
        pa["event_line"] = EventLineConfig(**pa["event_line"])

    specs   = [PlotSpec.from_dict(s) for s in config["specs"]]
    plotter = DashPlotter(plots=specs, **pa)

    internal = config["internal"]
    plotter._resolved_types           = internal["resolved_types"]
    plotter._trace_offsets            = internal["trace_offsets"]
    plotter._strip_trace_idx          = internal.get("strip_trace_idx")
    plotter._analysis_trace_start_idx = internal.get("analysis_trace_start_idx")
    plotter._is_monitor               = internal.get("is_monitor", False)
    plotter._final_elapsed            = config["meta"].get("elapsed")
    plotter._fig_dict                 = fig_dict
    plotter._stopped                  = True

    proc_name    = config.get("meta", {}).get("proc_name", "Orchid")
    plotter._proc = type("_ProcStub", (), {"name": proc_name})()

    return plotter, fig_dict


# ══════════════════════════════════════════════════════════════════════
#  Layout builder helpers
# ══════════════════════════════════════════════════════════════════════

def _br_header(theme_name: str) -> object:
    """Top bar — reuses .lp-header CSS."""
    from ._themes import THEMES
    from dash import dcc, html

    options = []
    for key, td in THEMES.items():
        traces = td.get("traces", ["#888"])
        swatches = [
            html.Span(style={
                "backgroundColor": traces[i] if i < len(traces) else "#ccc",
                "display": "inline-block",
                "width": "14px", "height": "28px",
            })
            for i in range(3)
        ]
        label = html.Div([
            html.Span(swatches, style={
                "display": "inline-flex",
                "border": "1px solid rgba(128,128,128,0.2)",
            }),
            html.Div([
                html.Div(td["name"], className="lp-theme-name"),
                html.Div(td.get("sub", ""), className="lp-theme-sub"),
            ]),
        ], className="lp-theme-option")
        options.append({"label": label, "value": key})

    cur_td     = THEMES.get(theme_name, THEMES["orchid"])
    cur_traces = cur_td.get("traces", ["#888"])
    summary_swatches = [
        html.Span(style={
            "backgroundColor": cur_traces[i] if i < len(cur_traces) else "#ccc",
            "display": "inline-block", "width": "8px", "height": "12px",
        })
        for i in range(3)
    ]

    return html.Div(className="lp-header", children=[
        html.Div(className="lp-brand", children=[
            html.Div(className="lp-mark"),
            html.Div([
                html.Div("orchid", className="lp-brand-name"),
                html.Div("Browse", className="lp-brand-sub"),
            ]),
        ]),
        html.Div(className="lp-divider"),
        html.Div(className="lp-exp-block", children=[
            html.Span("Experiment Browser", className="lp-exp-name"),
        ]),
        html.Div(className="lp-spacer"),
        html.Details(className="lp-appearance", children=[
            html.Summary([
                html.Span("Appearance", className="lp-appearance-current"),
                html.Span(
                    summary_swatches,
                    className="lp-appearance-swatches",
                    style={"marginLeft": "8px"},
                ),
                html.Span(className="lp-chev"),
            ]),
            html.Div(className="lp-appearance-panel", children=[
                html.Div("Theme", className="lp-appearance-heading"),
                dcc.RadioItems(
                    id="br-theme-radio",
                    options=options,
                    value=theme_name,
                    labelClassName="lp-theme-option-label",
                    inputStyle={"display": "none"},
                ),
            ]),
        ]),
    ])


def _br_entry(exp: dict, *, is_selected: bool) -> object:
    """Single experiment card — shows folder name + date + elapsed."""
    from dash import html

    live_dirs = _port_registry_dirs()
    is_live   = exp["data_dir"] in live_dirs
    dot_cls   = "lp-dot lp-dot-pulse" if is_live else "lp-dot lp-dot-idle"
    entry_cls = "lp-br-entry" + (" lp-br-entry-sel" if is_selected else "")

    return html.Div(
        id={"type": "br-entry", "dir": exp["data_dir"]},
        className=entry_cls,
        n_clicks=0,
        children=[
            html.Div(className="lp-br-entry-top", children=[
                html.Span(exp["folder_name"], className="lp-br-entry-name"),
                html.Span(className=dot_cls, title="Live" if is_live else "Done"),
            ]),
            html.Div(className="lp-br-entry-meta", children=[
                html.Span(exp["date_str"],    className="lp-br-entry-date"),
                html.Span(exp["elapsed_str"], className="lp-br-entry-elapsed lp-mono"),
            ]),
        ],
    )


def _br_info_children(exp: dict | None, data_dir: str | None) -> list:
    """Children for the info panel below the selector."""
    from dash import html

    if exp is None or data_dir is None:
        return []   # empty when nothing selected

    p         = Path(data_dir)
    live_dirs = _port_registry_dirs()
    is_live   = data_dir in live_dirs
    dot_cls   = "lp-dot lp-dot-pulse" if is_live else "lp-dot lp-dot-idle"
    status    = "Live" if is_live else "Done"

    parts       = p.parts
    parent_abbr = os.path.join(*parts[-3:-1]) + "/" if len(parts) > 2 else str(p.parent) + "/"

    return [html.Div([
        html.Div([
            html.Span("Info", className="lp-group-title"),
        ], className="lp-group-head"),
        html.Div(exp["name"], className="lp-br-info-name"),
        html.Div([
            html.Span("Date",    className="lp-kv-k"),
            html.Span(exp["date_str"], className="lp-kv-v"),
        ], className="lp-kv"),
        html.Div([
            html.Span("Elapsed", className="lp-kv-k"),
            html.Span(exp["elapsed_str"], className="lp-kv-v lp-mono"),
        ], className="lp-kv"),
        html.Div([
            html.Span("Status",  className="lp-kv-k"),
            html.Div([
                html.Span(className=dot_cls),
                html.Span(status, style={"marginLeft": "6px"}),
            ], className="lp-kv-v", style={"display": "flex", "alignItems": "center"}),
        ], className="lp-kv"),
        html.Div([
            html.Span("Path",    className="lp-kv-k"),
            html.Div([
                html.Span(parent_abbr, className="lp-data-dir"),
                html.Span(p.name,      className="lp-data-id"),
            ], className="lp-br-info-path"),
        ], className="lp-kv lp-kv-path"),
    ])]


# ══════════════════════════════════════════════════════════════════════
#  BrowseApp
# ══════════════════════════════════════════════════════════════════════

class BrowseApp:
    """Standalone experiment browser served on a background Werkzeug thread.

    Do not instantiate directly — use ``DashPlotter.browse(root_dir)``.
    """

    def __init__(self, root_dir: Path, port: int, theme: str = "orchid"):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.port     = port
        self._current_theme   = theme
        self._current_plotter = None   # frozen plotter for the selected experiment
        self._wsgi_server     = None
        self._server_thread: threading.Thread | None = None

    # ── Public ───────────────────────────────────────────────────────

    def stop(self, _silent: bool = False) -> None:
        """Shut down the browse server."""
        if self._wsgi_server is not None:
            self._wsgi_server.shutdown()
            self._wsgi_server.server_close()
            self._wsgi_server = None
        self._server_thread = None
        _browse_registry.pop(self.port, None)
        if not _silent:
            print(f"Orchid browse server stopped (port {self.port}).")

    # ── Internal ─────────────────────────────────────────────────────

    def _start_server(self) -> None:
        """Build the Dash app, register callbacks, start Werkzeug."""
        import logging
        from dash import Dash, dcc, html, no_update, ctx
        from dash.dependencies import Input, Output, State, ALL
        from werkzeug.serving import make_server

        for logger_name in ("werkzeug", "dash", "dash.dash", "flask", "flask.app"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        try:
            import flask.cli
            flask.cli.show_server_banner = lambda *a, **kw: None
        except (ImportError, AttributeError):
            pass

        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        app = Dash(
            __name__,
            update_title=None,
            assets_folder=assets_dir,
            suppress_callback_exceptions=True,
        )
        app.title = f"Orchid Browse — {self.root_dir.name}"
        browse_app = self

        # ── Layout ───────────────────────────────────────────────────

        def serve_layout():
            theme_name          = browse_app._current_theme
            initial_experiments = _scan_experiments(browse_app.root_dir)
            return html.Div(
                id="lp-root",
                className=f"theme-{theme_name}",
                children=[
                    _br_header(theme_name),
                    html.Div(
                        className="lp-br-body",
                        children=[
                            # ── Left column: selector + info ──────────
                            html.Div(
                                className="lp-br-left-col",
                                children=[
                                    # Selector rail
                                    html.Div(
                                        className="lp-rail lp-br-left",
                                        children=[html.Div(
                                            className="lp-br-left-inner",
                                            children=[
                                                html.Div(
                                                    className="lp-group-head",
                                                    children=[
                                                        html.Span("Experiments",
                                                                  className="lp-group-title"),
                                                        html.Span(
                                                            str(len(initial_experiments)),
                                                            id="br-count",
                                                            className="lp-group-right",
                                                        ),
                                                    ],
                                                ),
                                                dcc.Input(
                                                    id="br-search",
                                                    placeholder="Filter…",
                                                    debounce=True,
                                                    className="lp-br-search",
                                                    value="",
                                                ),
                                                html.Div(
                                                    id="br-list",
                                                    className="lp-br-list",
                                                    children=[
                                                        _br_entry(e, is_selected=False)
                                                        for e in initial_experiments
                                                    ],
                                                ),
                                            ],
                                        )],
                                    ),
                                    # Info panel
                                    html.Div(
                                        id="br-info-panel",
                                        className="lp-rail lp-br-info-panel",
                                        children=[],
                                    ),
                                ],
                            ),
                            # ── Right column: plot + rail ─────────────
                            html.Div(
                                className="lp-br-right-col",
                                children=[
                                    html.Div(
                                        className="lp-plots",
                                        children=[html.Div(
                                            className="lp-panel",
                                            children=[html.Div(
                                                className="lp-graph",
                                                children=[dcc.Graph(
                                                    id="br-graph",
                                                    figure={},
                                                    config={
                                                        "responsive": True,
                                                        "displayModeBar": True,
                                                    },
                                                    style={"height": "100%"},
                                                )],
                                            )],
                                        )],
                                    ),
                                    html.Div(
                                        id="br-plot-rail",
                                        className="",
                                        children=[],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Interval(id="br-interval", interval=4000, n_intervals=0),
                    dcc.Store(id="br-selected-dir", data=None),
                    dcc.Store(id="br-scan-cache",   data=initial_experiments),
                ],
            )

        app.layout = serve_layout

        # ── Callback 1: Refresh list ──────────────────────────────────

        @app.callback(
            Output("br-scan-cache", "data"),
            Output("br-list",       "children"),
            Output("br-count",      "children"),
            Input("br-interval",    "n_intervals"),
            Input("br-search",      "value"),
            Input("br-selected-dir","data"),
            State("br-scan-cache",  "data"),
        )
        def refresh_list(n, search_text, selected_dir, prev_cache):
            experiments = _scan_experiments(browse_app.root_dir)
            q = (search_text or "").strip().lower()
            if q:
                experiments = [
                    e for e in experiments
                    if q in e["folder_name"].lower()
                    or q in e["name"].lower()
                    or q in e["data_dir"].lower()
                ]

            prev_dirs    = [e["data_dir"] for e in (prev_cache or [])]
            curr_dirs    = [e["data_dir"] for e in experiments]
            list_changed = (
                (curr_dirs != prev_dirs)
                or ctx.triggered_id in ("br-search", "br-selected-dir")
            )

            if list_changed:
                if experiments:
                    cards = [
                        _br_entry(e, is_selected=(e["data_dir"] == selected_dir))
                        for e in experiments
                    ]
                else:
                    msg = (
                        f'No matches for "{q}"' if q
                        else f"No saved experiments found in\n{browse_app.root_dir}"
                    )
                    cards = [html.Div(msg, className="lp-br-empty")]
            else:
                cards = no_update

            return experiments, cards, str(len(experiments))

        # ── Callback 2: Click entry → load figure + rail ──────────────

        @app.callback(
            Output("br-graph",        "figure"),
            Output("br-plot-rail",    "children"),
            Output("br-plot-rail",    "className"),
            Output("br-selected-dir", "data"),
            Output("br-info-panel",   "children"),
            Input({"type": "br-entry", "dir": ALL}, "n_clicks"),
            State("br-scan-cache",    "data"),
            prevent_initial_call=True,
        )
        def on_entry_click(all_clicks, experiments):
            from ._dash import _lp_has_rail, _lp_rail_children

            if not all_clicks or not any(all_clicks):
                return no_update, no_update, no_update, no_update, no_update

            trig = ctx.triggered_id
            if not isinstance(trig, dict) or trig.get("type") != "br-entry":
                return no_update, no_update, no_update, no_update, no_update

            data_dir = trig["dir"]
            exp      = next(
                (e for e in (experiments or []) if e["data_dir"] == data_dir),
                None,
            )

            try:
                plotter, fig_dict = _reconstruct_frozen_plotter(data_dir)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                err = [html.Div(f"Error loading experiment:\n{exc}", className="lp-br-empty")]
                return no_update, err, "lp-rail", data_dir, _br_info_children(exp, data_dir)

            # Apply the browse app's current theme to the reconstructed figure.
            plotter._retheme_fig_dict(browse_app._current_theme)
            browse_app._current_plotter = plotter

            has_rail     = _lp_has_rail(plotter)
            rail_cls     = "lp-rail" if has_rail else ""
            rail_content = _lp_rail_children(plotter) if has_rail else []

            return (
                plotter._fig_dict or no_update,
                rail_content,
                rail_cls,
                data_dir,
                _br_info_children(exp, data_dir),
            )

        # ── Callback 3: Theme switch ───────────────────────────────────

        @app.callback(
            Output("lp-root",    "className"),
            Output("br-graph",   "figure", allow_duplicate=True),
            Input("br-theme-radio", "value"),
            prevent_initial_call=True,
        )
        def update_theme(theme_name):
            browse_app._current_theme = theme_name
            # Retheme the currently displayed figure, if any.
            p = browse_app._current_plotter
            if p is not None:
                p._retheme_fig_dict(theme_name)
                new_fig = p._fig_dict
            else:
                new_fig = no_update
            return f"theme-{theme_name}", new_fig

        # ── Start Werkzeug ────────────────────────────────────────────

        prev = _browse_registry.get(self.port)
        if prev is not None and prev is not self:
            prev.stop(_silent=True)

        srv = make_server("127.0.0.1", self.port, app.server)
        self._wsgi_server = srv
        _browse_registry[self.port] = self

        self._server_thread = threading.Thread(
            target=srv.serve_forever, daemon=True
        )
        self._server_thread.start()
        time.sleep(0.5)

        import webbrowser
        webbrowser.open(f"http://localhost:{self.port}")
        n_found = len(_scan_experiments(self.root_dir))
        print(f"Orchid experiment browser at http://localhost:{self.port}")
        print(f"  Scanning: {self.root_dir}  ({n_found} experiment(s) found)")
