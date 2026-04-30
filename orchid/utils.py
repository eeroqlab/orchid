"""Utility functions for post-experiment operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ── Plot theme ────────────────────────────────────────────────────────

PALETTE: dict[str, list[str]] = {
    # Muted, desaturated — reads well in print and on screen
    "muted": [
        "#4878d0",  # slate blue
        "#ee854a",  # terracotta
        "#6acc64",  # sage green
        "#d65f5f",  # muted red
        "#956cb4",  # dusty purple
        "#8c613c",  # warm brown
        "#dc7ec0",  # mauve
        "#797979",  # mid grey
    ],
    # Vivid but balanced — good for presentations and screens
    "vivid": [
        "#003049",  # deep navy
        "#D62828",  # vivid red
        "#2FA084",  # teal
        "#F77F00",  # vivid orange
        "#FCBF49",  # vivid yellow
        "#76D2DB",  # sky blue
        "#263B6A",  # dark blue
        "#6984A9",  # steel blue
        "#A0D585",  # light green
    ],
    # Pastel / soft — nice for dense overlapping traces
    "pastel": [
        "#7eb0d5",  # powder blue
        "#fd7f6f",  # salmon
        "#b2e061",  # lime
        "#bd7ebe",  # lilac
        "#ffb55a",  # peach
        "#ffee65",  # butter
        "#beb9db",  # periwinkle
        "#fdcce5",  # blush
    ],
    # High contrast minimal — 4 colors, works in grayscale too
    "minimal": [
        "#2166ac",  # deep blue
        "#d6604d",  # brick red
        "#4dac26",  # forest green
        "#1a1a1a",  # near black
    ],
}


def apply_theme(
    palette: str | list[str] = "vivid",
    colorscale: str = "Inferno",
    name: str = "sw_clean",
    base: str = "simple_white",
) -> str:
    """Register an orchid plot theme and set it as the Plotly default.

    Call this once (it is applied automatically when orchid.plotting is
    imported). Call again with different arguments to switch themes.

    Parameters
    ----------
    palette : str or list of str
        Colour cycle. Pass a key from ``orchid.PALETTE`` or a custom list
        of CSS/hex colour strings.
    colorscale : str
        Plotly colorscale name for heatmaps (e.g. ``"Inferno"``, ``"RdBu_r"``).
    name : str
        Name under which the template is registered in ``pio.templates``.
    base : str
        Base Plotly template to stack under the orchid template
        (e.g. ``"simple_white"``, ``"plotly_dark"``).

    Returns
    -------
    str
        The full template string that was set as default
        (e.g. ``"simple_white+sw_clean"``).

    Examples
    --------
    >>> from orchid import apply_theme, PALETTE
    >>> apply_theme()                          # defaults
    >>> apply_theme(palette="muted")           # different palette
    >>> apply_theme(palette=PALETTE["pastel"], colorscale="RdBu_r")
    >>> apply_theme(base="plotly_dark", colorscale="Plasma")
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    colors = PALETTE[palette] if isinstance(palette, str) else palette

    _axis = dict(
        showline=True,
        linewidth=1.0,
        linecolor="#111111",
        mirror=True,        # border on all four sides
        showgrid=True,
        gridcolor="#eeeeee",
        gridwidth=0.5,
        zeroline=False,
    )

    pio.templates[name] = go.layout.Template(
        layout=go.Layout(
            xaxis=_axis,
            yaxis=_axis,
            colorway=colors,
        ),
        data=go.layout.template.Data(
            heatmap=[go.Heatmap(colorscale=colorscale)]
        ),
    )

    full = f"{base}+{name}"
    pio.templates.default = full
    return full

# 1. Define bold escape sequences
BOLD = "\033[1m"
RESET = "\033[0m"
SPACES = "\x20\x20\x20\x20"


def update_metadata(data_dir: str | Path, **kwargs) -> dict:
    """Add or update fields in an experiment's metadata.yaml.

    Use this after an experiment to annotate results with additional
    information (notes, analysis results, quality flags, etc.).

    Parameters
    ----------
    data_dir : str or Path
        Path to the experiment directory (containing metadata.yaml).
    **kwargs
        Key-value pairs to add or overwrite in the metadata.

    Returns
    -------
    dict
        The full updated metadata dictionary.

    Examples
    --------
    >>> update_metadata("./data/0001", notes="good data", quality="A")
    >>> update_metadata(data_dir, T_mc=0.015, B_field="1T")
    """
    path = Path(data_dir) / "metadata.yaml"
    if path.exists():
        meta = yaml.safe_load(path.read_text()) or {}
    else:
        meta = {}

    meta.update(kwargs)
    path.write_text(yaml.safe_dump(meta, sort_keys=False))
    return meta


def read_limit_log(data_dir: str | Path) -> list[dict]:
    """Read controller limit violations recorded during an experiment.

    Returns a list of dicts, each with keys:
    ``controller`` (name), ``index`` (sweep index list), ``requested``
    (value that was asked for), ``clamped`` (value that was applied).
    Returns an empty list if no violations occurred or ``limit_log.yaml``
    does not exist.

    Parameters
    ----------
    data_dir : str or Path
        Path to the experiment directory (containing limit_log.yaml).

    Examples
    --------
    >>> entries = read_limit_log("./data/0005")
    >>> for e in entries:
    ...     print(f"{e['controller']}[{e['index']}]: {e['requested']} → {e['clamped']}")
    """
    path = Path(data_dir) / "limit_log.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()) or []


def read_events(data_dir: str | Path) -> list[dict]:
    """Read parameter change events recorded during a monitor run.

    Returns a list of event dicts, each with keys:
    ``time`` (Unix timestamp), ``elapsed`` (seconds from start),
    ``param`` (parameter name), ``value`` (value that was set).
    Returns an empty list if no events were recorded.

    Parameters
    ----------
    data_dir : str or Path
        Path to the experiment directory (containing events.yaml).

    Examples
    --------
    >>> events = read_events("./data/0005")
    >>> for e in events:
    ...     print(f"t={e['elapsed']:.1f}s  {e['param']} → {e['value']}")
    """
    path = Path(data_dir) / "events.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()) or []


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.4g} s"
    elif seconds < 3600:
        return f"{seconds / 60:.4g} min"
    else:
        return f"{seconds / 3600:.4g} hr"


def _format_procedure_summary(d: dict) -> str:
    """Format a procedure dict (from Procedure.to_dict()) as a plain-text table.

    Uses a fixed 4-column layout — (identifier, value, meta, unit) — populated
    from the dict and rendered in a single tabulate call so all columns are
    globally aligned. Blank sentinel rows are post-processed into ─ separators.
    """
    from tabulate import tabulate as _tabulate

    kind = d.get("kind", "sweep")
    sweeps = d.get("sweeps", [])
    readouts = d.get("readouts", [])
    hooks = d.get("hooks", {})
    active_hooks = {k: v for k, v in hooks.items() if v is not None}

    BLANK = ("", "", "", "")
    rows: list[tuple] = []
    bold_indices: set[int] = set()  # row indices that should be bold in the output

    def _header(label: str) -> None:
        """Append a bold section-header row and record its index."""
        bold_indices.add(len(rows))
        rows.append((label, "", "", ""))

    # ── Header ────────────────────────────────────────────
    rows.append(("Experiment" if kind == "sweep" else "Monitor", d["name"], "", ""))
    tags = d.get("tags") or []
    if tags:
        rows.append(("Tags", ", ".join(tags), "", ""))

    # ── Sweeps ────────────────────────────────────────────
    if kind == "sweep":
        rows.append(BLANK)
        total = d.get("total_points", 0)
        ndim = d.get("ndim", 0)
        rows.append((f"Sweeps {ndim}D", f"{total:,} pts", "", ""))
        for sw in sweeps:
            ax = sw["axis"]
            n = sw["n"]
            rev_tag = " ↔" if sw.get("reverse") else ""
            if sw["type"] == "multi":
                for j, p in enumerate(sw["parameters"]):
                    rng = f"{p['min']:>10.4g} → {p['max']:<10.4g}"
                    unit_str = p.get("unit") or ""
                    if j == 0:
                        rows.append((f"  [{ax}] {p['name']}", rng, f"{n} pts{rev_tag}", unit_str))
                    else:
                        rows.append((f"      └── {p['name']}", rng, "", unit_str))
            else:
                rng = f"{sw['min']:>10.4g} → {sw['max']:<10.4g}"
                unit_str = sw.get("unit") or ""
                rows.append((f"  [{ax}] {sw['parameter']}", rng, f"{n} pts{rev_tag}", unit_str))

    # ── Readouts ──────────────────────────────────────────
    rows.append(BLANK)
    _header("Readouts")
    for r in readouts:
        unit_str = r.get("unit") or ""
        shape_str = f"shape={r['shape']}" if r.get("shape") else ""
        rows.append((f"  {r['name']}", r["kind"], shape_str, unit_str))

    # ── Settings ──────────────────────────────────────────
    rows.append(BLANK)
    _header("Settings")
    s = d.get("settings", {})
    if kind == "sweep":
        rows.append(("  write_mode", s.get("write_mode", ""), "", ""))
        settle = s.get("settle_time", 0) or 0
        if settle > 0:
            settle_str = f"{settle * 1000:.4g} ms" if settle < 1 else f"{settle:.4g} s"
            rows.append(("  settle_time", settle_str, "", ""))
        if s.get("snake"):
            rows.append(("  snake", "True", "", ""))
        rows.append(("  error_policy", s.get("error_policy", ""), "", ""))
        est = d.get("estimated_duration_s", 0) or 0
        if est > 0:
            rows.append(("  est. duration", f"~{_format_duration(est)}", "", ""))
    else:
        dur = s.get("duration")
        rows.append(("  interval",   f"{s.get('interval', 1.0)} s", "", ""))
        rows.append(("  duration",   _format_duration(dur) if dur else "unlimited", "", ""))
        rows.append(("  chunk_size", str(s.get("chunk_size", 256)), "", ""))

    # ── Hooks ─────────────────────────────────────────────
    if active_hooks:
        rows.append(BLANK)
        _header("Hooks")
        for hname, hinfo in active_hooks.items():
            fn_name = hinfo.get("name", "?")
            doc = hinfo.get("doc") or ""
            note = hinfo.get("note") or ""
            first_line = doc.split("\n")[0] if doc else ""
            extra = f'"{first_line}"' if first_line else f"({note})" if note else ""
            rows.append((f"  {hname}", fn_name, extra, ""))

    # ── Render ────────────────────────────────────────────
    # tabulate receives plain text only — no ANSI codes — so column widths
    # are measured correctly. Bold and separators are applied in post-processing.
    text = _tabulate(rows, tablefmt="plain")
    lines = text.splitlines()
    W = max((len(line) for line in lines if line.strip()), default=52)
    sep = "─" * W

    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            out.append(sep)
        elif i in bold_indices:
            out.append(f"{BOLD}{line}{RESET}")
        else:
            out.append(line)
    return "\n".join(out)


def read_procedure(data_dir: str | Path, print_summary: bool = True) -> dict:
    """Read procedure.yaml from an experiment data directory.

    Parameters
    ----------
    data_dir : str or Path
        Path to the experiment directory (containing procedure.yaml).
    print_summary : bool
        If True (default), print a formatted summary table.

    Returns
    -------
    dict
        The procedure dictionary.

    Examples
    --------
    >>> d = read_procedure("./data/0042")
    >>> d = read_procedure("./data/0042", print_summary=False)
    """
    path = Path(data_dir) / "procedure.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No procedure.yaml in {data_dir}")
    d = yaml.safe_load(path.read_text()) or {}
    if print_summary:
        print(_format_procedure_summary(d))
    return d


def read_metadata(data_dir: str | Path) -> dict:
    """Read an experiment's metadata.yaml.

    Parameters
    ----------
    data_dir : str or Path
        Path to the experiment directory.

    Returns
    -------
    dict
        The metadata dictionary.
    """
    path = Path(data_dir) / "metadata.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No metadata.yaml in {data_dir}")
    return yaml.safe_load(path.read_text()) or {}
