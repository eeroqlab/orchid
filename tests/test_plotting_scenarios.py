"""Showcase and regression tests for all PlotSpec + PlotterBase scenarios.

No real instruments or browser servers — only the figure-dict logic,
type resolution, and data-update paths are exercised.

Run with:
    /Users/helium/miniconda3/envs/fem/bin/python -m pytest tests/test_plotting_scenarios.py -v
"""

import sys
sys.path.insert(0, "src")

import time
import pytest
import numpy as np

from orchid.plotting import (
    PlotSpec,
    PlotterBase,
    _resolve_plot_type,
    _default_update_every,
    _normalize_y_col,
    _resolve_col,
    _find_sweep_by_ctrl,
    _sweep_ctrl_names,
)
from orchid.controller import DataKind


# ══════════════════════════════════════════════════════════════════════
#  Test doubles
# ══════════════════════════════════════════════════════════════════════

class _MockReadout:
    def __init__(self, name, kind, shape=None, contains=None):
        self.name = name
        self.kind = kind
        self.shape = shape
        self.contains = contains


class _MockController:
    def __init__(self, name):
        self.name = name


class _MockSweep:
    def __init__(self, name, values):
        self.controller = _MockController(name)
        self.values = np.asarray(values, dtype=np.float64)
        self.length = len(self.values)


class _MockBench:
    def __init__(self, readouts: dict):
        self.readouts = readouts


class _MockProc:
    """Generic mock procedure — sweeps=None means monitor (no sweeps attr)."""
    def __init__(self, name, readouts: dict, sweeps=None):
        self.name = name
        self.bench = _MockBench(readouts)
        if sweeps is not None:
            self.sweeps = sweeps


class NullPlotter(PlotterBase):
    """PlotterBase subclass that never starts a server."""

    def _start_server(self) -> None:
        pass

    def stop(self, _silent: bool = False) -> None:
        self._stopped = True

    @property
    def is_running(self) -> bool:
        return not self._stopped


def make_plotter(*specs: PlotSpec, **kw) -> NullPlotter:
    p = NullPlotter(list(specs), **kw)
    p._stopped = False
    return p


# ══════════════════════════════════════════════════════════════════════
#  Helper-function unit tests
# ══════════════════════════════════════════════════════════════════════

class TestResolveColHelper:
    def test_none_returns_none(self):
        assert _resolve_col(None, None) is None

    def test_int_passthrough(self):
        ro = _MockReadout("r", DataKind.TRACE)
        assert _resolve_col(1, ro) == 1

    def test_str_found_in_contains(self):
        ro = _MockReadout("r", DataKind.TRACE, contains=["X", "Y"])
        assert _resolve_col("Y", ro) == 1

    def test_str_not_found_raises(self):
        ro = _MockReadout("r", DataKind.TRACE, contains=["X", "Y"])
        with pytest.raises(ValueError, match="z_col='Z' not found"):
            _resolve_col("Z", ro, "z_col")

    def test_str_no_contains_raises(self):
        ro = _MockReadout("r", DataKind.TRACE, contains=None)
        with pytest.raises(ValueError, match="no contains list"):
            _resolve_col("X", ro)

    def test_bad_type_raises(self):
        ro = _MockReadout("r", DataKind.TRACE, contains=["X"])
        with pytest.raises(TypeError):
            _resolve_col(1.5, ro)


class TestNormalizeYCol:
    def _trace_ro(self, contains=None):
        return _MockReadout("r", DataKind.TRACE, contains=contains)

    def test_scalar_readout_always_none(self):
        ro = _MockReadout("r", DataKind.SCALAR)
        assert _normalize_y_col("X", ro) is None
        assert _normalize_y_col(0, ro) is None
        assert _normalize_y_col(["X"], ro) is None

    def test_none_readout_returns_none(self):
        assert _normalize_y_col(None, None) is None

    def test_none_y_col_returns_none(self):
        ro = self._trace_ro(["X", "Y"])
        assert _normalize_y_col(None, ro) is None

    def test_int_normalised_to_list(self):
        ro = self._trace_ro()
        assert _normalize_y_col(1, ro) == [1]

    def test_str_resolved_via_contains(self):
        ro = self._trace_ro(["X", "Y"])
        assert _normalize_y_col("Y", ro) == [1]

    def test_list_int_passthrough(self):
        ro = self._trace_ro()
        assert _normalize_y_col([0, 1], ro) == [0, 1]

    def test_list_str_resolved(self):
        ro = self._trace_ro(["X", "Y", "Z"])
        assert _normalize_y_col(["X", "Z"], ro) == [0, 2]

    def test_mixed_list_raises(self):
        ro = self._trace_ro(["X", "Y"])
        with pytest.raises(TypeError):
            _normalize_y_col([0, 1.5], ro)


class TestResolvePlotType:
    def _proc(self, n_sweeps=1):
        sweeps = [_MockSweep(f"s{i}", [0, 1]) for i in range(n_sweeps)]
        return _MockProc("p", {}, sweeps=sweeps)

    def test_explicit_type_passes_through(self):
        for t in ("line", "heatmap", "live_trace", "trace_heatmap"):
            spec = PlotSpec(x="x", y="y", plot_type=t)
            assert _resolve_plot_type(spec, self._proc()) == t

    def test_array_x_gives_live_trace(self):
        spec = PlotSpec(x=np.linspace(0, 1, 10), y="vna")
        assert _resolve_plot_type(spec, self._proc()) == "live_trace"

    def test_array_y_gives_trace_heatmap(self):
        spec = PlotSpec(x="Vgt", y=np.linspace(0, 1, 10), z="vna")
        assert _resolve_plot_type(spec, self._proc(2)) == "trace_heatmap"

    def test_z_with_2d_sweep_gives_heatmap(self):
        spec = PlotSpec(x="s0", y="s1", z="sig")
        assert _resolve_plot_type(spec, self._proc(2)) == "heatmap"

    def test_z_with_1d_sweep_gives_line(self):
        spec = PlotSpec(x="s0", y="sig", z="sig")
        assert _resolve_plot_type(spec, self._proc(1)) == "line"

    def test_scalar_gives_line(self):
        spec = PlotSpec(x="Vgt", y="sig")
        assert _resolve_plot_type(spec, self._proc(1)) == "line"

    def test_monitor_scalar_gives_line(self):
        proc = _MockProc("m", {})  # no sweeps attribute
        spec = PlotSpec(x="_time", y="sig")
        assert _resolve_plot_type(spec, proc) == "line"


class TestDefaultUpdateEvery:
    def _proc(self, n_sweeps):
        sweeps = [_MockSweep(f"s{i}", [0, 1]) for i in range(n_sweeps)]
        return _MockProc("p", {}, sweeps=sweeps)

    def test_line_always_point(self):
        assert _default_update_every("line", self._proc(1)) == "point"
        assert _default_update_every("line", self._proc(2)) == "point"

    def test_live_trace_always_point(self):
        assert _default_update_every("live_trace", self._proc(1)) == "point"

    def test_trace_heatmap_always_point(self):
        assert _default_update_every("trace_heatmap", self._proc(2)) == "point"

    def test_heatmap_2d_sweep_is_sweep(self):
        assert _default_update_every("heatmap", self._proc(2)) == "sweep"

    def test_heatmap_3d_sweep_is_plane(self):
        assert _default_update_every("heatmap", self._proc(3)) == "plane"


class TestFindSweepByCtrl:
    def _proc(self, names):
        sweeps = [_MockSweep(n, [0, 1]) for n in names]
        return _MockProc("p", {}, sweeps=sweeps)

    def test_found_by_name(self):
        proc = self._proc(["Vgt", "fac"])
        s = _find_sweep_by_ctrl(proc, "fac")
        assert s is not None
        assert s.controller.name == "fac"

    def test_not_found_returns_none(self):
        proc = self._proc(["Vgt"])
        assert _find_sweep_by_ctrl(proc, "missing") is None

    def test_no_sweeps_returns_none(self):
        proc = _MockProc("m", {})
        assert _find_sweep_by_ctrl(proc, "x") is None


# ══════════════════════════════════════════════════════════════════════
#  Scenario fixtures
# ══════════════════════════════════════════════════════════════════════

FREQS = np.linspace(4e9, 8e9, 51)
VGT_VALS = np.linspace(-1.0, 0.0, 21)
FAC_VALS = np.linspace(0.0, 1.0, 11)


def _proc1d(readouts):
    sweeps = [_MockSweep("Vgt", VGT_VALS)]
    return _MockProc("proc1d", readouts, sweeps=sweeps)


def _proc2d(readouts):
    sweeps = [_MockSweep("Vgt", VGT_VALS), _MockSweep("fac", FAC_VALS)]
    return _MockProc("proc2d", readouts, sweeps=sweeps)


def _monitor(readouts):
    return _MockProc("monitor", readouts)  # no sweeps


# ══════════════════════════════════════════════════════════════════════
#  Scenario 1 — Procedure1D / SCALAR / line
# ══════════════════════════════════════════════════════════════════════

class TestScenario1_1D_Scalar_Line:
    """PlotSpec(x='Vgt', y='sig') — basic scalar sweep plot."""

    def setup_method(self):
        ro = {"sig": _MockReadout("sig", DataKind.SCALAR)}
        self.proc = _proc1d(ro)
        self.spec = PlotSpec(x="Vgt", y="sig")
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_resolves_to_line(self):
        assert self.plotter._resolved_types[0] == "line"

    def test_update_every_is_point(self):
        assert self.plotter._resolved_update_every[0] == "point"

    def test_single_trace_in_figure(self):
        assert len(self.plotter._fig_dict["data"]) == 1

    def test_trace_name_is_readout_name(self):
        assert self.plotter._fig_dict["data"][0]["name"] == "sig"

    def test_state_has_correct_readout_and_no_col(self):
        st = self.plotter._sweep_data[0][0]
        assert st["_readout"] == "sig"
        assert st["_col"] is None

    def test_update_line_appends_points(self):
        for i, vgt in enumerate(VGT_VALS):
            self.plotter.update_point(
                (i,),
                {"sig": float(i) * 0.1},
                {"Vgt": vgt},
            )
        n = self.plotter._sweep_data[0][0]["_n"]
        assert n == len(VGT_VALS)

    def test_x_axis_label_is_sweep_param(self):
        layout = self.plotter._fig_dict["layout"]
        assert layout["xaxis"]["title"]["text"] == "Vgt"


# ══════════════════════════════════════════════════════════════════════
#  Scenario 2 — Procedure1D / TRACE / line single-col extraction
# ══════════════════════════════════════════════════════════════════════

class TestScenario2_1D_Trace_Line_SingleCol:
    """PlotSpec(x='Vgt', y='lockin', y_col='X') — extract scalar from TRACE."""

    def setup_method(self):
        ro = {"lockin": _MockReadout("lockin", DataKind.TRACE, shape=(2,), contains=["X", "Y"])}
        self.proc = _proc1d(ro)
        self.spec = PlotSpec(x="Vgt", y="lockin", y_col="X")
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_resolves_to_line(self):
        assert self.plotter._resolved_types[0] == "line"

    def test_single_trace_with_col_label(self):
        assert len(self.plotter._fig_dict["data"]) == 1
        assert self.plotter._fig_dict["data"][0]["name"] == "X"

    def test_state_col_is_zero(self):
        st = self.plotter._sweep_data[0][0]
        assert st["_col"] == 0

    def test_update_extracts_first_element(self):
        self.plotter.update_point(
            (0,),
            {"lockin": np.array([3.14, 2.72])},
            {"Vgt": -1.0},
        )
        y_data = self.plotter._fig_dict["data"][0]["y"]
        assert abs(float(y_data[0]) - 3.14) < 1e-9


# ══════════════════════════════════════════════════════════════════════
#  Scenario 3 — Procedure1D / TRACE / multi-line (same readout, multi-col)
# ══════════════════════════════════════════════════════════════════════

class TestScenario3_1D_Trace_MultiLine:
    """PlotSpec(x='Vgt', y='lockin', y_col=['X','Y']) — two traces, one readout."""

    def setup_method(self):
        ro = {"lockin": _MockReadout("lockin", DataKind.TRACE, shape=(2,), contains=["X", "Y"])}
        self.proc = _proc1d(ro)
        self.spec = PlotSpec(x="Vgt", y="lockin", y_col=["X", "Y"])
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_two_traces_in_figure(self):
        assert len(self.plotter._fig_dict["data"]) == 2

    def test_trace_names_match_y_col_strings(self):
        names = [t["name"] for t in self.plotter._fig_dict["data"]]
        assert names == ["X", "Y"]

    def test_both_states_point_to_same_readout(self):
        for st in self.plotter._sweep_data[0]:
            assert st["_readout"] == "lockin"

    def test_col_indices_are_zero_and_one(self):
        cols = [st["_col"] for st in self.plotter._sweep_data[0]]
        assert cols == [0, 1]

    def test_update_extracts_correct_elements(self):
        trace_data = np.array([1.11, 2.22])
        self.plotter.update_point(
            (0,),
            {"lockin": trace_data},
            {"Vgt": -1.0},
        )
        assert abs(float(self.plotter._fig_dict["data"][0]["y"][0]) - 1.11) < 1e-9
        assert abs(float(self.plotter._fig_dict["data"][1]["y"][0]) - 2.22) < 1e-9


# ══════════════════════════════════════════════════════════════════════
#  Scenario 4 — Procedure1D / legacy multi-readout line
# ══════════════════════════════════════════════════════════════════════

class TestScenario4_1D_MultiReadout_Line:
    """PlotSpec(x='Vgt', y=['sig_X','sig_Y']) — two scalar readouts, one subplot."""

    def setup_method(self):
        ro = {
            "sig_X": _MockReadout("sig_X", DataKind.SCALAR),
            "sig_Y": _MockReadout("sig_Y", DataKind.SCALAR),
        }
        self.proc = _proc1d(ro)
        self.spec = PlotSpec(x="Vgt", y=["sig_X", "sig_Y"])
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_two_traces_from_two_readouts(self):
        assert len(self.plotter._fig_dict["data"]) == 2

    def test_state_readouts_are_distinct(self):
        readouts = [st["_readout"] for st in self.plotter._sweep_data[0]]
        assert readouts == ["sig_X", "sig_Y"]

    def test_both_cols_are_none(self):
        for st in self.plotter._sweep_data[0]:
            assert st["_col"] is None

    def test_update_each_readout_independently(self):
        self.plotter.update_point(
            (0,), {"sig_X": 1.0, "sig_Y": 2.0}, {"Vgt": -1.0}
        )
        assert float(self.plotter._fig_dict["data"][0]["y"][0]) == pytest.approx(1.0)
        assert float(self.plotter._fig_dict["data"][1]["y"][0]) == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 5 — Procedure1D / TRACE / live_trace (whole array)
# ══════════════════════════════════════════════════════════════════════

class TestScenario5_1D_Trace_LiveTrace:
    """PlotSpec(x=freqs, y='vna_mag') — full TRACE refreshed each point."""

    def setup_method(self):
        ro = {"vna_mag": _MockReadout("vna_mag", DataKind.TRACE, shape=(len(FREQS),))}
        self.proc = _proc1d(ro)
        self.spec = PlotSpec(x=FREQS, y="vna_mag")
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_resolves_to_live_trace(self):
        assert self.plotter._resolved_types[0] == "live_trace"

    def test_update_every_is_point(self):
        assert self.plotter._resolved_update_every[0] == "point"

    def test_x_axis_has_freq_values(self):
        x_data = self.plotter._fig_dict["data"][0]["x"]
        assert len(x_data) == len(FREQS)
        assert abs(x_data[0] - FREQS[0]) < 1.0

    def test_initial_y_is_nan(self):
        y_data = self.plotter._fig_dict["data"][0]["y"]
        assert all(np.isnan(y) for y in y_data)

    def test_update_replaces_full_array(self):
        spectrum = np.sin(FREQS / 1e9)
        self.plotter.update_point(
            (0,), {"vna_mag": spectrum}, {"Vgt": -1.0}
        )
        y_data = np.array(self.plotter._fig_dict["data"][0]["y"])
        np.testing.assert_allclose(y_data, spectrum)

    def test_state_col_is_none(self):
        assert self.plotter._sweep_data[0][0]["_col"] is None


# ══════════════════════════════════════════════════════════════════════
#  Scenario 6 — Procedure1D / IMAGE / live_trace single-col extraction
# ══════════════════════════════════════════════════════════════════════

class TestScenario6_1D_Image_LiveTrace_Col:
    """PlotSpec(x=freqs, y='vna', y_col='mag') — extract one channel from IMAGE."""

    def setup_method(self):
        ro = {"vna": _MockReadout("vna", DataKind.IMAGE, shape=(3, len(FREQS)),
                                  contains=["freq", "mag", "phase"])}
        self.proc = _proc1d(ro)
        self.spec = PlotSpec(x=FREQS, y="vna", y_col="mag")
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_single_trace_named_mag(self):
        assert len(self.plotter._fig_dict["data"]) == 1
        assert self.plotter._fig_dict["data"][0]["name"] == "mag"

    def test_state_col_is_one(self):
        assert self.plotter._sweep_data[0][0]["_col"] == 1

    def test_update_extracts_mag_row(self):
        image = np.stack([FREQS, np.ones(len(FREQS)) * -30.0, np.zeros(len(FREQS))])
        self.plotter.update_point(
            (0,), {"vna": image}, {"Vgt": -1.0}
        )
        y_data = np.array(self.plotter._fig_dict["data"][0]["y"])
        np.testing.assert_allclose(y_data, -30.0)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 7 — Procedure1D / IMAGE / multi-col live_trace
# ══════════════════════════════════════════════════════════════════════

class TestScenario7_1D_Image_MultiCol_LiveTrace:
    """PlotSpec(x=freqs, y='vna', y_col=['mag','phase']) — two live traces."""

    def setup_method(self):
        ro = {"vna": _MockReadout("vna", DataKind.IMAGE, shape=(3, len(FREQS)),
                                  contains=["freq", "mag", "phase"])}
        self.proc = _proc1d(ro)
        self.spec = PlotSpec(x=FREQS, y="vna", y_col=["mag", "phase"])
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_two_traces(self):
        assert len(self.plotter._fig_dict["data"]) == 2

    def test_trace_names_match_col_strings(self):
        names = [t["name"] for t in self.plotter._fig_dict["data"]]
        assert names == ["mag", "phase"]

    def test_state_cols_are_1_and_2(self):
        cols = [st["_col"] for st in self.plotter._sweep_data[0]]
        assert cols == [1, 2]

    def test_update_extracts_separate_rows(self):
        mag = np.linspace(-20, -10, len(FREQS))
        phase = np.linspace(-180, 180, len(FREQS))
        image = np.stack([FREQS, mag, phase])
        self.plotter.update_point(
            (0,), {"vna": image}, {"Vgt": -1.0}
        )
        np.testing.assert_allclose(
            np.array(self.plotter._fig_dict["data"][0]["y"]), mag
        )
        np.testing.assert_allclose(
            np.array(self.plotter._fig_dict["data"][1]["y"]), phase
        )


# ══════════════════════════════════════════════════════════════════════
#  Scenario 8 — Procedure2D / SCALAR / heatmap
# ══════════════════════════════════════════════════════════════════════

class TestScenario8_2D_Scalar_Heatmap:
    """PlotSpec(x='Vgt', y='fac', z='sig') — standard 2D heatmap."""

    def setup_method(self):
        ro = {"sig": _MockReadout("sig", DataKind.SCALAR)}
        self.proc = _proc2d(ro)
        self.spec = PlotSpec(x="Vgt", y="fac", z="sig")
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_resolves_to_heatmap(self):
        assert self.plotter._resolved_types[0] == "heatmap"

    def test_update_every_is_sweep(self):
        assert self.plotter._resolved_update_every[0] == "sweep"

    def test_heatmap_trace_in_figure(self):
        assert self.plotter._fig_dict["data"][0]["type"] == "heatmap"

    def test_z_matrix_shape(self):
        z = np.array(self.plotter._fig_dict["data"][0]["z"])
        # rows = fac (outer sweep), cols = Vgt (inner sweep)
        assert z.shape == (len(FAC_VALS), len(VGT_VALS))

    def test_z_initially_nan(self):
        z = np.array(self.plotter._fig_dict["data"][0]["z"])
        assert np.all(np.isnan(z))

    def test_update_heatmap_fills_row(self):
        row_data = np.linspace(0.0, 1.0, len(VGT_VALS))
        self.plotter.update_sweep(
            (2,), {"sig": row_data}, {"fac": FAC_VALS[2]}
        )
        z = np.array(self.plotter._fig_dict["data"][0]["z"])
        np.testing.assert_allclose(z[2, :], row_data)

    def test_update_heatmap_point_fills_cell(self):
        self.plotter.update_sweep(
            (1, 3), {"sig": 0.42}, {"fac": FAC_VALS[1], "Vgt": VGT_VALS[3]}
        )
        z = np.array(self.plotter._fig_dict["data"][0]["z"])
        assert z[1, 3] == pytest.approx(0.42)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 9 — Procedure2D / TRACE / trace_heatmap (no z_col)
# ══════════════════════════════════════════════════════════════════════

class TestScenario9_2D_Trace_TraceHeatmap:
    """PlotSpec(x='Vgt', y=freqs, z='vna_mag') — TRACE accumulated into heatmap."""

    def setup_method(self):
        ro = {"vna_mag": _MockReadout("vna_mag", DataKind.TRACE, shape=(len(FREQS),))}
        self.proc = _proc2d(ro)
        self.spec = PlotSpec(x="Vgt", y=FREQS, z="vna_mag")
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_resolves_to_trace_heatmap(self):
        assert self.plotter._resolved_types[0] == "trace_heatmap"

    def test_z_matrix_shape(self):
        z = np.array(self.plotter._fig_dict["data"][0]["z"])
        assert z.shape == (len(FREQS), len(VGT_VALS))

    def test_update_fills_column(self):
        col_data = np.sin(FREQS / 1e9)
        self.plotter.update_point(
            (5,), {"vna_mag": col_data}, {"Vgt": VGT_VALS[5]}
        )
        z = np.array(self.plotter._fig_dict["data"][0]["z"])
        np.testing.assert_allclose(z[:, 5], col_data)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 10 — Procedure2D / IMAGE / trace_heatmap with z_col
# ══════════════════════════════════════════════════════════════════════

class TestScenario10_2D_Image_TraceHeatmap_ZCol:
    """PlotSpec(x='Vgt', y=freqs, z='vna', z_col='mag') — IMAGE col extraction."""

    def setup_method(self):
        ro = {"vna": _MockReadout("vna", DataKind.IMAGE, shape=(3, len(FREQS)),
                                  contains=["freq", "mag", "phase"])}
        self.proc = _proc2d(ro)
        self.spec = PlotSpec(x="Vgt", y=FREQS, z="vna", z_col="mag")
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_resolves_to_trace_heatmap(self):
        assert self.plotter._resolved_types[0] == "trace_heatmap"

    def test_update_extracts_mag_row(self):
        mag_col = np.linspace(-30, -20, len(FREQS))
        image = np.stack([FREQS, mag_col, np.zeros(len(FREQS))])
        self.plotter.update_point(
            (3,), {"vna": image}, {"Vgt": VGT_VALS[3]}
        )
        z = np.array(self.plotter._fig_dict["data"][0]["z"])
        np.testing.assert_allclose(z[:, 3], mag_col)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 11 — Monitor / SCALAR / line vs time
# ══════════════════════════════════════════════════════════════════════

class TestScenario11_Monitor_Scalar_LineVsTime:
    """PlotSpec(x='_time', y='sig') — rolling buffer time-series."""

    def setup_method(self):
        ro = {"sig": _MockReadout("sig", DataKind.SCALAR)}
        self.proc = _monitor(ro)
        self.spec = PlotSpec(x="_time", y="sig")
        self.plotter = make_plotter(self.spec, max_display_pts=10)
        self.plotter.setup(self.proc)

    def test_resolves_to_line(self):
        assert self.plotter._resolved_types[0] == "line"

    def test_buffer_capacity_equals_max_display_pts(self):
        st = self.plotter._sweep_data[0][0]
        assert st["_cap"] == 10

    def test_state_has_raw_t_buffer(self):
        st = self.plotter._sweep_data[0][0]
        assert "_raw_t" in st

    def test_monitor_accumulates_samples(self):
        t0 = time.time()
        for k in range(5):
            self.plotter.update_monitor(k, {"sig": float(k)}, t0 + k)
        n = self.plotter._sweep_data[0][0]["_n"]
        assert n == 5

    def test_monitor_rolls_buffer_when_full(self):
        t0 = time.time()
        for k in range(15):
            self.plotter.update_monitor(k, {"sig": float(k)}, t0 + k)
        st = self.plotter._sweep_data[0][0]
        assert st["_n"] == 10
        # Oldest retained sample should be sample 5 (0-indexed)
        assert float(st["y"][0]) == pytest.approx(5.0)
        assert float(st["y"][-1]) == pytest.approx(14.0)

    def test_x_axis_starts_as_seconds(self):
        t0 = time.time()
        self.plotter.update_monitor(0, {"sig": 1.0}, t0)
        layout = self.plotter._fig_dict["layout"]
        title_text = layout.get("xaxis", {}).get("title", {}).get("text", "")
        assert "s" in title_text


# ══════════════════════════════════════════════════════════════════════
#  Scenario 12 — Monitor / TRACE / line vs time single-col
# ══════════════════════════════════════════════════════════════════════

class TestScenario12_Monitor_Trace_LineVsTime_Col:
    """PlotSpec(x='_time', y='lockin', y_col='X') — extract scalar from TRACE, rolling."""

    def setup_method(self):
        ro = {"lockin": _MockReadout("lockin", DataKind.TRACE, shape=(2,), contains=["X", "Y"])}
        self.proc = _monitor(ro)
        self.spec = PlotSpec(x="_time", y="lockin", y_col="X")
        self.plotter = make_plotter(self.spec, max_display_pts=20)
        self.plotter.setup(self.proc)

    def test_single_trace_named_X(self):
        assert len(self.plotter._fig_dict["data"]) == 1
        assert self.plotter._fig_dict["data"][0]["name"] == "X"

    def test_state_col_is_zero(self):
        assert self.plotter._sweep_data[0][0]["_col"] == 0

    def test_monitor_extracts_X_component(self):
        t0 = time.time()
        for k in range(5):
            self.plotter.update_monitor(k, {"lockin": np.array([float(k), float(k) * 2])}, t0 + k)
        y_data = self.plotter._fig_dict["data"][0]["y"]
        # Y buffer contains X components: 0,1,2,3,4
        for i, val in enumerate(y_data):
            assert float(val) == pytest.approx(float(i))


# ══════════════════════════════════════════════════════════════════════
#  Scenario 13 — Monitor / TRACE / multi-line vs time
# ══════════════════════════════════════════════════════════════════════

class TestScenario13_Monitor_Trace_MultiLine_VsTime:
    """PlotSpec(x='_time', y='lockin', y_col=['X','Y']) — two rolling traces."""

    def setup_method(self):
        ro = {"lockin": _MockReadout("lockin", DataKind.TRACE, shape=(2,), contains=["X", "Y"])}
        self.proc = _monitor(ro)
        self.spec = PlotSpec(x="_time", y="lockin", y_col=["X", "Y"])
        self.plotter = make_plotter(self.spec, max_display_pts=20)
        self.plotter.setup(self.proc)

    def test_two_traces(self):
        assert len(self.plotter._fig_dict["data"]) == 2

    def test_trace_names(self):
        names = [t["name"] for t in self.plotter._fig_dict["data"]]
        assert names == ["X", "Y"]

    def test_monitor_updates_both_traces(self):
        t0 = time.time()
        self.plotter.update_monitor(0, {"lockin": np.array([1.11, 2.22])}, t0)
        y0 = float(self.plotter._fig_dict["data"][0]["y"][0])
        y1 = float(self.plotter._fig_dict["data"][1]["y"][0])
        assert y0 == pytest.approx(1.11)
        assert y1 == pytest.approx(2.22)

    def test_monitor_rolls_independently(self):
        t0 = time.time()
        for k in range(25):
            self.plotter.update_monitor(k, {"lockin": np.array([float(k), -float(k)])}, t0 + k)
        st0 = self.plotter._sweep_data[0][0]
        st1 = self.plotter._sweep_data[0][1]
        assert st0["_n"] == 20
        assert st1["_n"] == 20
        assert float(st0["y"][-1]) == pytest.approx(24.0)
        assert float(st1["y"][-1]) == pytest.approx(-24.0)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 14 — Monitor / live_trace (spectrum refresh in monitor)
# ══════════════════════════════════════════════════════════════════════

class TestScenario14_Monitor_LiveTrace:
    """PlotSpec(x=freqs, y='vna_mag') — spectrum refreshed each monitor sample."""

    def setup_method(self):
        ro = {"vna_mag": _MockReadout("vna_mag", DataKind.TRACE, shape=(len(FREQS),))}
        self.proc = _monitor(ro)
        self.spec = PlotSpec(x=FREQS, y="vna_mag")
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_resolves_to_live_trace(self):
        assert self.plotter._resolved_types[0] == "live_trace"

    def test_initial_y_all_nan(self):
        y = self.plotter._fig_dict["data"][0]["y"]
        assert all(np.isnan(v) for v in y)

    def test_monitor_update_replaces_spectrum(self):
        spectrum = np.cos(FREQS / 1e9)
        t0 = time.time()
        self.plotter.update_monitor(0, {"vna_mag": spectrum}, t0)
        y_data = np.array(self.plotter._fig_dict["data"][0]["y"])
        np.testing.assert_allclose(y_data, spectrum)

    def test_repeated_monitor_updates_overwrite(self):
        t0 = time.time()
        for k in range(5):
            s = np.full(len(FREQS), float(k))
            self.plotter.update_monitor(k, {"vna_mag": s}, t0 + k)
        y_data = np.array(self.plotter._fig_dict["data"][0]["y"])
        np.testing.assert_allclose(y_data, 4.0)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 15 — update_every auto-resolution and dispatch routing
# ══════════════════════════════════════════════════════════════════════

class TestScenario15_UpdateEveryAutoRoute:
    """Verify that dispatch() only fires at the right event cadence."""

    def test_line_fires_on_point_not_sweep(self):
        ro = {"sig": _MockReadout("sig", DataKind.SCALAR)}
        proc = _proc1d(ro)
        spec = PlotSpec(x="Vgt", y="sig")
        plotter = make_plotter(spec)
        plotter.setup(proc)

        # point event → should update
        plotter.dispatch("point", (0,), {"sig": 1.0}, {"Vgt": -1.0})
        assert plotter._sweep_data[0][0]["_n"] == 1

        # sweep event → should NOT update (line resolves to point)
        plotter.dispatch("sweep", (0,), {"sig": 2.0}, {"Vgt": -1.0})
        assert plotter._sweep_data[0][0]["_n"] == 1

    def test_heatmap_fires_on_sweep_not_point(self):
        ro = {"sig": _MockReadout("sig", DataKind.SCALAR)}
        proc = _proc2d(ro)
        spec = PlotSpec(x="Vgt", y="fac", z="sig")
        plotter = make_plotter(spec)
        plotter.setup(proc)

        # point event → should NOT update heatmap (resolves to sweep)
        plotter.dispatch("point", (0, 3), {"sig": 1.0}, {"fac": FAC_VALS[0], "Vgt": VGT_VALS[3]})
        z = np.array(plotter._fig_dict["data"][0]["z"])
        assert np.all(np.isnan(z))

        # sweep event → should update
        plotter.dispatch("sweep", (2,), {"sig": np.ones(len(VGT_VALS))}, {"fac": FAC_VALS[2]})
        z = np.array(plotter._fig_dict["data"][0]["z"])
        assert not np.any(np.isnan(z[2, :]))

    def test_explicit_update_every_overrides_default(self):
        ro = {"sig": _MockReadout("sig", DataKind.SCALAR)}
        proc = _proc1d(ro)
        spec = PlotSpec(x="Vgt", y="sig", update_every="sweep")
        plotter = make_plotter(spec)
        plotter.setup(proc)

        assert plotter._resolved_update_every[0] == "sweep"

        # point event → should NOT update
        plotter.dispatch("point", (0,), {"sig": 1.0}, {"Vgt": -1.0})
        assert plotter._sweep_data[0][0]["_n"] == 0


# ══════════════════════════════════════════════════════════════════════
#  Scenario 16 — inner-sweep reset on x wrap-around
# ══════════════════════════════════════════════════════════════════════

class TestScenario16_InnerSweepReset:
    """Line plot resets when x goes back to start (2D outer sweep scenario)."""

    def setup_method(self):
        ro = {"sig": _MockReadout("sig", DataKind.SCALAR)}
        self.proc = _proc2d(ro)
        self.spec = PlotSpec(x="Vgt", y="sig")
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_reset_on_second_inner_sweep(self):
        # Fill one inner sweep
        for i, v in enumerate(VGT_VALS):
            self.plotter.update_point((0, i), {"sig": float(i)}, {"Vgt": v})
        assert self.plotter._sweep_data[0][0]["_n"] == len(VGT_VALS)

        # Start second outer row — x goes back to VGT_VALS[0], should reset
        self.plotter.update_point((1, 0), {"sig": 99.0}, {"Vgt": VGT_VALS[0]})
        assert self.plotter._sweep_data[0][0]["_n"] == 1
        assert float(self.plotter._fig_dict["data"][0]["y"][0]) == pytest.approx(99.0)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 17 — unknown readout in PlotSpec raises on setup
# ══════════════════════════════════════════════════════════════════════

class TestScenario17_BadReadoutName:
    def test_unknown_readout_raises_value_error(self):
        ro = {"sig": _MockReadout("sig", DataKind.SCALAR)}
        proc = _proc1d(ro)
        spec = PlotSpec(x="Vgt", y="not_registered")
        plotter = make_plotter(spec)
        with pytest.raises(ValueError, match="not_registered"):
            plotter.setup(proc)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 18 — y_col with int indices (no contains needed)
# ══════════════════════════════════════════════════════════════════════

class TestScenario18_YColInt:
    """y_col as plain integers — no readout.contains required."""

    def setup_method(self):
        ro = {"data": _MockReadout("data", DataKind.TRACE, shape=(4,), contains=None)}
        self.proc = _proc1d(ro)
        self.spec = PlotSpec(x="Vgt", y="data", y_col=[0, 2])
        self.plotter = make_plotter(self.spec)
        self.plotter.setup(self.proc)

    def test_two_traces_with_int_labels(self):
        assert len(self.plotter._fig_dict["data"]) == 2

    def test_trace_names_use_index_notation(self):
        names = [t["name"] for t in self.plotter._fig_dict["data"]]
        assert names == ["data[0]", "data[2]"]

    def test_update_extracts_correct_elements(self):
        raw = np.array([10.0, 20.0, 30.0, 40.0])
        self.plotter.update_point((0,), {"data": raw}, {"Vgt": -1.0})
        assert float(self.plotter._fig_dict["data"][0]["y"][0]) == pytest.approx(10.0)
        assert float(self.plotter._fig_dict["data"][1]["y"][0]) == pytest.approx(30.0)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 19 — two specs in same plotter (multi-subplot)
# ══════════════════════════════════════════════════════════════════════

class TestScenario19_MultiSubplot:
    """Two PlotSpec objects → two subplots, independent trace offsets."""

    def setup_method(self):
        ro = {
            "sig_X": _MockReadout("sig_X", DataKind.SCALAR),
            "sig_Y": _MockReadout("sig_Y", DataKind.SCALAR),
        }
        self.proc = _proc1d(ro)
        spec_x = PlotSpec(x="Vgt", y="sig_X")
        spec_y = PlotSpec(x="Vgt", y="sig_Y")
        self.plotter = make_plotter(spec_x, spec_y)
        self.plotter.setup(self.proc)

    def test_two_traces_total(self):
        assert len(self.plotter._fig_dict["data"]) == 2

    def test_trace_offsets_are_0_and_1(self):
        assert self.plotter._trace_offsets == [0, 1]

    def test_update_routes_to_correct_subplot(self):
        self.plotter.update_point(
            (0,), {"sig_X": 1.0, "sig_Y": 2.0}, {"Vgt": -1.0}
        )
        assert float(self.plotter._fig_dict["data"][0]["y"][0]) == pytest.approx(1.0)
        assert float(self.plotter._fig_dict["data"][1]["y"][0]) == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 20 — mixed subplot: line + live_trace + heatmap
# ══════════════════════════════════════════════════════════════════════

class TestScenario20_MixedSubplots:
    """Three specs of different types in one plotter."""

    def setup_method(self):
        ro = {
            "sig":     _MockReadout("sig",     DataKind.SCALAR),
            "vna_mag": _MockReadout("vna_mag", DataKind.TRACE, shape=(len(FREQS),)),
            "hm_sig":  _MockReadout("hm_sig",  DataKind.SCALAR),
        }
        self.proc = _proc2d(ro)
        specs = [
            PlotSpec(x="Vgt", y="sig"),
            PlotSpec(x=FREQS, y="vna_mag"),
            PlotSpec(x="Vgt", y="fac", z="hm_sig"),
        ]
        self.plotter = make_plotter(*specs)
        self.plotter.setup(self.proc)

    def test_resolved_types(self):
        assert self.plotter._resolved_types == ["line", "live_trace", "heatmap"]

    def test_update_every_per_type(self):
        assert self.plotter._resolved_update_every == ["point", "point", "sweep"]

    def test_three_traces_total(self):
        assert len(self.plotter._fig_dict["data"]) == 3

    def test_trace_offsets(self):
        assert self.plotter._trace_offsets == [0, 1, 2]

    def test_heatmap_only_updates_on_sweep_event(self):
        # point fires line and live_trace but not heatmap
        self.plotter.dispatch(
            "point", (0, 0), {"sig": 1.0, "vna_mag": np.zeros(len(FREQS)), "hm_sig": 9.9},
            {"fac": FAC_VALS[0], "Vgt": VGT_VALS[0]}
        )
        z = np.array(self.plotter._fig_dict["data"][2]["z"])
        assert np.all(np.isnan(z))  # heatmap untouched

        # sweep event fills heatmap row
        self.plotter.dispatch(
            "sweep", (0,), {"hm_sig": np.ones(len(VGT_VALS))},
            {"fac": FAC_VALS[0]}
        )
        z = np.array(self.plotter._fig_dict["data"][2]["z"])
        assert not np.any(np.isnan(z[0, :]))
