"""Live-plot demo — one runnable experiment per PlotSpec scenario.

Usage
-----
    /Users/helium/miniconda3/envs/fem/bin/python examples/demo_plot_scenarios.py <N>

where N is a scenario number 1–16 (or "list" to print the menu).

Each demo opens a browser tab at http://localhost:8050, runs a short
simulated experiment, then freezes the final plot until you press Enter.

Scenarios
---------
 1  Procedure1D / SCALAR   / line                         — Lorentzian peak
 2  Procedure1D / TRACE    / line  single y_col           — lock-in X component
 3  Procedure1D / TRACE    / line  multi  y_col           — lock-in X and Y
 4  Procedure1D / SCALAR×2 / line  legacy multi-readout   — two independent channels
 5  Procedure1D / TRACE    / live_trace (whole array)     — resonance spectrum
 6  Procedure1D / IMAGE    / live_trace single y_col      — VNA magnitude channel
 7  Procedure1D / IMAGE    / live_trace multi  y_col      — VNA mag + phase
 8  Procedure2D / SCALAR   / heatmap                      — Coulomb diamond
 9  Procedure2D / TRACE    / trace_heatmap                — spectrum vs gate
10  Procedure2D / IMAGE    / trace_heatmap + z_col        — VNA image col extraction
11  Monitor     / SCALAR   / line vs time                 — drifting signal
12  Monitor     / TRACE    / line vs time single y_col    — lock-in X vs time
13  Monitor     / TRACE    / line vs time multi  y_col    — lock-in X and Y vs time
14  Monitor     / TRACE    / live_trace in monitor        — refreshing spectrum
15  Monitor     / IMAGE    / live_trace y_col in monitor  — VNA image, mag channel
16  Monitor     / SCALAR×2 / readout vs readout           — lock-in IQ semi-circle
"""

import sys
import time
import tempfile
import webbrowser

import numpy as np

sys.path.insert(0, "src")

from orchid import Bench, ExperimentRunner
from orchid.controller import DataKind
from orchid.procedure import Procedure, MonitorProcedure, Sweep, WriteMode
from orchid.plotting import PlotSpec, DashPlotter


# ══════════════════════════════════════════════════════════════════════
#  Shared physics helpers
# ══════════════════════════════════════════════════════════════════════

FREQS = np.linspace(4.0e9, 8.0e9, 61)          # GHz-range frequency axis
RNG   = np.random.default_rng(42)


def lorentzian(x, x0=0.0, width=0.1, amp=1.0) -> float:
    return amp / (1.0 + ((x - x0) / width) ** 2)


def resonance_dip(freqs, f0, width=0.4e9, depth=30.0) -> np.ndarray:
    """Transmission dip in dB."""
    return -depth * lorentzian(freqs, f0, width)


def resonance_phase(freqs, f0, width=0.4e9) -> np.ndarray:
    """Phase response (degrees)."""
    return -180.0 / np.pi * np.arctan2(freqs - f0, width)


def noise(sigma=0.02):
    return float(RNG.normal(0, sigma))


def vnoise(n, sigma=0.5):
    return RNG.normal(0, sigma, n)


# ══════════════════════════════════════════════════════════════════════
#  DemoPlotter — keeps Dash server alive after run() so user can inspect
# ══════════════════════════════════════════════════════════════════════

class DemoPlotter(DashPlotter):
    """Freezes instead of closing after the runner finishes."""

    def stop(self, _silent=False):
        self.finalize()             # pause live updates, keep server alive
        if not _silent:
            print("  [plot frozen — press Enter in the terminal to close]")

    def really_stop(self):
        super().stop()


# ══════════════════════════════════════════════════════════════════════
#  Shared setup helpers
# ══════════════════════════════════════════════════════════════════════

def _make_bench() -> tuple[Bench, dict]:
    """Return (bench, state) where state is the shared instrument state dict."""
    data_dir = tempfile.mkdtemp(prefix="orchid_demo_")
    bench = Bench(data_root=data_dir)
    state = {"Vgt": 0.0, "fac": 0.0}
    return bench, state, data_dir


def _make_plotter(*specs, port=8050) -> DemoPlotter:
    p = DemoPlotter(list(specs), port=port, open_browser=True,
                    height=380, width=750, update_interval=150)
    return p


def _wait_and_close(plotter: DemoPlotter):
    input("\n  Press Enter to close the plot server and continue…")
    plotter.really_stop()


def _run_sweep(proc, plotter):
    runner = ExperimentRunner(use_experiment_id=False)
    runner.run(proc, plotter=plotter)
    _wait_and_close(plotter)


def _run_monitor(proc, plotter):
    runner = ExperimentRunner(use_experiment_id=False)
    runner.run_monitor(proc, plotter=plotter)
    _wait_and_close(plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 1 — Procedure1D / SCALAR / line
#  Simple Lorentzian peak measured at each gate voltage point.
# ══════════════════════════════════════════════════════════════════════

def scenario_01():
    """1D sweep, scalar readout → basic line plot."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"],
        unit="V")

    bench.add_readout("signal", kind=DataKind.SCALAR,
        get_func=lambda: lorentzian(state["Vgt"], x0=-0.4, width=0.08) + noise())

    proc = Procedure(
        name="s01_scalar_line",
        bench=bench,
        sweeps=[Sweep("Vgt", np.linspace(-1.0, 0.0, 41))],
        readouts=["signal"],
        settle_time=0.04,
    )
    plotter = _make_plotter(PlotSpec(x="Vgt", y="signal"))
    print("\nScenario 1: 1D SCALAR → line plot")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 2 — Procedure1D / TRACE / line, single y_col
#  Lock-in returns [X, Y]. Plot only X vs gate.
# ══════════════════════════════════════════════════════════════════════

def scenario_02():
    """TRACE readout [X, Y] — extract X component into a line plot."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"], unit="V")

    def lockin_xy():
        v = state["Vgt"]
        x = lorentzian(v, x0=-0.5, width=0.1) + noise()
        y = (v + 0.5) * 0.5 + noise(0.03)      # slowly varying background
        return np.array([x, y])

    bench.add_readout("lockin", kind=DataKind.TRACE, shape=(2,),
        contains=["X", "Y"], get_func=lockin_xy)

    proc = Procedure(
        name="s02_trace_line_col",
        bench=bench,
        sweeps=[Sweep("Vgt", np.linspace(-1.0, 0.0, 41))],
        readouts=["lockin"],
        settle_time=0.04,
    )
    plotter = _make_plotter(PlotSpec(x="Vgt", y="lockin", y_col="X"))
    print("\nScenario 2: 1D TRACE → line, single y_col='X'")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 3 — Procedure1D / TRACE / multi-line y_col list
#  Lock-in returns [X, Y]. Plot both on the same subplot.
# ══════════════════════════════════════════════════════════════════════

def scenario_03():
    """TRACE readout [X, Y] — plot both components on one subplot."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"], unit="V")

    def lockin_xy():
        v = state["Vgt"]
        x = lorentzian(v, x0=-0.5, width=0.1) + noise()
        y = lorentzian(v, x0=-0.3, width=0.08, amp=0.6) + noise(0.02)
        return np.array([x, y])

    bench.add_readout("lockin", kind=DataKind.TRACE, shape=(2,),
        contains=["X", "Y"], get_func=lockin_xy)

    proc = Procedure(
        name="s03_trace_multiline",
        bench=bench,
        sweeps=[Sweep("Vgt", np.linspace(-1.0, 0.0, 41))],
        readouts=["lockin"],
        settle_time=0.04,
    )
    plotter = _make_plotter(PlotSpec(x="Vgt", y="lockin", y_col=["X", "Y"]))
    print("\nScenario 3: 1D TRACE → multi-line y_col=['X','Y']")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 4 — Procedure1D / two SCALAR readouts / legacy multi-readout
#  Two independent channels in one subplot via y=[list of names].
# ══════════════════════════════════════════════════════════════════════

def scenario_04():
    """Two scalar readouts overlaid on one subplot (legacy y=list mode)."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"], unit="V")

    bench.add_readout("ch_A", kind=DataKind.SCALAR,
        get_func=lambda: lorentzian(state["Vgt"], x0=-0.6, width=0.08) + noise())
    bench.add_readout("ch_B", kind=DataKind.SCALAR,
        get_func=lambda: lorentzian(state["Vgt"], x0=-0.3, width=0.12, amp=0.7) + noise())

    proc = Procedure(
        name="s04_multi_readout_line",
        bench=bench,
        sweeps=[Sweep("Vgt", np.linspace(-1.0, 0.0, 41))],
        readouts=["ch_A", "ch_B"],
        settle_time=0.04,
    )
    plotter = _make_plotter(PlotSpec(x="Vgt", y=["ch_A", "ch_B"]))
    print("\nScenario 4: 1D SCALAR×2 → multi-readout line, y=['ch_A','ch_B']")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 5 — Procedure1D / TRACE / live_trace (whole array)
#  VNA-like: full spectrum refreshed every gate step.
# ══════════════════════════════════════════════════════════════════════

def scenario_05():
    """TRACE spectrum refreshed at every point — live_trace with whole array."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"], unit="V")

    def vna_mag():
        vgt = state["Vgt"]
        f0  = 6.0e9 + vgt * 1.0e9          # resonance shifts with gate
        return resonance_dip(FREQS, f0) + vnoise(len(FREQS), sigma=0.3)

    bench.add_readout("vna_mag", kind=DataKind.TRACE, shape=(len(FREQS),),
        unit="dB", get_func=vna_mag)

    proc = Procedure(
        name="s05_trace_live",
        bench=bench,
        sweeps=[Sweep("Vgt", np.linspace(-1.0, 0.5, 31))],
        readouts=["vna_mag"],
        settle_time=0.08,
    )
    plotter = _make_plotter(PlotSpec(x=FREQS / 1e9, y="vna_mag"))
    print("\nScenario 5: 1D TRACE → live_trace (whole spectrum each step)")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 6 — Procedure1D / IMAGE / live_trace single y_col
#  VNA returns (freq, mag, phase) as IMAGE. Show magnitude channel.
# ══════════════════════════════════════════════════════════════════════

def scenario_06():
    """IMAGE readout — extract magnitude row for live_trace."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"], unit="V")

    def vna_image():
        vgt = state["Vgt"]
        f0  = 6.0e9 + vgt * 1.0e9
        mag   = resonance_dip(FREQS, f0)   + vnoise(len(FREQS), 0.3)
        phase = resonance_phase(FREQS, f0) + vnoise(len(FREQS), 1.5)
        return np.stack([FREQS / 1e9, mag, phase])   # shape (3, N_freq)

    bench.add_readout("vna", kind=DataKind.IMAGE, shape=(3, len(FREQS)),
        contains=["freq", "mag", "phase"], get_func=vna_image)

    proc = Procedure(
        name="s06_image_live_col",
        bench=bench,
        sweeps=[Sweep("Vgt", np.linspace(-1.0, 0.5, 31))],
        readouts=["vna"],
        settle_time=0.08,
    )
    plotter = _make_plotter(PlotSpec(x=FREQS / 1e9, y="vna", y_col="mag"))
    print("\nScenario 6: 1D IMAGE → live_trace y_col='mag'")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 7 — Procedure1D / IMAGE / multi-col live_trace
#  Same IMAGE readout — show magnitude AND phase simultaneously.
# ══════════════════════════════════════════════════════════════════════

def scenario_07():
    """IMAGE readout — two live_trace subplots (mag and phase)."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"], unit="V")

    def vna_image():
        vgt = state["Vgt"]
        f0  = 6.0e9 + vgt * 1.0e9
        mag   = resonance_dip(FREQS, f0)   + vnoise(len(FREQS), 0.3)
        phase = resonance_phase(FREQS, f0) + vnoise(len(FREQS), 1.5)
        return np.stack([FREQS / 1e9, mag, phase])

    bench.add_readout("vna", kind=DataKind.IMAGE, shape=(3, len(FREQS)),
        contains=["freq", "mag", "phase"], get_func=vna_image)

    proc = Procedure(
        name="s07_image_multicol_live",
        bench=bench,
        sweeps=[Sweep("Vgt", np.linspace(-1.0, 0.5, 31))],
        readouts=["vna"],
        settle_time=0.08,
    )
    spec_mag   = PlotSpec(x=FREQS / 1e9, y="vna", y_col="mag")
    spec_phase = PlotSpec(x=FREQS / 1e9, y="vna", y_col="phase")
    plotter = _make_plotter(spec_mag, spec_phase)
    print("\nScenario 7: 1D IMAGE → two live_trace subplots (mag + phase)")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 8 — Procedure2D / SCALAR / heatmap
#  Coulomb diamond pattern in gate vs frequency-coupling space.
# ══════════════════════════════════════════════════════════════════════

def scenario_08():
    """2D SCALAR sweep → heatmap (Coulomb diamond)."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"], unit="V")
    bench.add_controller("fac",
        set_func=lambda v: state.__setitem__("fac", v),
        get_func=lambda: state["fac"], unit="")

    def diamond():
        vgt = state["Vgt"]
        fac = state["fac"]
        # Two overlapping Lorentzians shifted by fac → diamond pattern
        a = lorentzian(vgt, x0=-0.5 + 0.4 * fac, width=0.06)
        b = lorentzian(vgt, x0=-0.2 - 0.4 * fac, width=0.06)
        return float(a + b) + noise(0.02)

    bench.add_readout("cond", kind=DataKind.SCALAR, unit="e²/h", get_func=diamond)

    vgt_vals = np.linspace(-1.0, 0.2, 25)
    fac_vals = np.linspace(0.0,  1.0, 16)

    proc = Procedure(
        name="s08_heatmap",
        bench=bench,
        # sweeps[0] is the outer (slow) axis; sweeps[-1] is the inner (fast) axis
        sweeps=[Sweep("fac", fac_vals), Sweep("Vgt", vgt_vals)],
        readouts=["cond"],
        settle_time=0.01,
        write_mode=WriteMode.SWEEPWISE,
    )
    plotter = _make_plotter(PlotSpec(x="Vgt", y="fac", z="cond"))
    print("\nScenario 8: 2D SCALAR → heatmap (Coulomb diamond)")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 9 — Procedure2D / TRACE / trace_heatmap
#  VNA spectrum accumulated as columns — resonance shift vs gate.
# ══════════════════════════════════════════════════════════════════════

def scenario_09():
    """TRACE accumulated per gate step → trace_heatmap."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"], unit="V")
    bench.add_controller("fac",
        set_func=lambda v: state.__setitem__("fac", v),
        get_func=lambda: state["fac"], unit="")

    def vna_mag():
        vgt = state["Vgt"]
        f0  = 6.0e9 + vgt * 0.8e9
        return resonance_dip(FREQS, f0) + vnoise(len(FREQS), 0.4)

    bench.add_readout("vna_mag", kind=DataKind.TRACE, shape=(len(FREQS),),
        unit="dB", get_func=vna_mag)

    vgt_vals = np.linspace(-1.0, 0.5, 21)

    proc = Procedure(
        name="s09_trace_heatmap",
        bench=bench,
        sweeps=[Sweep("Vgt", vgt_vals)],
        readouts=["vna_mag"],
        settle_time=0.06,
    )
    spec = PlotSpec(x="Vgt", y=FREQS / 1e9, z="vna_mag")
    plotter = _make_plotter(spec)
    print("\nScenario 9: 1D TRACE → trace_heatmap (spectrum vs gate)")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 10 — Procedure2D / IMAGE / trace_heatmap + z_col
#  VNA IMAGE readout — accumulate only the magnitude channel.
# ══════════════════════════════════════════════════════════════════════

def scenario_10():
    """IMAGE readout, z_col='mag' → trace_heatmap of magnitude column."""
    bench, state, _ = _make_bench()

    bench.add_controller("Vgt",
        set_func=lambda v: state.__setitem__("Vgt", v),
        get_func=lambda: state["Vgt"], unit="V")

    def vna_image():
        vgt = state["Vgt"]
        f0  = 6.0e9 + vgt * 0.8e9
        mag   = resonance_dip(FREQS, f0)   + vnoise(len(FREQS), 0.4)
        phase = resonance_phase(FREQS, f0) + vnoise(len(FREQS), 1.5)
        return np.stack([FREQS / 1e9, mag, phase])

    bench.add_readout("vna", kind=DataKind.IMAGE, shape=(3, len(FREQS)),
        contains=["freq", "mag", "phase"], get_func=vna_image)

    vgt_vals = np.linspace(-1.0, 0.5, 21)

    proc = Procedure(
        name="s10_image_heatmap_zcol",
        bench=bench,
        sweeps=[Sweep("Vgt", vgt_vals)],
        readouts=["vna"],
        settle_time=0.06,
    )
    spec = PlotSpec(x="Vgt", y=FREQS / 1e9, z="vna", z_col="mag")
    plotter = _make_plotter(spec)
    print("\nScenario 10: 1D IMAGE, z_col='mag' → trace_heatmap")
    _run_sweep(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 11 — Monitor / SCALAR / line vs time
#  Slowly drifting signal + random noise, plotted vs elapsed time.
# ══════════════════════════════════════════════════════════════════════

def scenario_11():
    """Monitor SCALAR readout → rolling line vs time (15 s)."""
    bench, _, _ = _make_bench()
    t0 = [time.time()]

    def drifting_signal():
        elapsed = time.time() - t0[0]
        drift   = 0.5 * np.sin(2 * np.pi * elapsed / 20.0)
        return drift + float(RNG.normal(0, 0.05))

    bench.add_readout("signal", kind=DataKind.SCALAR, get_func=drifting_signal)

    proc = MonitorProcedure(
        name="s11_monitor_scalar",
        bench=bench,
        readouts=["signal"],
        interval=0.25,
        duration=20.0,
    )
    plotter = _make_plotter(PlotSpec(x="_time", y="signal"))
    print("\nScenario 11: Monitor SCALAR → line vs time (20 s)")
    t0[0] = time.time()
    _run_monitor(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 12 — Monitor / TRACE / line vs time single y_col
#  Lock-in returning [X, Y]. Plot only X component vs time.
# ══════════════════════════════════════════════════════════════════════

def scenario_12():
    """Monitor TRACE [X,Y] → extract X component, rolling line vs time."""
    bench, _, _ = _make_bench()
    t0 = [time.time()]

    def lockin_xy():
        elapsed = time.time() - t0[0]
        x = 0.6 * np.sin(2 * np.pi * elapsed / 8.0)  + float(RNG.normal(0, 0.04))
        y = 0.4 * np.cos(2 * np.pi * elapsed / 12.0) + float(RNG.normal(0, 0.04))
        return np.array([x, y])

    bench.add_readout("lockin", kind=DataKind.TRACE, shape=(2,),
        contains=["X", "Y"], get_func=lockin_xy)

    proc = MonitorProcedure(
        name="s12_monitor_trace_col",
        bench=bench,
        readouts=["lockin"],
        interval=0.2,
        duration=20.0,
    )
    plotter = _make_plotter(PlotSpec(x="_time", y="lockin", y_col="X"))
    print("\nScenario 12: Monitor TRACE → line vs time, y_col='X' (20 s)")
    t0[0] = time.time()
    _run_monitor(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 13 — Monitor / TRACE / multi-line vs time
#  Lock-in returning [X, Y]. Plot both vs time on the same subplot.
# ══════════════════════════════════════════════════════════════════════

def scenario_13():
    """Monitor TRACE [X,Y] → two rolling lines vs time."""
    bench, _, _ = _make_bench()
    t0 = [time.time()]

    def lockin_xy():
        elapsed = time.time() - t0[0]
        x = 0.6 * np.sin(2 * np.pi * elapsed / 8.0)  + float(RNG.normal(0, 0.04))
        y = 0.4 * np.cos(2 * np.pi * elapsed / 12.0) + float(RNG.normal(0, 0.04))
        return np.array([x, y])

    bench.add_readout("lockin", kind=DataKind.TRACE, shape=(2,),
        contains=["X", "Y"], get_func=lockin_xy)

    proc = MonitorProcedure(
        name="s13_monitor_multiline",
        bench=bench,
        readouts=["lockin"],
        interval=0.2,
        duration=20.0,
    )
    plotter = _make_plotter(PlotSpec(x="_time", y="lockin", y_col=["X", "Y"]))
    print("\nScenario 13: Monitor TRACE → multi-line vs time, y_col=['X','Y'] (20 s)")
    t0[0] = time.time()
    _run_monitor(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 14 — Monitor / live_trace refreshing spectrum
#  Spectrum drifts slowly over time — peak shifts left and right.
# ══════════════════════════════════════════════════════════════════════

def scenario_14():
    """Monitor TRACE → live_trace spectrum that drifts over time."""
    bench, _, _ = _make_bench()
    t0 = [time.time()]

    def vna_mag():
        elapsed = time.time() - t0[0]
        f0 = 6.0e9 + 0.6e9 * np.sin(2 * np.pi * elapsed / 15.0)
        return resonance_dip(FREQS, f0, width=0.35e9) + vnoise(len(FREQS), 0.3)

    bench.add_readout("vna_mag", kind=DataKind.TRACE, shape=(len(FREQS),),
        unit="dB", get_func=vna_mag)

    proc = MonitorProcedure(
        name="s14_monitor_live_trace",
        bench=bench,
        readouts=["vna_mag"],
        interval=0.15,
        duration=30.0,
    )
    plotter = _make_plotter(PlotSpec(x=FREQS / 1e9, y="vna_mag"))
    print("\nScenario 14: Monitor TRACE → live_trace (drifting resonance, 30 s)")
    t0[0] = time.time()
    _run_monitor(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 15 — Monitor / IMAGE / live_trace with y_col
#  VNA IMAGE readout (mag + phase channels). Display magnitude channel
#  as a live trace; resonance drifts during the monitor run.
# ══════════════════════════════════════════════════════════════════════

def scenario_15():
    """Monitor IMAGE → live_trace y_col='mag' (drifting resonance, 25 s)."""
    bench, _, _ = _make_bench()
    t0 = [time.time()]

    def vna_image():
        elapsed = time.time() - t0[0]
        f0 = 6.0e9 + 0.5e9 * np.sin(2 * np.pi * elapsed / 12.0)
        mag   = resonance_dip(FREQS, f0, width=0.35e9) + vnoise(len(FREQS), 0.3)
        phase = resonance_phase(FREQS, f0, width=0.35e9) + vnoise(len(FREQS), 1.0)
        return np.stack([mag, phase])   # shape (2, N_freq)

    bench.add_readout("vna", kind=DataKind.IMAGE, shape=(2, len(FREQS)),
        contains=["mag", "phase"], get_func=vna_image)

    proc = MonitorProcedure(
        name="s15_monitor_image_live_trace",
        bench=bench,
        readouts=["vna"],
        interval=0.15,
        duration=25.0,
    )
    plotter = _make_plotter(PlotSpec(x=FREQS / 1e9, y="vna", y_col="mag"))
    print("\nScenario 15: Monitor IMAGE → live_trace (y_col='mag', drifting resonance, 25 s)")
    t0[0] = time.time()
    _run_monitor(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Scenario 16 — Monitor / SCALAR×2 / readout vs readout (IQ semi-circle)
#  Two independent SCALAR readouts — lock-in X and Y. As time progresses
#  the IQ phase sweeps 0 → π, tracing a semi-circle in the XY plane.
# ══════════════════════════════════════════════════════════════════════

def scenario_16():
    """Monitor SCALAR × 2 → readout vs readout, IQ semi-circle (20 s)."""
    bench, _, _ = _make_bench()
    t0 = [time.time()]
    sweep_time = 18.0   # seconds for full π sweep

    def _phi():
        return np.pi * min((time.time() - t0[0]) / sweep_time, 1.0)

    bench.add_readout("lockin_X", kind=DataKind.SCALAR, unit="V",
        get_func=lambda: 0.7 * np.cos(_phi()) + float(RNG.normal(0, 0.02)))
    bench.add_readout("lockin_Y", kind=DataKind.SCALAR, unit="V",
        get_func=lambda: 0.7 * np.sin(_phi()) + float(RNG.normal(0, 0.02)))

    proc = MonitorProcedure(
        name="s16_monitor_iq_circle",
        bench=bench,
        readouts=["lockin_X", "lockin_Y"],
        interval=0.15,
        duration=20.0,
    )
    # x and y are both readout names → each sample plots (lockin_X, lockin_Y)
    plotter = _make_plotter(PlotSpec(x="lockin_X", y="lockin_Y"))
    print("\nScenario 16: Monitor SCALAR×2 → IQ semi-circle (lockin_X vs lockin_Y, 20 s)")
    t0[0] = time.time()
    _run_monitor(proc, plotter)


# ══════════════════════════════════════════════════════════════════════
#  Dispatch table and CLI
# ══════════════════════════════════════════════════════════════════════

SCENARIOS = {
    1:  (scenario_01, "Procedure1D / SCALAR   / line"),
    2:  (scenario_02, "Procedure1D / TRACE    / line single y_col"),
    3:  (scenario_03, "Procedure1D / TRACE    / multi-line y_col list"),
    4:  (scenario_04, "Procedure1D / SCALAR×2 / legacy multi-readout line"),
    5:  (scenario_05, "Procedure1D / TRACE    / live_trace whole array"),
    6:  (scenario_06, "Procedure1D / IMAGE    / live_trace single y_col"),
    7:  (scenario_07, "Procedure1D / IMAGE    / live_trace multi y_col (two subplots)"),
    8:  (scenario_08, "Procedure2D / SCALAR   / heatmap (Coulomb diamond)"),
    9:  (scenario_09, "Procedure2D / TRACE    / trace_heatmap (spectrum vs gate)"),
    10: (scenario_10, "Procedure2D / IMAGE    / trace_heatmap + z_col"),
    11: (scenario_11, "Monitor     / SCALAR   / line vs time"),
    12: (scenario_12, "Monitor     / TRACE    / line vs time single y_col"),
    13: (scenario_13, "Monitor     / TRACE    / multi-line vs time"),
    14: (scenario_14, "Monitor     / TRACE    / live_trace drifting spectrum"),
    15: (scenario_15, "Monitor     / IMAGE    / live_trace y_col='mag'"),
    16: (scenario_16, "Monitor     / SCALAR×2 / readout vs readout (IQ semi-circle)"),
}


def _print_menu():
    print("\nAvailable scenarios:")
    for n, (_, desc) in SCENARIOS.items():
        print(f"  {n:2d}  {desc}")
    print()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("list", "--list", "-l", "help", "--help"):
        _print_menu()
        return

    try:
        n = int(sys.argv[1])
    except ValueError:
        print(f"Error: argument must be a scenario number (1–{max(SCENARIOS)}). Got: {sys.argv[1]!r}")
        _print_menu()
        sys.exit(1)

    if n not in SCENARIOS:
        print(f"Error: scenario {n} not found. Valid range: 1–{max(SCENARIOS)}.")
        sys.exit(1)

    fn, desc = SCENARIOS[n]
    print(f"\n{'─'*60}")
    print(f"  Running scenario {n}: {desc}")
    print(f"  Browser: http://localhost:8050")
    print(f"{'─'*60}")
    fn()


if __name__ == "__main__":
    main()
