<p align="left">
<img src="docs/logo.png" alt="logo" width="350"/>
</p>

**Orchestrating Instruments & Data** — a Python package for automated lab experiment control.

Orchid connects your lab instruments (pymeasure, qcodes, or custom drivers) to a clean sweep/monitor engine with automatic data saving via [zarro](https://github.com/eeroqlab/zarro).

## Features

- **Multi-backend instruments** — pymeasure, qcodes, and custom Python objects with auto-detection
- **pymeasure-style access** — `ctx["Vgt"] = 0.4` to set, `ctx["Vgt"]` to read
- **Controller limits and bindings** — clamp, log, raise, or bind one logical control to multiple physical channels
- **1D / 2D / 3D sweeps** with snake scan and hysteresis support
- **Time-series monitoring** with configurable interval, duration, and stop conditions
- **Flexible write modes** — write per point, per sweep, per plane, all at once, or skip saving entirely (`NONE` for dry runs)
- **Custom hooks** — inject logic before/after experiments, sweeps, and measurement points
- **Async support** — both sync and async instrument drivers
- **Live plotting** — real-time Dash browser window with themes, sweep rail, and snapshot (line, heatmap, multi-trace, custom)
- **Configurable error handling** — stop, retry+skip, or ignore
- **Automatic data saving** — Zarr v3 (via zarro) with metadata YAML
- **Live snapshot** — `ctx.snapshot()` prints a formatted table of all current values

## Installation

```bash
pip install -e ./zarro    # data backend
pip install -e .          # orchid
```

## Quick Start

```python
import numpy as np
from orchid import *

# 1. Instrument (any Python object with properties works)
class VoltageSource:
    def __init__(self):
        self._v = 0.0
    @property
    def voltage(self):
        return self._v
    @voltage.setter
    def voltage(self, v):
        self._v = v

vs = VoltageSource()

# 2. Bench — register instruments, controllers, readouts
bench = Bench(data_root="./data", metadata={"sample": "chip_A1"})
bench.add_instrument("vs", vs)
bench.add_controller("Vgt", instrument="vs", attr="voltage", unit="V")
bench.add_readout("signal", kind="scalar", get_func=lambda: vs.voltage ** 2, unit="V")

# 3. Interact
bench["Vgt"] = 0.5        # set
print(bench["Vgt"])        # read -> 0.5
print(bench["signal"])     # measure -> 0.25
bench.snapshot()           # print table of all values

# 4. Define and run a sweep
proc = Procedure(
    name="gate_sweep",
    bench=bench,
    sweeps=[Sweep("Vgt", np.linspace(0, 1, 101))],
    readouts=["signal"],
)
data_dir = ExperimentRunner().run(proc)

# 5. Read back
import zarr
z = zarr.open(str(data_dir / "vault.zarr"))
print(z["signal"][:])    # shape (101,)
```

## Key Concepts

```
InstrumentAdapter          Controller / Readout
 (pymeasure/qcodes/       (named controls and
  custom drivers)           measurement channels)
        \                       /
         \                     /
       Bench
       bench["Vgt"] = 0.4  |  bench.snapshot()
                |
        Procedure / MonitorProcedure
        (sweeps, readouts, hooks, write_mode)
                |
        ExperimentRunner
        runner.run(proc)  |  runner.run_monitor(mon)
                |
            zarro
        vault.zarr + metadata.yaml
```

| Class                | Role                                                  |
|----------------------|-------------------------------------------------------|
| `InstrumentAdapter`  | Unified get/set wrapper for any instrument backend    |
| `Controller`          | Named control mapped to an instrument channel         |
| `Readout`            | Read-only measurement channel (scalar, trace, image)  |
| `Bench`  | Container for all instruments, parameters, readouts   |
| `Procedure`          | Defines sweeps, readouts, hooks, and write strategy   |
| `MonitorProcedure`   | Time-series monitoring with interval and stop logic   |
| `ExperimentRunner`   | Executes procedures, manages data flow to zarro       |

## Write Modes

Control when data is flushed to disk:

| Mode        | Writes to disk          | zarro method     | Best for              |
|-------------|-------------------------|------------------|-----------------------|
| `POINTWISE` | After every point       | `write_point()`  | Safety-critical scans |
| `SWEEPWISE` | After each inner sweep  | `write_trace()`  | 2D / 3D scans        |
| `PLANEWISE` | After each 2D plane     | `write_image()`  | 3D scans              |
| `ALL`       | Once at the end         | `write_all()`    | Fast, small scans     |
| `NONE`      | Never                   | —                | Dry runs, live-only   |

```python
proc = Procedure(..., write_mode=WriteMode.SWEEPWISE)
```

## Documentation

Full tutorial, cookbook, and API reference: **[docs/orchid.md](docs/orchid.md)**

## Dependencies

- `numpy`
- `tqdm`
- `tabulate`
- `zarro`
- `plotly`, `dash` (for live plotting),
- `qcodes`,
- `pymeasure`
