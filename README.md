<p align="left">
<img src="docs/logo.png" alt="logo" width="350"/>
</p>

**Orchestrating Instruments & Data** — a Python package for automated lab experiment control.

Orchid connects your lab instruments (pymeasure, qcodes, or custom drivers) to a clean sweep/monitor engine with automatic data saving via [zarro](https://github.com/eeroqlab/zarro).

## Features

- **Multi-backend instruments** — pymeasure, qcodes, and custom Python objects with auto-detection
- **pymeasure-style access** — `ctx["Vgt"] = 0.4` to set, `ctx["Vgt"]` to read
- **1D / 2D / 3D sweeps** with snake scan and hysteresis support
- **Time-series monitoring** with configurable interval, duration, and stop conditions
- **Flexible write modes** — write per point, per sweep, per plane, or all at once
- **Custom hooks** — inject logic before/after experiments, sweeps, and measurement points
- **Async support** — both sync and async instrument drivers
- **Live plotting** — real-time Dash browser window (line, heatmap, multi-trace, custom)
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

# 2. Context — register instruments, parameters, readouts
ctx = ExperimentContext(data_root="./data", metadata={"sample": "chip_A1"})
ctx.add_instrument("vs", vs)
ctx.add_parameter("Vgt", instrument="vs", attr="voltage", unit="V")
ctx.add_readout("signal", kind="scalar", get_func=lambda: vs.voltage ** 2, unit="V")

# 3. Interact
ctx["Vgt"] = 0.5        # set
print(ctx["Vgt"])        # read -> 0.5
print(ctx["signal"])     # measure -> 0.25
ctx.snapshot()           # print table of all values

# 4. Define and run a sweep
proc = Procedure(
    name="gate_sweep",
    context=ctx,
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
InstrumentAdapter          Parameter / Readout
 (pymeasure/qcodes/       (named controls and
  custom drivers)           measurement channels)
        \                       /
         \                     /
       ExperimentContext
       ctx["Vgt"] = 0.4  |  ctx.snapshot()
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
| `Parameter`          | Named control mapped to an instrument channel         |
| `Readout`            | Read-only measurement channel (scalar, trace, image)  |
| `ExperimentContext`  | Container for all instruments, parameters, readouts   |
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
- Optional: `plotly` (for live plotting), `qcodes`, `pymeasure`
