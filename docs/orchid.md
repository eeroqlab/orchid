# Orchid Documentation

**Orchestrating Instruments & Data** — a Python package for lab experiment control.

Orchid provides a clean pipeline for running automated lab experiments:  
**Instruments** &rarr; **Controllers & Readouts** &rarr; **Bench** &rarr; **Procedure** &rarr; **Runner** &rarr; **Data (zarro)**

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tutorial](#tutorial)
  - [Step 1: Instruments](#step-1-instruments)
  - [Step 2: Controllers & Readouts](#step-2-controllers--readouts)
  - [Step 3: Bench](#step-3-bench)
  - [Step 4: Procedures](#step-4-procedures)
  - [Step 5: Running Experiments](#step-5-running-experiments)
  - [Step 6: Reading Data Back](#step-6-reading-data-back)
  - [Step 7: Live Plotting](#step-7-live-plotting)
- [Cookbook](#cookbook)
  - [1D Sweep](#1d-sweep)
  - [2D Sweep](#2d-sweep)
  - [2D Sweep with Snake Scan](#2d-sweep-with-snake-scan)
  - [Hysteresis (Forward + Backward)](#hysteresis-forward--backward)
  - [Time-Series Monitoring](#time-series-monitoring)
  - [Controller Limits](#controller-limits)
  - [Controller Bindings](#controller-bindings)
  - [Controller Event Logging](#controller-event-logging)
  - [Background Monitoring](#background-monitoring)
  - [Custom Hooks](#custom-hooks)
  - [Mixed Readouts (Scalar + Trace)](#mixed-readouts-scalar--trace)
  - [Async Usage](#async-usage)
  - [Inspecting a Procedure](#inspecting-a-procedure)
  - [Saving and Loading Bench Configuration](#saving-and-loading-bench-configuration)
  - [Interactive Control Panel](#interactive-control-panel)
- [API Reference](#api-reference)
  - [InstrumentAdapter](#instrumentadapter)
  - [Controller](#controller)
  - [LimitPolicy](#limitpolicy)
  - [Readout](#readout)
  - [DataKind](#datakind)
  - [Bench](#bench)
  - [Sweep](#sweep)
  - [MultiSweep](#multisweep)
  - [Procedure](#procedure)
  - [MonitorProcedure](#monitorprocedure)
  - [EventLineConfig](#eventlineconfig)
  - [PlotSpec](#plotspec)
  - [PlotterBase](#plotterbase)
  - [DashPlotter / LivePlotter](#dashplotter--liveplotter)
  - [TaipyPlotter](#taipyplotter)
  - [ControlPanel](#controlpanel)
  - [WriteMode](#writemode)
  - [ErrorPolicy](#errorpolicy)
  - [ExperimentRunner](#experimentrunner)
  - [Utility Functions](#utility-functions)
- [Data Backend (zarro)](#data-backend-zarro)
- [Architecture](#architecture)

---

## Installation

```bash
pip install -e ./zarro    # data backend (if not already installed)
pip install -e .          # orchid
```

Optional extras for instrument frameworks:

```bash
pip install -e ".[qcodes]"      # adds qcodes
pip install -e ".[pymeasure]"   # adds pymeasure
```

**Dependencies:** `numpy`, `tqdm`, `plotly`, `dash`, `zarro`, `tabulate`

---

## Quick Start

```python
import numpy as np
from orchid import *

# Mock instrument
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

# Set up bench
bench = Bench(data_root="./data", metadata={"sample": "chip_A1"})
bench.add_instrument("vs", vs)
bench.add_controller("Vgt", instrument="vs", attr="voltage", unit="V")
bench.add_readout("signal", kind="scalar", get_func=lambda: vs.voltage ** 2, unit="V")

# Set/get values (pymeasure style)
bench["Vgt"] = 0.5
print(bench["Vgt"])        # 0.5
print(bench["signal"])     # 0.25

# Snapshot
bench.snapshot()

# Define and run a 1D sweep
proc = Procedure(
    name="gate_sweep",
    bench=bench,
    sweeps=[Sweep("Vgt", np.linspace(0, 1, 101))],
    readouts=["signal"],
)
data_dir = ExperimentRunner().run(proc)
```

---

## Tutorial

### Step 1: Instruments

Orchid supports three instrument backends: **pymeasure**, **qcodes**, and **custom** (any Python object). Register instruments through `Bench.add_instrument()` or create adapters directly.

#### Custom instruments (any Python object)

Any object with properties or attributes works:

```python
class MyKeithley:
    def __init__(self):
        self._voltage = 0.0
    
    @property
    def voltage(self):
        return self._voltage
    
    @voltage.setter
    def voltage(self, v):
        self._voltage = v

keithley = MyKeithley()
bench.add_instrument("keithley", keithley)
# backend is auto-detected as "custom"
```

#### pymeasure instruments

```python
from pymeasure.instruments.keithley import Keithley2400

keithley = Keithley2400("GPIB::24")
bench.add_instrument("keithley", keithley, backend="pymeasure")
# get/set uses property access: keithley.source_voltage
```

#### qcodes instruments

```python
from qcodes.instrument_drivers.stanford_research import SR830

lockin = SR830("lockin", "GPIB::8")
bench.add_instrument("lockin", lockin, backend="qcodes")
# get/set uses qcodes Parameter API: lockin.frequency.get() / .set()
```

#### Auto-detection

When `backend="auto"` (the default), orchid inspects the class MRO to detect pymeasure or qcodes instruments automatically:

```python
bench.add_instrument("lockin", lockin)  # auto-detects qcodes
```

#### Registering connection info for save/load

Pass `args` and `kwargs` to record how the instrument was constructed. This allows `bench.save()` to write a fully self-contained YAML that `Bench.load()` can replay without any extra code:

```python
# pymeasure — single positional address arg
bench.add_instrument("keithley", keithley,
    args=["GPIB::24"])

# qcodes — name + address as positional args
bench.add_instrument("lockin", lockin,
    args=["lockin", "GPIB::8"])

# custom with keyword args
bench.add_instrument("vs", vs,
    kwargs={"port": "COM3", "timeout": 5000})
```

If `class_path` is omitted, orchid auto-detects it from `type(instrument).__module__`. Instruments defined directly in a Jupyter notebook (`__main__`) emit a warning and are saved as a stub — you must provide the importable path manually before using `Bench.load()`.

#### Direct adapter creation

You can also create adapters without a context:

```python
from orchid import InstrumentAdapter

adapter = InstrumentAdapter.from_qcodes("lockin", lockin)
adapter = InstrumentAdapter.from_pymeasure("keithley", keithley)
adapter = InstrumentAdapter.auto("lockin", lockin)
```

---

### Step 2: Controllers & Readouts

**Controllers** are named controls that map to instrument channels. They support both get and set.  
**Readouts** are read-only measurement channels that acquire data.

There are several ways to define a controller depending on your instrument driver.

#### Controllers for pymeasure instruments

pymeasure instruments expose parameters as Python properties. Pass the property name as `attr`:

```python
from pymeasure.instruments.keithley import Keithley2400
from pymeasure.instruments.srs import SR830

keithley = Keithley2400("GPIB::24")
lockin = SR830("GPIB::8")

bench.add_instrument("keithley", keithley)
bench.add_instrument("lockin", lockin)

# attr matches the pymeasure property name
# internally uses: keithley.source_voltage / keithley.source_voltage = val
bench.add_controller("Vgt", instrument="keithley", attr="source_voltage", unit="V")
bench.add_controller("I_compliance", instrument="keithley", attr="compliance_current", unit="A")

# lockin frequency
# internally uses: lockin.frequency / lockin.frequency = val
bench.add_controller("fac", instrument="lockin", attr="frequency", unit="Hz")
bench.add_controller("sensitivity", instrument="lockin", attr="sensitivity", unit="V")
```

To find available property names, check the pymeasure docs or use `dir(instrument)`.

#### Controllers for qcodes instruments

qcodes instruments expose parameters as `qcodes.Parameter` objects with `.get()` / `.set()` methods. The adapter handles this automatically:

```python
from qcodes.instrument_drivers.stanford_research import SR830
from qcodes.instrument_drivers.yokogawa import GS200

yoko = GS200("yoko", "GPIB::1")
lockin = SR830("lockin", "GPIB::8")

bench.add_instrument("yoko", yoko)
bench.add_instrument("lockin", lockin)

# attr matches the qcodes parameter name
# internally uses: yoko.voltage.get() / yoko.voltage.set(val)
bench.add_controller("Vgt", instrument="yoko", attr="voltage", unit="V")
bench.add_controller("I_range", instrument="yoko", attr="current_range", unit="A")

# internally uses: lockin.frequency.get() / lockin.frequency.set(val)
bench.add_controller("fac", instrument="lockin", attr="frequency", unit="Hz")
bench.add_controller("amplitude", instrument="lockin", attr="amplitude", unit="V")
```

To find available parameter names, use `instrument.print_readable_snapshot()` or `instrument.parameters.keys()`.

#### Controllers via InstrumentAdapter directly

You can create an `InstrumentAdapter` first and pass it to `add_controller` instead of using a registered name:

```python
from orchid import InstrumentAdapter

adapter = InstrumentAdapter.from_pymeasure("keithley", keithley)
# or
adapter = InstrumentAdapter.from_qcodes("yoko", yoko)
# or
adapter = InstrumentAdapter.from_custom("my_device", my_device)

bench.add_controller("Vgt", instrument=adapter, attr="voltage", unit="V")
```

This is useful when you want to manage adapters outside the context, or use the same adapter for multiple controllers without registering it.

#### Controllers via custom callables

For full flexibility — or when the get/set logic doesn't map cleanly to a single attribute — pass `get_func` and/or `set_func`:

```python
# Custom getter + setter (any arbitrary logic)
fac = bench.add_controller(
    "fac",
    get_func=lambda: lockin.driver.frequency,
    set_func=lambda v: setattr(lockin.driver, 'frequency', v),
    unit="Hz",
)

# Read-only controller (no set_func)
bench.add_controller(
    "T_mc",
    get_func=lambda: fridge.get_temperature("MC"),
    unit="K",
)

# Computed controller (e.g., converting DAC codes to voltage)
bench.add_controller(
    "Vgt_actual",
    get_func=lambda: dac.read_channel(3) * 10.0 / 65535,
    set_func=lambda v: dac.write_channel(3, int(v * 65535 / 10.0)),
    unit="V",
)
```

#### Summary: which approach to use

| Situation                              | Approach                                    |
|----------------------------------------|---------------------------------------------|
| pymeasure instrument, standard property | `instrument="name", attr="property_name"`   |
| qcodes instrument, standard parameter  | `instrument="name", attr="param_name"`      |
| Custom Python object with properties   | `instrument="name", attr="property_name"`   |
| Adapter created externally             | `instrument=adapter_obj, attr="attr_name"`  |
| Non-standard access pattern            | `get_func=..., set_func=...`                |
| Read-only value                        | `get_func=...` (no `set_func`)              |
| Computed / derived quantity            | `get_func=..., set_func=...` with logic     |
| One logical control for multiple channels | `add_controller_binding("name", [...])`  |

#### Readouts

Readouts can be defined with a `get_func` callable **or** with `instrument + attr` — just like controllers. The `instrument + attr` form is fully serializable by `bench.save()` / `Bench.load()`.

```python
# Scalar — get_func form (arbitrary logic, not serializable)
bench.add_readout("lockin_X", kind="scalar",
    get_func=lockin_amplifier.read_x, unit="V")

# Scalar — instrument+attr form (serializable)
bench.add_readout("lockin_X", kind="scalar",
    instrument="lockin", attr="x", unit="V")

# Trace readout (1D array per point)
bench.add_readout("mag", kind="trace", shape=(1601,),
    instrument="vna", attr="magnitude", unit="dB",
    contains="transmission magnitude")

# Image readout (2D array per point — e.g. VNA returning freq + mag + phase)
# Pass contains as a list of column names so live plotters can select columns by name
bench.add_readout("S21", kind="image", shape=(3, 1601),
    get_func=vna.get_trace, unit=["Hz", "dB", "deg"],
    contains=["f", "mag", "phase"])

# Simple 2D image
bench.add_readout("camera", kind="image", shape=(480, 640),
    get_func=camera.capture, unit="counts")
```

| `kind`   | `shape`        | Data per point      | `contains`                                 |
|----------|----------------|---------------------|--------------------------------------------|
| `scalar` | not needed     | single `float`      | optional string description                |
| `trace`  | `(N,)`         | 1D `ndarray`        | optional string description                |
| `image`  | `(H, W)`       | 2D `ndarray`        | optional `list[str]` of column names for `W` |

For `IMAGE` readouts, passing `contains` as a `list[str]` (one name per column) lets the `LivePlotter` select a column by name via `z_col` — see [Trace heatmap (VNA)](#trace-heatmap-vna) below.

---

### Step 3: Bench

The `Bench` is the central container that holds your entire lab bench configuration: instruments, parameters, readouts, and metadata.

```python
bench = Bench(
    data_root="./data",
    metadata={
        "sample": "chip_A1",
        "operator": "alice",
        "fridge": "BlueFors_XLD",
        "cooldown": 42,
    },
)
```

#### pymeasure-style access

```python
# Set a parameter
bench["Vgt"] = 0.4

# Read a parameter or readout
voltage = bench["Vgt"]       # reads from instrument
signal  = bench["lockin_X"]  # acquires measurement
```

#### Snapshot

Print a table of current parameter values:

```python
bench.snapshot()
```

Output:
```
Name      Type    Value    Unit
--------  ------  -------  ------
Vgt       param   0.4      V
fac       param   2500.0   Hz
```

By default `snapshot()` reads only **controllers** (fast: no instrument queries for readouts). Pass `include_readouts=True` to also acquire readout values — useful for a quick sanity check, but slow if readouts involve VNA sweeps or camera captures:

```python
bench.snapshot(include_readouts=True)
# Also prints readout rows:
# lockin_X  scalar  0.0023   V
# S21       trace   [...]    dB
```

Print only specific parameters or readouts by name:

```python
bench.snapshot(["Vgt", "lockin_X"])
# explicit list overrides include_readouts; reads exactly what is named
```

#### Saving and loading bench configuration

`bench.save()` writes the full bench configuration to a YAML file — instrument class paths, connection args, all controllers and `instrument + attr` readouts, plus source hints for any custom callables. `Bench.load()` reads it back and re-instantiates everything via Python's `importlib` (no `eval`). Entries that cannot be auto-wired (custom `get_func`, failed connections, `__main__` classes) are collected as *stubs* — call `bench.show_stubs()` to review them.

```python
bench.save("bench.yaml")          # write once

# In any future session:
bench = Bench.load("bench.yaml")  # instruments connected, controllers wired
# Bench loaded ← bench.yaml  ·  2 stubs — call bench.show_stubs()

bench.show_stubs()    # compact table with source hints for custom callables
bench["Vgt"] = 0.5
```

See the [Saving and Loading Bench Configuration](#saving-and-loading-bench-configuration) cookbook section for the full workflow, generated YAML format, and stub re-registration.

#### Removing instruments, controllers, and readouts

```python
bench.remove_controller("Vgt")
bench.remove_readout("S21")
bench.remove_instrument("keithley")
# also removes any controllers that depend on it
```

#### Accessing raw objects

The `Controller` and `Readout` objects are accessible when you need them (e.g., for `Sweep` setup):

```python
bench.controllers["Vgt"]   # -> Controller object
bench.readouts["S21"]      # -> Readout object
bench.instruments["keithley"]  # -> InstrumentAdapter object
```

---

### Step 4: Procedures

Procedures define **what** to do. Two types are available:

#### Procedure (sweeps)

```python
proc = Procedure(
    name="gate_sweep",
    bench=bench,
    sweeps=[
        Sweep("Vgt", np.linspace(0, 1, 101)),
    ],
    readouts=["lockin_X"],
    settle_time=0.01,          # 10ms settle after each set
    tags=["transport", "1d"],
    metadata={"field": "0T"},
)
```

**Sweep ordering:** `sweeps[0]` is the **outermost** (slowest) axis, `sweeps[-1]` is the **innermost** (fastest). The dimensionality is determined by the number of sweeps:

| Sweeps              | Dimensionality |
|---------------------|----------------|
| `[Sweep(Vgt, ...)]` | 1D             |
| `[Sweep(Vgt, ...), Sweep(fac, ...)]` | 2D |
| `[Sweep(Vgt, ...), Sweep(fac, ...), Sweep(power, ...)]` | 3D |

You can reference controllers by name (`"Vgt"`) or by the `Controller` object directly.

#### Write mode

The `write_mode` parameter controls when data is flushed to disk. This maps directly to the zarro `WriteType`:

```python
from orchid import WriteMode

# Default — write after every point (safest, most I/O)
proc = Procedure(..., write_mode=WriteMode.POINTWISE)

# Write after each inner sweep completes (good for 2D/3D scans)
proc = Procedure(..., write_mode=WriteMode.SWEEPWISE)

# Write after each 2D plane completes (good for 3D scans)
proc = Procedure(..., write_mode=WriteMode.PLANEWISE)

# Buffer everything, write once at the end (fastest, but data lost on crash)
proc = Procedure(..., write_mode=WriteMode.ALL)
```

| Mode        | Writes to disk                | zarro method          | Min. sweeps | Trade-off                        |
|-------------|-------------------------------|-----------------------|-------------|----------------------------------|
| `POINTWISE` | After every point             | `write_point(index)`  | 1           | Safest; most I/O overhead        |
| `SWEEPWISE` | After each inner sweep        | `write_trace(index)`  | 1           | Good balance for 2D/3D scans     |
| `PLANEWISE` | After each 2D plane           | `write_image(index)`  | 2           | Good for 3D scans                |
| `ALL`       | Once at the end               | `write_all(data)`     | 1           | Fastest; all data lost on crash  |

**Example — 2D scan with SWEEPWISE:**

```python
proc = Procedure(
    name="gate_freq_map",
    bench=bench,
    sweeps=[
        Sweep("Vgt", np.linspace(0, 5, 50)),     # outer (slow)
        Sweep("fac", np.linspace(1e3, 1e6, 200)), # inner (fast)
    ],
    readouts=["lockin_X"],
    write_mode=WriteMode.SWEEPWISE,
    # Data for all 200 inner points is buffered in memory,
    # then written as one trace after each inner sweep completes.
    # Only 50 writes to disk instead of 10,000.
)
```

**Example — 3D scan with PLANEWISE:**

```python
proc = Procedure(
    name="power_gate_freq_cube",
    bench=bench,
    sweeps=[
        Sweep("power", np.linspace(-30, 0, 20)),    # outer (slowest)
        Sweep("Vgt", np.linspace(0, 5, 50)),         # middle
        Sweep("fac", np.linspace(1e3, 1e6, 200)),    # inner (fastest)
    ],
    readouts=["lockin_X"],
    write_mode=WriteMode.PLANEWISE,
    # The full 50x200 plane is buffered, then written at once
    # after the two inner sweeps complete. Only 20 writes to disk.
)
```

#### MonitorProcedure (time-series)

```python
monitor = MonitorProcedure(
    name="temperature_log",
    bench=bench,
    readouts=["lockin_X", "temperature"],
    interval=0.5,        # read every 0.5 seconds
    duration=60.0,       # run for 60 seconds (None = until stopped)
    stop_condition=lambda data: data["temperature"] > 4.2,
    tags=["monitoring"],
)
```

---

### Step 5: Running Experiments

The `ExperimentRunner` executes procedures and handles data saving:

```python
runner = ExperimentRunner()

# Sweep experiment
data_dir = runner.run(proc)            # sync
data_dir = await runner.arun(proc)     # async

# Monitor experiment
data_dir = runner.run_monitor(monitor)            # sync (blocking)
data_dir = await runner.arun_monitor(monitor)     # async

# Monitor in background (non-blocking) — change parameters while running
runner.run_monitor(monitor, background=True)
bench["Vgt"] = 0.5   # change parameters from the next cell
data_dir = runner.stop_monitor()  # stop and get data path
```

The runner saves a `procedure.yaml` to the data directory. To also print a formatted summary, pass `print_summary=True`:

```python
runner.run(proc, print_summary=True)
```

Output:

```
Experiment : gate_sweep
Tags       : cooldown_3
────────────────────────────────────────────────────
Sweeps  2D · 5,000 points
  [0] Vbias         -1.0 →  1.0       100 pts  V
  [1] Vgt           -3.0 →  3.0        50 pts  V
────────────────────────────────────────────────────
Readouts
  lockin_X           scalar   V
  lockin_Y           scalar   V
────────────────────────────────────────────────────
Settings
  write_mode    sweepwise
  settle_time   10 ms
  error_policy  stop_and_save
  est. duration ~50 s
```

You can also call `proc.summary()` manually at any time without running.

#### Interrupt handling

Pressing `Ctrl+C` (or interrupting the Jupyter kernel) cleanly stops the experiment, saves all collected data with `status: "interrupted"` in the metadata, and prints a short message — no tracebacks:

```
my_sweep:   5%|█         | 5/100 [00:02<00:45]

Experiment 'my_sweep' interrupted. Data saved to: data/0003
```

The runner:
1. Creates a numbered output directory (e.g., `data/0001/`)
2. Builds a zarro `MeasurementSchema` from the procedure
3. Executes the sweep loops with a tqdm progress bar
4. At each point: sets parameters, waits `settle_time`, reads all readouts
5. Writes data via `ZarrWriter.write_point()` (or `StreamingWriter.append()` for monitors)
6. Saves metadata on completion (or on error)
7. Returns the `Path` to the data directory

#### How write modes execute internally

The runner uses recursive sweep loops that change behavior depending on the `write_mode`. Here is how each mode works for a 3D scan with `sweeps = [power, Vgt, fac]`:

**POINTWISE** — flat recursion, writes at every leaf:

```
for each power[i]:          # axis 0
  for each Vgt[j]:          # axis 1
    for each fac[k]:        # axis 2
      set, settle, measure
      write_point((i,j,k))  # one write per point
```

**SWEEPWISE** — recurses through outer axes, buffers the innermost:

```
for each power[i]:          # axis 0 (outer loop)
  for each Vgt[j]:          # axis 1 (outer loop)
    buffer = []
    for each fac[k]:        # axis 2 (buffered)
      set, settle, measure
      buffer[k] = data
    write_trace((i,j), buffer)  # one write per inner sweep
```

The runner walks axes 0 and 1 with normal recursion. When it reaches the innermost axis (2), it switches to a buffering loop that collects all `fac` points into a numpy array, then writes them in one shot via `write_trace(outer_index, buffer)`. For a 20x50x200 scan, that is 1,000 writes instead of 200,000.

**PLANEWISE** — recurses through outer axes, buffers the two innermost:

```
for each power[i]:          # axis 0 (outer loop)
  buffer = np.empty((50, 200))
  for each Vgt[j]:          # axis 1 (buffered)
    for each fac[k]:        # axis 2 (buffered)
      set, settle, measure
      buffer[j, k] = data
  write_image((i,), buffer)  # one write per 2D plane
```

Same idea but buffers a full 2D plane (the two innermost axes). For a 20x50x200 scan, only 20 writes to disk.

**ALL** — buffers everything, single write:

```
buffer = np.empty((20, 50, 200))
for each power[i]:
  for each Vgt[j]:
    for each fac[k]:
      set, settle, measure
      buffer[i, j, k] = data
write_all(buffer)             # one single write
```

#### Experiment ID numbering

By default, data is saved in auto-numbered subdirectories:

```
data/
  0001/       # first run
    vault.zarr/
    metadata.yaml
  0002/       # second run
    vault.zarr/
    metadata.yaml
```

Disable with:

```python
runner = ExperimentRunner(use_experiment_id=False)
# saves to data/<procedure_name>/ instead
```

---

### Step 6: Reading Data Back

All data is stored using the [zarro](https://github.com/eeroqlab/zarro) package in Zarr v3 format:

```python
import zarr
import yaml

# Open vault
z = zarr.open("./data/0001/vault.zarr", mode="r")

# List arrays
print(list(z.keys()))    # ['Vgt', 'lockin_X']

# Read arrays (standard NumPy indexing)
vgt = z["Vgt"][:]          # control values, shape (101,)
signal = z["lockin_X"][:]  # measured data, shape (101,)

# For 2D: shape is (outer_len, inner_len)
data_2d = z["lockin_X"][:]  # shape (50, 101)
row_5 = z["lockin_X"][5, :]  # one outer slice

# Read metadata
from orchid import read_metadata
meta = read_metadata("./data/0001")
print(meta["sample"])     # "chip_A1"
print(meta["status"])     # "completed"
```

#### Reading procedure info

Every run saves a `procedure.yaml` alongside `metadata.yaml`. Use `read_procedure()` to retrieve and display it:

```python
from orchid import read_procedure

# Prints summary table and returns the dict
d = read_procedure("./data/0001")

# Just the dict, no print
d = read_procedure("./data/0001", print_summary=False)
print(d["total_points"])          # 5000
print(d["settings"]["write_mode"])  # "sweepwise"
```

The dict structure mirrors what `proc.to_dict()` returns — full sweep ranges, readout specs, settings, estimated duration, and hook source code.

#### Updating metadata after an experiment

Use `update_metadata` to annotate an experiment with additional information at any time after it completes — notes, analysis results, quality flags, etc.:

```python
from orchid import update_metadata, read_metadata

data_dir = runner.run(proc)

# Later (even in a different script/session):
update_metadata(data_dir, notes="clean Coulomb diamonds", quality="A")
update_metadata(data_dir, T_mc=0.015, B_field="1T")

# Keys are merged — existing keys are overwritten, new keys are added
meta = read_metadata(data_dir)
print(meta["notes"])      # "clean Coulomb diamonds"
print(meta["T_mc"])       # 0.015
```

---

### Step 7: Live Plotting

Orchid provides built-in live plotting using a Dash server that opens in a **separate browser window**. The plots update in real time as data is acquired. Create a `DashPlotter` (or its alias `LivePlotter`) with one or more `PlotSpec` objects and pass it to the runner.

The plotter architecture separates data logic from the display backend: `PlotterBase` holds all figure-building and data-update code; `DashPlotter` adds the Dash/Werkzeug server on top. You can subclass `PlotterBase` directly to add other backends (e.g. Taipy) without touching the runner.

#### 1D sweep — line plot

```python
from orchid import LivePlotter, PlotSpec

plotter = LivePlotter([PlotSpec(x="Vgt", y="lockin_X")])
runner.run(proc, plotter=plotter)
# A live line plot appears and updates as data is acquired
```

#### 2D sweep — heatmap

For 2D scans, specify `x` (inner sweep), `y` (outer sweep), and `z` (readout for color). The heatmap fills row-by-row as each inner sweep completes:

```python
plotter = LivePlotter([PlotSpec(x="fac", y="Vgt", z="lockin_X")])
runner.run(proc_2d, plotter=plotter)
```

#### Multiple subplots

```python
plotter = LivePlotter([
    PlotSpec(x="Vgt", y="lockin_X"),
    PlotSpec(x="Vgt", y="lockin_Y"),
])
runner.run(proc, plotter=plotter)
```

#### Multiple traces in one subplot

Pass a list to `y` to overlay several readouts in the same axes — useful for comparing
channels (e.g. X and Y from a lock-in, or signal vs. reference):

```python
plotter = LivePlotter([
    PlotSpec(x="Vgt", y=["lockin_X", "lockin_Y"]),   # both on one subplot
])
runner.run(proc, plotter=plotter)
```

Mix single and multi-trace subplots freely:

```python
plotter = LivePlotter([
    PlotSpec(x="Vgt", y=["lockin_X", "lockin_Y"]),   # XY on top
    PlotSpec(x="Vgt", y="temperature"),               # temperature below
])
```

All readout names passed in `y` must be present in the procedure's `readouts` list.
If a readout is missing from a particular data point (e.g. due to an error), that
trace is silently skipped for that update.

#### Heatmap colorscale

Override the default colorscale per subplot with the `colorscale` argument
(any Plotly colorscale name or custom list). Leave it as `None` to use the
active template default (set by `apply_theme`):

```python
plotter = LivePlotter([
    PlotSpec(x="fac", y="Vgt", z="lockin_X", colorscale="RdBu_r"),
    PlotSpec(x="fac", y="Vgt", z="lockin_Y", colorscale="Viridis"),
])
```

#### Live trace (VNA / spectrum at each step)

When `x` is a fixed array (e.g. a frequency list), the plot type auto-detects as `live_trace` — a line plot whose y values are **overwritten** at each sweep step to show the most recent trace. Set `update_every="point"` so it refreshes at every measurement, and use `z_col` to pick a column from a multi-column (`IMAGE`) readout:

```python
flist = np.linspace(4e9, 8e9, 1601)

# VNA trace readout (IMAGE, 3 columns: f, mag, phase)
bench.add_readout("S21", kind="image", shape=(3, 1601),
    get_func=vna.get_trace, contains=["f", "mag", "phase"])

# Live line plot: shows current VNA magnitude, refreshing at each power step
plotter = LivePlotter([
    PlotSpec(x=flist, y="S21", z_col="mag", update_every="point"),
])
runner.run(proc_power_sweep, plotter=plotter)
```

For a plain `TRACE` readout (1D array, no columns), `z_col` is not needed:

```python
bench.add_readout("mag", kind="trace", shape=(1601,), get_func=vna.get_magnitude)

plotter = LivePlotter([
    PlotSpec(x=flist, y="mag", update_every="point"),
])
```

#### Trace heatmap (VNA)

When `y` is a fixed array (e.g. frequencies) and `x` is a sweep parameter, the plot auto-detects as `trace_heatmap` — a heatmap that **accumulates** one column per sweep step. The x-axis is the sweep parameter (e.g. VNA power) and the y-axis is the fixed array (frequencies). Useful for visualising how a VNA trace evolves with a control parameter:

```python
flist = np.linspace(4e9, 8e9, 1601)

bench.add_readout("S21", kind="image", shape=(3, 1601),
    get_func=vna.get_trace, contains=["f", "mag", "phase"])

# Heatmap: x=power step, y=frequency, color=magnitude
plotter = LivePlotter([
    PlotSpec(x="vna_power", y=flist, z="S21", z_col="mag",
             update_every="point"),
])
runner.run(proc_power_sweep, plotter=plotter)
```

> **Note on axes:** `x` is the sweep parameter — it maps to the *x-axis* of the heatmap (each step adds a new column). `y` is the fixed array — it maps to the *y-axis*. This matches how VNA waterfall plots are normally displayed (frequency on y, power on x, color = magnitude).

**`z_col` selector:**

| `z_col` value | Behaviour |
|---------------|-----------|
| `None`        | Use the whole array (valid for `TRACE` readouts) |
| `int`         | Use that column index (0-based) |
| `str`         | Look up column by name in `readout.contains` list |

Combine `live_trace` and `trace_heatmap` in one plotter for a complete VNA dashboard:

```python
plotter = LivePlotter([
    PlotSpec(x=flist, y="S21", z_col="mag", update_every="point"),   # current trace
    PlotSpec(x="vna_power", y=flist, z="S21", z_col="mag",
             update_every="point"),                                    # waterfall
])
runner.run(proc_power_sweep, plotter=plotter)
```

#### Time-series monitoring

When `x="_time"`, the x-axis shows elapsed time starting from zero. The axis label auto-scales: seconds (s) for < 2 min, minutes (min) for < 2 hr, hours (hr) beyond that.

```python
plotter = LivePlotter([PlotSpec(x="_time", y="temperature")])
runner.run_monitor(monitor, plotter=plotter)
```

With background mode — change parameters while monitoring:

```python
plotter = LivePlotter([PlotSpec(x="_time", y="lockin_X")])
runner.run_monitor(monitor, plotter=plotter, background=True)

# In the next cell:
bench["Vgt"] = 0.5   # change gate voltage while monitoring

# When done:
data_dir = runner.stop_monitor()
```

#### Custom update function

When `update_func` is set on a PlotSpec, the default line/heatmap logic is skipped entirely. Instead, your function is called every time new data is written. It receives three arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `fig` | `FigureWidget` | The plotly figure. `fig.data[0]` is the trace for the first PlotSpec, `fig.data[1]` for the second, etc. |
| `index` | `tuple` | Current sweep index, e.g. `(3,)` for 1D, `(2, 5)` for 2D. For monitors, this is `sample_idx` (int). |
| `data` | `dict` | Readout name to value(s) just measured. Scalar for POINTWISE, array for SWEEPWISE. |

**Plot magnitude of a complex trace:**

```python
import numpy as np

def plot_s21_mag(fig, index, data):
    s21 = data["S21"]
    mag_db = 20 * np.log10(np.abs(s21))
    fig.data[0].y = mag_db

plotter = LivePlotter([PlotSpec(x="freq", y="S21", update_func=plot_s21_mag)])
runner.run(proc, plotter=plotter)
```

**Accumulate points with color coding:**

```python
xs, ys = [], []

def accumulate(fig, index, data):
    xs.append(index[0])
    ys.append(data["lockin_X"])
    colors = ["red" if v > 0.5 else "blue" for v in ys]

    with fig.batch_update():
        fig.data[0].x = xs
        fig.data[0].y = ys
        fig.data[0].marker.color = colors

plotter = LivePlotter([PlotSpec(x="Vgt", y="lockin_X", update_func=accumulate)])
runner.run(proc, plotter=plotter)
```

**2D heatmap with derivative (SWEEPWISE):**

```python
import numpy as np

# In SWEEPWISE mode, data["lockin_X"] is a full row (array)
z_matrix = np.full((50, 200), np.nan)

def plot_derivative(fig, index, data):
    row = index[0]  # outer sweep index
    trace = data["lockin_X"]  # shape (200,)
    z_matrix[row, :] = np.gradient(trace)
    fig.data[0].z = z_matrix

plotter = LivePlotter([PlotSpec(x="fac", y="lockin_X", update_func=plot_derivative)])
runner.run(proc_2d, plotter=plotter)
```

**Key points:**
- `fig.data[N]` corresponds to the Nth `PlotSpec` in the list
- Use `with fig.batch_update():` when updating multiple properties at once (prevents flicker)
- Capture external state via closures (like `z_matrix` or `xs, ys` above)
- For POINTWISE: called every point, `data` values are scalars
- For SWEEPWISE: called every inner sweep, `data` values are arrays

#### Plot update frequency vs write mode

The plot update frequency (`update_every`) is **independent** of the data write mode (`write_mode`). You can save data efficiently in large batches while still seeing every point appear live:

```python
# Save per sweep (fast I/O), but plot every point (real-time visual)
proc = Procedure(..., write_mode=WriteMode.SWEEPWISE)
plotter = LivePlotter([PlotSpec(x="Vgt", y="sig", update_every="point")])
runner.run(proc, plotter=plotter)

# Save per sweep, plot per sweep (both aligned — default)
plotter = LivePlotter([PlotSpec(x="Vgt", y="sig", update_every="sweep")])

# Save per plane, plot per sweep (see rows fill in on a 3D scan)
proc = Procedure(..., write_mode=WriteMode.PLANEWISE)
plotter = LivePlotter([PlotSpec(x="fac", y="sig", update_every="sweep")])

# Multiple subplots with different update rates
plotter = LivePlotter([
    PlotSpec(x="Vgt", y="lockin_X", update_every="point"),   # real-time
    PlotSpec(x="fac", y="S21", update_every="sweep"),         # per row
])
```

| `update_every` | Updates on                           |
|-----------------|--------------------------------------|
| `"point"`       | Every measurement point              |
| `"sweep"`       | Each inner sweep completion          |
| `"plane"`       | Each 2D plane completion (3D scans)  |

#### Configuration

```python
plotter = LivePlotter(
    [PlotSpec(x="Vgt", y="sig")],
    port=8050,            # Dash server port (default 8050)
    height=400,           # pixels per subplot
    width=800,            # figure width
    open_browser=True,    # auto-open browser (default True)
    update_interval=500,  # poll interval in ms (default 500)
)
```

The Dash server runs on a background daemon thread. After the experiment completes, the server stops refreshing (zoom/pan is preserved) but stays running for inspection. Shut it down to free the port with:

```python
plotter.stop()
```

A `LivePlotter` can be reused across experiments — `setup()` automatically stops the previous server and resets all state:

```python
plotter = LivePlotter([PlotSpec(x="Vgt", y="sig", update_every="point")])

runner.run(proc1, plotter=plotter)  # first experiment
runner.run(proc2, plotter=plotter)  # fresh plot, old server cleaned up
```

---

## Cookbook

### 1D Sweep

```python
proc = Procedure(
    name="iv_curve",
    bench=bench,
    sweeps=[Sweep("Vbias", np.linspace(-1, 1, 201))],
    readouts=["current"],
    settle_time=0.01,
)
data_dir = ExperimentRunner().run(proc)
```

### 2D Sweep

```python
proc = Procedure(
    name="gate_frequency_map",
    bench=bench,
    sweeps=[
        Sweep("Vgt", np.linspace(0, 5, 50)),    # outer (slow)
        Sweep("fac", np.linspace(1e3, 1e6, 200)),  # inner (fast)
    ],
    readouts=["lockin_X", "lockin_Y"],
    settle_time=0.005,
)
data_dir = ExperimentRunner().run(proc)
# Data shape: lockin_X -> (50, 200)
```

### 2D Sweep with Snake Scan

Snake scan reverses the inner sweep direction on alternating outer iterations, reducing backlash and improving speed:

```python
proc = Procedure(
    name="gate_map_snake",
    bench=bench,
    sweeps=[
        Sweep("Vgt", np.linspace(0, 5, 50)),
        Sweep("fac", np.linspace(1e3, 1e6, 200)),
    ],
    readouts=["lockin_X"],
    snake=True,   # <-- enables snake scan
)
```

```
Outer index 0: fac sweeps  0 → 199   (forward)
Outer index 1: fac sweeps  199 → 0   (backward)
Outer index 2: fac sweeps  0 → 199   (forward)
...
```

### Hysteresis (Forward + Backward)

Use `reverse=True` on a `Sweep` to automatically append the reversed values:

```python
proc = Procedure(
    name="hysteresis",
    bench=bench,
    sweeps=[
        Sweep("Vgt", np.linspace(0, 1, 100), reverse=True),
        # values become: [0, 0.01, ..., 1.0, 1.0, 0.99, ..., 0]
        # total length: 200
    ],
    readouts=["lockin_X"],
)
```

### Time-Series Monitoring

Monitor instruments in real time without sweeping:

```python
monitor = MonitorProcedure(
    name="stability_check",
    bench=bench,
    readouts=["lockin_X", "temperature"],
    interval=1.0,       # 1 second between reads
    duration=3600.0,    # 1 hour
    tags=["stability"],
)
data_dir = ExperimentRunner().run_monitor(monitor)
```

With a stop condition:

```python
monitor = MonitorProcedure(
    name="cooldown_watch",
    bench=bench,
    readouts=["temperature"],
    interval=5.0,
    duration=None,  # run indefinitely
    stop_condition=lambda data: data["temperature"] < 0.01,
)
```

Stop manually with `Ctrl+C` — data is always saved.

#### Controlling disk flush frequency

Samples are buffered in memory and written to disk in batches. The `chunk_size` parameter
controls how many samples accumulate before a flush:

```python
# Default: flush every 256 samples
monitor = MonitorProcedure(..., interval=1.0)

# Flush every 10 samples — safer on crash, more I/O
monitor = MonitorProcedure(..., interval=0.1, chunk_size=10)

# Flush every 1000 samples — less I/O, good for fast logging
monitor = MonitorProcedure(..., interval=0.01, chunk_size=1000)

# Flush after every sample — maximum safety
monitor = MonitorProcedure(..., chunk_size=1)
```

Up to `chunk_size - 1` samples may be in memory at any time. On a clean exit or `Ctrl+C`,
the remaining buffer is always flushed before the file is closed, so no data is lost under
normal conditions. Only an abrupt process kill (power loss, `SIGKILL`) could lose buffered
samples.

### Controller Limits

Attach soft or hard bounds to any controller. When `set()` or `aset()` is called with a value outside the range, the value is clamped and the violation is handled according to the `limit_policy`.

```python
from orchid import LimitPolicy

# Soft limit — warn once, then log silently (default)
bench.add_controller("Vgt", instrument="yoko", attr="voltage", unit="V",
                     limits=(-3.0, 3.0))

# Hard safety limit — stop the experiment immediately
bench.add_controller("heater", instrument="tc", attr="power", unit="W",
                     limits=(0.0, 0.1), limit_policy=LimitPolicy.RAISE)

# Silent log — inspect after the run
bench.add_controller("Vbg", instrument="yoko", attr="voltage", unit="V",
                     limits=(-5.0, 5.0), limit_policy=LimitPolicy.LOG)
```

The runner resets all limit logs at the start of each run and saves `limit_log.yaml` when violations occurred:

```python
runner.run(proc)

# Read violations after the run:
from orchid import read_limit_log

entries = read_limit_log(data_dir)
for e in entries:
    print(f"{e['controller']}[{e['index']}]: {e['requested']} → {e['clamped']}")
# Vgt[(42,)]: 3.2 → 3.0
# Vgt[(43,)]: 3.5 → 3.0
```

Inspect live (without a file):

```python
bench.controllers["Vgt"].limit_log
# [LimitEntry(index=(42,), requested=3.2, clamped=3.0),
#  LimitEntry(index=(43,), requested=3.5, clamped=3.0)]

len(bench.controllers["Vgt"].limit_log)  # how many points were clamped
```

`limit_log.yaml` is only written when at least one violation occurred. Returns `[]` if the file does not exist.

---

### Controller Bindings

Use `Bench.add_controller_binding()` when one logical control should fan out to several existing physical controllers. Setting the binding applies the same value to every bound controller. Reading the binding returns a dictionary of current bound-controller values.

```python
bench.add_controller("stp", instrument="yoko1", attr="voltage", unit="V")
bench.add_controller("stm", instrument="yoko2", attr="voltage", unit="V")

bench.add_controller_binding("reservoir_v", ["stp", "stm"])

bench["reservoir_v"] = -0.25
print(bench["reservoir_v"])
# {"stp": -0.25, "stm": -0.25}
```

Bindings are regular `Controller` objects, so they can be used in `Sweep` or `MultiSweep`. Bound physical controllers still enforce their own limits when they receive the value.

`add_controller_binding()` raises `KeyError` if any bound name is missing and `ValueError` if the bound-controller list is empty.

---

### Controller Event Logging

While a background monitor is running, every `bench["param"] = value` call is automatically recorded as a timestamped event. Events are saved to `events.yaml` alongside the data when the monitor finishes.

```python
runner.run_monitor(monitor, plotter=plotter, background=True)

# These are recorded automatically:
bench["Vgt"] = 0.5    # t=12.3s
bench["Vgt"] = 1.0    # t=45.7s
bench["fac"] = 3000   # t=60.1s

data_dir = runner.stop_monitor()
```

**Reading events back:**

```python
from orchid import read_events

events = read_events(data_dir)
for e in events:
    print(f"t={e['elapsed']:.1f}s  {e['param']} → {e['value']}")
# t=12.3s  Vgt → 0.5
# t=45.7s  Vgt → 1.0
# t=60.1s  fac → 3000
```

Each event dict contains:

| Key       | Description                          |
|-----------|--------------------------------------|
| `time`    | Unix timestamp (float)               |
| `elapsed` | Seconds from monitor start (float)   |
| `param`   | Parameter name (str)                 |
| `value`   | Value that was set                   |

**Live plot integration:**

If a `LivePlotter` is passed with a `PlotSpec(x="_time", ...)`, each parameter change appears as a vertical dashed line annotated with the parameter name and value — visible in real time as you change parameters:

```python
plotter = LivePlotter([PlotSpec(x="_time", y="lockin_X")])
runner.run_monitor(monitor, plotter=plotter, background=True)

bench["Vgt"] = 0.5   # dashed marker line appears on the plot immediately
```

If no events occurred during the run, no `events.yaml` is written.

---

### Background Monitoring

Run monitoring in the background so you can change parameters from other cells:

```python
runner = ExperimentRunner()
plotter = LivePlotter([PlotSpec(x="_time", y="lockin_X")])

# Cell 1: start
runner.run_monitor(monitor, plotter=plotter, background=True)
```

```python
# Cell 2: change parameters while monitoring
bench["Vgt"] = 0.5
```

```python
# Cell 3: change again
bench["Vgt"] = 1.0
```

```python
# Cell 4: stop and save
data_dir = runner.stop_monitor()
```

### Custom Hooks

Hooks let you inject custom logic at specific points in the experiment:

```python
def ramp_field():
    """Ramp magnet before experiment starts."""
    magnet.set_field(1.0)
    time.sleep(10)

def log_point(index):
    """Print every 100th point."""
    if sum(index) % 100 == 0:
        print(f"Point {index}: Vgt={bench['Vgt']:.3f}")

proc = Procedure(
    name="with_hooks",
    bench=bench,
    sweeps=[Sweep("Vgt", np.linspace(0, 1, 500))],
    readouts=["lockin_X"],
    before_experiment=ramp_field,
    after_point=log_point,
)
```

Available hooks:

| Hook                | When called                  | Signature                    |
|---------------------|------------------------------|------------------------------|
| `before_experiment` | Once, before first point     | `() -> None`                 |
| `after_experiment`  | Once, after last point       | `() -> None`                 |
| `before_sweep`      | Before each sweep axis starts| `(axis_index) -> None`       |
| `after_sweep`       | After each sweep axis ends   | `(axis_index) -> None`       |
| `before_point`      | Before each measurement      | `(index_tuple) -> None`      |
| `after_point`       | After each measurement       | `(index_tuple) -> None`      |

All hooks support both sync and async callables.

### Mixed Readouts (Scalar + Trace)

Record different data types in a single experiment:

```python
bench.add_readout("lockin_X", kind="scalar", get_func=lockin.read_x, unit="V")
bench.add_readout("S21", kind="trace", shape=(1601,), get_func=vna.get_trace, unit="dB")
bench.add_readout("frame", kind="image", shape=(480, 640), get_func=camera.snap, unit="counts")

proc = Procedure(
    name="mixed",
    bench=bench,
    sweeps=[Sweep("Vgt", np.linspace(0, 1, 50))],
    readouts=["lockin_X", "S21", "frame"],
)
# Resulting shapes in vault:
#   lockin_X -> (50,)
#   S21      -> (50, 1601)
#   frame    -> (50, 480, 640)
```

### Async Usage

For use inside Jupyter notebooks or async applications:

```python
runner = ExperimentRunner()

# In a notebook cell:
data_dir = await runner.arun(proc)

# Or for monitoring:
data_dir = await runner.arun_monitor(monitor)
```

Async instrument drivers are supported natively — if your get/set functions are `async def`, orchid will `await` them directly instead of wrapping in a thread.

### Inspecting a Procedure

#### Before running — print summary manually

```python
proc.summary()
```

```
Experiment : gate_sweep
Tags       : cooldown_3
────────────────────────────────────────────────────
Sweeps  2D · 5,000 points
  [0] Vbias         -1.0 →  1.0       100 pts  V
  [1] Vgt           -3.0 →  3.0        50 pts  V
────────────────────────────────────────────────────
Readouts
  lockin_X           scalar   V
  lockin_Y           scalar   V
────────────────────────────────────────────────────
Settings
  write_mode    sweepwise
  settle_time   10 ms
  error_policy  stop_and_save
  est. duration ~50 s
────────────────────────────────────────────────────
Hooks
  after_point              auto_phase
                           "Re-phases lockin every 10 rows."
```

Call `proc.summary()` at any time to print the table. Pass `print_summary=True` to `runner.run()` to print it automatically before each run.

#### After running — read from data directory

```python
from orchid import read_procedure

# Prints summary and returns dict
d = read_procedure("./data/0042")

# Access specific fields
print(d["total_points"])                # 5000
print(d["settings"]["settle_time"])     # 0.01
print(d["sweeps"][0]["min"])            # -1.0
print(d["hooks"]["after_point"]["doc"]) # "Re-phases lockin every 10 rows."
```

#### Hook source in procedure.yaml

If a named function (not a lambda) is registered as a hook, its full source is captured:

```yaml
# procedure.yaml (excerpt)
hooks:
  after_point:
    name: auto_phase
    doc: Re-phases lockin every 10 rows.
    source: |
      def auto_phase(index):
          """Re-phases lockin every 10 rows."""
          if index[-1] == 0:
              lockin.auto_phase()
  before_experiment: null
  after_experiment: null
```

Lambdas are noted but not serialised:

```yaml
  before_point:
    name: "<lambda>"
    note: "lambda — source not recorded"
```

Source extraction works for functions defined in `.py` files and in Jupyter notebook cells (IPython keeps cell source in memory). If source is unavailable, the function name and module are recorded as a fallback.

### Saving and Loading Bench Configuration

`bench.save()` / `Bench.load()` persist the entire bench setup as a human-readable YAML file, so you can reconnect to instruments and recreate all controllers and readouts in one line — no copy-pasting setup code between notebooks.

#### What is and isn't serialized

| Thing | Serialized | On load |
|---|---|---|
| Instrument class + connection args | ✅ | Auto-connected via `importlib` |
| Controllers (`instrument + attr`) | ✅ | Fully wired |
| Readouts (`instrument + attr`) | ✅ | Fully wired |
| Custom `get_func` / `set_func` | ⚠️ source hint | Collected as stub — re-register manually |
| `__main__` instruments | ⚠️ stub | Collected as stub — update YAML `class` field |
| Failed connections | ⚠️ stub | Collected as stub — fix address and reload |

Stubs are collected silently during `load()`. A single summary line tells you how many there are, and `bench.show_stubs()` displays them in a compact table.

#### Full round-trip example

```python
# ── Session 1: first setup ─────────────────────────────────────────
from pymeasure.instruments.keithley import Keithley2400
from qcodes.instrument_drivers.stanford_research import SR830

smu  = Keithley2400("GPIB::24")
lockin = SR830("lockin", "GPIB::8")

bench = Bench(data_root="./data", metadata={"sample": "chip_A1"})

bench.add_instrument("smu",   smu,   args=["GPIB::24"])
bench.add_instrument("lockin", lockin, args=["lockin", "GPIB::8"])

bench.add_controller("Vbias", instrument="smu",   attr="source_voltage", unit="V",  limits=(-1, 1))
bench.add_controller("fac",   instrument="lockin", attr="frequency",      unit="Hz")
bench.add_readout(   "X",     kind="scalar", instrument="lockin", attr="x", unit="V")

bench.save("bench.yaml")   # ← write once
```

```python
# ── Session 2 onwards: one-liner ──────────────────────────────────
bench = Bench.load("bench.yaml")
# Bench loaded ← bench.yaml

bench["Vbias"] = 0.1
bench.snapshot()
```

#### Generated YAML

```yaml
bench:
  data_root: ./data
  metadata:
    sample: chip_A1

instruments:
  smu:
    class: pymeasure.instruments.keithley.Keithley2400
    args: [GPIB::24]
    kwargs: {}
    backend: pymeasure
  lockin:
    class: qcodes.instrument_drivers.stanford_research.SR830
    args: [lockin, GPIB::8]
    kwargs: {}
    backend: qcodes

controllers:
  Vbias:
    instrument: smu
    attr: source_voltage
    unit: V
    limits: [-1.0, 1.0]
    limit_policy: warn
  fac:
    instrument: lockin
    attr: frequency
    unit: Hz
    limits: null
    limit_policy: warn

readouts:
  X:
    kind: scalar
    instrument: lockin
    attr: x
    shape: null
    unit: V
    contains: null
```

#### Stubs — entries that need manual re-registration

`load()` is quiet: it collects everything it cannot auto-wire as a *stub* and prints a single summary line:

```
Bench loaded ← bench.yaml  ·  3 stubs — call bench.show_stubs()
```

Call `bench.show_stubs()` to see a compact table. Source strings for custom callables are truncated to the first line by default:

```python
bench.show_stubs()
```

```
Name    Kind        Reason                       Info          Source (hint)
------  ----------  ---------------------------  ------------  ----------------------------------------
smu     instrument  connection failed: No VISA   —             —
fac     controller  custom get_func/set_func     Hz  100…1e6   get: lambda: lockin.driver.frequency…
                                                               set: lambda v: setattr(lockin.driver…
power   readout     custom get_func              scalar  W     get: lambda: smu.current * smu.voltage…
  (pass full_source=True to see complete source)
```

Pass `full_source=True` to unfold multiline source in full:

```python
bench.show_stubs(full_source=True)
```

For programmatic access:

```python
bench.stubs          # dict keyed by name
bench.stubs["fac"]   # {'kind': 'controller', 'get_func_src': '...', ...}
```

#### Re-registering stubs

After reviewing `show_stubs()`, re-add custom callables by hand:

```python
# Re-add a controller with custom get/set (unit and limits from the YAML hint)
bench.add_controller("fac",
    get_func=lambda: lockin.driver.frequency,
    set_func=lambda v: setattr(lockin.driver, "frequency", v),
    unit="Hz", limits=(100, 1e6))

# Re-add a custom readout
bench.add_readout("power", kind="scalar",
    get_func=lambda: smu.current * smu.voltage, unit="W")
```

#### `__main__` instruments

Instruments defined inline in a Jupyter notebook have `class = __main__.MyClass`, which cannot be imported. `bench.save()` records a stub in the YAML:

```yaml
instruments:
  vs:
    class: null
    args: []
    kwargs: {}
    _note: "Set 'class' to the full module path before calling Bench.load()."
```

Move the class to an importable `.py` module, update the `class` field, and `Bench.load()` will connect it automatically on the next load.

---

### Interactive Control Panel

`ControlPanel` opens a Dash browser window with one vertical strip per controller, grouped by instrument. Writes are queued through a dedicated setter thread so the UI never blocks on slow instrument I/O.

```python
from orchid import ControlPanel

panel = ControlPanel(bench, port=8051)
panel.start()
# Browser opens at http://localhost:8051
```

#### Layout

Each controller gets a vertical strip containing (top to bottom):

- **Header** — colour-coded dot, controller name, unit
- **LCD display** — 7-segment readback (DSEG7 font), shown when `readback=True`
- **SP row** — current setpoint with `+`/`−` sign prefix
- **Vertical slider** — for controllers with `limits`; sets on mouse release only
- **Limit warning** — `⚠ NEAR LIMIT` / `⚠ OUT OF LIMIT` when close to or beyond bounds
- **Numeric input** — free-type value, respects `min`/`max` when limits are set
- **Step chips** — click to change the active step size (auto-computed from range)
- **Nudge buttons** — `−` / `+` bump the value by the active step

Strips are grouped by instrument. A separator line divides groups in the rack.

#### Instrument tabs

Tabs appear automatically in the header — one per instrument, plus an `ALL` tab:

```
[ALL] [lockin] [vna] [smu]
```

Clicking a tab shows only the strips for that instrument. `ALL` shows everything. Tab switching is clientside (instant, no server round-trip).

#### Selecting controllers

```python
# Show all controllers registered in bench (default)
panel = ControlPanel(bench)

# Show a specific subset
panel = ControlPanel(bench, controllers=["Vgt", "Vbg", "fac"])
```

#### Step sizes

Step chips are auto-computed from each controller's range (`(hi - lo) / 100`, `/ 10`, `/ 1`, `× 10`). Override per controller:

```python
panel = ControlPanel(bench, steps={"Vgt": 0.01, "freq": 1e6})
```

#### Readback

By default the panel polls each controller's current value every 2 s and shows it on the LCD. Disable if reads are slow:

```python
panel = ControlPanel(bench, readback=False)
panel = ControlPanel(bench, readback=True, readback_interval=5000)  # 5 s
```

#### Appearance

The **APPEARANCE** button in the header opens a dropdown with:

- **Theme** — Dark / Light
- **LCD accent** — 6 colour swatches (Blue, Red, Amber, Green, Cyan, Magenta); drives the LCD digits, slider track, active tab highlight, and step chip borders via a single `--accent` CSS variable

Theme and accent choice persist across browser refreshes via `localStorage`.

#### Status LEDs

| LED | Meaning |
|-----|---------|
| **PWR** | Solid green — panel server is running |
| **RUN** | Pulsing green — idle; pulsing amber — setter queue has pending writes |
| **FAULT** | Off — no errors; pulsing red — last instrument write threw an exception (clears on next successful write) |

#### Programmatic set

`panel.set()` is safe to call from any thread:

```python
panel.set("Vgt", 0.5)
```

#### Stopping

```python
panel.stop()
print(panel.is_running)  # False
```

#### Integration with live plots

Because `panel.set()` writes via `bench[name] = val`, every change fires the bench event log automatically. If a `LivePlotter` with `x="_time"` is running alongside, parameter changes appear as annotated vertical lines on the plot without any extra wiring.

```python
plotter = LivePlotter([PlotSpec(x="_time", y="lockin_X")])
runner.run_monitor(monitor, plotter=plotter, background=True)

panel = ControlPanel(bench)
panel.start()
# Adjust Vgt in the browser → event line appears on the live plot
```

---

## API Reference

### InstrumentAdapter

```python
from orchid import InstrumentAdapter
```

Thin wrapper normalizing pymeasure/qcodes/custom instruments into a uniform interface.

| Constructor        | Description                                     |
|--------------------|-------------------------------------------------|
| `InstrumentAdapter(name, driver, backend="custom")` | Direct construction |
| `.from_pymeasure(name, instrument)` | Wrap a pymeasure instrument   |
| `.from_qcodes(name, instrument)`    | Wrap a qcodes instrument     |
| `.from_custom(name, obj)`           | Wrap any Python object        |
| `.auto(name, instrument)`           | Auto-detect backend from MRO  |

| Attribute         | Type   | Description                                           |
|-------------------|--------|-------------------------------------------------------|
| `name`            | `str`  | Human-readable instrument name                        |
| `driver`          | `Any`  | Raw instrument object                                 |
| `backend`         | `str`  | `"pymeasure"`, `"qcodes"`, or `"custom"`              |
| `connection_info` | `dict` | Serialization metadata: `{"class": "...", "args": [...], "kwargs": {...}}`. Populated by `Bench.add_instrument()` when `args`/`kwargs` are provided; used by `bench.save()`. |

| Method                          | Description                                  |
|---------------------------------|----------------------------------------------|
| `get(attr) -> Any`              | Read attribute value (sync)                  |
| `set(attr, value) -> None`      | Write attribute value (sync)                 |
| `await aget(attr) -> Any`       | Read attribute value (async)                 |
| `await aset(attr, value) -> None` | Write attribute value (async)              |

**Backend behavior:**

| Backend    | `get(attr)`                     | `set(attr, val)`                  |
|------------|---------------------------------|-----------------------------------|
| pymeasure  | `getattr(driver, attr)`         | `setattr(driver, attr, val)`      |
| qcodes     | `driver.attr.get()`             | `driver.attr.set(val)`            |
| custom     | `getattr(driver, attr)`         | `setattr(driver, attr, val)`      |

---

### Controller

```python
from orchid import Controller
```

A named control parameter mapped to an instrument channel.

| Argument       | Type                            | Default             | Description                                      |
|----------------|---------------------------------|---------------------|--------------------------------------------------|
| `name`         | `str`                           | required            | Short label (e.g. `"Vgt"`)                       |
| `instrument`   | `InstrumentAdapter` or `None`   | `None`              | Instrument this controller uses                  |
| `attr`         | `str` or `None`                 | `None`              | Attribute name on instrument                     |
| `get_func`     | `callable` or `None`            | `None`              | Custom getter (overrides adapter)                |
| `set_func`     | `callable` or `None`            | `None`              | Custom setter (overrides adapter)                |
| `unit`         | `str` or `None`                 | `None`              | Physical unit                                    |
| `limits`       | `tuple[float, float]` or `None` | `None`              | `(lo, hi)` inclusive bounds. `None` = unconstrained. |
| `limit_policy` | `LimitPolicy`                   | `LimitPolicy.WARN`  | Response to limit violations — see `LimitPolicy`. |

| Method                        | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `get() -> Any`                | Read current value (no clamping)                         |
| `set(value) -> None`          | Clamp to limits (if set) then apply                      |
| `await aget() -> Any`         | Async read                                               |
| `await aset(value) -> None`   | Async clamp + apply                                      |
| `clear_limit_log() -> None`   | Reset violation log and warn-once flag. Called automatically by the runner at the start of each run. |

| Property      | Type               | Description                              |
|---------------|--------------------|------------------------------------------|
| `limit_log`   | `list[LimitEntry]` | All violations recorded since last reset |

**Precedence:** `get_func`/`set_func` override `instrument.get(attr)`/`instrument.set(attr)`.

---

### LimitPolicy

```python
from orchid import LimitPolicy
```

| Value               | Behaviour                                                                   |
|---------------------|-----------------------------------------------------------------------------|
| `LimitPolicy.WARN`  | Clamp the value; emit a `warnings.warn` on the **first** violation per run, then log silently. Default. |
| `LimitPolicy.RAISE` | Raise `ValueError` immediately — use for hard safety limits.                |
| `LimitPolicy.LOG`   | Clamp silently and record every violation; inspect via `controller.limit_log`. |

Each violation is stored as a `LimitEntry(index, requested, clamped)` named tuple where `index` is the sweep position (e.g. `(42,)` for 1D, `(3, 17)` for 2D, `()` for manual calls).

---

### Readout

```python
from orchid import Readout
```

A read-only measurement channel. Supply either `get_func` **or** `instrument + attr`; the latter is fully serializable by `bench.save()` / `Bench.load()`.

| Argument     | Type                          | Default  | Description                            |
|--------------|-------------------------------|----------|----------------------------------------|
| `name`       | `str`                         | required | Label (e.g. `"S21"`)                   |
| `kind`       | `DataKind`                    | required | `SCALAR`, `TRACE`, or `IMAGE`          |
| `get_func`   | `callable` or `None`          | `None`   | Acquisition function. Mutually exclusive with `instrument + attr`. |
| `instrument` | `InstrumentAdapter` or `None` | `None`   | Instrument to read from. Used together with `attr`. |
| `attr`       | `str` or `None`               | `None`   | Attribute name on the instrument. Used together with `instrument`. |
| `shape`      | `tuple` or `None`             | `None`   | Required for `TRACE` and `IMAGE`       |
| `unit`       | `str` or `None`               | `None`   | Physical unit                          |
| `contains`   | `str`, `list[str]`, or `None` | `None`   | For `IMAGE` readouts: list of column names (e.g. `["f", "mag", "phase"]`) enabling `z_col` string lookup in `PlotSpec`. For other kinds: plain string description. |

| Method                         | Description           |
|--------------------------------|-----------------------|
| `read() -> ndarray or float`   | Acquire measurement   |
| `await aread() -> ndarray or float` | Async acquire    |

---

### DataKind

```python
from orchid import DataKind
```

| Value            | Meaning                          |
|------------------|----------------------------------|
| `DataKind.SCALAR` | Single number per point          |
| `DataKind.TRACE`  | 1D array per point (shape `(N,)`) |
| `DataKind.IMAGE`  | 2D array per point (shape `(H, W)`) |

---

### Bench

```python
from orchid import Bench
```

Central container for the entire lab bench configuration.

| Argument    | Type              | Default   | Description                        |
|-------------|-------------------|-----------|------------------------------------|
| `data_root` | `str` or `Path`   | `"./data"`| Root directory for saved data      |
| `metadata`  | `dict`            | `{}`      | User metadata (sample, operator, etc.) |

| Attribute      | Type                             | Description            |
|----------------|----------------------------------|------------------------|
| `instruments`  | `dict[str, InstrumentAdapter]`   | Registered instruments |
| `controllers`  | `dict[str, Controller]`          | Registered controllers |
| `readouts`     | `dict[str, Readout]`             | Registered readouts    |
| `data_root`    | `Path`                           | Data output root       |
| `metadata`     | `dict`                           | User metadata          |

#### Methods

**`add_instrument(name, instrument, backend="auto", *, class_path=None, args=None, kwargs=None) -> InstrumentAdapter`**

Register an instrument. `backend` can be `"auto"`, `"pymeasure"`, `"qcodes"`, or `"custom"`. Pass `args` and/or `kwargs` to record how the instrument was constructed — this enables `bench.save()` / `Bench.load()` to reconnect it automatically. `class_path` overrides the auto-detected fully-qualified class name (useful when the detected path is wrong).

**`add_controller(name, instrument=None, attr=None, get_func=None, set_func=None, unit=None, limits=None, limit_policy=LimitPolicy.WARN) -> Controller`**

Register a control parameter. `instrument` can be an `InstrumentAdapter` object or the string name of a registered instrument. `limits` is an optional inclusive `(lo, hi)` range and `limit_policy` controls clamp, raise, or log behavior.

**`add_controller_binding(name, gate_names, *, unit=None) -> Controller`**

Register a virtual controller that sets all controllers named in `gate_names` to the same value and reads them back as `{controller_name: value}`. If `unit` is omitted, it is inferred from bound controllers.

**`add_readout(name, kind, get_func=None, instrument=None, attr=None, shape=None, unit=None, contains=None) -> Readout`**

Register a measurement readout. `kind` can be a `DataKind` enum or string (`"scalar"`, `"trace"`, `"image"`). Supply either `get_func` **or** `instrument + attr` (the latter is fully serializable by `bench.save()` / `Bench.load()`). `instrument` can be an `InstrumentAdapter` object or the string name of a registered instrument. For `IMAGE` readouts, pass `contains` as a `list[str]` of column names to enable named `z_col` selection in `PlotSpec`.

**`remove_instrument(name) -> None`**

Remove an instrument and **all controllers that depend on it**. Raises `KeyError` if not found.

**`remove_controller(name) -> None`**

Remove a controller. Raises `KeyError` if not found.

**`remove_readout(name) -> None`**

Remove a readout. Raises `KeyError` if not found.

**`bench[name]`** — Get current value (calls `parameter.get()` or `readout.read()`).

**`bench[name] = value`** — Set parameter value (calls `parameter.set(value)`).

**`snapshot(names=None, *, include_readouts=False) -> None`**

Print a formatted table of current values using `tabulate`. By default reads only controllers (fast). Pass `include_readouts=True` to also acquire readout values.

| Argument           | Type                  | Default | Description                                                      |
|--------------------|-----------------------|---------|------------------------------------------------------------------|
| `names`            | `list[str]` or `None` | `None`  | Explicit name list; `None` = all (uses `include_readouts` flag)  |
| `include_readouts` | `bool`                | `False` | When `names=None`, also read registered readouts. Ignored when `names` is given. |

**`save(path) -> None`**

Write the bench configuration to a YAML file. Instruments are stored by fully-qualified class path + constructor `args`/`kwargs`. Controllers and `instrument + attr` readouts are fully serialized. Custom `get_func` / `set_func` callables are not executable after load, but their source text is saved as a human-readable hint (`get_func_src` / `set_func_src` fields). Instruments defined in `__main__` emit a `UserWarning` and are written as a stub with `class: null`.

**`Bench.load(path) -> Bench`** *(classmethod)*

Reconstruct a bench from a YAML file saved by `save()`. Each instrument is instantiated by importing its class and calling it with the stored `args`/`kwargs`. Controllers and readouts are wired automatically. Entries that cannot be auto-loaded (custom callables, failed connections, `__main__` stubs) are collected silently in `bench._stubs`. A single summary line is printed if any stubs exist.

```python
bench.save("bench.yaml")
bench = Bench.load("bench.yaml")
# Bench loaded ← bench.yaml  ·  2 stubs — call bench.show_stubs()
```

**`show_stubs(*, full_source=False) -> None`**

Print a compact table of all entries that need manual re-registration after `load()`. Each row shows the name, kind (`instrument` / `controller` / `readout`), reason it was skipped, metadata (unit, limits, shape), and a source hint. Source strings are truncated to the first line by default; pass `full_source=True` to print them in full.

```python
bench.show_stubs()                 # compact — first line of source only
bench.show_stubs(full_source=True) # full multiline source
```

**`stubs`** *(property)* → `dict`

Raw dict of stub entries keyed by name. Each value has at least `"kind"` and `"reason"`. Controller stubs include `"get_func_src"` / `"set_func_src"` when source was recorded at `save()` time.

---

### Sweep

```python
from orchid import Sweep
```

Defines a sweep over one controller.

| Argument     | Type                   | Default  | Description                           |
|--------------|------------------------|----------|---------------------------------------|
| `controller` | `Controller` or `str`  | required | Controller to sweep (or its name)     |
| `values`     | `array-like`           | required | Sweep values                          |
| `reverse`    | `bool`                 | `False`  | Append reversed values (hysteresis)   |

| Property  | Type  | Description                         |
|-----------|-------|-------------------------------------|
| `length`  | `int` | Number of sweep points              |

When `reverse=True`, the values array is doubled: `[forward, reversed]`.

---

### MultiSweep

```python
from orchid import MultiSweep
```

Sweep multiple controllers simultaneously along a shared axis. All controllers step together at each point.

| Argument      | Type                      | Default  | Description                                          |
|---------------|---------------------------|----------|------------------------------------------------------|
| `controllers` | `list[Controller or str]` | required | Controllers to sweep simultaneously                  |
| `values`      | `list[array-like]`        | required | One values array per controller, all the same length |
| `reverse`     | `bool`                    | `False`  | Append reversed values (hysteresis) on all arrays    |

```python
proc = Procedure(
    name="gate_pair_sweep",
    bench=bench,
    sweeps=[
        MultiSweep(
            controllers=["Vgt", "Vbg"],
            values=[np.linspace(0, 1, 100), np.linspace(0, 5, 100)],
        )
    ],
    readouts=["lockin_X"],
)
# At point i: Vgt = linspace(0,1,100)[i]  AND  Vbg = linspace(0,5,100)[i]
# Result shape: lockin_X -> (100,)
```

Can be freely mixed with regular `Sweep` in the same procedure:

```python
proc = Procedure(
    sweeps=[
        Sweep("power", np.linspace(-30, 0, 20)),            # outer: slow axis
        MultiSweep(["Vgt", "Vbg"], [vgt_vals, vbg_vals]),   # inner: fast axis
    ],
    ...
)
# Result shape: lockin_X -> (20, 100)
```

| Property     | Description                                      |
|--------------|--------------------------------------------------|
| `values`     | First controller's values array (for iteration)  |
| `all_values` | List of all controllers' value arrays            |
| `length`     | Number of points                                 |
| `name`       | Combined name, e.g. `"Vgt+Vbg"`                 |
| `controller` | First controller (for axis labelling)            |

---

### Procedure

```python
from orchid import Procedure
```

Experiment procedure for sweep-based measurements.

| Argument           | Type                        | Default              | Description                                 |
|--------------------|-----------------------------|----------------------|---------------------------------------------|
| `name`             | `str`                       | required             | Experiment name                             |
| `bench`            | `Bench`                      | required             | Lab bench configuration                     |
| `sweeps`           | `list[Sweep or MultiSweep]` | `[]`                 | Sweep axes (outer-first ordering)           |
| `readouts`         | `list[str]`                 | `[]`                 | Readout names to record                     |
| `settle_time`      | `float`                     | `0.0`                | Seconds to wait after set, before read      |
| `snake`            | `bool`                      | `False`              | Alternate inner sweep direction             |
| `write_mode`       | `WriteMode`                 | `POINTWISE`          | When to flush data to disk                  |
| `error_policy`     | `ErrorPolicy`               | `STOP_AND_SAVE`      | How to handle measurement errors            |
| `max_retries`      | `int`                       | `3`                  | Retries for `RETRY_AND_SKIP` policy         |
| `tags`             | `list[str]`                 | `[]`                 | Free-form tags for metadata                 |
| `metadata`         | `dict`                      | `{}`                 | Additional metadata to save                 |
| `before_experiment`| `callable` or `None`        | `None`               | Hook `()`: once before start                |
| `after_experiment` | `callable` or `None`        | `None`               | Hook `()`: once after finish                |
| `before_point`     | `callable` or `None`        | `None`               | Hook `(index_tuple)`: before each measurement |
| `after_point`      | `callable` or `None`        | `None`               | Hook `(index_tuple)`: after each measurement  |
| `before_sweep`     | `callable` or `None`        | `None`               | Hook `(axis_index)`: before each sweep axis   |
| `after_sweep`      | `callable` or `None`        | `None`               | Hook `(axis_index)`: after each sweep axis    |

| Property  | Type            | Description                    |
|-----------|-----------------|--------------------------------|
| `ndim`    | `int`           | Number of sweep axes           |
| `shape`   | `tuple[int,...]`| Sweep grid shape               |

| Method        | Description                                                                          |
|---------------|--------------------------------------------------------------------------------------|
| `summary()`   | Print a formatted table of sweeps, readouts, settings, and hooks. Called automatically by the runner before each experiment. |
| `to_dict()`   | Serialize to a plain dict. Saved as `procedure.yaml` in the data directory. Read back with `read_procedure()`. |

---

### MonitorProcedure

```python
from orchid import MonitorProcedure
```

Time-series monitoring procedure (no sweeps).

| Argument           | Type                | Default              | Description                                 |
|--------------------|---------------------|----------------------|---------------------------------------------|
| `name`             | `str`               | required             | Session name                                |
| `bench`            | `Bench`              | required             | Lab bench configuration                     |
| `readouts`         | `list[str]`         | `[]`                 | Readout names to record                     |
| `interval`         | `float`             | `1.0`                | Seconds between reads                       |
| `duration`         | `float` or `None`   | `None`               | Total duration; `None` = run until stopped  |
| `stop_condition`   | `callable` or `None`| `None`               | `(data_dict) -> bool`; return `True` to stop|
| `chunk_size`       | `int`               | `256`                | Samples buffered in memory before flushing to disk. Smaller = safer on crash, more I/O. |
| `tags`             | `list[str]`         | `[]`                 | Free-form tags                              |
| `metadata`         | `dict`              | `{}`                 | Additional metadata                         |
| `before_experiment`| `callable` or `None`| `None`               | Hook `()`: once before start                |
| `after_experiment` | `callable` or `None`| `None`               | Hook `()`: once after finish                |
| `after_point`      | `callable` or `None`| `None`               | Hook `(sample_index, data_dict)`: after each read |

Data is saved via zarro's `StreamingWriter` with a `_time` timestamp array. Samples are buffered
in memory and flushed to disk in batches of `chunk_size` to minimise I/O overhead. A final flush
happens on `close()`, so no data is lost on normal exit or `Ctrl+C`.

| Method        | Description                                                                          |
|---------------|--------------------------------------------------------------------------------------|
| `summary()`   | Print a formatted table of readouts, settings, and hooks. Called automatically by the runner before each monitor run. |
| `to_dict()`   | Serialize to a plain dict. Saved as `procedure.yaml` in the data directory. Read back with `read_procedure()`. |

---

### EventLineConfig

```python
from orchid import EventLineConfig
```

Visual properties for parameter-change event markers drawn on time-series plots.

| Argument      | Type  | Default                    | Description                                                         |
|---------------|-------|----------------------------|---------------------------------------------------------------------|
| `color`       | `str` | `"#444444"`                | Line and label font color. Any CSS/plotly color string.             |
| `width`       | `int` | `2`                        | Line width in pixels.                                               |
| `dash`        | `str` | `"dash"`                   | Line style: `"solid"`, `"dot"`, `"dash"`, `"longdash"`, `"dashdot"`. |
| `font_size`   | `int` | `15`                       | Label font size in points.                                          |
| `bgcolor`     | `str` | `"rgba(255,255,255,0.85)"` | Label box background color. Use `rgba(r,g,b,a)` for transparency.  |
| `bordercolor` | `str` | `"#000000"`                | Label box border color.                                             |
| `borderwidth` | `int` | `1`                        | Label box border width in pixels.                                   |
| `borderpad`   | `int` | `3`                        | Padding in pixels between the label text and the box border.        |

Labels are rotated 90° and centered vertically on the event line.

```python
plotter = LivePlotter(
    [PlotSpec(x="_time", y="lockin_X")],
    event_line=EventLineConfig(
        color="#2255cc",
        dash="dot",
        bgcolor="rgba(255,255,255,0.0)",  # transparent box (no box)
    ),
)
```

---

### PlotSpec

```python
from orchid import PlotSpec
```

Describes one subplot in a `LivePlotter`.

| Argument      | Type                               | Default   | Description                                      |
|---------------|------------------------------------|-----------|--------------------------------------------------|
| `x`           | `str` or `array-like`              | required  | **str**: sweep parameter name, readout name, or `"_time"` (monitor). **array**: fixed axis values (e.g. frequency list) — triggers `live_trace` auto-detection. |
| `y`           | `str`, `list[str]`, or `array-like`| required  | **str**: readout name. **list[str]**: multiple readout names overlaid on one subplot. **array**: fixed axis values (e.g. frequency list) — triggers `trace_heatmap` auto-detection. Heatmap: outer sweep parameter name (string). |
| `z`           | `str` or `None`                    | `None`    | Readout name for color values. Required for `"heatmap"` and `"trace_heatmap"`. |
| `z_col`       | `int`, `str`, or `None`            | `None`    | Column selector for `IMAGE` or `TRACE` readouts used as `z` (or `y` for `live_trace`). `None` = use whole array (valid for `TRACE`). `int` = column index. `str` = column name looked up in `readout.contains`. |
| `plot_type`   | `str`                              | `"auto"`  | `"line"`, `"heatmap"`, `"live_trace"`, `"trace_heatmap"`, or `"auto"` (infer from types of `x` and `y`). |
| `update_every`| `str`                              | `"sweep"` | `"point"`, `"sweep"`, or `"plane"`               |
| `update_func` | `callable` or `None`               | `None`    | Custom `(fig_dict, index, data) -> None`         |
| `colorscale`  | `str`, `list`, or `None`           | `None`    | Plotly colorscale name (heatmaps only). `None` uses the active template default. |

**Auto-detection rules** (when `plot_type="auto"`):

| `x`    | `y`    | procedure ndim | resolved type    |
|--------|--------|----------------|------------------|
| str    | str    | 1              | `line`           |
| str    | str    | ≥ 2            | `heatmap`        |
| array  | str    | any            | `live_trace`     |
| str    | array  | any            | `trace_heatmap`  |

---

### PlotterBase

```python
from orchid import PlotterBase
```

Abstract base class for live plotters. Holds all figure-building and data-update logic — no server code. Subclass it to add a new display backend; you only need to implement three things:

```python
class MyPlotter(PlotterBase):
    def _start_server(self) -> None: ...   # start your server
    def stop(self, _silent=False) -> None: ...   # stop it
    @property
    def is_running(self) -> bool: ...      # check if alive
```

Override `on_data_changed()` to push updates to your server after each data write:

```python
    def on_data_changed(self) -> None:
        # e.g. Taipy: broadcast_callback(self._gui, lambda s: ...)
        # e.g. Dash: self._data_version += 1  (already done in DashPlotter)
```

#### Lifecycle methods (called by the runner)

| Method / Property                              | Description                                                                 |
|------------------------------------------------|-----------------------------------------------------------------------------|
| `setup(proc)`                                  | Reset state, build figure dict, call `_start_server()` if not running      |
| `update_point(index, data, sweep_values)`      | After every measurement point                                               |
| `update_sweep(outer_index, data, sweep_values)`| After each inner sweep completes                                            |
| `update_plane(outer_index, data, sweep_values)`| After each 2D plane completes                                               |
| `update_monitor(sample_idx, data, timestamp)`  | After each monitor sample. `x="_time"` auto-scales to s/min/hr from zero.  |
| `notify_event(timestamp, param, value)`        | Draw a vertical event line on all `x="_time"` subplots                      |
| `finalize()`                                   | Mark experiment done (stops polling; server stays up for zoom/pan)          |
| `stop()`                                       | *(abstract)* Stop server, free resources                                    |
| `is_running` *(property)*                      | *(abstract)* `True` if the server is running                                |
| `on_data_changed()`                            | Hook called after every write to `_fig_dict`. No-op in base; override for push/poll. |

#### Data helpers (available to subclasses)

| Method                                          | Description                                         |
|-------------------------------------------------|-----------------------------------------------------|
| `build_figure_dict(proc, n) -> dict`            | Build initial Plotly figure as a plain dict         |
| `dispatch(event, index, data, sweep_values)`    | Route update to matching subplots by `update_every` |
| `update_line(spec_idx, spec, data, sweep_values)` | Update a line trace buffer                        |
| `update_heatmap(spec_idx, spec, index, data)`   | Fill one row/cell of a heatmap                      |
| `update_live_trace(spec_idx, spec, data)`       | Overwrite a live trace                              |
| `update_trace_heatmap(spec_idx, spec, index, data)` | Fill one column of a trace heatmap              |
| `resolve_col(z_col, readout) -> int or None`    | Resolve `z_col` string/int/None to a column index   |
| `extract_col(raw, z_col, readout) -> ndarray`   | Extract one channel from a raw readout array        |
| `format_elapsed(seconds) -> (value, unit)`      | *(static)* Scale seconds to s/min/hr                |
| `unit_divisor(unit) -> float`                   | *(static)* Seconds per display unit                 |

#### Constructor parameters

| Argument          | Type                       | Default             | Description                          |
|-------------------|----------------------------|---------------------|--------------------------------------|
| `plots`           | `list[PlotSpec]`           | required            | Subplot specifications               |
| `height`          | `int`                      | `350`               | Height in pixels per subplot         |
| `width`           | `int`                      | `700`               | Figure width in pixels               |
| `open_browser`    | `bool`                     | `True`              | Auto-open browser when server starts |
| `event_line`      | `EventLineConfig` or `None`| `EventLineConfig()` | Style for parameter-change markers   |
| `max_display_pts` | `int`                      | `5000`              | Rolling window size for monitor line plots |

---

### DashPlotter / LivePlotter

```python
from orchid import DashPlotter   # preferred
from orchid import LivePlotter   # backward-compat alias: LivePlotter = DashPlotter
```

Concrete implementation of `PlotterBase` using a Dash/Werkzeug server. The browser polls for updates every `update_interval` milliseconds.

Adds two constructor arguments on top of `PlotterBase`:

| Argument          | Type  | Default | Description                                  |
|-------------------|-------|---------|----------------------------------------------|
| `port`            | `int` | `8050`  | Dash server port                             |
| `update_interval` | `int` | `500`   | Browser polling interval in milliseconds     |

All `PlotterBase` constructor arguments (`plots`, `height`, `width`, `open_browser`, `event_line`, `max_display_pts`) are also accepted.

#### Additional methods

| Method / Property | Description                                                                                     |
|-------------------|-------------------------------------------------------------------------------------------------|
| `stop()`          | Shut down the Dash server and free the port. Called automatically by the runner after each run. |
| `is_running`      | `True` if the Dash server thread is alive                                                       |

**Reusable across experiments:** `setup()` resets figure state but keeps the server alive, so the browser reconnects without a page reload. Call `stop()` manually to free the port entirely.

#### Time axis formatting

When `x="_time"` is used (monitoring mode), the x-axis shows elapsed time from zero with auto-scaling units:

| Elapsed time | x-axis unit | Label      |
|--------------|-------------|------------|
| < 2 minutes  | seconds     | Time (s)   |
| < 2 hours    | minutes     | Time (min) |
| ≥ 2 hours    | hours       | Time (hr)  |

The label updates automatically. Existing data points are rescaled when the unit changes.

#### Usage

```python
from orchid import DashPlotter, PlotSpec

plotter = DashPlotter([PlotSpec(x="Vgt", y="lockin_X")])
runner.run(proc, plotter=plotter)
# Browser opens at http://localhost:8050 with live-updating plot

plotter.stop()  # shut down server to free port (optional)
```

Requires `plotly` and `dash` (included in the default dependencies).

---

### TaipyPlotter

Push-based live plotting backend using [Taipy GUI](https://docs.taipy.io/en/latest/).  
Unlike `DashPlotter` (which polls for changes), `TaipyPlotter` pushes each update directly to every connected browser tab via WebSocket — zero polling latency.

```python
class TaipyPlotter(PlotterBase)
```

**Constructor parameters** — same as `PlotterBase` plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | `int` | `5000` | TCP port for the Taipy GUI server |
| `host` | `str` | `"localhost"` | Hostname / IP to bind |

**Usage:**

```python
from orchid import TaipyPlotter, PlotSpec

# 1-D sweep
plotter = TaipyPlotter([PlotSpec(x="Vgt", y="lockin_X")])
runner.run(proc, plotter=plotter)
# Browser opens at http://localhost:5000

# 2-D sweep — heatmap auto-detected
plotter = TaipyPlotter([PlotSpec(x="fac", y="lockin_X")], port=5001)
runner.run(proc_2d, plotter=plotter)

# Multiple subplots
plotter = TaipyPlotter([
    PlotSpec(x="Vgt", y="lockin_X"),
    PlotSpec(x="Vgt", y="lockin_Y"),
])

plotter.stop()   # shut down server (optional)
```

**How updates work:**

Every time the experiment writes new data, `PlotterBase` calls `on_data_changed()`.  
`TaipyPlotter` deep-copies `_fig_dict` for snapshot safety, then calls:

```python
broadcast_callback(gui, lambda state: setattr(state, "figure", fig_snapshot))
```

This sends the new figure to **all** browser tabs simultaneously over WebSocket — no polling interval, no stale reads.

**Installation:**

```bash
pip install "orchid[taipy]"
# or directly:
pip install "taipy-gui>=3.1"
```

**DashPlotter vs TaipyPlotter — when to choose which:**

| | DashPlotter | TaipyPlotter |
|---|---|---|
| Update mechanism | Poll every 500 ms | Push (WebSocket) |
| Extra dependency | `dash`, `plotly` | `taipy-gui` |
| Multi-tab support | Each tab polls independently | All tabs update in sync |
| Default port | 8050 | 5000 |
| Alias | `LivePlotter` | — |

---

### ControlPanel

```python
from orchid import ControlPanel
```

Standalone Dash browser UI for interactively adjusting bench controllers. Displays one vertical strip per controller, grouped by instrument with tab switching. All instrument writes are routed through a dedicated setter thread with last-value-wins semantics, so the UI never blocks on slow instrument I/O.

```python
panel = ControlPanel(bench, port=8051)
panel.start()           # opens http://localhost:8051
panel.set("Vgt", 0.5)  # programmatic write from any thread
panel.stop()
```

#### Constructor parameters

| Argument            | Type                        | Default | Description                                                                      |
|---------------------|-----------------------------|---------|----------------------------------------------------------------------------------|
| `bench`             | `Bench`                     | required | The lab bench whose controllers this panel controls                             |
| `port`              | `int`                       | `8051`  | TCP port for the Dash server                                                     |
| `controllers`       | `list[str]` or `None`       | `None`  | Names to show; `None` shows all controllers in `bench`                           |
| `open_browser`      | `bool`                      | `True`  | Open a browser tab automatically on `start()`                                   |
| `readback`          | `bool`                      | `True`  | Poll each controller's current value periodically and display on the LCD        |
| `readback_interval` | `int`                       | `2000`  | Readback poll period in milliseconds. Reads run in parallel via a thread pool.  |
| `steps`             | `dict[str, float]` or `None`| `None`  | Override the default active step size per controller (used by slider and nudge buttons) |

#### Methods and properties

| Method / Property   | Description                                               |
|---------------------|-----------------------------------------------------------|
| `start()`           | Start the setter thread and Dash server                   |
| `stop()`            | Drain the setter queue and shut down the server           |
| `set(name, value)`  | Queue a controller write. Safe to call from any thread.   |
| `is_running`        | `True` while the Dash server thread is alive              |

#### UI behaviour

- **Instrument tabs** — auto-generated from `ctrl.instrument.name`; controllers with no instrument appear under `"Custom"`. Clientside switching (instant).
- **Bounded controllers** (`limits` set) — vertical slider (sets on mouse release) + step chips + nudge `−`/`+` buttons + numeric input.
- **Unbounded controllers** — numeric input only.
- **Step chips** — 4 options auto-computed from the range; click to change the active step for slider snapping and nudge size. Override with `steps=`.
- **LCD readback** — 7-segment DSEG7 font; polled every `readback_interval` ms via a thread pool.
- **Limit warnings** — `⚠ NEAR LIMIT` (within 5 % of a bound) and `⚠ OUT OF LIMIT` displayed below the slider.
- **Last-value-wins** — if the same controller is changed multiple times before the setter thread drains, only the final value is applied.
- **Status LEDs** — PWR (always green), RUN (amber while queue has pending writes), FAULT (red on setter exception, clears on next success).
- **Appearance menu** — dark/light theme toggle + 6 accent colour swatches; persists via `localStorage`.

#### Event log integration

`ControlPanel` writes via `bench[name] = val`, which fires the bench event log. If a `LivePlotter` with `x="_time"` is running, every manual panel adjustment automatically appears as an annotated vertical line on the plot.

#### Dependencies

```bash
pip install dash
```

No `dash-daq` required.

---

### WriteMode

```python
from orchid import WriteMode
```

Controls when data is flushed to disk during a sweep. Maps to zarro `WriteType`.

| Value                | zarro method     | Min. sweeps | Behavior                                                    |
|----------------------|------------------|-------------|-------------------------------------------------------------|
| `WriteMode.POINTWISE`| `write_point()`  | 1           | Write after every measurement point (safest, most I/O)      |
| `WriteMode.SWEEPWISE`| `write_trace()`  | 1           | Buffer innermost sweep, write after it completes            |
| `WriteMode.PLANEWISE`| `write_image()`  | 2           | Buffer two innermost sweeps, write after they complete      |
| `WriteMode.ALL`      | `write_all()`    | 1           | Buffer entire experiment, write once at the end (fastest)   |

---

### ErrorPolicy

```python
from orchid import ErrorPolicy
```

| Value                         | Behavior                                           |
|-------------------------------|----------------------------------------------------|
| `ErrorPolicy.STOP_AND_SAVE`   | Stop experiment, save collected data, re-raise error |
| `ErrorPolicy.RETRY_AND_SKIP`  | Retry up to `max_retries` times, skip point with NaN on failure |
| `ErrorPolicy.IGNORE`          | Log error, fill NaN, continue                      |

---

### ExperimentRunner

```python
from orchid import ExperimentRunner
```

Executes procedures and manages data flow to zarro. Internally delegates sweep execution to a **write strategy** class selected by `procedure.write_mode`.

| Argument             | Type   | Default | Description                           |
|----------------------|--------|---------|---------------------------------------|
| `use_experiment_id`  | `bool` | `True`  | Auto-number output directories        |

#### Methods

| Method / Property                       | Description                              |
|-----------------------------------------|------------------------------------------|
| `run(procedure, plotter=None, print_summary=False, return_path=False) -> Path or None` | Run sweep experiment (sync) |
| `await arun(procedure, plotter=None, print_summary=False) -> Path` | Run sweep experiment (async) |
| `run_monitor(procedure, plotter=None, background=False, print_summary=False, return_path=False) -> Path or None` | Run time-series monitor |
| `await arun_monitor(procedure, plotter=None, print_summary=False) -> Path` | Run monitor (async) |
| `stop_monitor() -> Path`                | Stop a background monitor and return data path |
| `is_monitoring` *(property)*            | `True` if a background monitor is currently running |

All methods return the `Path` to the output data directory.

**`run` / `arun` shared parameters:**

| Argument        | Type                    | Default | Description                                   |
|-----------------|-------------------------|---------|-----------------------------------------------|
| `procedure`     | `Procedure`             | required| The sweep procedure                           |
| `plotter`       | `PlotterBase` or `None` | `None`  | Live plotter (any `PlotterBase` subclass)     |
| `print_summary` | `bool`                  | `False` | If `True`, print the procedure summary table before running. |
| `return_path`   | `bool`                  | `False` | If `True`, return the `Path` to the saved data directory (`run()` only). |

**`run_monitor` parameters:**

| Argument        | Type                    | Default | Description                                    |
|-----------------|-------------------------|---------|------------------------------------------------|
| `procedure`     | `MonitorProcedure`      | required| The monitoring procedure                       |
| `plotter`       | `LivePlotter` or `None` | `None`  | Live plotter                                   |
| `background`    | `bool`                  | `False` | If `True`, run in background thread and return immediately. Use `bench["Vgt"] = 0.5` to change parameters, `runner.stop_monitor()` to stop. |
| `print_summary` | `bool`                  | `False` | If `True`, print the procedure summary table before running. |
| `return_path`   | `bool`                  | `False` | If `True`, return the `Path` to the saved data directory. In background mode, always returns `None`; use `stop_monitor()` to get the path. |

**Interrupt handling:** `Ctrl+C` cleanly stops any running experiment, saves collected data with `status: "interrupted"` in metadata, and prints a single-line message. No tracebacks in Jupyter.

#### Write strategies

The runner dispatches sweep execution to a strategy class based on `write_mode`:

```python
strategy = _STRATEGY_MAP[proc.write_mode](proc, writer, pbar)
await strategy.execute()
```

| `WriteMode`  | Strategy class       | Description                                    |
|--------------|----------------------|------------------------------------------------|
| `POINTWISE`  | `PointwiseStrategy`  | Recursive loop, `write_point()` at each leaf   |
| `SWEEPWISE`  | `SweepwiseStrategy`  | Buffer innermost axis, `write_trace()` per row |
| `PLANEWISE`  | `PlanewiseStrategy`  | Buffer two innermost axes, `write_image()` per plane |
| `ALL`        | `AllStrategy`        | Buffer everything, single `write_all()`        |

All strategies inherit from `WriteStrategy`, which provides shared helpers:

| Method               | Description                                           |
|----------------------|-------------------------------------------------------|
| `_maybe_snake()`     | Reverse sweep values on odd parent index (snake scan) |
| `_safe_read()`       | Read with error policy (retry/skip/ignore)            |
| `_nan_value()`       | NaN placeholder matching readout shape                |
| `_allocate_buffers()` | Pre-allocate numpy arrays for all readouts           |
| `_outer_loop()`      | Generic recursive loop through outer sweep axes       |

**Monitor execution flow** (not strategy-based — handled directly by ExperimentRunner):

```
loop:
  read all readouts
  append(data)
  check stop_condition / duration
  sleep(interval)
```

---

### Utility Functions

```python
from orchid import apply_theme, PALETTE, read_events, read_limit_log, read_metadata, read_procedure, update_metadata
```

**`apply_theme(palette="vivid", colorscale="Inferno", name="sw_clean", base="simple_white") -> str`**

Register a Plotly template and set it as the global default. Called automatically when
`orchid.plotting` is imported (with default arguments), so plots look consistent without
any extra setup. Call again to switch themes.

| Argument     | Type              | Default          | Description                                          |
|--------------|-------------------|------------------|------------------------------------------------------|
| `palette`    | `str` or `list[str]` | `"vivid"`     | Colour cycle. Key from `PALETTE` or a custom list of CSS/hex strings. |
| `colorscale` | `str`             | `"Inferno"`      | Plotly colorscale for heatmaps (e.g. `"RdBu_r"`, `"Plasma"`). |
| `name`       | `str`             | `"sw_clean"`     | Template name registered in `pio.templates`.         |
| `base`       | `str`             | `"simple_white"` | Base Plotly template to stack under the orchid template. |

Returns the full template string set as default (e.g. `"simple_white+sw_clean"`).

```python
from orchid import apply_theme, PALETTE

apply_theme()                                   # defaults (vivid palette, Inferno heatmap)
apply_theme(palette="muted")                    # switch colour cycle
apply_theme(palette=PALETTE["pastel"], colorscale="RdBu_r")
apply_theme(base="plotly_dark", colorscale="Plasma")   # dark theme
```

**`PALETTE`** — `dict[str, list[str]]`

Built-in named colour cycles for use with `apply_theme`:

| Key       | Description                                         |
|-----------|-----------------------------------------------------|
| `"vivid"` | Vivid but balanced — good for presentations (default) |
| `"muted"` | Desaturated — reads well in print and on screen     |
| `"pastel"`| Soft tones — nice for dense overlapping traces      |
| `"minimal"`| 4-colour high-contrast set, works in greyscale     |

```python
from orchid import PALETTE

apply_theme(palette=PALETTE["muted"])
# or build your own:
apply_theme(palette=["#003049", "#D62828", "#2FA084"])
```

---

**`read_procedure(data_dir, print_summary=True) -> dict`**

Read `procedure.yaml` from an experiment directory. By default also prints the same formatted summary table that was shown before the experiment ran.

```python
d = read_procedure("./data/0042")              # prints summary + returns dict
d = read_procedure("./data/0042", print_summary=False)  # silent

# Useful fields
d["kind"]                        # "sweep" or "monitor"
d["name"]                        # procedure name
d["total_points"]                # e.g. 5000
d["shape"]                       # e.g. [100, 50]
d["settings"]["write_mode"]      # e.g. "sweepwise"
d["settings"]["settle_time"]     # in seconds
d["estimated_duration_s"]        # lower-bound estimate
d["sweeps"][0]["controller"]      # first sweep's controller name
d["sweeps"][0]["min"]            # sweep start value
d["sweeps"][0]["max"]            # sweep end value
d["hooks"]["after_point"]        # None, or dict with "name", "doc", "source"
```

For a `MonitorProcedure` the dict contains `kind="monitor"`, `settings["interval"]`, `settings["duration"]`, `settings["chunk_size"]`, and no `sweeps` key.

**`read_limit_log(data_dir) -> list[dict]`**

Read controller limit violations recorded during an experiment. Returns a list of dicts with keys `controller`, `index`, `requested`, `clamped`. Returns `[]` if no violations occurred or `limit_log.yaml` does not exist.

```python
entries = read_limit_log("./data/0005")
for e in entries:
    print(f"{e['controller']}[{e['index']}]: {e['requested']} → {e['clamped']}")
```

---

**`read_events(data_dir) -> list[dict]`**

Read parameter change events recorded during a monitor run. Returns a list of dicts with keys `time`, `elapsed`, `param`, `value`. Returns `[]` if no events were recorded or `events.yaml` does not exist.

```python
events = read_events("./data/0005")
for e in events:
    print(f"t={e['elapsed']:.1f}s  {e['param']} → {e['value']}")
```

**`update_metadata(data_dir, **kwargs) -> dict`**

Add or overwrite fields in an experiment's `metadata.yaml`. Returns the full updated dict.

```python
update_metadata("./data/0001", notes="good data", quality="A", T_mc=0.015)
```

**`read_metadata(data_dir) -> dict`**

Read an experiment's `metadata.yaml`. Raises `FileNotFoundError` if missing.

```python
meta = read_metadata("./data/0001")
```

---

## Data Backend (zarro)

Orchid uses [zarro](https://github.com/eeroqlab/zarro) for all data persistence. Here is how orchid maps to zarro concepts:

| Orchid concept               | zarro class                     |
|------------------------------|---------------------------------|
| `Sweep`                      | `ControlVar` + `AxisSpecs`      |
| `Readout`                    | `ReadoutSpecs`                  |
| `Procedure` sweeps + readouts| `MeasurementSchema`             |
| Sweep experiment data        | `ZarrWriter.write_point()`      |
| Monitor data                 | `StreamingWriter.append()`      |
| Output numbering             | `ExperimentID`                  |

**Output files per run:**

```
data/0001/
  vault.zarr/         # Zarr v3 group with all arrays
    Vgt/              # control values
    lockin_X/         # measurement data
    S21/              # trace data
  metadata.yaml       # status, date, schema, tags, user metadata
  procedure.yaml      # full procedure spec: sweeps, readouts, settings, hook source
  events.yaml         # parameter changes during monitor runs (only if any occurred)
  limit_log.yaml      # controller limit violations (only if any occurred)
```

**Metadata includes:** date (ISO 8601), schema, tags, and all user metadata from both the context and the procedure.

---

## Architecture

```
                     +-----------------------+
                     |   InstrumentAdapter   |
                     |  pymeasure / qcodes   |
                     |      / custom         |
                     +-----------+-----------+
                                 |
                    +------------+------------+
                    |                         |
              +-----+------+          +------+-----+
              | Controller |          |   Readout  |
              |  (get/set) |          | (read-only)|
              +-----+------+          +------+-----+
                    |                         |
                    +------------+------------+
                                 |
                     +-----------+-----------+
                     |         Bench        |
                     |  bench["Vgt"] = 0.4  |
                     |  bench["S21"]        |
                     |  bench.snapshot()    |
                     +-----------+-----------+
                                 |
                    +------------+------------+
                    |                         |
              +-----+------+       +----------+---------+
              |  Procedure |       | MonitorProcedure   |
              | (1D/2D/3D) |       | (time-series)      |
              +-----+------+       +----------+---------+
                    |                         |
                    +------------+------------+
                                 |
                     +-----------+-----------+
                     |  ExperimentRunner     |
                     |  .run() / .arun()    |
                     |  .run_monitor()      |
                     +-----------+-----------+
                                 |
              +------------------+------------------+
              |                  |                   |
    +---------+------+ +--------+--------+ +--------+--------+
    | WriteStrategy  | | SweepwiseSt.    | | AllStrategy     |
    | (base class)   | | PlanewiseSt.    | | (buffer all)    |
    | PointwiseSt.   | | (buffer inner)  | |                 |
    +--------+-------+ +--------+--------+ +--------+--------+
              |                  |                   |
              +------------------+------------------+
                                 |
                     +-----------+-----------+
                     |       zarro          |
                     |  ZarrWriter          |
                     |  StreamingWriter     |
                     |  vault.zarr +        |
                     |  metadata.yaml       |
                     +-----------------------+

Live plotting (optional, passed to runner as plotter=...):

              +----------------------------+
              |       PlotterBase          |
              |  build_figure_dict()       |
              |  update_line/heatmap/...   |
              |  on_data_changed() [hook]  |
              |  _start_server() [abstract]|
              +-------------+--------------+
                            |
              +-------------+--------------+
              |                            |
    +---------+----------+    +-----------+-----------+
    |    DashPlotter      |    |    TaipyPlotter       |
    |  (LivePlotter)      |    |                       |
    |  Werkzeug + Dash    |    |  taipy.gui + push     |
    |  dcc.Interval poll  |    |  broadcast_callback   |
    +---------------------+    +-----------------------+
```
