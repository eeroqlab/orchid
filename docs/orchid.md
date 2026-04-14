# Orchid Documentation

**Orchestrating Instruments & Data** — a Python package for lab experiment control.

Orchid provides a clean pipeline for running automated lab experiments:  
**Instruments** &rarr; **Parameters & Readouts** &rarr; **ExperimentContext** &rarr; **Procedure** &rarr; **Runner** &rarr; **Data (zarro)**

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tutorial](#tutorial)
  - [Step 1: Instruments](#step-1-instruments)
  - [Step 2: Parameters & Readouts](#step-2-parameters--readouts)
  - [Step 3: ExperimentContext](#step-3-experimentcontext)
  - [Step 4: Procedures](#step-4-procedures)
  - [Step 5: Running Experiments](#step-5-running-experiments)
  - [Step 6: Reading Data Back](#step-6-reading-data-back)
- [Cookbook](#cookbook)
  - [1D Sweep](#1d-sweep)
  - [2D Sweep](#2d-sweep)
  - [2D Sweep with Snake Scan](#2d-sweep-with-snake-scan)
  - [Hysteresis (Forward + Backward)](#hysteresis-forward--backward)
  - [Time-Series Monitoring](#time-series-monitoring)
  - [Custom Hooks](#custom-hooks)
  - [Mixed Readouts (Scalar + Trace)](#mixed-readouts-scalar--trace)
  - [Async Usage](#async-usage)
- [API Reference](#api-reference)
  - [InstrumentAdapter](#instrumentadapter)
  - [Parameter](#parameter)
  - [Readout](#readout)
  - [DataKind](#datakind)
  - [ExperimentContext](#experimentcontext)
  - [Sweep](#sweep)
  - [Procedure](#procedure)
  - [MonitorProcedure](#monitorprocedure)
  - [WriteMode](#writemode)
  - [ErrorPolicy](#errorpolicy)
  - [ExperimentRunner](#experimentrunner)
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

**Dependencies:** `numpy`, `tqdm`, `zarro`, `tabulate`

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

# Set up context
ctx = ExperimentContext(data_root="./data", metadata={"sample": "chip_A1"})
ctx.add_instrument("vs", vs)
ctx.add_parameter("Vgt", instrument="vs", attr="voltage", unit="V")
ctx.add_readout("signal", kind="scalar", get_func=lambda: vs.voltage ** 2, unit="V")

# Set/get values (pymeasure style)
ctx["Vgt"] = 0.5
print(ctx["Vgt"])        # 0.5
print(ctx["signal"])     # 0.25

# Snapshot
ctx.snapshot()

# Define and run a 1D sweep
proc = Procedure(
    name="gate_sweep",
    context=ctx,
    sweeps=[Sweep("Vgt", np.linspace(0, 1, 101))],
    readouts=["signal"],
)
data_dir = ExperimentRunner().run(proc)
```

---

## Tutorial

### Step 1: Instruments

Orchid supports three instrument backends: **pymeasure**, **qcodes**, and **custom** (any Python object). Register instruments through `ExperimentContext.add_instrument()` or create adapters directly.

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
ctx.add_instrument("keithley", keithley)
# backend is auto-detected as "custom"
```

#### pymeasure instruments

```python
from pymeasure.instruments.keithley import Keithley2400

keithley = Keithley2400("GPIB::24")
ctx.add_instrument("keithley", keithley, backend="pymeasure")
# get/set uses property access: keithley.source_voltage
```

#### qcodes instruments

```python
from qcodes.instrument_drivers.stanford_research import SR830

lockin = SR830("lockin", "GPIB::8")
ctx.add_instrument("lockin", lockin, backend="qcodes")
# get/set uses qcodes Parameter API: lockin.frequency.get() / .set()
```

#### Auto-detection

When `backend="auto"` (the default), orchid inspects the class MRO to detect pymeasure or qcodes instruments automatically:

```python
ctx.add_instrument("lockin", lockin)  # auto-detects qcodes
```

#### Direct adapter creation

You can also create adapters without a context:

```python
from orchid import InstrumentAdapter

adapter = InstrumentAdapter.from_qcodes("lockin", lockin)
adapter = InstrumentAdapter.from_pymeasure("keithley", keithley)
adapter = InstrumentAdapter.auto("lockin", lockin)
```

---

### Step 2: Parameters & Readouts

**Parameters** are named controls that map to instrument channels. They support both get and set.  
**Readouts** are read-only measurement channels that acquire data.

There are several ways to define a parameter depending on your instrument driver.

#### Parameters for pymeasure instruments

pymeasure instruments expose parameters as Python properties. Pass the property name as `attr`:

```python
from pymeasure.instruments.keithley import Keithley2400
from pymeasure.instruments.srs import SR830

keithley = Keithley2400("GPIB::24")
lockin = SR830("GPIB::8")

ctx.add_instrument("keithley", keithley)
ctx.add_instrument("lockin", lockin)

# attr matches the pymeasure property name
# internally uses: keithley.source_voltage / keithley.source_voltage = val
ctx.add_parameter("Vgt", instrument="keithley", attr="source_voltage", unit="V")
ctx.add_parameter("I_compliance", instrument="keithley", attr="compliance_current", unit="A")

# lockin frequency
# internally uses: lockin.frequency / lockin.frequency = val
ctx.add_parameter("fac", instrument="lockin", attr="frequency", unit="Hz")
ctx.add_parameter("sensitivity", instrument="lockin", attr="sensitivity", unit="V")
```

To find available property names, check the pymeasure docs or use `dir(instrument)`.

#### Parameters for qcodes instruments

qcodes instruments expose parameters as `qcodes.Parameter` objects with `.get()` / `.set()` methods. The adapter handles this automatically:

```python
from qcodes.instrument_drivers.stanford_research import SR830
from qcodes.instrument_drivers.yokogawa import GS200

yoko = GS200("yoko", "GPIB::1")
lockin = SR830("lockin", "GPIB::8")

ctx.add_instrument("yoko", yoko)
ctx.add_instrument("lockin", lockin)

# attr matches the qcodes parameter name
# internally uses: yoko.voltage.get() / yoko.voltage.set(val)
ctx.add_parameter("Vgt", instrument="yoko", attr="voltage", unit="V")
ctx.add_parameter("I_range", instrument="yoko", attr="current_range", unit="A")

# internally uses: lockin.frequency.get() / lockin.frequency.set(val)
ctx.add_parameter("fac", instrument="lockin", attr="frequency", unit="Hz")
ctx.add_parameter("amplitude", instrument="lockin", attr="amplitude", unit="V")
```

To find available parameter names, use `instrument.print_readable_snapshot()` or `instrument.parameters.keys()`.

#### Parameters via InstrumentAdapter directly

You can create an `InstrumentAdapter` first and pass it to `add_parameter` instead of using a registered name:

```python
from orchid import InstrumentAdapter

adapter = InstrumentAdapter.from_pymeasure("keithley", keithley)
# or
adapter = InstrumentAdapter.from_qcodes("yoko", yoko)
# or
adapter = InstrumentAdapter.from_custom("my_device", my_device)

ctx.add_parameter("Vgt", instrument=adapter, attr="voltage", unit="V")
```

This is useful when you want to manage adapters outside the context, or use the same adapter for multiple parameters without registering it.

#### Parameters via custom callables

For full flexibility — or when the get/set logic doesn't map cleanly to a single attribute — pass `get_func` and/or `set_func`:

```python
# Custom getter + setter (any arbitrary logic)
fac = ctx.add_parameter(
    "fac",
    get_func=lambda: lockin.driver.frequency,
    set_func=lambda v: setattr(lockin.driver, 'frequency', v),
    unit="Hz",
)

# Read-only parameter (no set_func)
ctx.add_parameter(
    "T_mc",
    get_func=lambda: fridge.get_temperature("MC"),
    unit="K",
)

# Computed parameter (e.g., converting DAC codes to voltage)
ctx.add_parameter(
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
| Computed / derived quantity             | `get_func=..., set_func=...` with logic     |

#### Readouts

Readouts always use a `get_func` callable. The `kind` determines the data shape:

```python
# Scalar readout (single number per point)
ctx.add_readout("lockin_X", kind="scalar", 
    get_func=lockin_amplifier.read_x, unit="V")

# Trace readout (1D array per point)
ctx.add_readout("S21", kind="trace", shape=(1601,),
    get_func=vna.get_trace, unit="dB",
    contains="transmission magnitude")

# Image readout (2D array per point)
ctx.add_readout("camera", kind="image", shape=(480, 640),
    get_func=camera.capture, unit="counts")
```

| `kind`   | `shape`        | Data per point      |
|----------|----------------|---------------------|
| `scalar` | not needed     | single `float`      |
| `trace`  | `(N,)`         | 1D `ndarray`        |
| `image`  | `(H, W)`       | 2D `ndarray`        |

---

### Step 3: ExperimentContext

The `ExperimentContext` is the central container that holds your entire lab bench configuration: instruments, parameters, readouts, and metadata.

```python
ctx = ExperimentContext(
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
ctx["Vgt"] = 0.4

# Read a parameter or readout
voltage = ctx["Vgt"]       # reads from instrument
signal  = ctx["lockin_X"]  # acquires measurement
```

#### Snapshot

Print a table of all current values:

```python
ctx.snapshot()
```

Output:
```
Name      Type    Value    Unit
--------  ------  -------  ------
Vgt       param   0.4      V
fac       param   2500.0   Hz
lockin_X  scalar  0.0023   V
S21       trace   [...]    dB
```

Print only specific parameters:

```python
ctx.snapshot(["Vgt", "lockin_X"])
```

#### Removing instruments, parameters, and readouts

```python
ctx.remove_parameter("Vgt")
ctx.remove_readout("S21")
ctx.remove_instrument("keithley")
# also removes any parameters that depend on it
```

#### Accessing raw objects

The `Parameter` and `Readout` objects are accessible when you need them (e.g., for `Sweep` setup):

```python
ctx.parameters["Vgt"]   # -> Parameter object
ctx.readouts["S21"]      # -> Readout object
ctx.instruments["keithley"]  # -> InstrumentAdapter object
```

---

### Step 4: Procedures

Procedures define **what** to do. Two types are available:

#### Procedure (sweeps)

```python
proc = Procedure(
    name="gate_sweep",
    context=ctx,
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

You can reference parameters by name (`"Vgt"`) or by the `Parameter` object directly.

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
    context=ctx,
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
    context=ctx,
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
    context=ctx,
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
data_dir = runner.run_monitor(monitor)            # sync
data_dir = await runner.arun_monitor(monitor)     # async
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

## Cookbook

### 1D Sweep

```python
proc = Procedure(
    name="iv_curve",
    context=ctx,
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
    context=ctx,
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
    context=ctx,
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
    context=ctx,
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
    context=ctx,
    readouts=["lockin_X", "temperature"],
    interval=1.0,       # 1 second between reads
    duration=3600.0,     # 1 hour
    tags=["stability"],
)
data_dir = ExperimentRunner().run_monitor(monitor)
```

With a stop condition:

```python
monitor = MonitorProcedure(
    name="cooldown_watch",
    context=ctx,
    readouts=["temperature"],
    interval=5.0,
    duration=None,  # run indefinitely
    stop_condition=lambda data: data["temperature"] < 0.01,
)
```

Stop manually with `Ctrl+C` — data is always saved.

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
        print(f"Point {index}: Vgt={ctx['Vgt']:.3f}")

proc = Procedure(
    name="with_hooks",
    context=ctx,
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
ctx.add_readout("lockin_X", kind="scalar", get_func=lockin.read_x, unit="V")
ctx.add_readout("S21", kind="trace", shape=(1601,), get_func=vna.get_trace, unit="dB")
ctx.add_readout("frame", kind="image", shape=(480, 640), get_func=camera.snap, unit="counts")

proc = Procedure(
    name="mixed",
    context=ctx,
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

| Attribute  | Type   | Description                    |
|------------|--------|--------------------------------|
| `name`     | `str`  | Human-readable instrument name |
| `driver`   | `Any`  | Raw instrument object          |
| `backend`  | `str`  | `"pymeasure"`, `"qcodes"`, or `"custom"` |

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

### Parameter

```python
from orchid import Parameter
```

A named control parameter mapped to an instrument channel.

| Argument     | Type                          | Default | Description                     |
|--------------|-------------------------------|---------|---------------------------------|
| `name`       | `str`                         | required| Short label (e.g. `"Vgt"`)     |
| `instrument` | `InstrumentAdapter` or `None` | `None`  | Instrument this parameter uses  |
| `attr`       | `str` or `None`               | `None`  | Attribute name on instrument    |
| `get_func`   | `callable` or `None`          | `None`  | Custom getter (overrides adapter)|
| `set_func`   | `callable` or `None`          | `None`  | Custom setter (overrides adapter)|
| `unit`       | `str` or `None`               | `None`  | Physical unit                   |

| Method                        | Description             |
|-------------------------------|-------------------------|
| `get() -> Any`                | Read current value      |
| `set(value) -> None`          | Write value             |
| `await aget() -> Any`         | Async read              |
| `await aset(value) -> None`   | Async write             |

**Precedence:** `get_func`/`set_func` override `instrument.get(attr)`/`instrument.set(attr)`.

---

### Readout

```python
from orchid import Readout
```

A read-only measurement channel.

| Argument   | Type                 | Default  | Description                            |
|------------|----------------------|----------|----------------------------------------|
| `name`     | `str`                | required | Label (e.g. `"S21"`)                   |
| `kind`     | `DataKind`           | required | `SCALAR`, `TRACE`, or `IMAGE`          |
| `get_func` | `callable`           | `None`   | Acquisition function                   |
| `shape`    | `tuple` or `None`    | `None`   | Required for `TRACE` and `IMAGE`       |
| `unit`     | `str` or `None`      | `None`   | Physical unit                          |
| `contains` | `str` or `None`      | `None`   | Description of what is measured        |

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

### ExperimentContext

```python
from orchid import ExperimentContext
```

Central container for the entire lab bench configuration.

| Argument    | Type              | Default   | Description                        |
|-------------|-------------------|-----------|------------------------------------|
| `data_root` | `str` or `Path`   | `"./data"`| Root directory for saved data      |
| `metadata`  | `dict`            | `{}`      | User metadata (sample, operator, etc.) |

| Attribute      | Type                             | Description           |
|----------------|----------------------------------|-----------------------|
| `instruments`  | `dict[str, InstrumentAdapter]`   | Registered instruments|
| `parameters`   | `dict[str, Parameter]`           | Registered parameters |
| `readouts`     | `dict[str, Readout]`             | Registered readouts   |
| `data_root`    | `Path`                           | Data output root      |
| `metadata`     | `dict`                           | User metadata         |

#### Methods

**`add_instrument(name, instrument, backend="auto") -> InstrumentAdapter`**

Register an instrument. `backend` can be `"auto"`, `"pymeasure"`, `"qcodes"`, or `"custom"`.

**`add_parameter(name, instrument=None, attr=None, get_func=None, set_func=None, unit=None) -> Parameter`**

Register a control parameter. `instrument` can be an `InstrumentAdapter` object or the string name of a registered instrument.

**`add_readout(name, kind, get_func, shape=None, unit=None, contains=None) -> Readout`**

Register a measurement readout. `kind` can be a `DataKind` enum or string (`"scalar"`, `"trace"`, `"image"`).

**`remove_instrument(name) -> None`**

Remove an instrument and **all parameters that depend on it**. Raises `KeyError` if not found.

**`remove_parameter(name) -> None`**

Remove a parameter. Raises `KeyError` if not found.

**`remove_readout(name) -> None`**

Remove a readout. Raises `KeyError` if not found.

**`ctx[name]`** — Get current value (calls `parameter.get()` or `readout.read()`).

**`ctx[name] = value`** — Set parameter value (calls `parameter.set(value)`).

**`snapshot(names=None) -> None`**

Print a formatted table of current values using `tabulate`.

| Argument | Type               | Default | Description                    |
|----------|--------------------|---------|--------------------------------|
| `names`  | `list[str]` or `None` | `None`  | Filter to these names; `None` = all |

---

### Sweep

```python
from orchid import Sweep
```

Defines a sweep over one parameter.

| Argument    | Type                  | Default | Description                          |
|-------------|-----------------------|---------|--------------------------------------|
| `parameter` | `Parameter` or `str`  | required| Parameter to sweep (or its name)     |
| `values`    | `array-like`          | required| Sweep values                         |
| `reverse`   | `bool`                | `False` | Append reversed values (hysteresis)  |

| Property  | Type  | Description                         |
|-----------|-------|-------------------------------------|
| `length`  | `int` | Number of sweep points              |

When `reverse=True`, the values array is doubled: `[forward, reversed]`.

---

### Procedure

```python
from orchid import Procedure
```

Experiment procedure for sweep-based measurements.

| Argument           | Type                | Default              | Description                                 |
|--------------------|---------------------|----------------------|---------------------------------------------|
| `name`             | `str`               | required             | Experiment name                             |
| `context`          | `ExperimentContext`  | required             | Lab bench configuration                     |
| `sweeps`           | `list[Sweep]`       | `[]`                 | Sweep axes (outer-first ordering)           |
| `readouts`         | `list[str]`         | `[]`                 | Readout names to record                     |
| `settle_time`      | `float`             | `0.0`                | Seconds to wait after set, before read      |
| `snake`            | `bool`              | `False`              | Alternate inner sweep direction             |
| `write_mode`       | `WriteMode`         | `POINTWISE`          | When to flush data to disk                  |
| `error_policy`     | `ErrorPolicy`       | `STOP_AND_SAVE`      | How to handle measurement errors            |
| `max_retries`      | `int`               | `3`                  | Retries for `RETRY_AND_SKIP` policy         |
| `tags`             | `list[str]`         | `[]`                 | Free-form tags for metadata                 |
| `metadata`         | `dict`              | `{}`                 | Additional metadata to save                 |
| `before_experiment`| `callable` or `None`| `None`               | Hook `()`: once before start                |
| `after_experiment` | `callable` or `None`| `None`               | Hook `()`: once after finish                |
| `before_point`     | `callable` or `None`| `None`               | Hook `(index_tuple)`: before each measurement |
| `after_point`      | `callable` or `None`| `None`               | Hook `(index_tuple)`: after each measurement  |
| `before_sweep`     | `callable` or `None`| `None`               | Hook `(axis_index)`: before each sweep axis   |
| `after_sweep`      | `callable` or `None`| `None`               | Hook `(axis_index)`: after each sweep axis    |

| Property  | Type            | Description                    |
|-----------|-----------------|--------------------------------|
| `ndim`    | `int`           | Number of sweep axes           |
| `shape`   | `tuple[int,...]`| Sweep grid shape               |

---

### MonitorProcedure

```python
from orchid import MonitorProcedure
```

Time-series monitoring procedure (no sweeps).

| Argument           | Type                | Default              | Description                                 |
|--------------------|---------------------|----------------------|---------------------------------------------|
| `name`             | `str`               | required             | Session name                                |
| `context`          | `ExperimentContext`  | required             | Lab bench configuration                     |
| `readouts`         | `list[str]`         | `[]`                 | Readout names to record                     |
| `interval`         | `float`             | `1.0`                | Seconds between reads                       |
| `duration`         | `float` or `None`   | `None`               | Total duration; `None` = run until stopped  |
| `stop_condition`   | `callable` or `None`| `None`               | `(data_dict) -> bool`; return `True` to stop|
| `tags`             | `list[str]`         | `[]`                 | Free-form tags                              |
| `metadata`         | `dict`              | `{}`                 | Additional metadata                         |
| `before_experiment`| `callable` or `None`| `None`               | Hook `()`: once before start                |
| `after_experiment` | `callable` or `None`| `None`               | Hook `()`: once after finish                |
| `after_point`      | `callable` or `None`| `None`               | Hook `(sample_index, data_dict)`: after each read |

Data is saved via zarro's `StreamingWriter` with a `_time` timestamp array.

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

Executes procedures and manages data flow to zarro.

| Argument             | Type   | Default | Description                           |
|----------------------|--------|---------|---------------------------------------|
| `use_experiment_id`  | `bool` | `True`  | Auto-number output directories        |

#### Methods

| Method                                  | Description                              |
|-----------------------------------------|------------------------------------------|
| `run(procedure) -> Path`                | Run sweep experiment (sync)              |
| `await arun(procedure) -> Path`         | Run sweep experiment (async)             |
| `run_monitor(procedure) -> Path`        | Run time-series monitor (sync)           |
| `await arun_monitor(procedure) -> Path` | Run time-series monitor (async)          |

All methods return the `Path` to the output data directory.

**Sweep execution flow:**

```
for each outer value:           # sweeps[0]
  set sweeps[0].parameter
  for each inner value:         # sweeps[1]
    set sweeps[1].parameter
    ...                         # sweeps[N] (recursive)
      wait settle_time
      read all readouts
      write_point(index, data)
```

**Monitor execution flow:**

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
from orchid import update_metadata, read_metadata
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
  metadata.yaml       # human-readable metadata
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
              |  Parameter |          |   Readout  |
              |  (get/set) |          | (read-only)|
              +-----+------+          +------+-----+
                    |                         |
                    +------------+------------+
                                 |
                     +-----------+-----------+
                     |  ExperimentContext    |
                     |  ctx["Vgt"] = 0.4    |
                     |  ctx["S21"]          |
                     |  ctx.snapshot()      |
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
                     +-----------+-----------+
                     |       zarro          |
                     |  ZarrWriter          |
                     |  StreamingWriter     |
                     |  vault.zarr +        |
                     |  metadata.yaml       |
                     +-----------------------+
```
