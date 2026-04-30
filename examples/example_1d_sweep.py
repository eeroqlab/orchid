"""Example: 1D sweep with simulated instruments.

Demonstrates the full orchid pipeline:
  1. Create mock instruments
  2. Set up ExperimentContext with parameters and readouts
  3. Define a Procedure with a 1D sweep
  4. Run the experiment
  5. Read back saved data
"""

import numpy as np
from src.orchid import (
    DataKind,
    ExperimentContext,
    ExperimentRunner,
    Parameter,
    Procedure,
    Readout,
    Sweep,
)


# ── 1. Mock instruments ──────────────────────────────────────────────

class MockVoltageSource:
    """Simulated voltage source with one channel."""
    def __init__(self):
        self._voltage = 0.0

    @property
    def voltage(self):
        return self._voltage

    @voltage.setter
    def voltage(self, v):
        self._voltage = v


class MockLockin:
    """Simulated lockin amplifier returning a signal dependent on voltage."""
    def __init__(self, source: MockVoltageSource):
        self._source = source

    def read_x(self) -> float:
        """Return X channel — a Lorentzian peak at V=0.5."""
        v = self._source.voltage
        return 1.0 / (1.0 + ((v - 0.5) / 0.05) ** 2) + np.random.normal(0, 0.01)


# ── 2. Set up context ────────────────────────────────────────────────

vs = MockVoltageSource()
lockin = MockLockin(vs)

ctx = ExperimentContext(data_root="./example_data", metadata={"sample": "test_chip"})

# Register instruments
ctx.add_instrument("voltage_source", vs)
ctx.add_instrument("lockin", lockin)

# Register control parameter
Vgt = ctx.add_parameter("Vgt", instrument="voltage_source", attr="voltage", unit="V")

# Register readout
lockin_X = ctx.add_readout(
    "lockin_X",
    kind=DataKind.SCALAR,
    get_func=lockin.read_x,
    unit="V",
    contains="X quadrature",
)


# ── 3. Define procedure ──────────────────────────────────────────────

proc = Procedure(
    name="gate_sweep",
    context=ctx,
    sweeps=[Sweep(parameter=Vgt, values=np.linspace(0, 1, 101))],
    readouts=["lockin_X"],
    settle_time=0.0,  # no settle needed for simulation
    tags=["example", "1d"],
)


# ── 4. Run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runner = ExperimentRunner()
    data_dir = runner.run(proc)

    # ── 5. Read back ──────────────────────────────────────────────────
    import zarr

    z = zarr.open(str(data_dir / "vault.zarr"), mode="r")
    print(f"\nArrays in vault: {list(z.keys())}")
    print(f"Vgt shape: {z['Vgt'][:].shape}")
    print(f"lockin_X shape: {z['lockin_X'][:].shape}")
    print(f"lockin_X first 5 values: {z['lockin_X'][:5]}")
