"""Tests for ExperimentRunner readout execution."""

import numpy as np
import zarr

from orchid import Bench, DataKind, ExperimentRunner, Procedure, Sweep, WriteMode


def test_runner_records_virtual_readout_without_recording_sources(tmp_path):
    state = {"gate": 0.0}
    bench = Bench(data_root=tmp_path)
    bench.add_controller(
        "gate",
        set_func=lambda value: state.__setitem__("gate", value),
        get_func=lambda: state["gate"],
        unit="V",
    )
    bench.add_readout("raw", kind=DataKind.SCALAR, get_func=lambda: state["gate"])
    bench.add_virtual_readout(
        "double",
        sources=["raw"],
        transform=lambda data: 2.0 * data["raw"],
        kind=DataKind.SCALAR,
    )
    procedure = Procedure(
        name="virtual_only",
        bench=bench,
        sweeps=[Sweep("gate", np.array([0.0, 1.0]))],
        readouts=["double"],
        write_mode=WriteMode.ALL,
    )

    data_dir = ExperimentRunner(use_experiment_id=False).run(
        procedure,
        return_path=True,
    )

    vault = zarr.open(str(data_dir / "vault.zarr"), mode="r")
    assert "double" in vault
    assert "raw" not in vault
    np.testing.assert_allclose(vault["double"][...], np.array([0.0, 2.0]))
