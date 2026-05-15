"""Tests for Bench controller registration and binding helpers."""

import numpy as np
import pytest

from orchid import Bench, Controller, DataKind, LimitPolicy, PhysicalReadout, VirtualReadout


def test_add_controller_registers_limits_and_limit_policy():
    bench = Bench()

    bench.add_controller(
        "gate",
        set_func=lambda value: None,
        get_func=lambda: 0.0,
        unit="V",
        limits=(-1.0, 1.0),
        limit_policy=LimitPolicy.RAISE,
    )

    controller = bench.controllers["gate"]
    assert isinstance(controller, Controller)
    assert controller.limits == (-1.0, 1.0)
    assert controller.limit_policy == LimitPolicy.RAISE


def test_add_controller_binding_fans_out_set_and_reads_bound_values():
    bench = Bench()
    stp_value = [0.0]
    stm_value = [0.0]
    bench.add_controller(
        "stp",
        set_func=lambda value: stp_value.__setitem__(0, value),
        get_func=lambda: stp_value[0],
        unit="V",
    )
    bench.add_controller(
        "stm",
        set_func=lambda value: stm_value.__setitem__(0, value),
        get_func=lambda: stm_value[0],
        unit="V",
    )

    bench.add_controller_binding("reservoir_v", ["stp", "stm"])
    controller = bench.controllers["reservoir_v"]

    controller.set(0.5)
    assert stp_value[0] == pytest.approx(0.5)
    assert stm_value[0] == pytest.approx(0.5)

    stm_value[0] = 0.3
    assert stp_value[0] == pytest.approx(0.5)
    assert stm_value[0] == pytest.approx(0.3)


def test_add_controller_binding_uses_physical_controller_limits():
    bench = Bench()
    values = {"left": 0.0, "right": 0.0}
    bench.add_controller(
        "left",
        set_func=lambda value: values.__setitem__("left", value),
        get_func=lambda: values["left"],
        unit="V",
        limits=(-0.25, 0.25),
        limit_policy=LimitPolicy.LOG,
    )
    bench.add_controller(
        "right",
        set_func=lambda value: values.__setitem__("right", value),
        get_func=lambda: values["right"],
        unit="V",
        limits=(-0.5, 0.5),
        limit_policy=LimitPolicy.LOG,
    )
    bench.add_controller_binding("pair", ["left", "right"])
    controller = bench.controllers["pair"]

    controller.set(1.0)

    assert values == {"left": 0.25, "right": 0.5}
    assert controller.limit_log == []
    assert bench.controllers["left"].limit_log[-1].requested == 1.0
    assert bench.controllers["left"].limit_log[-1].clamped == 0.25
    assert bench.controllers["right"].limit_log[-1].requested == 1.0
    assert bench.controllers["right"].limit_log[-1].clamped == 0.5


def test_add_controller_binding_raises_for_missing_controller():
    bench = Bench()
    bench.add_controller("stp", set_func=lambda value: None, get_func=lambda: 0.0)

    with pytest.raises(KeyError, match="stm"):
        bench.add_controller_binding("reservoir_v", ["stp", "stm"])


def test_add_controller_binding_rejects_empty_controller_list():
    bench = Bench()

    with pytest.raises(ValueError, match="at least one controller"):
        bench.add_controller_binding("empty", [])


# ── VirtualReadout tests ──────────────────────────────────────────────────────

def test_add_virtual_readout_registers_and_computes():
    bench = Bench()
    data_store = [0.0]
    bench.add_readout("raw", kind=DataKind.SCALAR, get_func=lambda: data_store[0])

    def double(data):
        return data["raw"] * 2.0

    vrd = bench.add_virtual_readout("doubled", sources=["raw"], transform=double)

    assert isinstance(vrd, VirtualReadout)
    assert bench.readouts["doubled"] is vrd
    assert vrd.compute({"raw": 3.0}) == pytest.approx(6.0)


def test_add_virtual_readout_raises_for_missing_source():
    bench = Bench()

    with pytest.raises(KeyError, match="missing_src"):
        bench.add_virtual_readout("vrd", sources=["missing_src"], transform=lambda d: d)


def test_add_virtual_readout_rejects_virtual_source():
    bench = Bench()
    bench.add_readout("raw", kind=DataKind.SCALAR, get_func=lambda: 0.0)
    bench.add_virtual_readout("v1", sources=["raw"], transform=lambda d: d["raw"])

    with pytest.raises(ValueError, match="virtual-to-virtual"):
        bench.add_virtual_readout("v2", sources=["v1"], transform=lambda d: d["v1"])


def test_virtual_readout_shape_required_for_non_scalar():
    bench = Bench()
    bench.add_readout("raw", kind=DataKind.TRACE, shape=(10,), get_func=lambda: np.zeros(10))

    with pytest.raises(ValueError, match="shape is required"):
        bench.add_virtual_readout(
            "vrd", sources=["raw"], transform=lambda d: d["raw"], kind=DataKind.TRACE
        )


def test_bench_getitem_raises_for_virtual_readout():
    bench = Bench()
    bench.add_readout("raw", kind=DataKind.SCALAR, get_func=lambda: 1.0)
    bench.add_virtual_readout("vrd", sources=["raw"], transform=lambda d: d["raw"])

    with pytest.raises(RuntimeError, match="VirtualReadout"):
        _ = bench["vrd"]


def test_virtual_readout_acompute_async():
    import asyncio

    bench = Bench()
    bench.add_readout("raw", kind=DataKind.SCALAR, get_func=lambda: 5.0)
    bench.add_virtual_readout("vrd", sources=["raw"], transform=lambda d: d["raw"] + 1)

    result = asyncio.run(bench.readouts["vrd"].acompute({"raw": 5.0}))
    assert result == pytest.approx(6.0)
