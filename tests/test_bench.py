"""Tests for Bench controller registration and binding helpers."""

import pytest

from orchid import Bench, Controller, LimitPolicy


def test_add_controller_registers_limits_and_limit_policy():
    bench = Bench()

    controller = bench.add_controller(
        "gate",
        set_func=lambda value: None,
        get_func=lambda: 0.0,
        unit="V",
        limits=(-1.0, 1.0),
        limit_policy=LimitPolicy.RAISE,
    )

    assert isinstance(controller, Controller)
    assert controller.limits == (-1.0, 1.0)
    assert controller.limit_policy == LimitPolicy.RAISE
    assert bench.controllers["gate"] is controller


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

    controller = bench.add_controller_binding("reservoir_v", ["stp", "stm"])

    controller.set(0.5)
    assert stp_value[0] == 0.5
    assert stm_value[0] == 0.5
    assert controller.get() == {
        "stp": pytest.approx(0.5),
        "stm": pytest.approx(0.5),
    }

    stm_value[0] = 0.3
    assert controller.get() == {
        "stp": pytest.approx(0.5),
        "stm": pytest.approx(0.3),
    }


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
    controller = bench.add_controller_binding("pair", ["left", "right"])

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
