"""Orchid — Orchestrating Instruments & Data for lab experiments."""

from .instrument import InstrumentAdapter
from .controller import (
    DataKind, Controller, ControllerBase, PhysicalController, VirtualController,
    LimitPolicy, LimitEntry, Readout, PhysicalReadout, VirtualReadout,
)
from .bench import Bench
from .procedure import ErrorPolicy, MonitorProcedure, MultiSweep, Procedure, Sweep, WriteMode
from .runner import ExperimentRunner
from .plotting import EventLineConfig, LivePlotter, DashPlotter, PlotterBase, PlotSpec, PostResult
from .control_panel import ControlPanel
from .utils import apply_theme, read_events, read_limit_log, read_metadata, read_procedure, update_metadata, PALETTE

__all__ = [
    "InstrumentAdapter",
    "DataKind",
    "Controller",
    "ControllerBase",
    "PhysicalController",
    "VirtualController",
    "LimitPolicy",
    "LimitEntry",
    "Readout",
    "PhysicalReadout",
    "VirtualReadout",
    "Bench",
    "ErrorPolicy",
    "MonitorProcedure",
    "MultiSweep",
    "Procedure",
    "Sweep",
    "WriteMode",
    "ExperimentRunner",
    "EventLineConfig",
    "PlotterBase",
    "DashPlotter",
    "LivePlotter",
    "PlotSpec",
    "PostResult",
    "apply_theme",
    "PALETTE",
    "read_events",
    "read_limit_log",
    "read_metadata",
    "read_procedure",
    "update_metadata",
    "ControlPanel",
]
