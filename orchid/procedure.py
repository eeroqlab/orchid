"""Procedure definitions — what to do in an experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

import numpy as np

from .context import ExperimentContext
from .parameter import Parameter


class ErrorPolicy(StrEnum):
    """How to handle errors during a sweep."""
    STOP_AND_SAVE = "stop_and_save"
    RETRY_AND_SKIP = "retry_and_skip"
    IGNORE = "ignore"


class WriteMode(StrEnum):
    """Controls when data is flushed to disk during a sweep.

    Maps directly to zarro WriteType and determines which ZarrWriter
    method is used.

    POINTWISE : write after every single measurement point.
        Uses ``ZarrWriter.write_point(index, data)``.
        Index length = number of sweep axes.
        Safest (no data loss on crash) but most I/O.

    SWEEPWISE : buffer the innermost sweep, write after it completes.
        Uses ``ZarrWriter.write_trace(outer_index, data)``.
        Index length = number of sweep axes - 1.
        Good balance for 2D/3D scans.

    PLANEWISE : buffer the two innermost sweeps, write after they complete.
        Uses ``ZarrWriter.write_image(outer_index, data)``.
        Index length = number of sweep axes - 2.
        Useful for 3D scans where a full 2D plane is acquired per outer step.

    ALL : buffer the entire experiment, write once at the end.
        Uses ``ZarrWriter.write_all(data)``.
        No index needed. Fastest I/O but all data lost on crash.
    """
    POINTWISE = "pointwise"
    SWEEPWISE = "sweepwise"
    PLANEWISE = "planewise"
    ALL = "all"


@dataclass
class Sweep:
    """Defines a sweep over one parameter.

    Parameters
    ----------
    parameter : Parameter or str
        The parameter to sweep (or its name in the context).
    values : array-like
        Sweep values.
    reverse : bool
        If True, append reversed values for hysteresis sweep.
    """

    parameter: Parameter | str
    values: np.ndarray
    reverse: bool = False

    def __post_init__(self):
        self.values = np.asarray(self.values)
        if self.reverse:
            self.values = np.concatenate([self.values, self.values[::-1]])

    @property
    def length(self) -> int:
        return len(self.values)


@dataclass
class Procedure:
    """Experiment procedure definition.

    Parameters
    ----------
    name : str
        Name for this experiment run.
    context : ExperimentContext
        The lab bench configuration.
    sweeps : list of Sweep
        Sweep axes. Length determines dimensionality: 1=1D, 2=2D, 3=3D.
        Outer sweeps first: sweeps[0] is slowest, sweeps[-1] is fastest.
    readouts : list of str
        Names of readouts to record (must be registered in context).
    settle_time : float
        Seconds to wait after setting parameters before reading.
    snake : bool
        If True, reverse inner sweep direction on alternating outer iterations.
    write_mode : WriteMode
        Controls when data is written to disk:
        - POINTWISE (default): write after every measurement point.
        - SWEEPWISE: buffer the innermost sweep, write after it completes.
        - ALL: buffer the entire experiment, write once at the end.
    error_policy : ErrorPolicy
        How to handle measurement errors.
    max_retries : int
        Number of retries when error_policy is RETRY_AND_SKIP.
    tags : list of str
        Free-form tags for metadata.
    metadata : dict
        Additional metadata to save.

    Hooks
    -----
    before_experiment, after_experiment : callable or None
        Called once at start/end. Signature: () -> None
    before_point, after_point : callable or None
        Called around each measurement point. Signature: (index_tuple) -> None
    before_sweep, after_sweep : callable or None
        Called before/after each sweep axis. Signature: (axis_index) -> None
    """

    name: str
    context: ExperimentContext
    sweeps: list[Sweep] = field(default_factory=list)
    readouts: list[str] = field(default_factory=list)
    settle_time: float = 0.0
    snake: bool = False
    write_mode: WriteMode = WriteMode.POINTWISE
    error_policy: ErrorPolicy = ErrorPolicy.STOP_AND_SAVE
    max_retries: int = 3
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    before_experiment: Callable | None = None
    after_experiment: Callable | None = None
    before_point: Callable | None = None
    after_point: Callable | None = None
    before_sweep: Callable | None = None
    after_sweep: Callable | None = None

    def __post_init__(self):
        # Resolve string references to Parameter objects
        for i, sweep in enumerate(self.sweeps):
            if isinstance(sweep.parameter, str):
                sweep.parameter = self.context.parameters[sweep.parameter]
        # Validate readout names
        for rname in self.readouts:
            if rname not in self.context.readouts:
                raise KeyError(f"Readout {rname!r} not found in context")

    @property
    def ndim(self) -> int:
        """Experiment dimensionality (number of sweep axes)."""
        return len(self.sweeps)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the sweep grid."""
        return tuple(s.length for s in self.sweeps)


@dataclass
class MonitorProcedure:
    """Time-series monitoring — periodic reads without sweeps.

    Parameters
    ----------
    name : str
        Name for this monitoring session.
    context : ExperimentContext
        The lab bench configuration.
    readouts : list of str
        Names of readouts to record.
    interval : float
        Seconds between reads.
    duration : float or None
        Total duration in seconds. None = run until stopped.
    stop_condition : callable or None
        Called after each read with (data_dict) -> bool. Return True to stop.
    chunk_size : int
        Number of samples buffered in memory before flushing to disk (default 256).
        Smaller values reduce potential data loss on crash; larger values reduce I/O.
    tags : list of str
        Free-form tags.
    metadata : dict
        Additional metadata.

    Hooks
    -----
    before_experiment, after_experiment : callable or None
        Called once at start/end. Signature: () -> None
    after_point : callable or None
        Called after each read. Signature: (sample_index, data_dict) -> None
    """

    name: str
    context: ExperimentContext
    readouts: list[str] = field(default_factory=list)
    interval: float = 1.0
    duration: float | None = None
    stop_condition: Callable | None = None
    chunk_size: int = 256
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    before_experiment: Callable | None = None
    after_experiment: Callable | None = None
    after_point: Callable | None = None

    def __post_init__(self):
        for rname in self.readouts:
            if rname not in self.context.readouts:
                raise KeyError(f"Readout {rname!r} not found in context")
