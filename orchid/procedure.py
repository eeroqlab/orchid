"""Procedure definitions — what to do in an experiment."""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

import numpy as np

from .context import ExperimentContext
from .parameter import Parameter


def _describe_hook(fn) -> dict | None:
    """Return a serializable description of a hook callable, or None.

    Strategy (best-effort, graceful degradation):
      - Named function in .py file  → full source captured via inspect
      - Named function in Jupyter   → full source via IPython linecache patch
      - Lambda                      → name + note, no source (too noisy)
      - Any callable                → name + module as fallback
    """
    if fn is None:
        return None

    result: dict = {"name": getattr(fn, "__name__", repr(fn))}

    # Docstring — clean one-liner for summary display
    doc = inspect.getdoc(fn)
    if doc:
        result["doc"] = doc

    if result["name"] == "<lambda>":
        result["note"] = "lambda — source not recorded"
        return result

    # Full source — works for .py files and Jupyter cells (IPython patches linecache)
    try:
        src = inspect.getsource(fn)
        result["source"] = textwrap.dedent(src)
    except (OSError, TypeError):
        module = getattr(fn, "__module__", None)
        if module:
            result["module"] = module

    return result


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


class MultiSweep:
    """Sweep multiple parameters simultaneously along a shared axis.

    All parameters step together: at point i,
    ``parameters[0]`` is set to ``values[0][i]``,
    ``parameters[1]`` is set to ``values[1][i]``, and so on.

    Parameters
    ----------
    parameters : list of Parameter or str
        Parameters to sweep simultaneously. String names are resolved
        against the context in ``Procedure.__post_init__``.
    values : list of array-like
        One values array per parameter. All arrays must have the same length.
    reverse : bool
        If True, append reversed values for hysteresis on all arrays.

    Examples
    --------
    Gate and back-gate swept together::

        MultiSweep(
            parameters=["Vgt", "Vbg"],
            values=[np.linspace(0, 1, 100), np.linspace(0, 5, 100)],
        )
    """

    def __init__(self, parameters: list, values: list, reverse: bool = False):
        self.all_values = [np.asarray(v) for v in values]
        self.reverse = reverse

        if len(parameters) != len(self.all_values):
            raise ValueError(
                f"MultiSweep: number of parameters ({len(parameters)}) "
                f"must match number of value arrays ({len(self.all_values)})"
            )
        lengths = [len(v) for v in self.all_values]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"MultiSweep: all value arrays must have the same length, got {lengths}"
            )
        if reverse:
            self.all_values = [np.concatenate([v, v[::-1]]) for v in self.all_values]

        # Keep parameters as-is — strings are resolved later by Procedure.__post_init__
        self.parameters = list(parameters)

    # ── Sweep-compatible interface ────────────────────────────────────

    @property
    def values(self) -> np.ndarray:
        """Primary (first) parameter's values — used for iteration and snake logic."""
        return self.all_values[0]

    @property
    def length(self) -> int:
        """Number of points along this axis."""
        return len(self.all_values[0])

    @property
    def parameter(self):
        """First parameter — used for naming and plotter axis labels."""
        return self.parameters[0]

    @property
    def name(self) -> str:
        """Combined name, e.g. ``'Vgt+Vbg'``."""
        return "+".join(
            p.name if hasattr(p, "name") else str(p) for p in self.parameters
        )

    def __repr__(self) -> str:
        names = [p.name if hasattr(p, "name") else str(p) for p in self.parameters]
        return f"MultiSweep({names}, length={self.length})"


@dataclass
class Procedure:
    """Experiment procedure definition.

    Parameters
    ----------
    name : str
        Name for this experiment run.
    context : ExperimentContext
        The lab bench configuration.
    sweeps : list of Sweep or MultiSweep
        Sweep axes. Length determines dimensionality: 1=1D, 2=2D, 3=3D.
        Outer sweeps first: sweeps[0] is slowest, sweeps[-1] is fastest.
        Use ``MultiSweep`` to step multiple parameters simultaneously along
        one axis.
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
        - PLANEWISE: buffer the two innermost sweeps, write after each plane.
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

    Properties
    ----------
    ndim : int
        Number of sweep axes.
    shape : tuple of int
        Shape of the sweep grid, e.g. ``(100, 50)`` for a 2D scan.

    Methods
    -------
    summary()
        Print a formatted table of sweeps, readouts, settings, and hooks.
        Called automatically by the runner before each experiment.
    to_dict()
        Serialize procedure to a plain dict. Saved as ``procedure.yaml``
        in the data directory alongside ``metadata.yaml``.
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
        for sweep in self.sweeps:
            if isinstance(sweep, MultiSweep):
                sweep.parameters = [
                    self.context.parameters[p] if isinstance(p, str) else p
                    for p in sweep.parameters
                ]
            elif isinstance(sweep.parameter, str):
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

    def to_dict(self) -> dict:
        """Serialize procedure to a plain dict for YAML / summary display."""
        sweeps_list = []
        for i, s in enumerate(self.sweeps):
            if isinstance(s, MultiSweep):
                params_info = [
                    {
                        "name": p.name,
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "unit": p.unit,
                    }
                    for p, vals in zip(s.parameters, s.all_values)
                ]
                sweeps_list.append({
                    "axis": i,
                    "type": "multi",
                    "n": s.length,
                    "reverse": s.reverse,
                    "parameters": params_info,
                })
            else:
                sweeps_list.append({
                    "axis": i,
                    "type": "single",
                    "parameter": s.parameter.name,
                    "min": float(np.min(s.values)),
                    "max": float(np.max(s.values)),
                    "n": s.length,
                    "unit": s.parameter.unit,
                    "reverse": s.reverse,
                })

        readouts_list = [
            {
                "name": self.context.readouts[rname].name,
                "kind": self.context.readouts[rname].kind.value,
                "unit": self.context.readouts[rname].unit,
                "shape": list(self.context.readouts[rname].shape)
                         if self.context.readouts[rname].shape else None,
            }
            for rname in self.readouts
        ]

        total_points = int(np.prod([s.length for s in self.sweeps])) if self.sweeps else 0

        hooks = {
            name: _describe_hook(getattr(self, name))
            for name in (
                "before_experiment", "after_experiment",
                "before_point", "after_point",
                "before_sweep", "after_sweep",
            )
        }

        return {
            "kind": "sweep",
            "name": self.name,
            "tags": list(self.tags),
            "ndim": self.ndim,
            "shape": list(self.shape),
            "total_points": total_points,
            "sweeps": sweeps_list,
            "readouts": readouts_list,
            "settings": {
                "write_mode": str(self.write_mode),
                "settle_time": self.settle_time,
                "snake": self.snake,
                "error_policy": str(self.error_policy),
                "max_retries": self.max_retries,
            },
            "hooks": hooks,
            "estimated_duration_s": round(total_points * self.settle_time, 2),
        }

    def summary(self) -> None:
        """Print a formatted summary table of this procedure."""
        from .utils import _format_procedure_summary
        print(_format_procedure_summary(self.to_dict()))


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

    Methods
    -------
    summary()
        Print a formatted table of readouts, settings, and hooks.
        Called automatically by the runner before each monitor run.
    to_dict()
        Serialize procedure to a plain dict. Saved as ``procedure.yaml``
        in the data directory alongside ``metadata.yaml``.
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

    def to_dict(self) -> dict:
        """Serialize monitor procedure to a plain dict for YAML / summary display."""
        readouts_list = [
            {
                "name": self.context.readouts[rname].name,
                "kind": self.context.readouts[rname].kind.value,
                "unit": self.context.readouts[rname].unit,
                "shape": list(self.context.readouts[rname].shape)
                         if self.context.readouts[rname].shape else None,
            }
            for rname in self.readouts
        ]

        hooks = {
            name: _describe_hook(getattr(self, name))
            for name in ("before_experiment", "after_experiment", "after_point")
        }

        return {
            "kind": "monitor",
            "name": self.name,
            "tags": list(self.tags),
            "readouts": readouts_list,
            "settings": {
                "interval": self.interval,
                "duration": self.duration,
                "chunk_size": self.chunk_size,
            },
            "hooks": hooks,
        }

    def summary(self) -> dict:
        """Print a formatted summary table of this monitor procedure.

        Returns the serialized dict so callers can reuse it without a
        second ``to_dict()`` call (e.g. to write ``procedure.yaml``).
        """
        from .utils import _format_procedure_summary
        d = self.to_dict()
        print(_format_procedure_summary(d))
        return d
