"""Controller and Readout classes — the bridge between instruments and experiments."""

from __future__ import annotations

import abc
import asyncio
import inspect
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, NamedTuple

import numpy as np

from .instrument import InstrumentAdapter


class DataKind(StrEnum):
    """Shape of a single measurement readout (mirrors zarro.DataKind)."""
    SCALAR = "scalar"
    TRACE = "trace"
    IMAGE = "image"


class LimitPolicy(StrEnum):
    """How to respond when a set value exceeds the controller limits.

    WARN  : clamp the value, emit a warning on the first violation per run,
            then log further violations silently.
    RAISE : raise ValueError immediately — use for hard safety limits.
    LOG   : clamp silently and record every violation; inspect via
            ``controller.limit_log`` after the experiment.
    """
    WARN  = "warn"
    RAISE = "raise"
    LOG   = "log"


class LimitEntry(NamedTuple):
    """One limit-violation record.

    Attributes
    ----------
    index : tuple
        Sweep index at the time of the violation, e.g. ``(42,)`` for 1D
        or ``(3, 17)`` for 2D. Empty tuple ``()`` for manual/monitor calls.
    requested : float
        The value that was requested.
    clamped : float
        The value that was actually applied after clamping.
    """
    index: tuple
    requested: float
    clamped: float


# ══════════════════════════════════════════════════════════════════════
#  ControllerBase — shared identity, limits, and bookkeeping
# ══════════════════════════════════════════════════════════════════════

class ControllerBase(abc.ABC):
    """Abstract base for all controller types.

    Holds shared state (name, unit, limits, limit log) and enforces the
    ``set`` / ``aset`` interface.  Subclasses implement the actual
    dispatch to hardware (:class:`PhysicalController`) or to other
    controllers (:class:`VirtualController`).
    """

    def __init__(
        self,
        name: str,
        unit: str | None = None,
        limits: tuple[float, float] | None = None,
        limit_policy: LimitPolicy = LimitPolicy.WARN,
    ) -> None:
        self.name = name
        self.unit = unit
        self.limits = limits
        self.limit_policy = limit_policy
        # Runtime state — not included in repr/eq
        self._limit_hit: bool = False
        self._limit_log: list[LimitEntry] = []
        self._sweep_index: tuple = ()   # set by runner before each aset()

    # ── Limit helpers ─────────────────────────────────────────────────

    @property
    def limit_log(self) -> list[LimitEntry]:
        """All limit violations recorded since the last reset."""
        return list(self._limit_log)

    def clear_limit_log(self) -> None:
        """Reset violation log and warn-once flag. Called by the runner before each run."""
        self._limit_log = []
        self._limit_hit = False

    def _clamp(self, value: float) -> float:
        """Return value clamped to limits; record/warn/raise on violation."""
        if self.limits is None:
            return value
        lo, hi = self.limits
        clamped = max(lo, min(hi, value))
        if clamped == value:
            return value

        if self.limit_policy == LimitPolicy.RAISE:
            raise ValueError(
                f"Controller {self.name!r}: value {value} outside limits [{lo}, {hi}]"
            )

        entry = LimitEntry(index=self._sweep_index, requested=value, clamped=clamped)
        self._limit_log.append(entry)

        if self.limit_policy == LimitPolicy.WARN and not self._limit_hit:
            warnings.warn(
                f"Controller {self.name!r}: value {value} clamped to {clamped} "
                f"(limits [{lo}, {hi}]). Further violations logged silently.",
                stacklevel=3,
            )
            self._limit_hit = True

        return clamped

    # ── Abstract interface ─────────────────────────────────────────────

    @abc.abstractmethod
    def set(self, value: float) -> None:
        """Apply ``value`` (after clamping)."""

    @abc.abstractmethod
    async def aset(self, value: float) -> None:
        """Apply ``value`` asynchronously (after clamping)."""

    def get(self) -> Any:
        """Read current value.

        Raises ``RuntimeError`` for virtual controllers which have no
        readback path. Override in :class:`PhysicalController`.
        """
        raise RuntimeError(
            f"Controller {self.name!r} is a virtual controller and has no readback."
        )


# ══════════════════════════════════════════════════════════════════════
#  PhysicalController — wraps a real instrument channel
# ══════════════════════════════════════════════════════════════════════

class PhysicalController(ControllerBase):
    """A named control parameter mapped to an instrument channel.

    Provide either an ``instrument + attr`` pair or custom
    ``get_func`` / ``set_func`` callables.

    Parameters
    ----------
    name : str
        Short label, e.g. ``"Vgt"``, ``"fac"``.
    instrument : InstrumentAdapter, optional
        Instrument this controller belongs to.
    attr : str, optional
        Attribute name on the instrument.
    get_func : callable, optional
        Custom getter (overrides ``instrument.get``).
    set_func : callable, optional
        Custom setter (overrides ``instrument.set``).
    unit : str, optional
        Physical unit.
    limits : tuple[float, float] or None, optional
        ``(lo, hi)`` inclusive bounds. ``None`` means unconstrained.
    limit_policy : LimitPolicy, optional
        How to respond to limit violations. Default is ``WARN``.

    Examples
    --------
    >>> ctrl = PhysicalController("Vgt", set_func=qdac.ch01.set, get_func=qdac.ch01.get)
    >>> ctrl.set(-0.5)
    >>> ctrl.get()
    -0.5
    """

    def __init__(
        self,
        name: str,
        instrument: InstrumentAdapter | None = None,
        attr: str | None = None,
        get_func: Callable[[], Any] | None = None,
        set_func: Callable[[Any], None] | None = None,
        unit: str | None = None,
        limits: tuple[float, float] | None = None,
        limit_policy: LimitPolicy = LimitPolicy.WARN,
    ) -> None:
        super().__init__(name, unit=unit, limits=limits, limit_policy=limit_policy)
        if instrument is None and get_func is None and set_func is None:
            raise ValueError(
                f"PhysicalController {name!r}: provide instrument+attr or get_func/set_func"
            )
        self.instrument = instrument
        self.attr = attr
        self.get_func = get_func
        self.set_func = set_func

    # ── Public API ────────────────────────────────────────────────────

    def get(self) -> Any:
        """Read current value from instrument or get_func."""
        if self.get_func is not None:
            return self.get_func()
        if self.instrument is not None:
            return self.instrument.get(self.attr)
        raise RuntimeError(f"Controller {self.name!r} is set-only (no getter)")

    def set(self, value: float) -> None:
        """Clamp value to limits (if set) and apply."""
        value = self._clamp(float(value))
        if self.set_func is not None:
            self.set_func(value)
        elif self.instrument is not None and self.attr is not None:
            self.instrument.set(self.attr, value)
        else:
            raise RuntimeError(f"Controller {self.name!r} is read-only (no setter)")

    async def aget(self) -> Any:
        """Read current value asynchronously."""
        if self.get_func is not None:
            if inspect.iscoroutinefunction(self.get_func):
                return await self.get_func()
            return await asyncio.to_thread(self.get_func)
        return await self.instrument.aget(self.attr)

    async def aset(self, value: float) -> None:
        """Clamp value to limits (if set) and apply asynchronously."""
        value = self._clamp(float(value))
        if self.set_func is not None:
            if inspect.iscoroutinefunction(self.set_func):
                await self.set_func(value)
            else:
                await asyncio.to_thread(self.set_func, value)
        elif self.instrument is not None and self.attr is not None:
            await self.instrument.aset(self.attr, value)
        else:
            raise RuntimeError(f"Controller {self.name!r} is read-only (no setter)")

    def __repr__(self) -> str:
        src = self.attr or "callable"
        lim = f", limits={self.limits}" if self.limits is not None else ""
        return f"PhysicalController({self.name!r}, {src}, unit={self.unit!r}{lim})"


# ══════════════════════════════════════════════════════════════════════
#  VirtualController — dispatches to physical controllers via a binding
# ══════════════════════════════════════════════════════════════════════

class VirtualController(ControllerBase):
    """A controller that dispatches to physical controllers via a binding function.

    No readback path — ``get()`` raises ``RuntimeError``.

    Two binding modes:

    **Linear** — ``binding`` is a ``dict[str, float]`` mapping each target
    controller name to a scale factor.  ``set(v)`` calls
    ``target.set(factor * v)`` for every entry::

        VirtualController("VP1", binding={"P1": 1.0, "P2": 0.5}, registry=...)

    **Callable** — ``binding`` is a function ``f(v) -> dict[str, float]``
    that maps the virtual value to a per-controller target dict::

        VirtualController("VP1",
            binding=lambda v: {"P1": np.tanh(v), "P2": v ** 2},
            registry=...)

    Parameters
    ----------
    name : str
        Short label, e.g. ``"VP1"``.
    binding : dict[str, float] or callable
        Linear weights or arbitrary mapping function.
    registry : dict[str, ControllerBase]
        Live reference to the bench's controller dict.  Target lookups
        happen at ``set`` time, so controllers added after construction
        are automatically available.
    unit : str, optional
        Physical unit.
    limits : tuple[float, float] or None, optional
        ``(lo, hi)`` bounds applied to the *virtual* value before dispatch.
    limit_policy : LimitPolicy, optional
        How to respond to limit violations. Default is ``WARN``.

    Examples
    --------
    Cross-capacitance correction (linear)::

        bench.add_virtual_controller("VP1", binding={"P1": 1.0, "B1": -0.3})
        bench["VP1"] = -0.5   # → P1.set(-0.5), B1.set(0.15)

    Non-linear binding (callable)::

        bench.add_virtual_controller(
            "VB",
            binding=lambda v: {"B1": v, "B2": v + 0.05 * v**2},
        )
    """

    def __init__(
        self,
        name: str,
        binding: dict[str, float] | Callable[[float], dict[str, float]],
        registry: dict[str, ControllerBase],
        unit: str | None = None,
        limits: tuple[float, float] | None = None,
        limit_policy: LimitPolicy = LimitPolicy.WARN,
    ) -> None:
        super().__init__(name, unit=unit, limits=limits, limit_policy=limit_policy)
        self._binding = binding
        self._registry = registry   # live reference — not a copy

    # ── Internal ──────────────────────────────────────────────────────

    def _resolve_targets(self, value: float) -> dict[str, float]:
        """Return {ctrl_name: target_value} for the given virtual value."""
        if callable(self._binding):
            return self._binding(value)
        return {name: coeff * value for name, coeff in self._binding.items()}

    # ── Public API ────────────────────────────────────────────────────

    def set(self, value: float) -> None:
        """Clamp to virtual limits then dispatch to all bound controllers."""
        value = self._clamp(float(value))
        for ctrl_name, ctrl_val in self._resolve_targets(value).items():
            self._registry[ctrl_name].set(ctrl_val)

    async def aset(self, value: float) -> None:
        """Clamp to virtual limits then set all bound controllers concurrently."""
        value = self._clamp(float(value))
        targets = self._resolve_targets(value)
        await asyncio.gather(*[
            self._registry[name].aset(val)
            for name, val in targets.items()
        ])

    def __repr__(self) -> str:
        if callable(self._binding):
            binding_repr = f"callable({getattr(self._binding, '__name__', '?')})"
        else:
            binding_repr = "{" + ", ".join(f"{k}: {v}" for k, v in self._binding.items()) + "}"
        lim = f", limits={self.limits}" if self.limits is not None else ""
        return f"VirtualController({self.name!r}, binding={binding_repr}{lim})"


# Backward-compatible alias — existing code using Controller keeps working
Controller = PhysicalController


# ══════════════════════════════════════════════════════════════════════
#  Readout — read-only measurement channel
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PhysicalReadout:
    """A read-only measurement channel backed by a real instrument.

    Supply either ``get_func`` **or** ``instrument + attr``; the latter is
    fully serializable by :py:meth:`Bench.save` / :py:meth:`Bench.load`.

    Parameters
    ----------
    name : str
        Label, e.g. "S21", "lockin_X".
    kind : DataKind
        SCALAR, TRACE, or IMAGE.
    get_func : callable, optional
        Function that acquires and returns the measurement data.
        Mutually exclusive with ``instrument + attr``.
    instrument : InstrumentAdapter, optional
        Instrument to read from.  Used together with ``attr``.
    attr : str, optional
        Attribute name on the instrument.  Used together with ``instrument``.
    shape : tuple, optional
        Trailing shape for TRACE (N,) or IMAGE (H, W). Required for non-scalar.
    unit : str, optional
        Physical unit.
    contains : str or list of str, optional
        Human-readable description of what is measured.
        For IMAGE readouts, pass a list of column names, e.g.
        ``["f", "mag", "phase"]``, so plotters can resolve columns by name.
    """

    name: str
    kind: DataKind
    get_func: Callable[[], Any] | None = None
    instrument: InstrumentAdapter | None = None
    attr: str | None = None
    shape: tuple[int, ...] | None = None
    unit: str | None = None
    contains: str | list[str] | None = None

    def __post_init__(self):
        has_func = self.get_func is not None
        has_instr = self.instrument is not None and self.attr is not None
        if not has_func and not has_instr:
            raise ValueError(
                f"Readout {self.name!r}: provide either get_func or instrument+attr"
            )
        if has_func and has_instr:
            raise ValueError(
                f"Readout {self.name!r}: provide get_func OR instrument+attr, not both"
            )
        if self.kind != DataKind.SCALAR and self.shape is None:
            raise ValueError(
                f"Readout {self.name!r}: shape is required for {self.kind} kind"
            )

    def read(self) -> np.ndarray | float:
        """Acquire one measurement."""
        if self.get_func is not None:
            return self.get_func()
        return self.instrument.get(self.attr)

    async def aread(self) -> np.ndarray | float:
        if self.get_func is not None:
            if inspect.iscoroutinefunction(self.get_func):
                return await self.get_func()
            return await asyncio.to_thread(self.get_func)
        return await self.instrument.aget(self.attr)

    def __repr__(self) -> str:
        src = self.attr if self.attr else "callable"
        return f"PhysicalReadout({self.name!r}, {self.kind}, src={src!r}, shape={self.shape})"


@dataclass
class VirtualReadout:
    """A derived measurement channel computed from physical readout data.

    VirtualReadout has no hardware connection — it calls ``transform`` on a
    subset of already-measured data.  This is useful for fitting, unit
    conversion, or any derived quantity (e.g. extracting resonance frequency
    from a VNA trace).

    The runner always measures physical readouts first, then computes virtuals
    in the order they appear in ``proc.readouts``.  All ``sources`` must be
    listed explicitly in ``proc.readouts``; the runner raises ``ValueError``
    otherwise so the user has full control over what is recorded.

    No ``read()`` or ``aread()`` — use ``compute(data)`` / ``acompute(data)``.

    Parameters
    ----------
    name : str
        Label, e.g. "resonance_freq".
    kind : DataKind
        SCALAR, TRACE, or IMAGE.
    sources : list of str
        Names of physical readouts whose data is passed to ``transform``.
        Must be :class:`PhysicalReadout` entries (no virtual-to-virtual).
    transform : callable
        ``f(data: dict) -> Any`` where the dict contains ``{source: value}``
        for each source name.  May be a coroutine function.
    shape : tuple, optional
        Trailing shape for TRACE/IMAGE output. Required for non-scalar.
    unit : str, optional
        Physical unit of the computed result.
    contains : str or list of str, optional
        Description of computed quantity.

    Examples
    --------
    Extract resonance frequency and linewidth from a VNA trace::

        def fit_resonance(data):
            f, mag = data["f_axis"], data["S21"]
            idx = np.argmin(mag)
            return np.array([f[idx], 0.1e6])  # freq, linewidth

        bench.add_virtual_readout(
            "res_fit",
            sources=["S21"],
            transform=fit_resonance,
            kind=DataKind.TRACE,
            shape=(2,),
            unit="Hz",
        )
    """

    name: str
    kind: DataKind
    sources: list[str]
    transform: Callable[[dict], Any]
    shape: tuple[int, ...] | None = None
    unit: str | None = None
    contains: str | list[str] | None = None

    def __post_init__(self):
        if self.kind != DataKind.SCALAR and self.shape is None:
            raise ValueError(
                f"VirtualReadout {self.name!r}: shape is required for {self.kind} kind"
            )

    def compute(self, data: dict) -> Any:
        """Compute the virtual readout from a dict of source data."""
        subset = {src: data[src] for src in self.sources}
        return self.transform(subset)

    async def acompute(self, data: dict) -> Any:
        """Compute the virtual readout asynchronously."""
        if inspect.iscoroutinefunction(self.transform):
            subset = {src: data[src] for src in self.sources}
            return await self.transform(subset)
        return await asyncio.to_thread(self.compute, data)

    def __repr__(self) -> str:
        return f"VirtualReadout({self.name!r}, {self.kind}, sources={self.sources})"


# Backward-compatible alias
Readout = PhysicalReadout
