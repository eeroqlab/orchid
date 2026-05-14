"""Controller and Readout classes — the bridge between instruments and experiments."""

from __future__ import annotations

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


@dataclass
class Controller:
    """A named control parameter mapped to an instrument channel.

    Can be used for both setting and reading values. Provide either
    an instrument+attr pair or custom get_func/set_func callables.

    Parameters
    ----------
    name : str
        Short label, e.g. "Vgt", "fac".
    instrument : InstrumentAdapter, optional
        Instrument this controller belongs to.
    attr : str, optional
        Attribute name on the instrument.
    get_func : callable, optional
        Custom getter (overrides instrument.get).
    set_func : callable, optional
        Custom setter (overrides instrument.set).
    unit : str, optional
        Physical unit.
    limits : tuple[float, float] or None, optional
        ``(lo, hi)`` inclusive bounds. ``None`` means unconstrained.
    limit_policy : LimitPolicy, optional
        How to respond to limit violations. Default is ``WARN``.
    """

    name: str
    instrument: InstrumentAdapter | None = None
    attr: str | None = None
    get_func: Callable[[], Any] | None = None
    set_func: Callable[[Any], None] | None = None
    unit: str | None = None
    limits: tuple[float, float] | None = None
    limit_policy: LimitPolicy = LimitPolicy.WARN

    def __post_init__(self):
        if self.instrument is None and self.get_func is None and self.set_func is None:
            raise ValueError(
                f"Controller {self.name!r}: provide instrument+attr or get_func/set_func"
            )
        # Runtime state — not dataclass fields so they don't appear in repr/eq
        self._limit_hit: bool = False
        self._limit_log: list[LimitEntry] = []
        self._sweep_index: tuple = ()  # set by runner before each aset()

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

    # ── Public API ────────────────────────────────────────────────────

    def get(self) -> Any:
        """Read current value."""
        if self.get_func is not None:
            return self.get_func()
        elif self.instrument is not None:
            return self.instrument.get(self.attr)
        raise RuntimeError(f"Controller {self.name!r} is set-only (no getter)")

    def set(self, value: Any) -> None:
        """Clamp value to limits (if set) and apply."""
        value = self._clamp(float(value))
        if self.set_func is not None:
            self.set_func(value)
        elif self.instrument is not None and self.attr is not None:
            self.instrument.set(self.attr, value)
        else:
            raise RuntimeError(f"Controller {self.name!r} is read-only (no setter)")

    async def aget(self) -> Any:
        if self.get_func is not None:
            if inspect.iscoroutinefunction(self.get_func):
                return await self.get_func()
            return await asyncio.to_thread(self.get_func)
        return await self.instrument.aget(self.attr)

    async def aset(self, value: Any) -> None:
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
        return f"Controller({self.name!r}, {src}, unit={self.unit!r}{lim})"


@dataclass
class Readout:
    """A read-only measurement channel (e.g. VNA trace, lockin reading).

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
        return f"Readout({self.name!r}, {self.kind}, src={src!r}, shape={self.shape})"
