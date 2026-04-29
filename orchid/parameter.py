"""Parameter and Readout classes — the bridge between instruments and experiments."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

import numpy as np

from .instrument import InstrumentAdapter


class DataKind(StrEnum):
    """Shape of a single measurement readout (mirrors zarro.DataKind)."""
    SCALAR = "scalar"
    TRACE = "trace"
    IMAGE = "image"


@dataclass
class Parameter:
    """A named control parameter mapped to an instrument channel.

    Can be used for both setting and reading values. Provide either
    an instrument+attr pair or custom get_func/set_func callables.

    Parameters
    ----------
    name : str
        Short label, e.g. "Vgt", "fac".
    instrument : InstrumentAdapter, optional
        Instrument this parameter belongs to.
    attr : str, optional
        Attribute name on the instrument.
    get_func : callable, optional
        Custom getter (overrides instrument.get).
    set_func : callable, optional
        Custom setter (overrides instrument.set).
    unit : str, optional
        Physical unit.
    """

    name: str
    instrument: InstrumentAdapter | None = None
    attr: str | None = None
    get_func: Callable[[], Any] | None = None
    set_func: Callable[[Any], None] | None = None
    unit: str | None = None

    def __post_init__(self):
        if self.instrument is None and self.get_func is None and self.set_func is None:
            raise ValueError(
                f"Parameter {self.name!r}: provide instrument+attr or get_func/set_func"
            )

    def get(self) -> Any:
        """Read current value."""
        if self.get_func is not None:
            return self.get_func()
        return self.instrument.get(self.attr)

    def set(self, value: Any) -> None:
        """Set value."""
        if self.set_func is not None:
            self.set_func(value)
        elif self.instrument is not None and self.attr is not None:
            self.instrument.set(self.attr, value)
        else:
            raise RuntimeError(f"Parameter {self.name!r} is read-only (no setter)")

    async def aget(self) -> Any:
        if self.get_func is not None:
            if inspect.iscoroutinefunction(self.get_func):
                return await self.get_func()
            return await asyncio.to_thread(self.get_func)
        return await self.instrument.aget(self.attr)

    async def aset(self, value: Any) -> None:
        if self.set_func is not None:
            if inspect.iscoroutinefunction(self.set_func):
                await self.set_func(value)
            else:
                await asyncio.to_thread(self.set_func, value)
        elif self.instrument is not None and self.attr is not None:
            await self.instrument.aset(self.attr, value)
        else:
            raise RuntimeError(f"Parameter {self.name!r} is read-only (no setter)")

    def __repr__(self) -> str:
        src = self.attr or "callable"
        return f"Parameter({self.name!r}, {src}, unit={self.unit!r})"


@dataclass
class Readout:
    """A read-only measurement channel (e.g. VNA trace, lockin reading).

    Parameters
    ----------
    name : str
        Label, e.g. "S21", "lockin_X".
    kind : DataKind
        SCALAR, TRACE, or IMAGE.
    get_func : callable
        Function that acquires and returns the measurement data.
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
    get_func: Callable[[], Any] = None
    shape: tuple[int, ...] | None = None
    unit: str | None = None
    contains: str | list[str] | None = None

    def __post_init__(self):
        if self.kind != DataKind.SCALAR and self.shape is None:
            raise ValueError(
                f"Readout {self.name!r}: shape is required for {self.kind} kind"
            )

    def read(self) -> np.ndarray | float:
        """Acquire one measurement."""
        return self.get_func()

    async def aread(self) -> np.ndarray | float:
        if inspect.iscoroutinefunction(self.get_func):
            return await self.get_func()
        return await asyncio.to_thread(self.get_func)

    def __repr__(self) -> str:
        return f"Readout({self.name!r}, {self.kind}, shape={self.shape})"
