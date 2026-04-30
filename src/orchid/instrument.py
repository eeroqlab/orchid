"""Instrument adapter layer for pymeasure, qcodes, and custom drivers."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InstrumentAdapter:
    """Thin wrapper normalizing pymeasure/qcodes/custom instruments.

    Provides a uniform get/set interface regardless of the underlying driver
    framework. Supports both synchronous and asynchronous access.

    Parameters
    ----------
    name : str
        Human-readable name for the instrument.
    driver : Any
        The raw instrument object from pymeasure, qcodes, or custom code.
    backend : str
        One of "pymeasure", "qcodes", or "custom".
    """

    name: str
    driver: Any
    backend: str = "custom"

    @classmethod
    def from_pymeasure(cls, name: str, instrument: Any) -> InstrumentAdapter:
        return cls(name=name, driver=instrument, backend="pymeasure")

    @classmethod
    def from_qcodes(cls, name: str, instrument: Any) -> InstrumentAdapter:
        return cls(name=name, driver=instrument, backend="qcodes")

    @classmethod
    def from_custom(cls, name: str, obj: Any) -> InstrumentAdapter:
        return cls(name=name, driver=obj, backend="custom")

    @classmethod
    def auto(cls, name: str, instrument: Any) -> InstrumentAdapter:
        """Auto-detect backend from instrument type."""
        type_name = type(instrument).__mro__
        type_names = [t.__module__ + "." + t.__qualname__ for t in type_name]
        for tn in type_names:
            if "qcodes" in tn:
                return cls.from_qcodes(name, instrument)
            if "pymeasure" in tn:
                return cls.from_pymeasure(name, instrument)
        return cls.from_custom(name, instrument)

    def get(self, attr: str) -> Any:
        """Get a parameter value from the instrument."""
        if self.backend == "qcodes":
            param = getattr(self.driver, attr)
            if callable(getattr(param, "get", None)):
                return param.get()
            return param()
        # pymeasure and custom: property access
        return getattr(self.driver, attr)

    def set(self, attr: str, value: Any) -> None:
        """Set a parameter value on the instrument."""
        if self.backend == "qcodes":
            param = getattr(self.driver, attr)
            if callable(getattr(param, "set", None)):
                param.set(value)
            else:
                param(value)
        else:
            # pymeasure and custom: property setter
            setattr(self.driver, attr, value)

    async def aget(self, attr: str) -> Any:
        """Async get — wraps sync get in executor if not natively async."""
        if self.backend == "qcodes":
            param = getattr(self.driver, attr)
            if hasattr(param, "get") and inspect.iscoroutinefunction(param.get):
                return await param.get()
        return await asyncio.to_thread(self.get, attr)

    async def aset(self, attr: str, value: Any) -> None:
        """Async set — wraps sync set in executor if not natively async."""
        if self.backend == "qcodes":
            param = getattr(self.driver, attr)
            if hasattr(param, "set") and inspect.iscoroutinefunction(param.set):
                await param.set(value)
                return
        await asyncio.to_thread(self.set, attr, value)

    def __repr__(self) -> str:
        return f"InstrumentAdapter({self.name!r}, backend={self.backend!r})"
