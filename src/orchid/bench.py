"""Bench — the lab bench configuration holding instruments, controllers, and readouts."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .instrument import InstrumentAdapter
from .controller import DataKind, Controller, LimitPolicy, Readout


@dataclass
class Bench:
    """Container for all instruments, controllers, and readouts.

    Acts as the "lab bench" configuration passed to procedures.

    Parameters
    ----------
    data_root : str or Path
        Root directory where experiment data will be saved.
    metadata : dict
        User metadata (sample name, operator, fridge, etc.).
    """

    data_root: str | Path = "./data"
    metadata: dict = field(default_factory=dict)

    instruments: dict[str, InstrumentAdapter] = field(default_factory=dict, init=False)
    controllers: dict[str, Controller] = field(default_factory=dict, init=False)
    readouts: dict[str, Readout] = field(default_factory=dict, init=False)

    # Event log — active only during run_monitor(); None when idle
    _event_log: list | None = field(default=None, init=False, repr=False)
    _event_callback: Callable | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.data_root = Path(self.data_root)

    def add_instrument(
        self,
        name: str,
        instrument: Any,
        backend: str = "auto",
    ) -> InstrumentAdapter:
        """Register an instrument.

        Parameters
        ----------
        name : str
            Unique name for the instrument.
        instrument : Any
            The raw driver object.
        backend : str
            "pymeasure", "qcodes", "custom", or "auto" (auto-detect).
        """
        if backend == "auto":
            adapter = InstrumentAdapter.auto(name, instrument)
        elif backend == "pymeasure":
            adapter = InstrumentAdapter.from_pymeasure(name, instrument)
        elif backend == "qcodes":
            adapter = InstrumentAdapter.from_qcodes(name, instrument)
        else:
            adapter = InstrumentAdapter.from_custom(name, instrument)
        self.instruments[name] = adapter
        return adapter

    def add_controller(
        self,
        name: str,
        instrument: InstrumentAdapter | str | None = None,
        attr: str | None = None,
        get_func=None,
        set_func=None,
        unit: str | None = None,
        limits: tuple[float, float] | None = None,
        limit_policy: LimitPolicy = LimitPolicy.WARN
    ) -> Controller:
        """Register a control parameter.

        Parameters
        ----------
        name : str
            Short label, e.g. "Vgt".
        instrument : InstrumentAdapter or str, optional
            Instrument or its registered name.
        attr : str, optional
            Attribute name on the instrument.
        get_func, set_func : callable, optional
            Custom getter/setter (override instrument access).
        unit : str, optional
            Physical unit.
        """
        if isinstance(instrument, str):
            instrument = self.instruments[instrument]
        ctrl = Controller(
            name=name,
            instrument=instrument,
            attr=attr,
            get_func=get_func,
            set_func=set_func,
            unit=unit,
            limits=limits,
            limit_policy=limit_policy,
        )
        self.controllers[name] = ctrl
        return ctrl

    def add_readout(
        self,
        name: str,
        kind: DataKind | str,
        get_func,
        shape: tuple[int, ...] | None = None,
        unit: str | list[str] | None = None,
        contains: str | list[str] | None = None,
    ) -> Readout:
        """Register a measurement readout.

        Parameters
        ----------
        name : str
            Label, e.g. "S21".
        kind : DataKind or str
            "scalar", "trace", or "image".
        get_func : callable
            Function that acquires and returns the measurement.
        shape : tuple, optional
            Trailing shape for trace/image readouts.
        unit : str, optional
            Physical unit.
        contains : str, optional
            Description of what is measured.
        """
        if isinstance(kind, str):
            kind = DataKind(kind)
        readout = Readout(
            name=name,
            kind=kind,
            get_func=get_func,
            shape=shape,
            unit=unit,
            contains=contains,
        )
        self.readouts[name] = readout
        return readout

    def remove_instrument(self, name: str) -> None:
        """Remove an instrument and all parameters/readouts that depend on it.

        Parameters
        ----------
        name : str
            Name of the instrument to remove.
        """
        if name not in self.instruments:
            raise KeyError(f"No instrument named {name!r}")
        adapter = self.instruments.pop(name)
        # Remove controllers that reference this instrument
        to_remove = [
            pname for pname, p in self.controllers.items()
            if p.instrument is adapter
        ]
        for pname in to_remove:
            del self.controllers[pname]
        if to_remove:
            print(f"Removed dependent controllers: {to_remove}")

    def remove_controller(self, name: str) -> None:
        """Remove a controller.

        Parameters
        ----------
        name : str
            Name of the controller to remove.
        """
        if name not in self.controllers:
            raise KeyError(f"No controller named {name!r}")
        del self.controllers[name]

    def remove_readout(self, name: str) -> None:
        """Remove a readout.

        Parameters
        ----------
        name : str
            Name of the readout to remove.
        """
        if name not in self.readouts:
            raise KeyError(f"No readout named {name!r}")
        del self.readouts[name]

    def __getitem__(self, name: str):
        """Get current value of a controller or readout.

        Usage::

            bench["Vgt"]       # reads voltage from instrument
            bench["lockin_X"]  # reads lockin X channel
        """
        if name in self.controllers:
            return self.controllers[name].get()
        if name in self.readouts:
            return self.readouts[name].read()
        raise KeyError(f"No controller or readout named {name!r}")

    def __setitem__(self, name: str, value) -> None:
        """Set a controller value.

        Usage::

            bench["Vgt"] = 0.4   # sets voltage on instrument
        """
        if name in self.controllers:
            self.controllers[name].set(value)
            if self._event_log is not None:
                entry = {
                    "time": _time.time(),
                    "param": name,
                    "value": value,
                }
                self._event_log.append(entry)
                if self._event_callback is not None:
                    self._event_callback(entry)
        else:
            raise KeyError(f"No controller named {name!r}")

    def _start_event_log(self, on_event: Callable | None = None) -> None:
        """Start recording controller change events.

        Called by ExperimentRunner at the start of a monitor run.
        """
        self._event_log = []
        self._event_callback = on_event

    def _stop_event_log(self) -> list:
        """Stop recording and return the collected events.

        Called by ExperimentRunner at the end of a monitor run.
        """
        log = self._event_log or []
        self._event_log = None
        self._event_callback = None
        return log

    def snapshot(
        self,
        names: list[str] | None = None,
        *,
        include_readouts: bool = False,
    ) -> None:
        """Print a table of current parameter and readout values.

        By default only controllers are read — readouts can involve slow
        instrument acquisitions (lock-in time constants, VNA sweeps, etc.)
        and are excluded unless explicitly requested.

        Parameters
        ----------
        names : list of str, optional
            If given, read exactly these controllers/readouts (by name).
            If None, read all controllers and, if ``include_readouts=True``,
            all readouts too.
        include_readouts : bool
            If True, include all registered readouts when ``names`` is None.
            Ignored when an explicit ``names`` list is supplied.
        """
        from tabulate import tabulate as _tabulate

        if names is None:
            names = list(self.controllers.keys())
            if include_readouts:
                names += list(self.readouts.keys())

        rows = []
        for name in names:
            if name in self.controllers:
                p = self.controllers[name]
                try:
                    val = p.get()
                except Exception as e:
                    val = f"ERR: {e}"
                rows.append([name, "param", val, p.unit or ""])
            elif name in self.readouts:
                r = self.readouts[name]
                try:
                    val = r.read()
                except Exception as e:
                    val = f"ERR: {e}"
                # Compact display for large array readouts
                if r.kind.value == "trace" and not isinstance(val, str):
                    val = "[...]"
                elif r.kind.value == "image" and not isinstance(val, str):
                    val = "[[...]]"
                rows.append([name, r.kind.value, val, r.unit or ""])
            else:
                rows.append([name, "?", "NOT FOUND", ""])

        print(_tabulate(rows, headers=["Name", "Type", "Value", "Unit"],
                        tablefmt="simple"))

    def __repr__(self) -> str:
        return (
            f"Bench("
            f"{len(self.instruments)} instruments, "
            f"{len(self.controllers)} controllers, "
            f"{len(self.readouts)} readouts)"
        )
