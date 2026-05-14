"""Bench — the lab bench configuration holding instruments, controllers, and readouts."""

from __future__ import annotations

import inspect
import textwrap
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .instrument import InstrumentAdapter
from .controller import DataKind, Controller, LimitPolicy, Readout


def _func_source(func) -> str | None:
    """Return a best-effort source string for *func* (informational only).

    Tries ``inspect.getsource()`` first — works for named functions and for
    lambdas defined in Jupyter cells (IPython keeps cell source in memory via
    ``linecache``).  Falls back to the qualified name if source is unavailable.
    Returns ``None`` for ``None`` input.
    """
    if func is None:
        return None
    try:
        src = inspect.getsource(func)
        return textwrap.dedent(src).strip()
    except (OSError, TypeError):
        pass
    try:
        return func.__qualname__
    except AttributeError:
        return repr(func)


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

    # Stubs — entries skipped during Bench.load() that need manual re-registration
    # Each value: {"kind": "controller"|"readout", "unit", "limits",
    #              "limit_policy", "get_func_src", "set_func_src"}
    _stubs: dict = field(default_factory=dict, init=False, repr=False)

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
        *,
        class_path: str | None = None,
        args: list | None = None,
        kwargs: dict | None = None,
    ):
        """Register an instrument.

        Parameters
        ----------
        name : str
            Unique name for the instrument.
        instrument : Any
            The raw driver object.
        backend : str
            "pymeasure", "qcodes", "custom", or "auto" (auto-detect).
        class_path : str, optional
            Full importable class path, e.g.
            ``"pymeasure.instruments.keithley.Keithley2400"``.
            Auto-detected from ``type(instrument)`` if omitted; instruments
            defined in ``__main__`` produce a stub with a warning.
        args : list, optional
            Positional arguments originally passed to the instrument
            constructor.  Stored for :py:meth:`save` / :py:meth:`load`.
        kwargs : dict, optional
            Keyword arguments originally passed to the instrument constructor.
        """
        if backend == "auto":
            adapter = InstrumentAdapter.auto(name, instrument)
        elif backend == "pymeasure":
            adapter = InstrumentAdapter.from_pymeasure(name, instrument)
        elif backend == "qcodes":
            adapter = InstrumentAdapter.from_qcodes(name, instrument)
        else:
            adapter = InstrumentAdapter.from_custom(name, instrument)

        # Build connection_info for serialisation
        if class_path is None:
            cls = type(instrument)
            class_path = f"{cls.__module__}.{cls.__qualname__}"
        adapter.connection_info = {
            "class": class_path,
            "args": list(args) if args else [],
            "kwargs": dict(kwargs) if kwargs else {},
        }
        self.instruments[name] = adapter

    def add_controller(
        self,
        name: str,
        instrument: InstrumentAdapter | str | None = None,
        attr: str | None = None,
        get_func=None,
        set_func=None,
        unit: str | None = None,
        limits: tuple[float, float] | None = None,
        limit_policy: LimitPolicy = LimitPolicy.WARN,
    ):
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
        limits : tuple[float, float], optional
            Inclusive bounds for the controller.
        limit_policy : LimitPolicy
            How to handle controller limit violations.
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

    def add_controller_binding(
        self,
        name: str,
        gate_names: list[str],
        *,
        unit: str | None = None,
        limits: tuple[float, float] | None = None,
        limit_policy: LimitPolicy = LimitPolicy.WARN,
    ):
        """Register a virtual controller bound to existing controllers.

        Setting the virtual controller sets all bound controllers to the same
        value. Reading it returns a ``{controller_name: value}`` mapping.

        Parameters
        ----------
        name : str
            Name for the virtual controller.
        gate_names : list[str]
            Existing controller names to bind.
        unit : str, optional
            Unit for the virtual controller. If omitted, inferred from bound
            controllers.
        """
        if not gate_names:
            raise ValueError("add_controller_binding requires at least one controller")

        missing = [gate for gate in gate_names if gate not in self.controllers]
        if missing:
            raise KeyError(f"Cannot bind {name!r}: missing controllers {missing}")

        physical = [self.controllers[gate] for gate in gate_names]
        if unit is None:
            units = list(dict.fromkeys(ctrl.unit for ctrl in physical if ctrl.unit))
            unit = (
                " / ".join(units)
                if len(units) > 1
                else (units[0] if units else None)
            )

        def set_bound(value):
            for ctrl in physical:
                ctrl.set(value)

        self.add_controller(
            name,
            get_func=None,
            set_func=set_bound,
            unit=unit,
            limits=limits,
            limit_policy=limit_policy
        )

    def add_readout(
        self,
        name: str,
        kind: DataKind | str,
        get_func=None,
        instrument: "InstrumentAdapter | str | None" = None,
        attr: str | None = None,
        shape: tuple[int, ...] | None = None,
        unit: str | list[str] | None = None,
        contains: str | list[str] | None = None,
    ):
        """Register a measurement readout.

        Supply either ``get_func`` **or** ``instrument + attr``.  The
        ``instrument + attr`` form is fully serializable by
        :py:meth:`save` / :py:meth:`load`.

        Parameters
        ----------
        name : str
            Label, e.g. "S21".
        kind : DataKind or str
            "scalar", "trace", or "image".
        get_func : callable, optional
            Function that acquires and returns the measurement.
        instrument : InstrumentAdapter or str, optional
            Instrument or its registered name.
        attr : str, optional
            Attribute name on the instrument.
        shape : tuple, optional
            Trailing shape for trace/image readouts.
        unit : str, optional
            Physical unit.
        contains : str, optional
            Description of what is measured.
        """
        if isinstance(kind, str):
            kind = DataKind(kind)
        if isinstance(instrument, str):
            instrument = self.instruments[instrument]
        readout = Readout(
            name=name,
            kind=kind,
            get_func=get_func,
            instrument=instrument,
            attr=attr,
            shape=shape,
            unit=unit,
            contains=contains,
        )
        self.readouts[name] = readout

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
                rows.append([name, "ctrl", val, p.unit or ""])
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
                rows.append([name, f"read / {r.kind.value}", val, r.unit or ""])
            else:
                rows.append([name, "?", "NOT FOUND", ""])

        print(_tabulate(rows, headers=["Name", "Type", "Value", "Unit"],
                        tablefmt="simple"))

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: "str | Path") -> None:
        """Save bench configuration to a YAML file.

        Instruments are stored by class path + constructor args so they can
        be re-instantiated without any ``eval``.  Controllers and
        ``instrument + attr`` readouts are fully round-tripped.  Custom
        ``get_func`` / ``set_func`` callables and ``__main__`` instrument
        classes cannot be serialized; a comment stub is emitted and a
        warning is printed.

        Parameters
        ----------
        path : str or Path
            Destination ``.yaml`` file.
        """
        import yaml

        config: dict = {
            "bench": {
                "data_root": str(self.data_root),
                "metadata": dict(self.metadata),
            },
            "instruments": {},
            "controllers": {},
            "readouts": {},
        }

        # ── instruments ───────────────────────────────────────────────
        for iname, adapter in self.instruments.items():
            ci = adapter.connection_info
            class_path: str | None = ci.get("class")
            is_main = (not class_path) or class_path.startswith("__main__.")
            entry: dict = {
                "class": None if is_main else class_path,
                "args": ci.get("args", []),
                "kwargs": ci.get("kwargs", {}),
                "backend": adapter.backend,
            }
            if is_main:
                entry["_note"] = (
                    f"Class {class_path!r} is not importable. "
                    "Set 'class' to the full module path before calling Bench.load()."
                )
                import warnings
                warnings.warn(
                    f"Bench.save: instrument {iname!r} class {class_path!r} is "
                    "defined in __main__ and cannot be auto-loaded. "
                    "Update the 'class' field in the saved YAML manually.",
                    stacklevel=2,
                )
            config["instruments"][iname] = entry

        # ── controllers ───────────────────────────────────────────────
        for cname, ctrl in self.controllers.items():
            if ctrl.instrument is not None and ctrl.attr is not None:
                # Find the registered name for this adapter
                instr_name = next(
                    (n for n, a in self.instruments.items() if a is ctrl.instrument),
                    None,
                )
                config["controllers"][cname] = {
                    "instrument": instr_name,
                    "attr": ctrl.attr,
                    "unit": ctrl.unit,
                    "limits": list(ctrl.limits) if ctrl.limits is not None else None,
                    "limit_policy": str(ctrl.limit_policy),
                }
            else:
                entry: dict = {
                    "instrument": None,
                    "attr": None,
                    "unit": ctrl.unit,
                    "limits": list(ctrl.limits) if ctrl.limits is not None else None,
                    "limit_policy": str(ctrl.limit_policy),
                    "_note": "Custom get_func/set_func — re-register manually after load.",
                }
                src_get = _func_source(ctrl.get_func)
                src_set = _func_source(ctrl.set_func)
                if src_get is not None:
                    entry["get_func_src"] = src_get
                if src_set is not None:
                    entry["set_func_src"] = src_set
                config["controllers"][cname] = entry

        # ── readouts ──────────────────────────────────────────────────
        for rname, ro in self.readouts.items():
            base: dict = {
                "kind": str(ro.kind),
                "shape": list(ro.shape) if ro.shape is not None else None,
                "unit": ro.unit,
                "contains": ro.contains,
            }
            if ro.instrument is not None and ro.attr is not None:
                instr_name = next(
                    (n for n, a in self.instruments.items() if a is ro.instrument),
                    None,
                )
                config["readouts"][rname] = {
                    **base,
                    "instrument": instr_name,
                    "attr": ro.attr,
                }
            else:
                ro_entry: dict = {
                    **base,
                    "instrument": None,
                    "attr": None,
                    "_note": "Custom get_func — re-register manually after load.",
                }
                src_get = _func_source(ro.get_func)
                if src_get is not None:
                    ro_entry["get_func_src"] = src_get
                config["readouts"][rname] = ro_entry

        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        print(f"Bench saved → {path}")

    @classmethod
    def load(cls, path: "str | Path") -> "Bench":
        """Load a bench configuration from a YAML file saved by :py:meth:`save`.

        Instruments are instantiated by importing their class and calling it
        with the stored ``args`` / ``kwargs``.  Controllers and
        ``instrument + attr`` readouts are wired automatically.

        Entries that cannot be auto-loaded (custom ``get_func``/``set_func``,
        ``__main__`` instruments, failed connections) are collected silently as
        *stubs* — call :py:meth:`show_stubs` after loading to review them.

        Parameters
        ----------
        path : str or Path
            Source ``.yaml`` file (produced by :py:meth:`save`).
        """
        import importlib
        import yaml

        path = Path(path)
        with open(path) as f:
            config = yaml.safe_load(f)

        bench_cfg = config.get("bench", {})
        bench = cls(
            data_root=bench_cfg.get("data_root", "./data"),
            metadata=bench_cfg.get("metadata", {}),
        )

        failed_instruments: set[str] = set()

        # ── instruments ───────────────────────────────────────────────
        for iname, icfg in config.get("instruments", {}).items():
            class_path = icfg.get("class")
            if not class_path:
                bench._stubs[iname] = {
                    "kind": "instrument",
                    "reason": "no class path (stub) — update YAML and reload",
                    "class": None,
                    "args": icfg.get("args", []),
                    "kwargs": icfg.get("kwargs", {}),
                }
                failed_instruments.add(iname)
                continue
            try:
                module_path, cls_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                instrument_cls = getattr(module, cls_name)
                i_args = icfg.get("args") or []
                i_kwargs = icfg.get("kwargs") or {}
                driver = instrument_cls(*i_args, **i_kwargs)
                bench.add_instrument(
                    iname, driver,
                    backend=icfg.get("backend", "auto"),
                    class_path=class_path,
                    args=i_args,
                    kwargs=i_kwargs,
                )
            except Exception as exc:
                bench._stubs[iname] = {
                    "kind": "instrument",
                    "reason": f"connection failed: {exc}",
                    "class": class_path,
                    "args": icfg.get("args", []),
                    "kwargs": icfg.get("kwargs", {}),
                }
                failed_instruments.add(iname)

        # ── controllers ───────────────────────────────────────────────
        for cname, ccfg in config.get("controllers", {}).items():
            instr_name = ccfg.get("instrument")
            attr = ccfg.get("attr")
            raw_limits = ccfg.get("limits")

            if not instr_name or not attr:
                bench._stubs[cname] = {
                    "kind": "controller",
                    "reason": "custom get_func/set_func",
                    "unit": ccfg.get("unit"),
                    "limits": raw_limits,
                    "limit_policy": ccfg.get("limit_policy", "warn"),
                    "get_func_src": ccfg.get("get_func_src"),
                    "set_func_src": ccfg.get("set_func_src"),
                }
                continue
            if instr_name in failed_instruments:
                bench._stubs[cname] = {
                    "kind": "controller",
                    "reason": f"instrument {instr_name!r} not loaded",
                    "unit": ccfg.get("unit"),
                    "limits": raw_limits,
                    "limit_policy": ccfg.get("limit_policy", "warn"),
                }
                continue
            bench.add_controller(
                cname,
                instrument=instr_name,
                attr=attr,
                unit=ccfg.get("unit"),
                limits=tuple(raw_limits) if raw_limits is not None else None,
                limit_policy=LimitPolicy(ccfg.get("limit_policy", "warn")),
            )

        # ── readouts ──────────────────────────────────────────────────
        for rname, rcfg in config.get("readouts", {}).items():
            instr_name = rcfg.get("instrument")
            attr = rcfg.get("attr")

            if not instr_name or not attr:
                bench._stubs[rname] = {
                    "kind": "readout",
                    "reason": "custom get_func",
                    "readout_kind": rcfg.get("kind", "scalar"),
                    "unit": rcfg.get("unit"),
                    "shape": rcfg.get("shape"),
                    "get_func_src": rcfg.get("get_func_src"),
                }
                continue
            if instr_name in failed_instruments:
                bench._stubs[rname] = {
                    "kind": "readout",
                    "reason": f"instrument {instr_name!r} not loaded",
                    "readout_kind": rcfg.get("kind", "scalar"),
                    "unit": rcfg.get("unit"),
                    "shape": rcfg.get("shape"),
                }
                continue
            raw_shape = rcfg.get("shape")
            bench.add_readout(
                rname,
                kind=rcfg.get("kind", "scalar"),
                instrument=instr_name,
                attr=attr,
                shape=tuple(raw_shape) if raw_shape is not None else None,
                unit=rcfg.get("unit"),
                contains=rcfg.get("contains"),
            )

        n_stubs = len(bench._stubs)
        stub_hint = f"  ·  {n_stubs} stub{'s' if n_stubs != 1 else ''} — call bench.show_stubs()" if n_stubs else ""
        print(f"Bench loaded ← {path}{stub_hint}")
        return bench

    # ── Stub inspection ───────────────────────────────────────────────

    @property
    def stubs(self) -> dict:
        """Raw dict of entries that could not be auto-loaded.

        Keys are names (instrument / controller / readout).  Each value is a
        dict with at least ``"kind"`` and ``"reason"``.  Controller stubs also
        carry ``"get_func_src"`` / ``"set_func_src"`` when source was recorded
        at :py:meth:`save` time.
        """
        return dict(self._stubs)

    def show_stubs(self, *, full_source: bool = False) -> None:
        """Print a formatted summary of entries that need manual re-registration.

        Parameters
        ----------
        full_source : bool
            If ``False`` (default), source strings are truncated to their first
            line (≤ 72 chars) so the table stays compact.  Pass ``True`` to
            print every line of recorded source.
        """
        from tabulate import tabulate as _tabulate

        if not self._stubs:
            print("No stubs — all entries loaded successfully.")
            return

        def _fmt_src(src: str | None) -> str:
            if not src:
                return "—"
            if full_source:
                return src
            first = src.split("\n")[0]
            return first[:72] + ("…" if len(first) > 72 or "\n" in src else "")

        rows = []
        for name, stub in self._stubs.items():
            kind = stub.get("kind", "?")
            reason = stub.get("reason", "")

            if kind == "instrument":
                rows.append([name, kind, reason, "—", "—"])

            elif kind == "controller":
                meta_parts = []
                if stub.get("unit"):
                    meta_parts.append(stub["unit"])
                lims = stub.get("limits")
                if lims:
                    meta_parts.append(f"{lims[0]}…{lims[1]}")
                meta = "  ".join(meta_parts) or "—"
                get_s = _fmt_src(stub.get("get_func_src"))
                set_s = _fmt_src(stub.get("set_func_src"))
                rows.append([name, kind, reason, meta, f"get: {get_s}\nset: {set_s}"])

            elif kind == "readout":
                meta_parts = [stub.get("readout_kind", "scalar")]
                if stub.get("unit"):
                    meta_parts.append(stub["unit"])
                if stub.get("shape"):
                    meta_parts.append(str(tuple(stub["shape"])))
                meta = "  ".join(meta_parts)
                get_s = _fmt_src(stub.get("get_func_src"))
                rows.append([name, kind, reason, meta, f"get: {get_s}"])

        print(_tabulate(rows,
                        headers=["Name", "Kind", "Reason", "Info", "Source (hint)"],
                        tablefmt="simple",
                        maxcolwidths=[None, None, None, None, 72]))
        if not full_source and any(
            (s.get("get_func_src") or "") + (s.get("set_func_src") or "")
            for s in self._stubs.values()
        ):
            print("  (pass full_source=True to see complete source)")

    def __repr__(self) -> str:
        return (
            f"Bench("
            f"{len(self.instruments)} instruments, "
            f"{len(self.controllers)} controllers, "
            f"{len(self.readouts)} readouts)"
        )
