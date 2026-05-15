"""Bench — the lab bench configuration holding instruments, controllers, and readouts."""

from __future__ import annotations

import inspect
import textwrap
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .instrument import InstrumentAdapter
from .controller import (
    DataKind, Controller, ControllerBase, PhysicalController, VirtualController,
    LimitPolicy, Readout, PhysicalReadout, VirtualReadout,
)


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
    controllers: dict[str, ControllerBase] = field(default_factory=dict, init=False)
    readouts: dict[str, PhysicalReadout | VirtualReadout] = field(default_factory=dict, init=False)

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
        ctrl = PhysicalController(
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

    def add_virtual_controller(
        self,
        name: str,
        binding: dict[str, float] | Callable[[float], dict[str, float]],
        *,
        unit: str | None = None,
        limits: tuple[float, float] | None = None,
        limit_policy: LimitPolicy = LimitPolicy.WARN,
    ) -> None:
        """Register a virtual controller that dispatches to physical controllers.

        Two binding modes:

        **Linear** — pass a ``dict`` mapping target controller names to scale
        factors.  ``bench["VP1"] = v`` calls ``target.set(factor * v)`` for
        each entry::

            bench.add_virtual_controller("VP1", binding={"P1": 1.0, "B1": -0.3})

        **Callable** — pass a function ``f(v) -> dict[str, float]`` for
        arbitrary (non-linear) mappings::

            bench.add_virtual_controller(
                "VB",
                binding=lambda v: {"B1": v, "B2": v + 0.05 * v**2},
            )

        Virtual controllers have no readback — ``bench["VP1"]`` raises
        ``RuntimeError``.  They can appear as sweep controllers in
        :class:`~orchid.procedure.Sweep` and their limits apply to the
        *virtual* value before dispatch.

        Parameters
        ----------
        name : str
            Name for the virtual controller.
        binding : dict[str, float] or callable
            Linear weights or arbitrary mapping function.
        unit : str, optional
            Unit for the virtual controller. If omitted, inferred from
            bound controllers (linear binding only).
        limits : tuple[float, float], optional
            Inclusive bounds applied to the virtual value before dispatch.
        limit_policy : LimitPolicy
            How to handle virtual-value limit violations.
        """
        if callable(binding):
            target_names: list[str] = []
        else:
            target_names = list(binding.keys())
            missing = [n for n in target_names if n not in self.controllers]
            if missing:
                raise KeyError(
                    f"add_virtual_controller {name!r}: unknown targets {missing}"
                )
            self._check_no_cycle(name, target_names)

        if unit is None and not callable(binding):
            units = list(dict.fromkeys(
                self.controllers[n].unit
                for n in binding
                if self.controllers[n].unit
            ))
            unit = units[0] if len(units) == 1 else (units[0] if units else None)

        self.controllers[name] = VirtualController(
            name=name,
            binding=binding,
            registry=self.controllers,
            unit=unit,
            limits=limits,
            limit_policy=limit_policy,
        )

    def _check_no_cycle(self, new_name: str, target_names: list[str]) -> None:
        """Raise ValueError if adding new_name → targets would create a cycle."""
        def deps(ctrl_name: str) -> list[str]:
            ctrl = self.controllers.get(ctrl_name)
            if isinstance(ctrl, VirtualController) and not callable(ctrl._binding):
                return list(ctrl._binding.keys())
            return []

        def dfs(node: str, visiting: set[str]) -> None:
            if node == new_name:
                raise ValueError(
                    f"add_virtual_controller {new_name!r}: binding would create a cycle"
                )
            if node in visiting:
                return
            visiting.add(node)
            for dep in deps(node):
                dfs(dep, visiting)

        for target in target_names:
            dfs(target, set())

    def add_controller_binding(
        self,
        name: str,
        gate_names: list[str],
        *,
        unit: str | None = None,
        limits: tuple[float, float] | None = None,
        limit_policy: LimitPolicy = LimitPolicy.WARN,
    ) -> None:
        """Register a virtual controller that mirrors multiple controllers at weight 1.

        Sets all bound controllers to the same value. Equivalent to
        ``add_virtual_controller(name, binding={g: 1.0 for g in gate_names})``.

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
        self.add_virtual_controller(
            name,
            binding={g: 1.0 for g in gate_names},
            unit=unit,
            limits=limits,
            limit_policy=limit_policy,
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

    def add_virtual_readout(
        self,
        name: str,
        sources: list[str],
        transform: Callable,
        *,
        kind: DataKind | str = DataKind.SCALAR,
        shape: tuple[int, ...] | None = None,
        unit: str | None = None,
        contains: str | list[str] | None = None,
    ) -> VirtualReadout:
        """Register a derived readout computed from physical readout data.

        The runner measures all physical readouts first, then calls each
        virtual readout's ``transform`` with the measured data as a dict.
        This keeps derived quantities (fits, unit conversions, etc.) in the
        same dataset as the raw data that produced them.

        All ``sources`` must be :class:`PhysicalReadout` entries registered
        on this bench — virtual-to-virtual chaining is not supported.  When
        used in a procedure, every source must also be listed in
        ``proc.readouts``; the runner validates this at setup time.

        Parameters
        ----------
        name : str
            Label for the derived readout.
        sources : list of str
            Names of physical readouts whose data is passed to ``transform``.
        transform : callable
            ``f(data: dict) -> Any``.  The dict contains ``{source: value}``
            for each listed source.  May be a coroutine function.
        kind : DataKind or str
            "scalar", "trace", or "image".
        shape : tuple, optional
            Trailing shape for trace/image outputs. Required for non-scalar.
        unit : str, optional
            Physical unit of the computed result.
        contains : str or list of str, optional
            Description of the computed quantity.

        Returns
        -------
        VirtualReadout
            The registered readout object.

        Examples
        --------
        ::

            def fit_dip(data):
                mag = data["S21"][1]   # mag row from IMAGE readout
                return float(np.min(mag))

            bench.add_virtual_readout(
                "res_depth",
                sources=["S21"],
                transform=fit_dip,
                kind=DataKind.SCALAR,
                unit="dB",
            )
        """
        if isinstance(kind, str):
            kind = DataKind(kind)
        missing = [s for s in sources if s not in self.readouts]
        if missing:
            raise KeyError(
                f"add_virtual_readout {name!r}: source readouts not in bench: {missing}"
            )
        non_physical = [s for s in sources if isinstance(self.readouts[s], VirtualReadout)]
        if non_physical:
            raise ValueError(
                f"add_virtual_readout {name!r}: sources must be PhysicalReadouts, "
                f"virtual-to-virtual chaining is not supported: {non_physical}"
            )
        vrd = VirtualReadout(
            name=name,
            kind=kind,
            sources=sources,
            transform=transform,
            shape=shape,
            unit=unit,
            contains=contains,
        )
        self.readouts[name] = vrd
        return vrd

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
            rd = self.readouts[name]
            if isinstance(rd, VirtualReadout):
                raise RuntimeError(
                    f"Readout {name!r} is a VirtualReadout — use compute(data) "
                    "with measured source data, not bench[name]"
                )
            return rd.read()
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
                if isinstance(p, VirtualController):
                    val = "—"
                    kind = "virtual"
                else:
                    try:
                        val = p.get()
                    except Exception as e:
                        val = f"ERR: {e}"
                    kind = "ctrl"
                rows.append([name, kind, val, p.unit or "", p.limits or ""])
            elif name in self.readouts:
                r = self.readouts[name]
                if isinstance(r, VirtualReadout):
                    rows.append([name, f"virtual / {r.kind.value}", "—", r.unit or "", f"sources={r.sources}"])
                else:
                    try:
                        val = r.read()
                    except Exception as e:
                        val = f"ERR: {e}"
                    # Compact display for large array readouts
                    if r.kind.value == "trace" and not isinstance(val, str):
                        val = "[...]"
                    elif r.kind.value == "image" and not isinstance(val, str):
                        val = "[[...]]"
                    rows.append([name, f"read / {r.kind.value}", val, r.unit or "", "N/A"])
            else:
                rows.append([name, "?", "NOT FOUND", ""])

        print(_tabulate(rows, headers=["Name", "Type", "Value", "Unit", "Limits"],
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
        import warnings as _warnings
        for cname, ctrl in self.controllers.items():
            lim = list(ctrl.limits) if ctrl.limits is not None else None
            base_meta: dict = {
                "unit": ctrl.unit,
                "limits": lim,
                "limit_policy": str(ctrl.limit_policy),
            }

            if isinstance(ctrl, VirtualController):
                if callable(ctrl._binding):
                    entry = {
                        "virtual": True,
                        "binding": None,
                        **base_meta,
                        "_note": "Callable binding — re-register manually after load.",
                    }
                    src = _func_source(ctrl._binding)
                    if src is not None:
                        entry["binding_src"] = src
                    _warnings.warn(
                        f"Bench.save: virtual controller {cname!r} has a callable "
                        "binding that cannot be serialized. A stub will be saved.",
                        stacklevel=2,
                    )
                else:
                    entry = {
                        "virtual": True,
                        "binding": dict(ctrl._binding),
                        **base_meta,
                    }
                config["controllers"][cname] = entry

            elif ctrl.instrument is not None and ctrl.attr is not None:
                instr_name = next(
                    (n for n, a in self.instruments.items() if a is ctrl.instrument),
                    None,
                )
                config["controllers"][cname] = {
                    "instrument": instr_name,
                    "attr": ctrl.attr,
                    **base_meta,
                }
            else:
                entry = {
                    "instrument": None,
                    "attr": None,
                    **base_meta,
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
        import warnings as _ro_warnings
        for rname, ro in self.readouts.items():
            base: dict = {
                "kind": str(ro.kind),
                "shape": list(ro.shape) if ro.shape is not None else None,
                "unit": ro.unit,
                "contains": ro.contains,
            }
            if isinstance(ro, VirtualReadout):
                vro_entry: dict = {
                    **base,
                    "virtual": True,
                    "sources": list(ro.sources),
                    "_note": "Virtual readout — transform cannot be serialized. Re-register manually after load.",
                }
                src_transform = _func_source(ro.transform)
                if src_transform is not None:
                    vro_entry["transform_src"] = src_transform
                config["readouts"][rname] = vro_entry
                _ro_warnings.warn(
                    f"Bench.save: virtual readout {rname!r} transform cannot be serialized. "
                    "A stub will be saved.",
                    stacklevel=2,
                )
            elif ro.instrument is not None and ro.attr is not None:
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
        all_ctrl_cfgs = config.get("controllers", {})
        physical_cfgs = {n: c for n, c in all_ctrl_cfgs.items() if not c.get("virtual")}
        virtual_cfgs  = {n: c for n, c in all_ctrl_cfgs.items() if c.get("virtual")}

        # Physical controllers first
        for cname, ccfg in physical_cfgs.items():
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

        # Virtual controllers second — dependency-ordered so chains load correctly.
        # Repeat passes until all are resolved or no progress is made.
        pending = dict(virtual_cfgs)
        while pending:
            progress = False
            for cname, ccfg in list(pending.items()):
                binding = ccfg.get("binding")
                raw_limits = ccfg.get("limits")

                if binding is None:
                    # Callable binding — cannot restore, becomes a stub
                    bench._stubs[cname] = {
                        "kind": "controller",
                        "reason": "callable virtual binding — re-register manually",
                        "unit": ccfg.get("unit"),
                        "limits": raw_limits,
                        "limit_policy": ccfg.get("limit_policy", "warn"),
                        "binding_src": ccfg.get("binding_src"),
                    }
                    del pending[cname]
                    progress = True
                    continue

                # Check all targets are already registered
                missing = [t for t in binding if t not in bench.controllers]
                if missing:
                    continue  # retry after other virtuals load

                try:
                    bench.add_virtual_controller(
                        cname,
                        binding={k: float(v) for k, v in binding.items()},
                        unit=ccfg.get("unit"),
                        limits=tuple(raw_limits) if raw_limits is not None else None,
                        limit_policy=LimitPolicy(ccfg.get("limit_policy", "warn")),
                    )
                except Exception as exc:
                    bench._stubs[cname] = {
                        "kind": "controller",
                        "reason": f"virtual load failed: {exc}",
                        "unit": ccfg.get("unit"),
                        "limits": raw_limits,
                        "binding": binding,
                    }
                del pending[cname]
                progress = True

            if not progress:
                # Remaining entries have unresolvable targets
                for cname, ccfg in pending.items():
                    missing = [t for t in (ccfg.get("binding") or {})
                               if t not in bench.controllers]
                    bench._stubs[cname] = {
                        "kind": "controller",
                        "reason": f"virtual targets not available: {missing}",
                        "unit": ccfg.get("unit"),
                        "limits": ccfg.get("limits"),
                        "binding": ccfg.get("binding"),
                    }
                break

        # ── readouts ──────────────────────────────────────────────────
        for rname, rcfg in config.get("readouts", {}).items():
            if rcfg.get("virtual"):
                bench._stubs[rname] = {
                    "kind": "readout",
                    "reason": "virtual readout — transform not serializable, re-register manually",
                    "readout_kind": rcfg.get("kind", "scalar"),
                    "unit": rcfg.get("unit"),
                    "shape": rcfg.get("shape"),
                    "sources": rcfg.get("sources", []),
                    "transform_src": rcfg.get("transform_src"),
                }
                continue

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
