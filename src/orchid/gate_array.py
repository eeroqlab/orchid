"""GateArray — a named group of bench controllers with compound operations.

A GateArray bundles multiple bench controllers (e.g. all the gate voltages on
a multi-channel DAC) into one object that supports simultaneous set, linear
ramps, and named configuration presets.

Example
-------
>>> gates = bench.add_gate_array("gates", ["P1", "B1", "B2", "ST"])
>>> gates.ramp({"P1": -0.4, "B1": -0.2}, steps=200, dt=0.005)
>>> gates.save_config("pinchoff")
>>> gates.save_config("near_pinchoff", base="pinchoff", P1=-0.35)
>>> gates.ramp_to_config("near_pinchoff")
>>> proc = Procedure(sweeps=[gates.to_multisweep({"P1": -0.5}, n_pts=50)])
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bench import Bench
    from .procedure import MultiSweep


class GateArray:
    """A named group of bench controllers with compound operations and presets.

    Parameters
    ----------
    bench : Bench
        The lab bench whose controllers this array manages.
    channels : list[str]
        Ordered list of bench controller names included in this array.
    name : str
        Display name used in repr, YAML files, and error messages.

    Notes
    -----
    All operation methods accept **partial dicts** — only the specified
    channels are touched; unspecified ones are left unchanged.

    Configs are partial by design: ``save_config("pinchoff")`` records the
    current state of all channels, but you can apply a config that only
    specifies a subset and the rest of the array is untouched.

    Virtual controllers (:class:`~orchid.controller.VirtualController`) are
    fully supported — their ``get()`` returns the last setpoint cached by
    ``set()``  (or ``nan`` if never written).  No special ``start=``
    argument is needed.
    """

    def __init__(
        self,
        bench: "Bench",
        channels: list[str],
        *,
        name: str = "gates",
    ) -> None:
        self._bench = bench
        self._channels: list[str] = list(channels)
        self.name = name
        self._configs: dict[str, dict[str, float]] = {}

    # ── Validation helpers ────────────────────────────────────────────────────

    def _validate_names(self, names) -> None:
        """Raise KeyError for any name not in this array's channel list."""
        keys = list(names.keys()) if hasattr(names, "keys") else list(names)
        unknown = [k for k in keys if k not in self._channels]
        if unknown:
            raise KeyError(
                f"GateArray '{self.name}': unknown channel(s) {unknown}. "
                f"Available: {self._channels}"
            )

    def _check_limits(self, values: dict[str, float]) -> None:
        """Raise ValueError if any target value would exceed a controller limit."""
        violations = []
        for name, val in values.items():
            ctrl = self._bench.controllers[name]
            if ctrl.limits is not None:
                lo, hi = ctrl.limits
                if not (lo <= val <= hi):
                    violations.append(f"  {name}: {val!r} outside [{lo}, {hi}]")
        if violations:
            raise ValueError(
                f"GateArray '{self.name}' limit violation(s):\n"
                + "\n".join(violations)
            )

    # ── Basic get / set ───────────────────────────────────────────────────────

    def get(self, names: list[str] | None = None) -> dict[str, float]:
        """Read current values from the bench.

        For physical controllers this queries the instrument.  For virtual
        controllers it returns the cached last-set value (``nan`` if not yet
        written — see :class:`~orchid.controller.VirtualController`).

        Parameters
        ----------
        names : list[str] or None
            Subset of channels to read. ``None`` reads all channels.

        Returns
        -------
        dict[str, float]
            ``{channel_name: current_value}``.
        """
        keys = names if names is not None else self._channels
        if names is not None:
            self._validate_names(names)
        return {k: self._bench[k] for k in keys}

    def set(self, values: dict[str, float]) -> None:
        """Set controller values immediately. Bench events fire normally.

        Parameters
        ----------
        values : dict[str, float]
            ``{channel_name: target_value}``.  Partial dicts are fine.
        """
        self._validate_names(values)
        self._check_limits(values)
        for k, v in values.items():
            self._bench[k] = v

    # ── Ramp ──────────────────────────────────────────────────────────────────

    def ramp(
        self,
        targets: dict[str, float],
        *,
        steps: int = 100,
        dt: float = 0.01,
        log_events: bool = True,
        start: dict[str, float] | None = None,
    ) -> None:
        """Interleaved linear ramp to target values.

        All specified channels step simultaneously at each iteration.
        Bench events are suppressed during the ramp; one summary event per
        channel is emitted at the end (when ``log_events=True`` and a monitor
        is running).

        A ``KeyboardInterrupt`` propagates cleanly — channels stay wherever
        the last completed step left them.

        Parameters
        ----------
        targets : dict[str, float]
            ``{channel_name: target_value}``.  Partial dicts are fine.
        steps : int
            Number of interpolation steps (default 100).
        dt : float
            Seconds between steps (default 0.01 s).
        log_events : bool
            Emit a bench event for each ramped channel when the ramp
            completes (default ``True``).
        start : dict[str, float] or None
            Explicit starting values.  ``None`` (default) reads current values
            via ``bench[name]`` — which for virtual controllers returns the
            cached last-set value.
        """
        self._validate_names(targets)
        self._check_limits(targets)
        starts = dict(start) if start is not None else {}
        for k in targets:
            if k not in starts:
                starts[k] = self._bench[k]
        with self._bench.suppress_events():
            for i in range(1, steps + 1):
                frac = i / steps
                for k in targets:
                    self._bench[k] = starts[k] + (targets[k] - starts[k]) * frac
                time.sleep(dt)
        if log_events:
            for k, v in targets.items():
                self._bench._fire_event(k, v)

    async def aramp(
        self,
        targets: dict[str, float],
        *,
        steps: int = 100,
        dt: float = 0.01,
        log_events: bool = True,
        start: dict[str, float] | None = None,
    ) -> None:
        """Async interleaved linear ramp. Drop-in replacement for :meth:`ramp`.

        Uses ``asyncio.sleep`` so the event loop is not blocked.

        Parameters
        ----------
        targets : dict[str, float]
            ``{channel_name: target_value}``.  Partial dicts are fine.
        steps : int
            Number of interpolation steps (default 100).
        dt : float
            Seconds between steps (default 0.01 s).
        log_events : bool
            Emit a bench event for each ramped channel when the ramp
            completes (default ``True``).
        start : dict[str, float] or None
            Explicit starting values.  ``None`` reads via ``bench[name]``.
        """
        self._validate_names(targets)
        self._check_limits(targets)
        starts = dict(start) if start is not None else {}
        for k in targets:
            if k not in starts:
                starts[k] = self._bench[k]
        with self._bench.suppress_events():
            for i in range(1, steps + 1):
                frac = i / steps
                for k in targets:
                    self._bench[k] = starts[k] + (targets[k] - starts[k]) * frac
                await asyncio.sleep(dt)
        if log_events:
            for k, v in targets.items():
                self._bench._fire_event(k, v)

    # ── Configs ───────────────────────────────────────────────────────────────

    def save_config(
        self,
        name: str,
        values: dict[str, float] | None = None,
        *,
        base: str | None = None,
        **overrides: float,
    ) -> None:
        """Save a named configuration preset.

        Priority order:

        1. ``values`` — explicit full dict (used as-is, then overrides applied).
        2. ``base`` — copy an existing config, then apply overrides.
        3. Neither — snapshot current bench state, then apply overrides.

        Parameters
        ----------
        name : str
            Config name to save (or overwrite).
        values : dict[str, float] or None
            Explicit channel values.  ``None`` → read from bench (or ``base``).
        base : str or None
            Existing config name to inherit from.
        **overrides : float
            Per-channel overrides applied on top of ``values`` or ``base``.
            Use the controller name as the keyword, e.g. ``P1=-0.45``.

        Examples
        --------
        >>> gates.save_config("pinchoff")                       # current state
        >>> gates.save_config("near_pinchoff", base="pinchoff", P1=-0.35)
        >>> gates.save_config("manual", {"P1": -0.5, "B1": -0.2})
        """
        if values is not None:
            config = dict(values)
        elif base is not None:
            if base not in self._configs:
                raise KeyError(
                    f"Base config '{base}' not found in gate array '{self.name}'. "
                    f"Available: {list(self._configs)}"
                )
            config = dict(self._configs[base])
        else:
            # Snapshot current state — for virtual controllers this returns the
            # cached last-set value (nan if never written); nan values are excluded.
            import math as _math
            config = {k: v for k, v in self.get().items() if not _math.isnan(v)}

        if overrides:
            config.update({k: float(v) for k, v in overrides.items()})

        if config:
            self._validate_names(config)
        self._configs[name] = config

    def load_config(self, name: str) -> dict[str, float]:
        """Return a copy of a named config without applying it.

        Parameters
        ----------
        name : str
            Config name to retrieve.

        Returns
        -------
        dict[str, float]
            A copy of the stored ``{channel: value}`` dict.
        """
        if name not in self._configs:
            raise KeyError(
                f"Config '{name}' not found in gate array '{self.name}'. "
                f"Available: {list(self._configs)}"
            )
        return dict(self._configs[name])

    def apply_config(
        self,
        name: str,
        *,
        ramp: bool = False,
        **ramp_kwargs,
    ) -> None:
        """Apply a named config by setting or ramping to it.

        Parameters
        ----------
        name : str
            Config name to apply.
        ramp : bool
            If ``True``, ramp to the config values instead of jumping.
        **ramp_kwargs
            Passed to :meth:`ramp` (``steps``, ``dt``, ``log_events``).
        """
        values = self.load_config(name)
        if ramp:
            self.ramp(values, **ramp_kwargs)
        else:
            self.set(values)

    def ramp_to_config(self, name: str, **ramp_kwargs) -> None:
        """Ramp to a named config. Shorthand for ``apply_config(name, ramp=True)``.

        Parameters
        ----------
        name : str
            Config name to ramp to.
        **ramp_kwargs
            Passed to :meth:`ramp` (``steps``, ``dt``, ``log_events``).
        """
        self.apply_config(name, ramp=True, **ramp_kwargs)

    def list_configs(self) -> list[str]:
        """Return a list of saved config names in insertion order."""
        return list(self._configs)

    def delete_config(self, name: str) -> None:
        """Delete a named config.

        Parameters
        ----------
        name : str
            Config name to remove.
        """
        if name not in self._configs:
            raise KeyError(
                f"Config '{name}' not found in gate array '{self.name}'."
            )
        del self._configs[name]

    # ── Sweep integration ─────────────────────────────────────────────────────

    def to_multisweep(
        self,
        targets: dict[str, float],
        n_pts: int,
        *,
        start: dict[str, float] | None = None,
    ) -> "MultiSweep":
        """Build a :class:`~orchid.procedure.MultiSweep` from current to target values.

        Parameters
        ----------
        targets : dict[str, float]
            ``{channel_name: final_value}`` for each channel to sweep.
        n_pts : int
            Number of points along the trajectory.
        start : dict[str, float] or None
            Starting values.  ``None`` (default) reads current bench values
            via ``bench[name]`` — which for virtual controllers returns the
            cached last-set value.

        Returns
        -------
        MultiSweep
            Ready to pass as an element of ``Procedure(sweeps=[...])``.

        Example
        -------
        >>> proc = Procedure(
        ...     sweeps=[gates.to_multisweep({"P1": -0.5, "B1": -0.3}, n_pts=50)],
        ...     readouts=["lockin_X"],
        ... )
        """
        import numpy as np
        from .procedure import MultiSweep

        self._validate_names(targets)
        starts = dict(start) if start is not None else {}
        for k in targets:
            if k not in starts:
                starts[k] = self._bench[k]
        controllers = list(targets)
        values = [np.linspace(float(starts[k]), float(targets[k]), n_pts)
                  for k in controllers]
        return MultiSweep(controllers=controllers, values=values)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a table of channel names, current values, units, and limits."""
        try:
            from tabulate import tabulate as _tabulate
            use_tabulate = True
        except ImportError:
            use_tabulate = False

        rows = []
        for ch in self._channels:
            ctrl = self._bench.controllers[ch]
            try:
                val = self._bench[ch]
                val_str = f"{val:.6g}"
            except Exception:
                val_str = "—"
            unit = ctrl.unit or ""
            limits = (
                f"[{ctrl.limits[0]}, {ctrl.limits[1]}]"
                if ctrl.limits is not None else "—"
            )
            rows.append([ch, val_str, unit, limits])

        print(f"GateArray '{self.name}'  ({len(self._channels)} channels,"
              f" {len(self._configs)} configs)")
        if use_tabulate:
            print(_tabulate(rows, headers=["Channel", "Value", "Unit", "Limits"],
                            tablefmt="simple"))
        else:
            print(f"{'Channel':<16} {'Value':>12}  {'Unit':<8} Limits")
            for ch, val, unit, lim in rows:
                print(f"{ch:<16} {val:>12}  {unit:<8} {lim}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write configs to a YAML file.

        The file stores the array name, channel list, and all saved configs.
        Load back with :meth:`load`.

        Parameters
        ----------
        path : str or Path
            Destination ``.yaml`` file.
        """
        import yaml

        data = {
            "name": self.name,
            "channels": list(self._channels),
            "configs": {n: dict(cfg) for n, cfg in self._configs.items()},
        }
        path = Path(path)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"GateArray '{self.name}': {len(self._configs)} config(s) saved → {path}")

    def load(self, path: str | Path) -> None:
        """Load configs from a YAML file (merges into existing configs).

        Existing configs with the same name are overwritten; others are kept.

        Parameters
        ----------
        path : str or Path
            Source ``.yaml`` file written by :meth:`save`.
        """
        import yaml

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        loaded = data.get("configs", {})
        self._configs.update(loaded)
        print(
            f"GateArray '{self.name}': {len(loaded)} config(s) loaded from {path}"
        )

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"GateArray('{self.name}', "
            f"channels={self._channels}, "
            f"configs={list(self._configs)})"
        )

    def __len__(self) -> int:
        """Number of channels in this array."""
        return len(self._channels)

    def __contains__(self, name: str) -> bool:
        """True if *name* is a channel in this array."""
        return name in self._channels

    def __iter__(self):
        """Iterate over channel names."""
        return iter(self._channels)
