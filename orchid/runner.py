"""Experiment runner — executes procedures and saves data via zarro."""

from __future__ import annotations

import abc
import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from .parameter import DataKind
from .procedure import ErrorPolicy, MonitorProcedure, MultiSweep, Procedure, WriteMode

# Zarro imports — use absolute import from the sibling package
from zarro import (
    AxisSpecs,
    ControlVar,
    DataKind as ZarroDataKind,
    ExperimentID,
    MeasurementSchema,
    ReadoutSpecs,
    WriteType,
    ZarrWriter,
)
from zarro.core2 import StreamingWriter


def _run_coro(coro):
    """Run a coroutine, handling both script and Jupyter (running loop) contexts.

    On KeyboardInterrupt, explicitly cancels the underlying task so the
    experiment stops immediately instead of continuing in the background.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running — normal script context
        return asyncio.run(coro)
    else:
        # Event loop already running (Jupyter, IPython, etc.)
        import nest_asyncio
        nest_asyncio.apply(loop)
        task = loop.create_task(coro)
        try:
            return loop.run_until_complete(task)
        except KeyboardInterrupt:
            task.cancel()
            # Give the task a chance to process the cancellation
            try:
                loop.run_until_complete(task)
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            raise


_WRITEMODE_TO_ZARR = {
    WriteMode.POINTWISE: WriteType.POINTWISE,
    WriteMode.SWEEPWISE: WriteType.TRACEWISE,
    WriteMode.PLANEWISE: WriteType.IMAGEWISE,
    WriteMode.ALL: WriteType.ALL,
}


def _to_zarro_kind(kind: DataKind) -> ZarroDataKind:
    """Map orchid DataKind to zarro DataKind."""
    return ZarroDataKind(kind.value)


def _build_schema(proc: Procedure) -> MeasurementSchema:
    """Build a zarro MeasurementSchema from a Procedure."""
    control_axes = []
    for sweep in proc.sweeps:
        if isinstance(sweep, MultiSweep):
            cvs = [
                ControlVar(name=p.name, values=v, unit=p.unit)
                for p, v in zip(sweep.parameters, sweep.all_values)
            ]
            control_axes.append(AxisSpecs(cvs))
        else:
            cv = ControlVar(
                name=sweep.parameter.name,
                values=sweep.values,
                unit=sweep.parameter.unit,
            )
            control_axes.append(AxisSpecs([cv]))

    readout_specs = []
    for rname in proc.readouts:
        rd = proc.context.readouts[rname]
        readout_specs.append(
            ReadoutSpecs(
                name=rd.name,
                kind=_to_zarro_kind(rd.kind),
                shape=rd.shape,
                unit=rd.unit,
                contains=rd.contains,
            )
        )

    return MeasurementSchema(
        control=control_axes,
        readout=readout_specs,
        writetype=_WRITEMODE_TO_ZARR[proc.write_mode],
    )


async def _acall_hook(hook, *args):
    """Call a hook if it's not None. Supports both sync and async callables."""
    if hook is not None:
        result = hook(*args)
        if asyncio.iscoroutine(result):
            await result


# ══════════════════════════════════════════════════════════════════════
#  Write strategies
# ══════════════════════════════════════════════════════════════════════


class WriteStrategy(abc.ABC):
    """Base class for sweep execution strategies.

    Holds shared state and helpers used by all write modes.
    """

    def __init__(self, proc: Procedure, writer: ZarrWriter, pbar, plotter=None):
        self.proc = proc
        self.writer = writer
        self.pbar = pbar
        self.plotter = plotter
        # Track the last-set index and reversal flag per axis so that
        # _sweep_current_values() can read values from arrays instead of
        # querying instruments (which may be slow GPIB/USB round-trips).
        self._current_indices: dict[int, int] = {}
        self._current_reversed_map: dict[int, bool] = {}

    @abc.abstractmethod
    async def execute(self) -> None:
        """Run the full sweep and write data."""

    # ── Shared helpers ────────────────────────────────────────────────

    def _maybe_snake(self, sweep, axis, index):
        """Return primary sweep values, reversed if snake scan on odd parent."""
        values = sweep.values  # works for both Sweep and MultiSweep (.values = first array)
        if self.proc.snake and axis > 0 and len(index) > 0 and index[-1] % 2 == 1:
            values = values[::-1]
        return values

    def _is_reversed(self, axis: int, index: tuple) -> bool:
        """True if snake scan reversal applies at this axis/index."""
        return self.proc.snake and axis > 0 and len(index) > 0 and index[-1] % 2 == 1

    async def _aset_sweep_point(self, axis: int, sweep, i: int, reversed_: bool = False) -> None:
        """Set all parameters of a sweep to their i-th values.

        Also records (axis, i, reversed_) so that _sweep_current_values()
        can look up the current value from the array without re-querying
        the instrument.
        """
        self._current_indices[axis] = i
        self._current_reversed_map[axis] = reversed_
        if isinstance(sweep, MultiSweep):
            for param, vals in zip(sweep.parameters, sweep.all_values):
                actual = vals[::-1] if reversed_ else vals
                await param.aset(actual[i])
        else:
            vals = sweep.values[::-1] if reversed_ else sweep.values
            await sweep.parameter.aset(vals[i])

    def _sweep_current_values(self) -> dict:
        """Return {param_name: current_value} for all sweep parameters.

        Reads from the in-memory value arrays using the last-recorded
        indices — no instrument round-trips.
        """
        result = {}
        for axis, sweep in enumerate(self.proc.sweeps):
            i = self._current_indices.get(axis, 0)
            rev = self._current_reversed_map.get(axis, False)
            if isinstance(sweep, MultiSweep):
                for param, vals in zip(sweep.parameters, sweep.all_values):
                    actual = vals[::-1] if rev else vals
                    result[param.name] = float(actual[i])
            else:
                vals = sweep.values[::-1] if rev else sweep.values
                result[sweep.parameter.name] = float(vals[i])
        return result

    async def _safe_read(self, readout):
        """Read a readout with error handling per the procedure's policy."""
        proc = self.proc
        for attempt in range(proc.max_retries + 1):
            try:
                return await readout.aread()
            except Exception as e:
                if proc.error_policy == ErrorPolicy.STOP_AND_SAVE:
                    raise
                if proc.error_policy == ErrorPolicy.RETRY_AND_SKIP:
                    if attempt < proc.max_retries:
                        print(f"  Retry {attempt + 1}/{proc.max_retries} for {readout.name}: {e}")
                        continue
                    print(f"  Skipping {readout.name} at point (max retries exceeded): {e}")
                    return self._nan_value(readout)
                if proc.error_policy == ErrorPolicy.IGNORE:
                    print(f"  Ignoring error for {readout.name}: {e}")
                    return self._nan_value(readout)
        return self._nan_value(readout)

    def _nan_value(self, readout):
        """Return a NaN-filled value matching the readout shape."""
        return _nan_for_readout(readout)

    def _allocate_buffers(self, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        """Pre-allocate numpy buffers for all readouts."""
        buffers = {}
        for rname in self.proc.readouts:
            rd = self.proc.context.readouts[rname]
            trailing = rd.shape if rd.kind != DataKind.SCALAR else ()
            buffers[rname] = np.empty(shape + trailing, dtype=np.float32)
        return buffers

    async def _measure_into(self, buffers, index):
        """Settle, read all readouts, store into buffers at index."""
        await _acall_hook(self.proc.before_point, index)
        if self.proc.settle_time > 0:
            await asyncio.sleep(self.proc.settle_time)

        for rname in self.proc.readouts:
            buffers[rname][index] = await self._safe_read(
                self.proc.context.readouts[rname]
            )

        await _acall_hook(self.proc.after_point, index)
        if self.pbar:
            self.pbar.update(1)

    def _notify_plotter_point(self, index, data):
        """Notify the plotter after a single measurement point."""
        if self.plotter is None:
            return
        self.plotter.update_point(index, data, self._sweep_current_values())

    def _notify_plotter_sweep(self, outer_index, buffers, inner_sweep):
        """Notify the plotter after a full inner sweep completes."""
        if self.plotter is None:
            return
        sweep_values = self._sweep_current_values()
        if isinstance(inner_sweep, MultiSweep):
            for param, vals in zip(inner_sweep.parameters, inner_sweep.all_values):
                sweep_values[param.name] = vals
        else:
            sweep_values[inner_sweep.parameter.name] = inner_sweep.values
        self.plotter.update_sweep(outer_index, buffers, sweep_values)

    def _notify_plotter_plane(self, outer_index, buffers):
        """Notify the plotter after a full 2D plane completes."""
        if self.plotter is None:
            return
        self.plotter.update_plane(outer_index, buffers, self._sweep_current_values())

    async def _outer_loop(self, axis, index, on_leaf):
        """Generic recursive loop through outer sweep axes.

        Calls ``on_leaf(index)`` when reaching ``leaf_axis``.
        """
        sweep = self.proc.sweeps[axis]
        n = len(self._maybe_snake(sweep, axis, index))
        reversed_ = self._is_reversed(axis, index)

        await _acall_hook(self.proc.before_sweep, axis)
        for i in range(n):
            await self._aset_sweep_point(axis, sweep, i, reversed_)
            await on_leaf(axis + 1, index + (i,))
        await _acall_hook(self.proc.after_sweep, axis)


class PointwiseStrategy(WriteStrategy):
    """Write after every measurement point."""

    async def execute(self):
        await self._loop(axis=0, index=())

    async def _loop(self, axis, index):
        if axis == self.proc.ndim:
            await _acall_hook(self.proc.before_point, index)
            if self.proc.settle_time > 0:
                await asyncio.sleep(self.proc.settle_time)

            data = {}
            for rname in self.proc.readouts:
                data[rname] = await self._safe_read(
                    self.proc.context.readouts[rname]
                )

            self.writer.write_point(index, data)
            self._notify_plotter_point(index, data)

            await _acall_hook(self.proc.after_point, index)
            if self.pbar:
                self.pbar.update(1)
            return

        await self._outer_loop(axis, index, self._loop)


class SweepwiseStrategy(WriteStrategy):
    """Buffer the innermost sweep, write one trace at a time."""

    async def execute(self):
        if self.proc.ndim < 1:
            raise ValueError("SWEEPWISE requires at least 1 sweep axis")
        await self._recurse(axis=0, index=())

    async def _recurse(self, axis, index):
        if axis == self.proc.ndim - 1:
            await self._collect_trace(index)
            return

        await self._outer_loop(axis, index, self._recurse)

    async def _collect_trace(self, outer_index):
        """Sweep innermost axis, buffer all points, write_trace."""
        inner_axis = self.proc.ndim - 1
        sweep = self.proc.sweeps[inner_axis]
        n = len(self._maybe_snake(sweep, inner_axis, outer_index))
        reversed_ = self._is_reversed(inner_axis, outer_index)

        buffers = self._allocate_buffers((sweep.length,))

        await _acall_hook(self.proc.before_sweep, inner_axis)

        for i in range(n):
            full_index = outer_index + (i,)
            await self._aset_sweep_point(inner_axis, sweep, i, reversed_)

            await _acall_hook(self.proc.before_point, full_index)
            if self.proc.settle_time > 0:
                await asyncio.sleep(self.proc.settle_time)

            point_data = {}
            for rname in self.proc.readouts:
                point_data[rname] = await self._safe_read(
                    self.proc.context.readouts[rname]
                )
                buffers[rname][i] = point_data[rname]

            self._notify_plotter_point(full_index, point_data)

            await _acall_hook(self.proc.after_point, full_index)
            if self.pbar:
                self.pbar.update(1)

        self.writer.write_trace(outer_index, buffers)
        self._notify_plotter_sweep(outer_index, buffers, sweep)

        await _acall_hook(self.proc.after_sweep, inner_axis)


class PlanewiseStrategy(WriteStrategy):
    """Buffer the two innermost sweeps, write one plane at a time."""

    async def execute(self):
        if self.proc.ndim < 2:
            raise ValueError("PLANEWISE requires at least 2 sweep axes")
        await self._recurse(axis=0, index=())

    async def _recurse(self, axis, index):
        if axis == self.proc.ndim - 2:
            await self._collect_plane(index)
            return

        await self._outer_loop(axis, index, self._recurse)

    async def _collect_plane(self, outer_index):
        """Sweep two innermost axes, buffer all points, write_image."""
        axis_row = self.proc.ndim - 2
        axis_col = self.proc.ndim - 1
        sweep_row = self.proc.sweeps[axis_row]
        sweep_col = self.proc.sweeps[axis_col]

        plane_shape = (sweep_row.length, sweep_col.length)
        buffers = self._allocate_buffers(plane_shape)

        await _acall_hook(self.proc.before_sweep, axis_row)

        n_rows = len(self._maybe_snake(sweep_row, axis_row, outer_index))
        reversed_row = self._is_reversed(axis_row, outer_index)

        for i in range(n_rows):
            await self._aset_sweep_point(axis_row, sweep_row, i, reversed_row)

            await _acall_hook(self.proc.before_sweep, axis_col)

            n_cols = len(self._maybe_snake(sweep_col, axis_col, outer_index + (i,)))
            reversed_col = self._is_reversed(axis_col, outer_index + (i,))
            for j in range(n_cols):
                full_index = outer_index + (i, j)
                await self._aset_sweep_point(axis_col, sweep_col, j, reversed_col)

                await _acall_hook(self.proc.before_point, full_index)
                if self.proc.settle_time > 0:
                    await asyncio.sleep(self.proc.settle_time)

                point_data = {}
                for rname in self.proc.readouts:
                    point_data[rname] = await self._safe_read(
                        self.proc.context.readouts[rname]
                    )
                    buffers[rname][i, j] = point_data[rname]

                self._notify_plotter_point(full_index, point_data)

                await _acall_hook(self.proc.after_point, full_index)
                if self.pbar:
                    self.pbar.update(1)

            # Notify plotter after each completed row
            row_data = {rname: buffers[rname][i, :] for rname in self.proc.readouts}
            self._notify_plotter_sweep(outer_index + (i,), row_data, sweep_col)

            await _acall_hook(self.proc.after_sweep, axis_col)

        self.writer.write_image(outer_index, buffers)
        self._notify_plotter_plane(outer_index, buffers)

        await _acall_hook(self.proc.after_sweep, axis_row)


class AllStrategy(WriteStrategy):
    """Buffer the entire experiment, write once at the end."""

    async def execute(self):
        buffers = self._allocate_buffers(self.proc.shape)
        await self._loop(buffers, axis=0, index=())
        self.writer.write_all(buffers)

    async def _loop(self, buffers, axis, index):
        if axis == self.proc.ndim:
            await _acall_hook(self.proc.before_point, index)
            if self.proc.settle_time > 0:
                await asyncio.sleep(self.proc.settle_time)

            point_data = {}
            for rname in self.proc.readouts:
                point_data[rname] = await self._safe_read(
                    self.proc.context.readouts[rname]
                )
                buffers[rname][index] = point_data[rname]

            self._notify_plotter_point(index, point_data)

            await _acall_hook(self.proc.after_point, index)
            if self.pbar:
                self.pbar.update(1)
            return

        sweep = self.proc.sweeps[axis]
        n = len(self._maybe_snake(sweep, axis, index))
        reversed_ = self._is_reversed(axis, index)

        await _acall_hook(self.proc.before_sweep, axis)
        for i in range(n):
            await self._aset_sweep_point(axis, sweep, i, reversed_)
            await self._loop(buffers, axis + 1, index + (i,))
        await _acall_hook(self.proc.after_sweep, axis)


def _nan_for_readout(readout) -> float | np.ndarray:
    """Return a NaN-filled value matching the readout's kind and shape."""
    if readout.kind == DataKind.SCALAR:
        return np.nan
    return np.full(readout.shape, np.nan, dtype=np.float32)


_STRATEGY_MAP: dict[WriteMode, type[WriteStrategy]] = {
    WriteMode.POINTWISE: PointwiseStrategy,
    WriteMode.SWEEPWISE: SweepwiseStrategy,
    WriteMode.PLANEWISE: PlanewiseStrategy,
    WriteMode.ALL: AllStrategy,
}


# ══════════════════════════════════════════════════════════════════════
#  ExperimentRunner
# ══════════════════════════════════════════════════════════════════════


class ExperimentRunner:
    """Executes experiment procedures and manages data flow to zarro.

    Parameters
    ----------
    use_experiment_id : bool
        If True, auto-create numbered subdirectories via ExperimentID.
    """

    def __init__(self, use_experiment_id: bool = True):
        self.use_experiment_id = use_experiment_id
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._monitor_result: Path | None = None
        self._monitor_plotter = None

    def _get_data_dir(self, proc: Procedure | MonitorProcedure) -> Path:
        """Determine the output directory for this run."""
        root = Path(proc.context.data_root)
        if self.use_experiment_id:
            eid = ExperimentID(root)
            return eid.next_dir()
        return root / proc.name

    def run(self, procedure: Procedure, plotter=None,
            print_summary: bool = False,
            return_path: bool = False) -> Path | None:
        """Run a sweep experiment synchronously.

        Works both from scripts (no event loop) and from Jupyter
        notebooks (event loop already running). Handles Ctrl+C
        cleanly — saves collected data and prints a short message
        instead of a long traceback.

        Parameters
        ----------
        procedure : Procedure
            The experiment procedure to run.
        plotter : LivePlotter, optional
            Live plotter for real-time visualization.
        print_summary : bool
            If True, print the procedure summary table before running.
            Default is False.
        return_path : bool
            If True, return the Path to the saved data directory.
            Default is False (returns None).

        Returns
        -------
        Path or None
            Data directory path if ``return_path=True``, otherwise None.
        """
        # Store references so the interrupt handler can do cleanup
        # even when asyncio.run() kills arun() before it can clean up.
        self._run_state = {
            "writer": None, "pbar": None, "data_dir": None,
            "proc": procedure, "plotter": plotter,
        }
        try:
            result = _run_coro(self.arun(procedure, plotter=plotter,
                                         print_summary=print_summary))
        except (KeyboardInterrupt, asyncio.CancelledError):
            result = self._handle_interrupt()

        return result if return_path else None

    def _handle_interrupt(self) -> Path | None:
        """Clean up after KeyboardInterrupt — save data, close progress bar."""
        s = self._run_state
        if s["pbar"]:
            s["pbar"].close()
        if s["writer"] and s["data_dir"]:
            meta = {**s["proc"].context.metadata, **s["proc"].metadata, "status": "interrupted"}
            try:
                s["writer"].overwrite = True
                s["writer"].write_metadata(meta=meta)
            except Exception:
                pass
        if s["plotter"]:
            try:
                s["plotter"].stop(_silent=True)
            except Exception:
                pass
        data_dir = s["data_dir"]
        name = s["proc"].name
        if data_dir:
            print(f"\nExperiment '{name}' interrupted. Data saved to: {data_dir}")
        else:
            print(f"\nExperiment '{name}' interrupted.")
        return data_dir

    async def arun(self, proc: Procedure, plotter=None,
                   print_summary: bool = False) -> Path:
        """Run a sweep experiment asynchronously."""
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        if print_summary:
            proc.summary()

        data_dir = self._get_data_dir(proc)
        schema = _build_schema(proc)
        writer = ZarrWriter(
            root=data_dir,
            schema=schema,
            tags=proc.tags,
            overwrite=False,
            initialize_arrays=True,
        )

        (data_dir / "procedure.yaml").write_text(
            yaml.safe_dump(proc.to_dict(), sort_keys=False, allow_unicode=True)
        )

        if plotter is not None:
            plotter.setup(proc)

        total_points = 1
        for s in proc.sweeps:
            total_points *= s.length

        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=total_points, desc=proc.name, unit="pt")

        # Populate run state for interrupt cleanup in run()
        if hasattr(self, "_run_state"):
            self._run_state.update(writer=writer, pbar=pbar, data_dir=data_dir)

        await _acall_hook(proc.before_experiment)

        strategy_cls = _STRATEGY_MAP[proc.write_mode]
        strategy = strategy_cls(proc, writer, pbar, plotter)

        try:
            await strategy.execute()
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Let it propagate — run() handles cleanup via _handle_interrupt()
            raise
        except Exception:
            meta = {**proc.context.metadata, **proc.metadata, "status": "error"}
            writer.overwrite = True
            writer.write_metadata(meta=meta)
            if pbar:
                pbar.close()
            raise

        if pbar:
            pbar.close()

        meta = {**proc.context.metadata, **proc.metadata, "status": "completed"}
        writer.overwrite = True
        writer.write_metadata(meta=meta)

        await _acall_hook(proc.after_experiment)

        if plotter is not None:
            plotter.stop()

        print(f"Experiment '{proc.name}' completed. Data saved to: {data_dir}")
        return data_dir

    # ── Monitor mode ──────────────────────────────────────────────────

    def run_monitor(self, procedure: MonitorProcedure, plotter=None,
                     background: bool = False,
                     print_summary: bool = False,
                     return_path: bool = False) -> Path | None:
        """Run time-series monitoring.

        Parameters
        ----------
        procedure : MonitorProcedure
            The monitoring procedure to run.
        plotter : LivePlotter, optional
            Live plotter for real-time visualization.
        background : bool
            If True, run in a background thread and return immediately.
            Use ``ctx["Vgt"] = 0.5`` in the next cell to change parameters
            while monitoring. Call ``runner.stop_monitor()`` to stop.
        print_summary : bool
            If True, print the procedure summary table before running.
            Default is False.
        return_path : bool
            If True, return the Path to the saved data directory.
            Default is False (returns None).

        Returns
        -------
        Path or None
            Data directory path if ``return_path=True``, otherwise None.
            In background mode, always returns None immediately; the path
            is available via ``runner.stop_monitor()`` after stopping.
        """
        if background:
            return self._run_monitor_background(procedure, plotter)

        self._monitor_stop.clear()
        self._run_state = {
            "writer": None, "pbar": None, "data_dir": None,
            "proc": procedure, "plotter": plotter,
        }
        try:
            result = _run_coro(self.arun_monitor(procedure, plotter=plotter,
                                                  print_summary=print_summary))
        except (KeyboardInterrupt, asyncio.CancelledError):
            result = self._handle_interrupt()

        return result if return_path else None

    def _run_monitor_background(self, procedure, plotter) -> None:
        """Start monitor in a background thread."""
        self._monitor_stop.clear()
        self._monitor_result = None
        self._monitor_plotter = plotter

        def _target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.arun_monitor(procedure, plotter=plotter)
                )
                self._monitor_result = result
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                loop.close()

        self._monitor_thread = threading.Thread(target=_target, daemon=True)
        self._monitor_thread.start()
        print(f"Monitor '{procedure.name}' running in background. Use runner.stop_monitor() to stop.")

    @property
    def is_monitoring(self) -> bool:
        """True if a background monitor is currently running."""
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def stop_monitor(self) -> Path | None:
        """Stop a background monitor and return the data directory.

        Returns
        -------
        Path or None
            Path to the saved data directory.
        """
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=10)
            self._monitor_thread = None
        if self._monitor_plotter is not None:
            self._monitor_plotter.stop(_silent=True)
            self._monitor_plotter = None
        result = self._monitor_result
        if result:
            print(f"Monitor stopped. Data saved to: {result}")
        return result

    async def arun_monitor(self, proc: MonitorProcedure, plotter=None,
                           print_summary: bool = False) -> Path:
        """Run time-series monitoring asynchronously."""
        if print_summary:
            proc.summary()

        data_dir = self._get_data_dir(proc)

        readout_shapes = {}
        for rname in proc.readouts:
            rd = proc.context.readouts[rname]
            if rd.kind == DataKind.SCALAR:
                readout_shapes[rname] = ()
            else:
                readout_shapes[rname] = rd.shape

        writer = StreamingWriter(
            root=data_dir,
            readouts=readout_shapes,
            chunk_size=proc.chunk_size,
            overwrite=False,
            tags=proc.tags,
        )

        (data_dir / "procedure.yaml").write_text(
            yaml.safe_dump(proc.to_dict(), sort_keys=False, allow_unicode=True)
        )

        if plotter is not None:
            plotter.setup(proc)

        # Populate run state so interrupt handler can print the data path
        if hasattr(self, "_run_state"):
            self._run_state.update(data_dir=data_dir)

        await _acall_hook(proc.before_experiment)

        start_time = time.time()
        sample_idx = 0

        # Register event callback — fires on every ctx["param"] = value
        def _on_event(entry):
            if plotter is not None:
                plotter.notify_event(entry["time"], entry["param"], entry["value"])

        proc.context._start_event_log(on_event=_on_event)

        interrupted = False
        try:
            while True:
                # Check stop signal from stop_monitor()
                if self._monitor_stop.is_set():
                    break

                # Check duration
                if proc.duration is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= proc.duration:
                        break

                # Read all readouts — errors are caught per-readout so a single
                # instrument hiccup doesn't abort a long-running monitor session.
                data = {}
                for rname in proc.readouts:
                    readout = proc.context.readouts[rname]
                    try:
                        data[rname] = await readout.aread()
                    except Exception as e:
                        print(f"  Warning: read error for '{rname}' at sample {sample_idx}: {e}")
                        data[rname] = _nan_for_readout(readout)

                timestamp = time.time()
                writer.append(data, timestamp=timestamp)

                if plotter is not None:
                    plotter.update_monitor(sample_idx, data, timestamp)

                if proc.after_point is not None:
                    await _acall_hook(proc.after_point, sample_idx, data)

                # Check stop condition
                if proc.stop_condition is not None and proc.stop_condition(data):
                    break

                sample_idx += 1

                if sample_idx % 100 == 0:
                    print(f"  Monitor: {sample_idx} samples collected")

                await asyncio.sleep(proc.interval)

        except KeyboardInterrupt:
            interrupted = True
        finally:
            event_log = proc.context._stop_event_log()

        status = "interrupted" if interrupted else "completed"
        meta = {**proc.context.metadata, **proc.metadata, "status": status}
        writer.close(meta=meta)

        # Write events.yaml if any parameter changes were recorded
        if event_log:
            for entry in event_log:
                entry["elapsed"] = round(entry["time"] - start_time, 3)
            (data_dir / "events.yaml").write_text(
                yaml.safe_dump(event_log, sort_keys=True, allow_unicode=True)
            )

        await _acall_hook(proc.after_experiment)

        if plotter is not None:
            plotter.finalize()

        if interrupted:
            print(f"\nMonitor '{proc.name}' stopped by user after {sample_idx} samples. Data saved to: {data_dir}")
        else:
            print(f"Monitor '{proc.name}' completed. Data saved to: {data_dir}")
        return data_dir
