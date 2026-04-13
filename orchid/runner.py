"""Experiment runner — executes procedures and saves data via zarro."""

from __future__ import annotations

import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .parameter import DataKind
from .procedure import ErrorPolicy, MonitorProcedure, Procedure, WriteMode

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


def _call_hook(hook, *args):
    """Call a hook if it's not None. Supports sync callables."""
    if hook is not None:
        result = hook(*args)
        if asyncio.iscoroutine(result):
            raise TypeError(
                f"Hook {hook} returned a coroutine. Use async hooks with the async runner."
            )


async def _acall_hook(hook, *args):
    """Call a hook if it's not None. Supports both sync and async callables."""
    if hook is not None:
        result = hook(*args)
        if asyncio.iscoroutine(result):
            await result


class ExperimentRunner:
    """Executes experiment procedures and manages data flow to zarro.

    Parameters
    ----------
    use_experiment_id : bool
        If True, auto-create numbered subdirectories via ExperimentID.
    """

    def __init__(self, use_experiment_id: bool = True):
        self.use_experiment_id = use_experiment_id

    def _get_data_dir(self, proc: Procedure | MonitorProcedure) -> Path:
        """Determine the output directory for this run."""
        root = Path(proc.context.data_root)
        if self.use_experiment_id:
            eid = ExperimentID(root)
            return eid.next_dir()
        return root / proc.name

    def run(self, procedure: Procedure) -> Path:
        """Run a sweep experiment synchronously.

        Returns the path to the saved data directory.
        """
        return asyncio.run(self.arun(procedure))

    async def arun(self, proc: Procedure) -> Path:
        """Run a sweep experiment asynchronously."""
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        data_dir = self._get_data_dir(proc)
        schema = _build_schema(proc)
        writer = ZarrWriter(
            root=data_dir,
            schema=schema,
            tags=proc.tags,
            overwrite=False,
            initialize_arrays=True,
        )

        total_points = 1
        for s in proc.sweeps:
            total_points *= s.length

        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=total_points, desc=proc.name, unit="pt")

        await _acall_hook(proc.before_experiment, proc)

        try:
            if proc.write_mode == WriteMode.ALL:
                await self._run_all(proc, writer, pbar)
            elif proc.write_mode == WriteMode.PLANEWISE:
                await self._run_planewise(proc, writer, pbar)
            elif proc.write_mode == WriteMode.SWEEPWISE:
                await self._run_sweepwise(proc, writer, pbar)
            else:
                await self._run_pointwise(proc, writer, pbar)
        except BaseException:
            # Always save metadata even on error
            meta = {**proc.context.metadata, **proc.metadata, "status": "error"}
            writer.write_metadata(meta=meta)
            if pbar:
                pbar.close()
            raise

        meta = {**proc.context.metadata, **proc.metadata, "status": "completed"}
        writer.write_metadata(meta=meta)

        await _acall_hook(proc.after_experiment, proc)

        if pbar:
            pbar.close()

        print(f"Experiment '{proc.name}' completed. Data saved to: {data_dir}")
        return data_dir

    # ── POINTWISE: write after every point ────────────────────────────

    async def _run_pointwise(self, proc, writer, pbar):
        """Nested sweep loop writing each point immediately."""
        await self._pointwise_loop(proc, writer, pbar, axis=0, index=())

    async def _pointwise_loop(self, proc, writer, pbar, axis, index):
        if axis == proc.ndim:
            await self._measure_and_write_point(proc, writer, index, pbar)
            return

        sweep = proc.sweeps[axis]
        values = self._maybe_snake(proc, sweep, axis, index)

        await _acall_hook(proc.before_sweep, axis, proc)
        for i, val in enumerate(values):
            await sweep.parameter.aset(val)
            await self._pointwise_loop(proc, writer, pbar, axis + 1, index + (i,))
        await _acall_hook(proc.after_sweep, axis, proc)

    async def _measure_and_write_point(self, proc, writer, index, pbar):
        await _acall_hook(proc.before_point, index, proc)
        if proc.settle_time > 0:
            await asyncio.sleep(proc.settle_time)

        data = {}
        for rname in proc.readouts:
            data[rname] = await self._safe_read(proc.context.readouts[rname], proc)

        writer.write_point(index, data)

        await _acall_hook(proc.after_point, index, proc)
        if pbar:
            pbar.update(1)

    # ── SWEEPWISE: buffer innermost sweep, write per trace ────────────

    async def _run_sweepwise(self, proc, writer, pbar):
        """Nested sweep loop that buffers the innermost axis and writes
        one full trace at a time via ``writer.write_trace()``."""
        if proc.ndim < 1:
            raise ValueError("SWEEPWISE requires at least 1 sweep axis")
        await self._sweepwise_outer(proc, writer, pbar, axis=0, index=())

    async def _sweepwise_outer(self, proc, writer, pbar, axis, index):
        """Recurse through outer axes; when we reach the innermost, hand off
        to ``_sweepwise_inner`` which collects a full trace."""
        inner_axis = proc.ndim - 1

        if axis == inner_axis:
            # We are at the innermost sweep — collect and write a trace
            await self._sweepwise_inner(proc, writer, pbar, index)
            return

        sweep = proc.sweeps[axis]
        values = self._maybe_snake(proc, sweep, axis, index)

        await _acall_hook(proc.before_sweep, axis, proc)
        for i, val in enumerate(values):
            await sweep.parameter.aset(val)
            await self._sweepwise_outer(proc, writer, pbar, axis + 1, index + (i,))
        await _acall_hook(proc.after_sweep, axis, proc)

    async def _sweepwise_inner(self, proc, writer, pbar, outer_index):
        """Sweep the innermost axis, buffer all points, then write_trace."""
        inner_axis = proc.ndim - 1
        sweep = proc.sweeps[inner_axis]
        values = self._maybe_snake(proc, sweep, inner_axis, outer_index)

        # Pre-allocate buffers for each readout
        buffers = {}
        for rname in proc.readouts:
            rd = proc.context.readouts[rname]
            if rd.kind == DataKind.SCALAR:
                buffers[rname] = np.empty(sweep.length, dtype=np.float32)
            elif rd.kind == DataKind.TRACE:
                buffers[rname] = np.empty((sweep.length,) + rd.shape, dtype=np.float32)
            elif rd.kind == DataKind.IMAGE:
                buffers[rname] = np.empty((sweep.length,) + rd.shape, dtype=np.float32)

        await _acall_hook(proc.before_sweep, inner_axis, proc)

        for i, val in enumerate(values):
            full_index = outer_index + (i,)
            await sweep.parameter.aset(val)

            await _acall_hook(proc.before_point, full_index, proc)
            if proc.settle_time > 0:
                await asyncio.sleep(proc.settle_time)

            for rname in proc.readouts:
                buffers[rname][i] = await self._safe_read(
                    proc.context.readouts[rname], proc
                )

            await _acall_hook(proc.after_point, full_index, proc)
            if pbar:
                pbar.update(1)

        # Write the full inner sweep at once
        writer.write_trace(outer_index, buffers)

        await _acall_hook(proc.after_sweep, inner_axis, proc)

    # ── PLANEWISE: buffer two innermost sweeps, write per plane ─────

    async def _run_planewise(self, proc, writer, pbar):
        """Nested sweep loop that buffers the two innermost axes and writes
        one full plane at a time via ``writer.write_image()``."""
        if proc.ndim < 2:
            raise ValueError("PLANEWISE requires at least 2 sweep axes")
        await self._planewise_outer(proc, writer, pbar, axis=0, index=())

    async def _planewise_outer(self, proc, writer, pbar, axis, index):
        """Recurse through outer axes; when we reach the second-to-last,
        hand off to ``_planewise_inner`` which collects a full 2D plane."""
        plane_start = proc.ndim - 2

        if axis == plane_start:
            await self._planewise_inner(proc, writer, pbar, index)
            return

        sweep = proc.sweeps[axis]
        values = self._maybe_snake(proc, sweep, axis, index)

        await _acall_hook(proc.before_sweep, axis, proc)
        for i, val in enumerate(values):
            await sweep.parameter.aset(val)
            await self._planewise_outer(proc, writer, pbar, axis + 1, index + (i,))
        await _acall_hook(proc.after_sweep, axis, proc)

    async def _planewise_inner(self, proc, writer, pbar, outer_index):
        """Sweep the two innermost axes, buffer all points, then write_image."""
        axis_row = proc.ndim - 2
        axis_col = proc.ndim - 1
        sweep_row = proc.sweeps[axis_row]
        sweep_col = proc.sweeps[axis_col]

        # Pre-allocate buffers: shape = (row_len, col_len [, ...readout shape])
        plane_shape = (sweep_row.length, sweep_col.length)
        buffers = {}
        for rname in proc.readouts:
            rd = proc.context.readouts[rname]
            if rd.kind == DataKind.SCALAR:
                buffers[rname] = np.empty(plane_shape, dtype=np.float32)
            elif rd.kind == DataKind.TRACE:
                buffers[rname] = np.empty(plane_shape + rd.shape, dtype=np.float32)
            elif rd.kind == DataKind.IMAGE:
                buffers[rname] = np.empty(plane_shape + rd.shape, dtype=np.float32)

        await _acall_hook(proc.before_sweep, axis_row, proc)

        for i, val_row in enumerate(self._maybe_snake(proc, sweep_row, axis_row, outer_index)):
            await sweep_row.parameter.aset(val_row)

            await _acall_hook(proc.before_sweep, axis_col, proc)

            col_values = self._maybe_snake(proc, sweep_col, axis_col, outer_index + (i,))
            for j, val_col in enumerate(col_values):
                full_index = outer_index + (i, j)
                await sweep_col.parameter.aset(val_col)

                await _acall_hook(proc.before_point, full_index, proc)
                if proc.settle_time > 0:
                    await asyncio.sleep(proc.settle_time)

                for rname in proc.readouts:
                    buffers[rname][i, j] = await self._safe_read(
                        proc.context.readouts[rname], proc
                    )

                await _acall_hook(proc.after_point, full_index, proc)
                if pbar:
                    pbar.update(1)

            await _acall_hook(proc.after_sweep, axis_col, proc)

        # Write the full 2D plane at once
        writer.write_image(outer_index, buffers)

        await _acall_hook(proc.after_sweep, axis_row, proc)

    # ── ALL: buffer everything, write once at the end ─────────────────

    async def _run_all(self, proc, writer, pbar):
        """Sweep all axes, buffer all data in memory, write once at the end."""
        # Pre-allocate full-size buffers
        shape = proc.shape
        buffers = {}
        for rname in proc.readouts:
            rd = proc.context.readouts[rname]
            if rd.kind == DataKind.SCALAR:
                buffers[rname] = np.empty(shape, dtype=np.float32)
            elif rd.kind == DataKind.TRACE:
                buffers[rname] = np.empty(shape + rd.shape, dtype=np.float32)
            elif rd.kind == DataKind.IMAGE:
                buffers[rname] = np.empty(shape + rd.shape, dtype=np.float32)

        await self._all_loop(proc, buffers, pbar, axis=0, index=())

        # Single write for all data
        writer.write_all(buffers)

    async def _all_loop(self, proc, buffers, pbar, axis, index):
        if axis == proc.ndim:
            await _acall_hook(proc.before_point, index, proc)
            if proc.settle_time > 0:
                await asyncio.sleep(proc.settle_time)

            for rname in proc.readouts:
                buffers[rname][index] = await self._safe_read(
                    proc.context.readouts[rname], proc
                )

            await _acall_hook(proc.after_point, index, proc)
            if pbar:
                pbar.update(1)
            return

        sweep = proc.sweeps[axis]
        values = self._maybe_snake(proc, sweep, axis, index)

        await _acall_hook(proc.before_sweep, axis, proc)
        for i, val in enumerate(values):
            await sweep.parameter.aset(val)
            await self._all_loop(proc, buffers, pbar, axis + 1, index + (i,))
        await _acall_hook(proc.after_sweep, axis, proc)

    # ── Shared helpers ────────────────────────────────────────────────

    def _maybe_snake(self, proc, sweep, axis, index):
        """Return sweep values, reversed if snake scan is active on odd parent."""
        values = sweep.values
        if proc.snake and axis > 0 and len(index) > 0 and index[-1] % 2 == 1:
            values = values[::-1]
        return values

    async def _safe_read(self, readout, proc):
        """Read a readout with error handling per the procedure's policy."""
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
        if readout.kind == DataKind.SCALAR:
            return np.nan
        return np.full(readout.shape, np.nan, dtype=np.float32)

    # ── Monitor mode ──────────────────────────────────────────────────

    def run_monitor(self, procedure: MonitorProcedure) -> Path:
        """Run time-series monitoring synchronously."""
        return asyncio.run(self.arun_monitor(procedure))

    async def arun_monitor(self, proc: MonitorProcedure) -> Path:
        """Run time-series monitoring asynchronously."""
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
            overwrite=False,
            tags=proc.tags,
        )

        await _acall_hook(proc.before_experiment, proc)

        start_time = time.time()
        sample_idx = 0

        try:
            while True:
                # Check duration
                if proc.duration is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= proc.duration:
                        break

                # Read all readouts
                data = {}
                for rname in proc.readouts:
                    readout = proc.context.readouts[rname]
                    data[rname] = await readout.aread()

                writer.append(data)

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
            print(f"\n  Monitor stopped by user after {sample_idx} samples")

        meta = {**proc.context.metadata, **proc.metadata, "status": "completed"}
        writer.close(meta=meta)

        await _acall_hook(proc.after_experiment, proc)

        print(f"Monitor '{proc.name}' completed. Data saved to: {data_dir}")
        return data_dir
