"""TaipyPlotter — Taipy GUI backend (push-based)."""
from __future__ import annotations

import copy
import threading
import time

from ._spec import PlotSpec, EventLineConfig
from ._base import PlotterBase


# ---------------------------------------------------------------------------
# TaipyPlotter — Taipy GUI backend (push-based via broadcast_callback)
# ---------------------------------------------------------------------------

class TaipyPlotter(PlotterBase):
    """Live plotting backend using Taipy GUI.

    Uses Taipy's ``broadcast_callback`` to push figure updates directly to
    every connected browser tab — no polling required.

    Requires ``pip install taipy-gui>=3.1``.

    Parameters
    ----------
    plots:
        List of :class:`PlotSpec` descriptors (same as :class:`DashPlotter`).
    port:
        TCP port for the Taipy GUI server (default ``5000``).
    host:
        Hostname / IP to bind (default ``"localhost"``).
    height, width:
        Figure dimensions in pixels.
    open_browser:
        Open a browser tab automatically when the server starts.
    event_line:
        Optional :class:`EventLineConfig` — draws a vertical marker when
        :py:meth:`notify_event` is called.
    max_display_pts:
        Down-sample display to at most this many points per trace.
    """

    def __init__(
        self,
        plots: "list[PlotSpec]",
        port: int = 5000,
        host: str = "localhost",
        height: int = 350,
        width: int = 700,
        open_browser: bool = True,
        event_line: "EventLineConfig | None" = None,
        max_display_pts: int = 5_000,
    ) -> None:
        super().__init__(
            plots=plots,
            height=height,
            width=width,
            open_browser=open_browser,
            event_line=event_line,
            max_display_pts=max_display_pts,
        )
        self.port = port
        self.host = host
        self._gui: "taipy.gui.Gui | None" = None  # type: ignore[name-defined]
        self._gui_thread: "threading.Thread | None" = None

    # ------------------------------------------------------------------
    # PlotterBase abstract interface
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        """Launch a Taipy GUI server in a daemon thread."""
        try:
            from taipy.gui import Gui  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "TaipyPlotter requires 'taipy-gui>=3.1'. "
                "Install it with:  pip install taipy-gui"
            ) from exc

        page = "<|chart|figure={figure}|>"
        self._gui = Gui(page)

        # Take a snapshot so the GUI thread's initial state is stable.
        fig_copy = copy.deepcopy(self._fig_dict)

        self._gui_thread = threading.Thread(
            target=self._gui.run,
            kwargs=dict(
                host=self.host,
                port=self.port,
                use_reloader=False,
                dark_mode=False,
                debug=False,
                # Pass the initial figure as a page-level variable.
                figure=fig_copy,
            ),
            daemon=True,
        )
        self._gui_thread.start()
        self._wait_for_startup()

        url = f"http://{self.host}:{self.port}"
        if self.open_browser:
            import webbrowser
            webbrowser.open(url)
        print(f"Taipy plot server started — {url}")

    def stop(self, _silent: bool = False) -> None:
        """Shut down the Taipy GUI server."""
        self._stopped = True
        if self._gui is not None:
            try:
                self._gui.stop()
            except Exception:
                pass
            self._gui = None
        self._gui_thread = None
        if not _silent:
            print("Taipy plot server stopped.")

    @property
    def is_running(self) -> bool:
        """``True`` while the Taipy GUI daemon thread is alive."""
        return self._gui_thread is not None and self._gui_thread.is_alive()

    # ------------------------------------------------------------------
    # Push updates via broadcast_callback
    # ------------------------------------------------------------------

    def on_data_changed(self) -> None:
        """Push the current figure to all connected browser clients.

        Called by :class:`PlotterBase` after every data mutation.  We
        deep-copy ``_fig_dict`` here so the GUI thread always reads a
        stable snapshot while the experiment thread continues writing.
        """
        if self._gui is None:
            return
        try:
            from taipy.gui import broadcast_callback  # noqa: PLC0415
        except ImportError:
            return

        fig_snapshot = copy.deepcopy(self._fig_dict)

        def _set_figure(state: object) -> None:
            state.figure = fig_snapshot  # type: ignore[attr-defined]

        broadcast_callback(self._gui, _set_figure)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_for_startup(self, timeout: float = 15.0) -> None:
        """Block until the HTTP server responds or *timeout* seconds elapse."""
        import urllib.request  # noqa: PLC0415

        url = f"http://{self.host}:{self.port}/"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=0.5)
                return
            except Exception:
                time.sleep(0.2)
        print(
            f"Warning: Taipy server may not have started yet on port {self.port}. "
            "Check for errors above."
        )
