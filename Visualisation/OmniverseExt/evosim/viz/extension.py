import omni.ext
import omni.ui as ui


EXTENSION_ID = "evosim.viz"
WINDOW_TITLE = "EvoSimViz"


class EvoSimVizExtension(omni.ext.IExt):
    """Minimal Python UI extension shell for the EvoSim visualization."""

    def on_startup(self, ext_id: str):
        print(f"[{EXTENSION_ID}] Extension startup ({ext_id})")

        self._window = ui.Window(WINDOW_TITLE, width=360, height=260)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label(
                    "Evolution simulation visualization (stub UI)",
                    word_wrap=True,
                )

                with ui.HStack(spacing=4):
                    ui.Button("Play", clicked_fn=self._on_play_clicked)
                    ui.Button("Pause", clicked_fn=self._on_pause_clicked)

                ui.Separator()
                ui.Label(
                    "Wire your NumPy-based sim into this panel\n"
                    "and/or drive a USD scene from here.",
                    word_wrap=True,
                )

    def on_shutdown(self):
        print(f"[{EXTENSION_ID}] Extension shutdown")
        self._window = None

    def _on_play_clicked(self):
        print(f"[{EXTENSION_ID}] Play clicked")

    def _on_pause_clicked(self):
        print(f"[{EXTENSION_ID}] Pause clicked")

