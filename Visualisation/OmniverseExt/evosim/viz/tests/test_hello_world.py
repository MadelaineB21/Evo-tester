import omni.kit.test
import omni.kit.ui_test as ui_test

import evosim.viz.extension as ext_mod


class Test(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._ext = ext_mod.EvoSimVizExtension()

    async def tearDown(self):
        if self._ext:
            self._ext.on_shutdown()
            self._ext = None

    async def test_startup_shutdown_smoke(self):
        self._ext.on_startup("evosim.viz")
        self._ext._on_play_clicked()
        self._ext._on_pause_clicked()
        self._ext.on_shutdown()

    async def test_window_label_exists(self):
        # Assumes the extension is enabled in the running Kit app.
        label = ui_test.find("EvoSimViz//Frame/**/Label[*]")
        self.assertIsNotNone(label)

