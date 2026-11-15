"""CPU-only display proxies; no segmentation, torch, GPU or research-image access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from leap_cell_counting.imaging import SliceSession, display_plane
from leap_cell_counting.errors import CountingError


class SliceDisplayProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="leap-display-proxy-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        (self.root / "output").mkdir()
        source = self.root / "input/source.stub"
        source.write_bytes(b"display fixture")
        self.planes = (np.arange(3 * 1024 * 2048, dtype=np.uint32).reshape(3, 1024, 2048) % 1024).astype(np.uint16)
        self.planes.setflags(write=False)
        owner = self
        class Reader:
            info = {"z_count": 2, "channel_count": 3}
            def plane(self, channel, z):
                return owner.planes[channel]
        self.session = SliceSession.__new__(SliceSession)
        self.session.source = source
        self.session.reader = Reader()
        self.session._check_source = lambda: None
        self.session.analysis = {"width": 2048, "height": 1024, "dapi_channel": 0, "display_channels": [0, 1, 2],
            "channels": [{"color": color, "display_range": [0, 4095], "slice_display_range": [0, 1023]}
                         for color in ("#2244FF", "#00BA2F", "#C5C5C5")],
            "points": [{"id": "auto-1", "plane_anchors": {"1": [1800, 800]}}]}

    def render(self, edge, *, channels=None):
        path = self.root / "output" / f"frame-{edge}.png"
        request = {"id": 1, "z": 1, "channels": [0, 1, 2] if channels is None else channels,
                   "max_edge": edge, "output": str(path)}
        result = self.session.render(request)
        return result, np.asarray(Image.open(path))

    def test_proxy_pixels_equal_native_pixel_center_samples_and_anchors_stay_native(self):
        native_meta, native = self.render(0)
        for edge in (1024, 1536):
            with self.subTest(edge=edge):
                meta, proxy = self.render(edge)
                rows = ((np.arange(meta["height"]) + .5) * 1024 / meta["height"]).astype(int)
                cols = ((np.arange(meta["width"]) + .5) * 2048 / meta["width"]).astype(int)
                np.testing.assert_array_equal(proxy, native[np.ix_(rows, cols)])
                self.assertEqual(meta["plane_anchors"], {"auto-1": [1800, 800]})
                self.assertEqual((meta["native_width"], meta["native_height"]), (2048, 1024))
                self.assertEqual((meta["width"], meta["height"]), (edge, edge // 2))
                self.assertEqual(meta["sampling"], "nearest_pixel_centers")
                self.assertEqual(meta["quality"], "proxy")
        self.assertEqual(native_meta["quality"], "native")
        self.assertFalse(self.planes.flags.writeable)
        self.assertNotIn("torch", sys.modules)

    def test_native_composition_preserves_saved_slice_range_and_original_arithmetic(self):
        _, native = self.render(0)
        expected = np.zeros((1024, 2048, 3), dtype=np.float32)
        for channel, settings in enumerate(self.session.analysis["channels"]):
            gray, _ = display_plane(self.planes[channel], settings["slice_display_range"])
            color = settings["color"]
            weights = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.
            expected += gray[..., None] * weights
        np.testing.assert_array_equal(native, np.clip(expected, 0, 255).astype(np.uint8))

    def test_historical_missing_range_uses_full_native_window_not_proxy_extrema(self):
        self.planes = self.planes.copy()
        self.planes[:, 1, 1] = 5000  # Center sampling skips this extreme at edge 1536.
        self.planes.setflags(write=False)
        for channel in self.session.analysis["channels"]:
            channel.pop("slice_display_range")
            channel.pop("display_range")
        _, native = self.render(0)
        meta, proxy = self.render(1536)
        rows = ((np.arange(meta["height"]) + .5) * 1024 / meta["height"]).astype(int)
        cols = ((np.arange(meta["width"]) + .5) * 2048 / meta["width"]).astype(int)
        np.testing.assert_array_equal(proxy, native[np.ix_(rows, cols)])

    def test_invalid_proxy_sizes_are_rejected_and_empty_channels_remain_black(self):
        for edge in (-1, 512, 2048, True):
            with self.subTest(edge=edge), self.assertRaises(CountingError):
                self.render(edge)
        meta, pixels = self.render(1024, channels=[])
        self.assertEqual(meta["width"], 1024)
        self.assertFalse(np.any(pixels))


if __name__ == "__main__":
    unittest.main()
