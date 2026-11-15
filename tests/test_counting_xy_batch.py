"""Native256 batching and genuine-source padding contracts, no biological claim."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))
from leap_cell_counting.xy_batch import _native_canvas, infer_xy_batch, TILING_VERSION, observed_boundary_labels
from leap_cell_counting.proposals import native_model_windows
from leap_cell_counting.errors import CountingError


class Handle:
    def remove(self): pass

class Net:
    def register_forward_pre_hook(self, fn): self.pre = fn; return Handle()
    def register_forward_hook(self, fn): self.post = fn; return Handle()

class Model:
    def __init__(self): self.net = Net(); self.calls = []; self.dynamics = []
    def forward(self, net, inputs):
        self.calls.append(inputs.copy()); net.pre(None, None)
        n = len(inputs); yy, xx = np.indices((256, 256), dtype=np.float32)
        fields = np.stack([np.broadcast_to(yy, (n, 256, 256)),
                          np.broadcast_to(xx, (n, 256, 256)), inputs[:, 0]], axis=1)
        net.post(None, None, None)
        return fields, np.zeros((n, 256), np.float32)
    def _compute_masks(self, shape, flow, score, **kwargs):
        self.dynamics.append({'shape': shape, 'flow': flow.copy(), 'score': score.copy(), 'kwargs': kwargs})
        return (score > .5).astype(np.uint16)


class XYBatchTests(unittest.TestCase):
    def test_internal_and_original_source_boundary_labels_are_distinct(self):
        labels = np.zeros((256, 256), np.uint32)
        labels[-10:, 50:80] = 2
        internal = observed_boundary_labels(labels, [[0, 5], [100, 356], [100, 356]], (20, 700, 700))
        self.assertEqual(internal[2]['internal_faces'], ['y_high'])
        self.assertEqual(internal[2]['source_faces'], [])
        source = observed_boundary_labels(labels, [[0, 5], [444, 700], [100, 356]], (20, 700, 700))
        self.assertEqual(source[2]['internal_faces'], [])
        self.assertEqual(source[2]['source_faces'], ['y_high'])
    def test_full_native256_has_no_artificial_border_and_normalizes_once(self):
        raw = np.arange(256*256, dtype=np.uint16).reshape(256, 256)
        before = raw.copy(); model = Model()
        with patch('cellpose.core._forward', side_effect=model.forward):
            result = infer_xy_batch(model, [{'image': raw, 'source_shape_yx': [4096, 4096],
                                            'normalization_limits': [0., 256.]}], include_weak=False)[0]
        self.assertEqual(model.calls[0].shape, (1, 1, 256, 256))
        np.testing.assert_array_equal(model.calls[0][0, 0], raw.astype(np.float32)/256.)
        np.testing.assert_array_equal(raw, before)
        self.assertEqual(result['computational_padding_yx'], [[0, 0], [0, 0]])
        self.assertEqual(result['tiling_version'], TILING_VERSION)
        self.assertEqual(len(model.dynamics), 1)

    def test_only_actual_small_sources_pad_and_fields_crop_before_dynamics(self):
        images = [np.arange(17*23, dtype=np.uint16).reshape(17, 23), np.full((255, 249), 137, np.uint16)]
        model = Model()
        with patch('cellpose.core._forward', side_effect=model.forward):
            outputs = infer_xy_batch(model, [{'image': image, 'source_shape_yx': list(image.shape),
                'normalization_limits': [0., 256.]} for image in images])
        self.assertEqual(model.calls[0].shape, (2, 1, 256, 256))
        for index, image in enumerate(images):
            self.assertEqual(outputs[index]['labels'].shape, image.shape)
            np.testing.assert_array_equal(outputs[index]['dapi_cellprob'], image.astype(np.float32)/256.)
            for call in model.dynamics[index*2:index*2+2]:
                self.assertEqual(call['shape'], (1, *image.shape, 1))
                self.assertEqual(call['flow'].shape, (2, 1, *image.shape))
                self.assertFalse(call['kwargs']['do_3D'])
                self.assertNotIn('diameter', call['kwargs'])

    def test_inadequate_large_source_crop_and_oversized_batch_rejected(self):
        raw = np.zeros((240, 240), np.float32)
        with self.assertRaises(CountingError): _native_canvas(raw, [4096, 4096])
        with self.assertRaises(CountingError): _native_canvas(np.zeros((257, 256), np.float32), [4096, 4096])
        with self.assertRaises(CountingError): infer_xy_batch(Model(), [{'image': raw}]*9)

    def test_native_model_context_translates_at_source_edges_and_preserves_z_events(self):
        windows = [{'window_id': 1, 'bounds_zyx': [[3, 6], [0, 40], [900, 950]],
                    'localization_seed_zyx': [4, 15, 930]}]
        result, ledger = native_model_windows(windows, (20, 1024, 1024))
        self.assertEqual(result[0]['bounds_zyx'], [[3, 6], [0, 256], [768, 1024]])
        self.assertEqual(result[0]['source_events'][0]['bounds_zyx'], windows[0]['bounds_zyx'])
        self.assertTrue(ledger['all_source_events_retained'])
        small, _ = native_model_windows([{'window_id': 2, 'bounds_zyx': [[0, 1], [0, 23], [0, 19]],
                                         'localization_seed_zyx': [0, 12, 8]}], (1, 23, 19))
        self.assertEqual(small[0]['bounds_zyx'], [[0, 1], [0, 23], [0, 19]])


if __name__ == '__main__': unittest.main()
