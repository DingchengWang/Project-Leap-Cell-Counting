"""Candidate accounting and native identity storage invariants; no biological validation."""
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))
from leap_cell_counting.label_store import NativeLabelStore
from leap_cell_counting.proposals import locate_windows
from leap_cell_counting.efficient import _prepare_source_and_model
from leap_cell_counting.errors import CountingError,CountingCancelled


def objects(label):
    return [{'label_id': label, 'flags': [], 'open_boundary_faces': []}]


class EfficientContracts(unittest.TestCase):
    def test_model_load_once_overlaps_scan_and_hands_object_to_main_thread(self):
        entered=threading.Event();scanning=threading.Event();calls=[];model=object()
        def loader(check):
            calls.append(threading.get_ident());entered.set()
            self.assertTrue(scanning.wait(2));check();return model,{}
        def scan(*args,**kwargs):
            self.assertTrue(entered.wait(2));scanning.set()
            return {'timings':{'total_seconds':0.}, 'shape_zyx':(1,10,10)}
        events=SimpleNamespace(check_cancel=lambda:None,emit=lambda *a,**k:None)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder, \
             patch('leap_cell_counting.source_scan.scan_source',side_effect=scan), \
             patch('leap_cell_counting.proposals.locate_windows',return_value=([],{})), \
             patch('leap_cell_counting.segmentation.load_model',side_effect=loader):
            result=_prepare_source_and_model(None,Path(folder),0,1,[],events)
        self.assertIs(result[3],model);self.assertEqual(len(calls),1)
        self.assertNotEqual(calls[0],threading.get_ident())
        self.assertTrue(result[4]['loaded_concurrently_with_source_statistics'])
        self.assertIn('model_wait_seconds',result[5])

    def test_scan_failure_or_cancel_stops_and_joins_model_loader(self):
        for error in (CountingError('fixture_scan_failure','fixture'),CountingCancelled('fixture')):
            entered=threading.Event();stopped=threading.Event()
            def loader(check):
                entered.set()
                try:
                    while True:check();time.sleep(.005)
                finally:stopped.set()
            def scan(*args,**kwargs):
                self.assertTrue(entered.wait(2));raise error
            events=SimpleNamespace(check_cancel=lambda:None,emit=lambda *a,**k:None)
            with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder, \
                 patch('leap_cell_counting.source_scan.scan_source',side_effect=scan), \
                 patch('leap_cell_counting.segmentation.load_model',side_effect=loader):
                with self.assertRaises(type(error)):
                    _prepare_source_and_model(None,Path(folder),0,1,[],events)
            self.assertTrue(stopped.is_set())

    def test_same_native_object_reused_and_distinct_z_identity_retained(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder:
            store = NativeLabelStore(Path(folder)/'labels.npy', (12, 30, 30))
            first = np.zeros((5, 12, 12), np.uint32)
            first[1:4, 3:8, 2:9] = 7
            store.add(first, objects(7), (0, 0, 0), 1)
            store.add(first, objects(7), (0, 0, 0), 2)
            store.add(first, objects(7), (7, 0, 0), 3)
            self.assertEqual(len(store.objects), 2)
            self.assertEqual(store.objects[1]['source_windows'], [1, 2])
            self.assertNotEqual(int(store.labels[2, 4, 4]), int(store.labels[9, 4, 4]))
            self.assertEqual(int(np.count_nonzero(store.labels)), 2*3*5*7)
            self.assertFalse(store.finish()['whole_volume_remapping_or_unique_scan'])
            store.close()

    def test_touching_original_masks_do_not_merge_by_proximity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder:
            store = NativeLabelStore(Path(folder)/'labels.npy', (5, 10, 10))
            local = np.zeros((5, 10, 10), np.uint32)
            local[1:4, 2:8, 2:5] = 1
            local[1:4, 2:8, 5:8] = 2
            store.add(local, objects(1)+objects(2), (0, 0, 0), 1)
            self.assertEqual(len(store.objects), 2)
            self.assertEqual(len(np.unique(store.labels)), 3)
            store.close()

    def test_one_mask_over_two_established_ids_is_not_a_dominant_identity_merge(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder:
            store=NativeLabelStore(Path(folder)/'labels.npy',(3,10,12))
            original=np.zeros((3,10,12),np.uint32)
            original[1,2:8,1:7]=1;original[1,2:8,7:11]=2
            store.add(original,objects(1)+objects(2),(0,0,0),1)
            combined=(original>0).astype(np.uint32)*9
            store.add(combined,objects(9),(0,0,0),2)
            self.assertEqual(len(store.objects),2)
            np.testing.assert_array_equal(store.labels,original)
            for record in store.objects.values():
                self.assertEqual(record['source_windows'],[1])
                self.assertIn('overlap_identity_ambiguous',record['flags'])
            self.assertEqual(len(store.audit),1)
            store.close()

    def test_locator_preserves_all_computation_foreground_and_native_xy_budget(self):
        sums = np.zeros((17, 48, 48), np.uint16)
        sums[0:5, 1:6, 1:6] = 10000
        sums[10:17, 39:45, 39:45] = 15000
        scan = {'moments': {16: {'dapi': {'sum': sums}}},
                'counts': {16: np.full((48, 48), 256, np.uint16)},
                'shape_zyx': (17, 768, 768)}
        windows, ledger = locate_windows(scan)
        self.assertGreater(len(windows), 0)
        self.assertTrue(ledger['all_windows_retained'])
        self.assertFalse(ledger['summary_is_model_or_segmentation_input'])
        self.assertTrue(any(w['bounds_zyx'][0][0] == 0 for w in windows))
        self.assertTrue(any(w['bounds_zyx'][0][1] == 17 for w in windows))
        for window in windows:
            for lo, hi in window['bounds_zyx'][1:]:
                self.assertLessEqual(hi-lo, 240)
            self.assertFalse(window['assigns_nucleus_identity'])


if __name__ == '__main__':
    unittest.main()
