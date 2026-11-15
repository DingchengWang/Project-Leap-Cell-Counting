"""Exact-result Support scheduling and bounded association geometry reuse."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import tempfile
import threading
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))
from leap_cell_counting import cell_pipeline, efficient
from leap_cell_counting.errors import CountingCancelled, CountingError


class SupportSchedulingTests(unittest.TestCase):
    def store(self, count, close=lambda: None):
        return SimpleNamespace(labels=np.zeros((1, 2, 2), np.uint32),
            objects={i: {'index': i} for i in range(count)}, finish=lambda: {}, close=close)

    def process(self, store, directory, pool, *, workers=2, check=lambda: None):
        reader = SimpleNamespace(info={'z_count': 1}, series=SimpleNamespace(shape=(1, 2, 2)))
        events = SimpleNamespace(check_cancel=check, emit=lambda *a, **k: None)
        with patch('leap_cell_counting.label_store.NativeLabelStore', return_value=store), \
             (patch.object(efficient, 'SUPPORT_WORKERS', workers) if workers is not None else nullcontext()):
            return efficient.process_windows(reader, [], Path(directory), 0, 1, 'custom',
                (.2, .2, None), object(), events, cpu_pool=pool)

    def test_completed_jobs_refill_before_slow_first_and_results_keep_record_order(self):
        first_started = threading.Event()
        third_started = threading.Event()
        lock = threading.Lock()
        active, maximum, completed = 0, 0, []

        def point(_reader, _lock, _store, record, *_args):
            nonlocal active, maximum
            index = record['index']
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                if index == 0:
                    first_started.set()
                    self.assertTrue(third_started.wait(2), 'An idle slot must not wait for record zero')
                elif index == 1:
                    self.assertTrue(first_started.wait(2))
                elif index == 2:
                    third_started.set()
                with lock:
                    completed.append(index)
                return index
            finally:
                with lock:
                    active -= 1

        with ThreadPoolExecutor(4) as pool, patch.object(efficient, '_point', side_effect=point):
            result = efficient._support_points(None, lock, self.store(20), 1, 'custom',
                (.2, .2, None), lambda: None, workers=2, cpu_pool=pool)
        self.assertEqual(result, list(range(20)))
        self.assertLess(completed.index(1), completed.index(0))
        self.assertLessEqual(maximum, 2)

    def test_error_and_cancel_join_running_jobs_before_label_store_closes(self):
        for error in (CountingError('fixture', 'fixture'), CountingCancelled('fixture')):
            with self.subTest(error=type(error).__name__):
                sibling_started = threading.Event()
                release_sibling = threading.Event()
                sibling_stopped = threading.Event()
                closed = threading.Event()

                def point(_reader, _lock, _store, record, *_args):
                    if record['index'] == 0:
                        self.assertTrue(sibling_started.wait(2))
                        threading.Timer(.05, release_sibling.set).start()
                        raise error
                    sibling_started.set()
                    try:
                        self.assertTrue(release_sibling.wait(2))
                    finally:
                        sibling_stopped.set()

                def close():
                    self.assertTrue(sibling_stopped.is_set())
                    closed.set()

                with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(4) as pool, \
                        patch.object(efficient, '_point', side_effect=point):
                    with self.assertRaises(type(error)) as raised:
                        self.process(self.store(2, close), directory, pool, workers=4)
                    self.assertIs(raised.exception, error)
                    self.assertTrue(closed.is_set())

    def test_cancellation_while_waiting_stops_submission_and_joins_workers(self):
        both_started = threading.Event()
        release = threading.Event()
        lock = threading.Lock()
        started, stopped = [], []
        cancelled = CountingCancelled('fixture')

        def point(_reader, _lock, _store, record, *_args):
            with lock:
                started.append(record['index'])
                if len(started) == 2:
                    both_started.set()
            self.assertTrue(release.wait(2))
            with lock:
                stopped.append(record['index'])

        def check():
            if both_started.is_set():
                threading.Timer(.05, release.set).start()
                raise cancelled

        def close():
            self.assertCountEqual(stopped, [0, 1])

        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(4) as pool, \
                patch.object(efficient, '_point', side_effect=point):
            with self.assertRaises(CountingCancelled) as raised:
                self.process(self.store(8, close), directory, pool, check=check)
        self.assertIs(raised.exception, cancelled)
        self.assertCountEqual(started, [0, 1])

    def test_default_support_stage_uses_available_cpus_without_changing_recovery(self):
        workers = os.cpu_count() or 1
        barrier = threading.Barrier(workers)

        def point(_reader, _lock, _store, record, *_args):
            barrier.wait(2)
            return record['index']

        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(4) as pool, \
                patch.object(efficient, '_point', side_effect=point):
            points, diagnostics = self.process(self.store(workers), directory, pool, workers=None)
        self.assertEqual(points, list(range(workers)))
        self.assertEqual(efficient.SUPPORT_WORKERS, workers)
        self.assertEqual(efficient.CPU_WORKERS, 4)
        self.assertEqual(diagnostics['maximum_pending_recovery_jobs'], 4)
        self.assertEqual(diagnostics['maximum_pending_support_jobs'], workers)


class BodyGeometryCacheTests(unittest.TestCase):
    def fixture(self, output):
        shape = (3, 8, 9)
        nuclei = np.zeros(shape, np.uint32)
        nuclei[1, 3, 3] = 1
        nuclei[1, 4, 4] = 2
        bodies = np.zeros(shape, np.uint32)
        bodies[:, 1:7, 1:4] = 1
        bodies[:, 1:7, 4:8] = 2
        np.save(output/'labels.npy', nuclei)
        np.save(output/'body-labels.npy', bodies)
        observations = []
        for label, y, x in ((1, 3, 3), (2, 4, 4)):
            observations.append({'id': f'nucleus-{label}', 'kind': 'nucleus',
                'label_id': label, 'bounds_zyx': [[1, 2], [y, y+1], [x, x+1]],
                'representative_xyz': [x, y, 1]})
        for label, x_bounds in ((1, [1, 4]), (2, [4, 8])):
            path = f'body-{label}.npz'
            np.savez_compressed(output/path, association_envelope=np.ones(shape, bool))
            observations.append({'id': f'body-{label}', 'kind': 'body', 'label_id': label,
                'artifact_id': 'support_bodies', 'bounds_zyx': [[0, 3], [1, 7], x_bounds],
                'metrics': {'local_observations': [{'bounds_zyx': [[0, 3], [0, 8], [0, 9]],
                    'geometry_artifact_id': 'support_bodies',
                    'geometry_path_relative_to': 'label_artifact_directory', 'geometry_path': path}]}})
        return observations

    def test_cached_evidence_is_exact_and_pair_order_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            observations = self.fixture(output)
            with patch.object(cell_pipeline, 'BODY_GEOMETRY_CACHE_BYTES', 0):
                original = cell_pipeline.collect_association_evidence(observations, output,
                    (.2, .2, .7), check_cancel=lambda: None)
            enclosure = cell_pipeline.enclosure_evidence

            def readonly_enclosure(nucleus, actual, origin, envelope, envelope_origin):
                self.assertFalse(actual.flags.writeable)
                self.assertFalse(envelope.flags.writeable)
                return enclosure(nucleus, actual, origin, envelope, envelope_origin)

            with patch.object(cell_pipeline, '_body_geometry', wraps=cell_pipeline._body_geometry) as load, \
                    patch.object(cell_pipeline, 'enclosure_evidence', side_effect=readonly_enclosure):
                cached = cell_pipeline.collect_association_evidence(observations, output,
                    (.2, .2, .7), check_cancel=lambda: None)
            self.assertEqual(cached, original)
            self.assertEqual(load.call_count, 2)
            self.assertEqual([(p['nucleus_id'], p['body_id']) for p in cached],
                [('nucleus-1', 'body-1'), ('nucleus-1', 'body-2'),
                 ('nucleus-2', 'body-1'), ('nucleus-2', 'body-2')])

    def test_byte_budget_evicts_arrays_and_oversize_geometry_stays_uncached(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            observations = self.fixture(output)
            for budget in (300, 1):
                with self.subTest(budget=budget), \
                        patch.object(cell_pipeline, 'BODY_GEOMETRY_CACHE_BYTES', budget), \
                        patch.object(cell_pipeline, '_body_geometry', wraps=cell_pipeline._body_geometry) as load:
                    evidence = cell_pipeline.collect_association_evidence(observations, output,
                        (.2, .2, .7), check_cancel=lambda: None)
                self.assertEqual(len(evidence), 4)
                self.assertEqual(load.call_count, 4)


if __name__ == '__main__':
    unittest.main()
