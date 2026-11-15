"""CPU orchestration contracts with injected masks, not biological validation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))
from leap_cell_counting.body_pipeline import process_body_regions, _discover_region, _core_work_coverage
from leap_cell_counting.errors import CountingCancelled, CountingError
from leap_cell_counting.identity import unify_cells, geometric_association
from leap_cell_counting.worker import Events
from leap_cell_counting.cell_pipeline import run_observation_routes, collect_association_evidence, resolve_observations


def run_body_fixture(*args, discover, **kwargs):
    with patch('leap_cell_counting.support_discovery.discover_support_bodies', side_effect=discover):
        return process_body_regions(*args, **kwargs)


def run_routes_fixture(*args, discover, nuclear_route, **kwargs):
    with patch('leap_cell_counting.support_discovery.discover_support_bodies', side_effect=discover), \
         patch('leap_cell_counting.efficient.process_windows', side_effect=nuclear_route):
        return run_observation_routes(*args, **kwargs)


def discover_region_fixture(reader, lock, channel, spacing, region, plan, cancel, discover):
    with patch('leap_cell_counting.support_discovery.discover_support_bodies', side_effect=discover):
        return _discover_region(reader, lock, channel, spacing, region, plan, cancel)


class Reader:
    def __init__(self, raw):
        self.raw = raw
        self.reads = []

    def read_region(self, channel, z, y, x, **kwargs):
        kwargs['check_cancel']()
        self.reads.append((channel, z, y, x))
        return self.raw[z, y, x]


def body(origin, shape=(3, 5, 5)):
    envelope = np.ones(shape, bool)
    mask = envelope.copy()
    mask[:, 2, 2] = False  # A dark cavity is geometry, never positive fluorescence.
    anchors = {str(z+origin[0]): [origin[2]+1, origin[1]+1] for z in range(shape[0])}
    return {'local_label_id': 1, 'origin_zyx': list(origin),
        'bounds_zyx': [[v, v+shape[i]] for i, v in enumerate(origin)],
        'mask_zyx': mask, 'association_envelope_zyx': envelope, 'core_mask_zyx': mask.copy(),
        'plane_anchors': anchors, 'representative_xyz': [origin[2]+1, origin[1]+1, origin[0]+1],
        'decision': 'accepted', 'reasons': ['synthetic_resolved_body'], 'open_faces': [],
        'requires_context_expansion': False, 'metrics': {'fixture': True}}


def proposal(shape, regions):
    return {'source_shape_zyx': shape,
        'regions': [{'region_id': i+1, 'bounds_zyx': box} for i, box in enumerate(regions)],
        'discovery_foreground_zyx': np.ones(shape, bool), 'discovery_grid_xy': 1,
        'background_profile': None, 'intensity_reference': None}


class BodyPipelineContracts(unittest.TestCase):
    def test_an_original_core_never_loses_already_observed_context(self):
        reader = Reader(np.zeros((9, 100, 100), np.uint8))
        plan = proposal(reader.raw.shape, [[[2, 7], [20, 40], [20, 40]]])
        original = body((3, 22, 35)); calls = []
        def discover(raw, spacing, **kwargs):
            origin = kwargs['origin_zyx']
            calls.append([[origin[a], origin[a]+raw.shape[a]] for a in range(3)])
            obj = dict(original)
            if len(calls) == 1:
                obj.update(requires_context_expansion=True,
                    open_faces=[{'axis': 'x', 'side': 'high', 'boundary': 'roi'}])
            elif len(calls) == 2:
                obj = body((3, 18, 36))
                obj.update(requires_context_expansion=True,
                    open_faces=[{'axis': 'y', 'side': 'low', 'boundary': 'roi'}])
            return {'objects': [obj], 'metrics': {}, 'rejected_regions': []}
        result = discover_region_fixture(reader, threading.Lock(), 1, (.2, .2, 1.), plan['regions'][0],
                                   plan, lambda: None, discover)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(a[0] <= b[0] and a[1] >= b[1] for a, b in zip(calls[2], calls[1])))
        self.assertEqual(len(result['result']['objects']), 1)

    def test_separate_truncated_objects_get_separate_local_contexts(self):
        reader = Reader(np.zeros((9, 100, 100), np.uint8))
        plan = proposal(reader.raw.shape, [[[2, 7], [20, 80], [20, 80]]])
        a, b = body((3, 20, 25)), body((3, 70, 75)); b['local_label_id'] = 2
        calls = []
        def discover(raw, spacing, **kwargs):
            origin = kwargs['origin_zyx']; calls.append((origin, raw.shape))
            selected = []
            for source in (a, b):
                if all(origin[i] <= source['bounds_zyx'][i][0] and
                       origin[i]+raw.shape[i] >= source['bounds_zyx'][i][1] for i in range(3)):
                    obj = dict(source)
                    if origin == (2, 20, 20):
                        obj.update(requires_context_expansion=True,
                            open_faces=[{'axis': 'y' if source is a else 'x',
                                         'side': 'low' if source is a else 'high', 'boundary': 'roi'}])
                    selected.append(obj)
            return {'objects': selected, 'metrics': {}, 'rejected_regions': []}
        result = discover_region_fixture(reader, threading.Lock(), 1, (.2, .2, 1.), plan['regions'][0],
                                   plan, lambda: None, discover)
        self.assertEqual(len(result['result']['objects']), 2)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(shape[1] < 40 and shape[2] < 40 for _, shape in calls[1:]))
        self.assertEqual(len(result['attempts'][0]['child_bounds_zyx']), 2)

    def test_new_neighbor_is_linked_to_real_source_work_without_expanding_owner(self):
        reader = Reader(np.zeros((9, 100, 100), np.uint8))
        plan = proposal(reader.raw.shape, [[[2, 7], [20, 40], [20, 40]], [[2, 7], [15, 45], [35, 65]]])
        original, neighbor = body((3, 22, 35)), body((3, 22, 43)); neighbor['local_label_id'] = 2
        calls = []
        def discover(raw, spacing, **kwargs):
            calls.append(kwargs['origin_zyx'])
            obj = dict(original)
            if len(calls) == 1:
                obj.update(requires_context_expansion=True,
                    open_faces=[{'axis': 'x', 'side': 'high', 'boundary': 'roi'}])
                objects = [obj]
            else:
                other = dict(neighbor); other.update(requires_context_expansion=True,
                    open_faces=[{'axis': 'x', 'side': 'high', 'boundary': 'roi'}])
                objects = [obj, other]
            return {'objects': objects, 'metrics': {}, 'rejected_regions': []}
        result = discover_region_fixture(reader, threading.Lock(), 1, (.2, .2, 1.), plan['regions'][0],
                                   plan, lambda: None, discover)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result['result']['objects']), 1)
        deferred = result['result']['metrics']['deferred_neighbor_work'][0]
        self.assertEqual([item['region_id'] for item in deferred['automatic_work_coverage']], [2])
        self.assertGreater(deferred['automatic_work_coverage'][0]['actual_core_voxels_in_region'], 0)
        with self.assertRaises(CountingError): _core_work_coverage(neighbor, plan['regions'][:1])

    def test_work_partition_changes_preserve_all_actual_body_identities(self):
        reader = Reader(np.zeros((8, 60, 60), np.uint8))
        actual = [body((2, 10, 10)), body((2, 40, 40))]
        def discover(raw, spacing, **kwargs):
            origin = kwargs['origin_zyx']; found = []
            for source in actual:
                if all(origin[i] <= source['bounds_zyx'][i][0] and
                       origin[i]+raw.shape[i] >= source['bounds_zyx'][i][1] for i in range(3)):
                    obj = dict(source); obj['local_label_id'] = len(found)+1; found.append(obj)
            return {'objects': found, 'metrics': {}, 'rejected_regions': []}
        comparisons = []
        for boxes in ([[[0, 8], [0, 60], [0, 60]]],
                      [[[0, 8], [0, 45], [0, 45]], [[0, 8], [35, 60], [35, 60]]]):
            plan = proposal(reader.raw.shape, boxes)
            self.assertTrue(all(_core_work_coverage(item, plan['regions']) for item in actual))
            with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder, ThreadPoolExecutor(4) as pool:
                observations, _ = run_body_fixture(reader, plan, Path(folder), 1, (.2, .2, 1.),
                    SimpleNamespace(check_cancel=lambda: None), cpu_pool=pool,
                    read_lock=threading.Lock(), discover=discover)
                labels = np.load(Path(folder)/'body-labels.npy')
                comparisons.append((len(observations), sorted((labels == i).tobytes()
                    for i in np.unique(labels) if i)))
                self.assertTrue(all(item['decision'] == 'accepted' for item in observations))
        self.assertEqual(comparisons[0], comparisons[1])
        self.assertEqual(comparisons[0][0], 2)

    def test_signal_sets_cooperative_cancellation_without_interrupting_future_tracking(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder:
            events = Events({'run_id': 'fixture', 'events_path': str(Path(folder)/'events.jsonl'),
                             'cancel_path': str(Path(folder)/'cancel')})
            events.signal_cancel(None, None)
            self.assertTrue(events.cancelled)
            with self.assertRaises(CountingCancelled): events.check_cancel()

    def test_whole_bright_signal_expands_even_without_any_body_object(self):
        reader = Reader(np.zeros((12, 20, 20), np.uint8))
        plan = proposal(reader.raw.shape, [[[4, 8], [0, 10], [0, 10]]])
        calls = []
        def discover(raw, spacing, **kwargs):
            origin = kwargs['origin_zyx']; calls.append(origin)
            return {'objects': [], 'metrics': {}, 'rejected_regions': [],
                'context_expansion_request': {'required': origin[0] == 4,
                    'reason': 'bright_foreground_fills_observed_roi',
                    'faces': [{'axis': 'z', 'side': 'low', 'boundary': 'roi'}]}}
        result = discover_region_fixture(reader, threading.Lock(), 1, (.1, .1, .3),
            plan['regions'][0], plan, lambda: None, discover)
        self.assertEqual([origin[0] for origin in calls], [4, 2])
        self.assertEqual(result['attempts'][0]['objects'], 0)

    def test_unresolved_original_boundary_cannot_be_published_as_zero_bodies(self):
        reader = Reader(np.zeros((4, 10, 10), np.uint8))
        plan = proposal(reader.raw.shape, [[[0, 4], [0, 10], [0, 10]]])
        def discover(*args, **kwargs):
            return {'objects': [], 'metrics': {'body_identity_count_known': False}, 'rejected_regions': [],
                    'context_expansion_request': {'required': False, 'faces': [],
                                                 'reason': 'bright_foreground_fills_observed_roi'}}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder, ThreadPoolExecutor(4) as pool:
            with self.assertRaises(CountingError) as caught:
                run_body_fixture(reader, plan, Path(folder), 1, (.1, .1, .3),
                    SimpleNamespace(check_cancel=lambda: None), cpu_pool=pool,
                    read_lock=threading.Lock(), discover=discover)
            self.assertEqual(caught.exception.code, 'body_identity_unresolved')
            detail = json.loads((Path(folder)/'body-progress.jsonl').read_text())
            self.assertFalse(detail['metrics']['body_identity_count_known'])

    def test_body_failure_interrupts_nuclear_route_and_joins_before_return(self):
        reader = Reader(np.zeros((10, 20, 20), np.uint8))
        plan = proposal(reader.raw.shape, [[[0, 5], [0, 10], [0, 10]]])
        nuclear_stopped = threading.Event()
        def discover(*args, **kwargs): raise ValueError('injected body failure')
        def nuclei(*args, **kwargs):
            events = args[8]
            try:
                while True:
                    events.check_cancel()
                    time.sleep(.002)
            finally:
                nuclear_stopped.set()
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder:
            with self.assertRaisesRegex(ValueError, 'injected body failure'):
                run_routes_fixture(reader, [], plan, Path(folder), 0, 1, 'custom', (.1, .1, .3),
                    object(), SimpleNamespace(check_cancel=lambda: None), discover=discover, nuclear_route=nuclei)
        self.assertTrue(nuclear_stopped.is_set())

    def test_shared_cpu_routes_do_not_gate_body_on_nuclear_success(self):
        reader = Reader(np.zeros((10, 20, 20), np.uint8))
        plan = proposal(reader.raw.shape, [[[0, 5], [0, 10], [0, 10]]])
        body_entered = threading.Event(); nuclear_entered = threading.Event()
        def discover(raw, spacing, **kwargs):
            body_entered.set()
            self.assertTrue(nuclear_entered.wait(2))
            return {'objects': [body((1, 2, 2))], 'metrics': {}, 'rejected_regions': []}
        def nuclei(*args, **kwargs):
            nuclear_entered.set()
            self.assertTrue(body_entered.wait(2))
            return [], {'nuclei': 0}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder:
            observations, diagnostics = run_routes_fixture(reader, [], plan, Path(folder), 0, 1,
                'custom', (.1, .1, .3), object(), SimpleNamespace(check_cancel=lambda: None),
                discover=discover, nuclear_route=nuclei)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]['decision'], 'accepted')
        self.assertEqual(diagnostics['shared_cpu_workers'], 4)
        self.assertTrue(diagnostics['route_times_overlap_and_must_not_be_added'])

    def test_saved_envelope_evidence_is_directional_and_policy_explicit(self):
        reader = Reader(np.zeros((10, 20, 20), np.uint8))
        plan = proposal(reader.raw.shape, [[[0, 5], [0, 10], [0, 10]]])
        def discover(raw, spacing, **kwargs):
            return {'objects': [body((1, 2, 2))], 'metrics': {}, 'rejected_regions': []}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder, ThreadPoolExecutor(4) as pool:
            output = Path(folder)
            bodies, _ = run_body_fixture(reader, plan, output, 1, (.1, .1, .3),
                SimpleNamespace(check_cancel=lambda: None), cpu_pool=pool,
                read_lock=threading.Lock(), discover=discover)
            labels = np.zeros(reader.raw.shape, np.uint32)
            labels[1:4, 4, 4] = 1
            np.save(output/'labels.npy', labels)
            nucleus = {'id': 'nucleus-1', 'kind': 'nucleus', 'artifact_id': 'dapi_nuclei', 'label_id': 1,
                'bounds_zyx': [[1, 4], [4, 5], [4, 5]], 'representative_xyz': [4, 4, 2],
                'plane_anchors': {str(z): [4, 4] for z in range(1, 4)}, 'decision': 'review', 'metrics': {}}
            observations = [nucleus]+bodies
            evidence = collect_association_evidence(observations, output, (.1, .1, .3), check_cancel=lambda: None)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]['evidence']['nucleus_fraction_in_body_envelope'], 1.)
            self.assertEqual(evidence[0]['evidence']['decision'], 'evidence_only')
            self.assertFalse(evidence[0]['evidence']['cross_kind_iou_used'])
            cells, relations = resolve_observations(observations, evidence, relation_decider=geometric_association)
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0]['source'], 'dapi_support')
            self.assertEqual(cells[0]['status'], 'accepted')

    def test_every_region_processed_and_same_native_body_not_counted_twice(self):
        raw = np.zeros((10, 20, 20), np.uint8)
        reader = Reader(raw)
        plan = proposal(raw.shape, [[[0, 5], [0, 10], [0, 10]]]*7)
        calls, active, maximum = [], [0], [0]
        lock = threading.Lock()
        def discover(pixels, spacing, **kwargs):
            with lock:
                active[0] += 1; maximum[0] = max(maximum[0], active[0])
                calls.append(kwargs)
            time.sleep(.005)
            with lock: active[0] -= 1
            return {'objects': [body((1, 2, 2))], 'metrics': {}, 'rejected_regions': []}
        events = SimpleNamespace(check_cancel=lambda: None)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder:
            with ThreadPoolExecutor(4) as pool:
                observations, diagnostics = run_body_fixture(reader, plan, Path(folder), 1,
                    (.1, .1, .3), events, cpu_pool=pool, read_lock=threading.Lock(), discover=discover)
            self.assertEqual(len(calls), 7)
            self.assertLessEqual(maximum[0], 2)
            self.assertEqual(len(observations), 1)
            self.assertEqual(unify_cells(observations)[0]['status'], 'accepted')
            self.assertFalse(diagnostics['DAPI_model_success_required'])
            labels = np.load(Path(folder)/'body-labels.npy', mmap_mode='r')
            self.assertEqual(int(labels[2, 4, 4]), 0)
            self.assertEqual(int(labels[2, 3, 3]), 1)
            for z, xy in observations[0]['plane_anchors'].items():
                self.assertEqual(int(labels[int(z), int(xy[1]), int(xy[0])]), 1)
            geometry = np.load(Path(folder)/'body-observations/region-1-object-1.npz')
            self.assertTrue(geometry['association_envelope'][1, 2, 2])
            self.assertEqual(len(diagnostics['regions']), 7)

    def test_distinct_z_bodies_keep_distinct_ids_without_dapi(self):
        reader = Reader(np.zeros((10, 20, 20), np.uint8))
        plan = proposal(reader.raw.shape, [[[0, 4], [0, 10], [0, 10]], [[5, 9], [0, 10], [0, 10]]])
        def discover(pixels, spacing, **kwargs):
            z = kwargs['origin_zyx'][0]
            return {'objects': [body((z, 2, 2))], 'metrics': {}, 'rejected_regions': []}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder, ThreadPoolExecutor(4) as pool:
            observations, _ = run_body_fixture(reader, plan, Path(folder), 1, (.1, .1, .3),
                SimpleNamespace(check_cancel=lambda: None), cpu_pool=pool,
                read_lock=threading.Lock(), discover=discover)
        self.assertEqual(len(unify_cells(observations)), 2)
        self.assertTrue(all(o['decision'] == 'accepted' for o in observations))
        self.assertTrue(set(observations[0]['plane_anchors']).isdisjoint(observations[1]['plane_anchors']))

    def test_open_core_expands_observed_source_context_and_retains_attempts(self):
        reader = Reader(np.zeros((12, 20, 20), np.uint8))
        plan = proposal(reader.raw.shape, [[[4, 8], [0, 10], [0, 10]]])
        calls = []
        def discover(raw, spacing, **kwargs):
            calls.append(kwargs['origin_zyx'])
            obj = body((4, 2, 2))
            if kwargs['origin_zyx'][0] == 4:
                obj.update(requires_context_expansion=True,
                           open_faces=[{'axis': 'z', 'side': 'low', 'boundary': 'roi'}])
            return {'objects': [obj], 'metrics': {}, 'rejected_regions': []}
        result = discover_region_fixture(reader, threading.Lock(), 1, (.1, .1, .3),
                                  plan['regions'][0], plan, lambda: None, discover)
        self.assertEqual(calls[0][0], 4)
        self.assertLess(calls[1][0], 4)
        self.assertEqual(len(result['attempts']), 2)

    def test_cancel_joins_all_workers_before_returning(self):
        reader = Reader(np.zeros((10, 20, 20), np.uint8))
        plan = proposal(reader.raw.shape, [[[0, 5], [0, 10], [0, 10]]]*5)
        cancel = threading.Event(); active = [0]; lock = threading.Lock()
        def check():
            if cancel.is_set(): raise CountingCancelled('fixture')
        def discover(raw, spacing, **kwargs):
            with lock: active[0] += 1
            try:
                cancel.set(); time.sleep(.02)
                return {'objects': [], 'metrics': {}, 'rejected_regions': []}
            finally:
                with lock: active[0] -= 1
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder, ThreadPoolExecutor(4) as pool:
            with self.assertRaises(CountingCancelled):
                run_body_fixture(reader, plan, Path(folder), 1, (.1, .1, .3),
                    SimpleNamespace(check_cancel=check), cpu_pool=pool,
                    read_lock=threading.Lock(), discover=discover)
            self.assertEqual(active[0], 0)

    def test_event_sequence_and_lines_are_atomic_between_producers(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d') as folder:
            path = Path(folder)
            events = Events({'run_id': 'fixture', 'events_path': str(path/'events.jsonl'),
                             'cancel_path': str(path/'cancel')})
            with ThreadPoolExecutor(4) as pool:
                list(pool.map(lambda n: events.emit('progress', 'count_inference', 'fixture', value=n), range(80)))
            values = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
            self.assertEqual([v['sequence'] for v in values], list(range(1, 81)))
            self.assertEqual(sorted(v['value'] for v in values), list(range(80)))


if __name__ == '__main__': unittest.main()
