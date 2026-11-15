"""Independent observation routes and native association evidence orchestration."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import numpy as np

from .body_pipeline import process_body_regions, MAX_BODY_ROI_VOXELS
from .errors import CountingError
from .identity import enclosure_evidence, nucleus_geometry, observation_from_nucleus, unify_cells

BODY_GEOMETRY_CACHE_BYTES = 128 * 1024 * 1024


class _RouteEvents:
    """Forward cancellation or the first route failure to both bounded branches."""
    def __init__(self, events):
        self.events = events
        self.failure = None
        self.lock = threading.Lock()

    def check_cancel(self):
        self.events.check_cancel()
        with self.lock:
            error = self.failure
        if error is not None:
            raise error

    def fail(self, error):
        with self.lock:
            if self.failure is None:
                self.failure = error

    def emit(self, *args, **kwargs):
        return self.events.emit(*args, **kwargs)


def run_observation_routes(reader, windows, body_proposal, output, dapi, support, marker, spacing,
                           model, events):
    """A single model thread shares a four-worker CPU pool with body discovery.

    CPU tasks acquire source ROI views only when executing. The coordinator
    queues at most four nuclear jobs and two body jobs, and both routes finish
    or cancel before their shared TIFF reader is released. A separate Support
    worker count applies only after nuclear model inference and recovery finish.
    """
    from .efficient import CPU_WORKERS, process_windows
    events = _RouteEvents(events)
    read_lock = threading.Lock()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CPU_WORKERS, thread_name_prefix='native-observations') as cpu_pool:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='body-coordinator') as coordinator:
            def bodies():
                try:
                    return process_body_regions(reader, body_proposal, output, support, spacing, events,
                        cpu_pool=cpu_pool, read_lock=read_lock)
                except BaseException as error:
                    events.fail(error)
                    raise
            body_future = coordinator.submit(bodies)
            try:
                nuclei, nuclear_diagnostics = process_windows(
                    reader, windows, output, dapi, support, marker, spacing, model, events,
                    cpu_pool=cpu_pool, read_lock=read_lock)
                body_observations, body_diagnostics = body_future.result()
            except BaseException as error:
                events.fail(error)
                # Coordinator's context joins its source reads and CPU work.
                raise
    observations = [observation_from_nucleus(point) for point in nuclei] + body_observations
    return observations, {'nuclear_route': nuclear_diagnostics, 'body_route': body_diagnostics,
        'shared_cpu_workers': CPU_WORKERS, 'model_instances': 1,
        'parallel_routes_wall_seconds': time.perf_counter()-started,
        'route_times_overlap_and_must_not_be_added': True}


def _intersects(a, b):
    return all(max(x[0], y[0]) < min(x[1], y[1]) for x, y in zip(a, b))


def _body_geometry(observation, body_artifact_path, labels):
    pieces = observation['metrics']['local_observations']
    bounds = [[min(item['bounds_zyx'][a][0] for item in pieces),
               max(item['bounds_zyx'][a][1] for item in pieces)] for a in range(3)]
    if np.prod([b-a for a, b in bounds], dtype=np.int64) > MAX_BODY_ROI_VOXELS:
        raise CountingError('body_association_memory_budget',
                            '同一胞体的局部关联包络超出容量；不发布未经完整关联的计数。')
    envelope = np.zeros(tuple(b-a for a, b in bounds), bool)
    for item in pieces:
        if (item['geometry_artifact_id'] != observation['artifact_id'] or
                item['geometry_path_relative_to'] != 'label_artifact_directory'):
            raise ValueError('Body geometry must be scoped to its registered label artifact directory')
        local = tuple(slice(item['bounds_zyx'][a][0]-bounds[a][0],
                            item['bounds_zyx'][a][1]-bounds[a][0]) for a in range(3))
        with np.load(Path(body_artifact_path).parent/item['geometry_path'], allow_pickle=False) as arrays:
            value = arrays['association_envelope']
            if value.shape != envelope[local].shape or value.dtype.kind != 'b':
                raise ValueError('Saved body association geometry disagrees with its original-coordinate bounds')
            envelope[local] |= value
    actual_bounds = observation['bounds_zyx']
    actual = labels[tuple(slice(*pair) for pair in actual_bounds)] == observation['label_id']
    return actual, [b[0] for b in actual_bounds], envelope, [b[0] for b in bounds]


def collect_association_evidence(observations, output, spacing, *, check_cancel):
    """Extract actual overlap/enclosure evidence; make no scientific match choice.

    Only object-local arrays are examined. A nucleus core is computed once for
    all candidate bodies. Body arrays are reused read-only within a 128 MiB
    cache, and no global label unique/remapping pass is needed.
    A bounding-box contact selects pairs to measure; it is never a match.
    """
    output = Path(output)
    nuclei = [o for o in observations if o['kind'] == 'nucleus']
    bodies = [o for o in observations if o['kind'] == 'body']
    nuclear_labels = np.load(output/'labels.npy', mmap_mode='r', allow_pickle=False)
    body_labels = np.load(output/'body-labels.npy', mmap_mode='r', allow_pickle=False)
    pairs = []
    body_geometry_cache = OrderedDict()
    cached_bytes = 0
    try:
        for nucleus in nuclei:
            check_cancel()
            # Include geometric cavity extent, which can differ from the
            # positive-fluorescence body bounds without inventing a nuclear mask.
            candidate_bodies = [body for body in bodies if any(
                _intersects(nucleus['bounds_zyx'], piece['bounds_zyx'])
                for piece in body['metrics']['local_observations'])]
            if not candidate_bodies:
                continue
            bounds = nucleus['bounds_zyx']
            mask = nuclear_labels[tuple(slice(*pair) for pair in bounds)] == nucleus['label_id']
            geometry = nucleus_geometry(mask, [pair[0] for pair in bounds], spacing,
                                        nucleus['representative_xyz'])
            for body in candidate_bodies:
                check_cancel()
                # Use the observation object itself as the cache identity so
                # reuse does not introduce any matching or deduplication rule.
                key = id(body)
                cached = body_geometry_cache.pop(key, None)
                if cached is None:
                    body_geometry = _body_geometry(body, output/'body-labels.npy', body_labels)
                    size = body_geometry[0].nbytes + body_geometry[2].nbytes
                    body_geometry[0].flags.writeable = False
                    body_geometry[2].flags.writeable = False
                    if size <= BODY_GEOMETRY_CACHE_BYTES:
                        while cached_bytes + size > BODY_GEOMETRY_CACHE_BYTES:
                            _, (_, evicted_bytes) = body_geometry_cache.popitem(last=False)
                            cached_bytes -= evicted_bytes
                        body_geometry_cache[key] = (body_geometry, size)
                        cached_bytes += size
                else:
                    body_geometry, size = cached
                    body_geometry_cache[key] = cached
                actual, origin, envelope, envelope_origin = body_geometry
                evidence = enclosure_evidence(geometry, actual, origin, envelope, envelope_origin)
                pairs.append({'nucleus_id': nucleus['id'], 'body_id': body['id'], 'evidence': evidence})
    finally:
        body_geometry_cache.clear()
        nuclear_labels._mmap.close()
        body_labels._mmap.close()
    return pairs


def resolve_observations(observations, measured_pairs, *, relation_decider):
    """Require an explicit scientific policy; never infer match from box overlap."""
    relations = [{**pair, **relation_decider(pair['evidence'])} for pair in measured_pairs]
    return unify_cells(observations, relations), relations


def label_artifacts(shape):
    return {key: {'path': filename, 'kind': kind, 'axes': 'ZYX',
                  'origin_zyx': [0, 0, 0], 'shape_zyx': list(shape)}
            for key, filename, kind in (('dapi_nuclei', 'labels.npy', 'nucleus'),
                                        ('support_bodies', 'body-labels.npy', 'body'))}
