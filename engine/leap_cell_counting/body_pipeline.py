"""Bounded Support-body I/O and persistence; detection stays a pure CPU function."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import json
import math
import time
import numpy as np

from .errors import CountingError
from .label_store import NativeLabelStore

MAX_PENDING_BODY_JOBS = 2
MAX_BODY_ROI_VOXELS = 8 * 1024 * 1024


def _bounds(value, shape):
    if (not isinstance(value, (list, tuple)) or len(value) != 3 or
            any(len(pair) != 2 or any(type(v) not in (int, np.int32, np.int64) for v in pair)
                for pair in value)):
        raise ValueError('Body work requires native integer ZYX half-open bounds')
    result = [[int(a), int(b)] for a, b in value]
    if any(a < 0 or a >= b or b > shape[i] for i, (a, b) in enumerate(result)):
        raise ValueError('Body work bounds must fit the original source')
    if math.prod(b-a for a, b in result) > MAX_BODY_ROI_VOXELS:
        raise CountingError('body_discovery_memory_budget',
                            '胞体局部区域超过有界容量；保留完整任务，不发布截断结果。')
    return result


def _core_intersection(owner, item):
    origin, mask = owner['origin_zyx'], owner['core_mask_zyx']
    other_origin, envelope = item['origin_zyx'], item['association_envelope_zyx']
    lo = [max(origin[a], other_origin[a]) for a in range(3)]
    hi = [min(origin[a]+mask.shape[a], other_origin[a]+envelope.shape[a]) for a in range(3)]
    if any(a >= b for a, b in zip(lo, hi)):
        return False
    left = tuple(slice(lo[a]-origin[a], hi[a]-origin[a]) for a in range(3))
    right = tuple(slice(lo[a]-other_origin[a], hi[a]-other_origin[a]) for a in range(3))
    return bool(np.any(mask[left] & envelope[right]))


def _core_work_coverage(item, regions):
    origin, core = item['origin_zyx'], item['core_mask_zyx']
    covered = []
    for region in regions:
        lo = [max(origin[a], region['bounds_zyx'][a][0]) for a in range(3)]
        hi = [min(origin[a]+core.shape[a], region['bounds_zyx'][a][1]) for a in range(3)]
        if any(a >= b for a, b in zip(lo, hi)):
            continue
        local = tuple(slice(lo[a]-origin[a], hi[a]-origin[a]) for a in range(3))
        count = int(core[local].sum())
        if count:
            covered.append({'region_id': region['region_id'], 'actual_core_voxels_in_region': count})
    if not covered:
        raise CountingError('body_proposal_coverage_error',
                            '新进入上下文的胞体 core 没有任何原始自动任务覆盖；不能静默丢弃。')
    return covered


def _object_context_bounds(item, shape, spacing, parameters):
    # Four Gaussian sigmas are an operator context, never a body-size prior.
    sigma = parameters.get('smoothing_sigma_um', .2)
    sampling = (spacing[2], spacing[1], spacing[0])
    margin = [0 if shape[a] == 1 else max(1, int(math.ceil(4*sigma/sampling[a]))) for a in range(3)]
    original = item['bounds_zyx']
    bounds = [[max(0, lo-margin[a]), min(shape[a], hi+margin[a])]
              for a, (lo, hi) in enumerate(original)]
    for face in item.get('open_faces', []):
        if face['boundary'] != 'roi':
            continue
        axis = 'zyx'.index(face['axis'])
        step = max(margin[axis], 1, (original[axis][1]-original[axis][0])//2)
        if face['side'] == 'low': bounds[axis][0] = max(0, bounds[axis][0]-step)
        elif face['side'] == 'high': bounds[axis][1] = min(shape[axis], bounds[axis][1]+step)
        else: raise ValueError('Unknown body boundary side')
    return _bounds(bounds, shape)


def _discover_region(reader, read_lock, channel, spacing, region, proposal, check_cancel):
    from .support_discovery import discover_support_bodies
    shape = tuple(proposal['source_shape_zyx'])
    initial = _bounds(region['bounds_zyx'], shape)
    queue = deque([{'bounds': initial, 'owner': None, 'fallback': None, 'lineage': 0}])
    attempts, retained, rejected, calls, deferred = [], [], [], [], []
    observed = {}; queued = {0: queue[0]}; owners = {}; reused_owners = []
    next_lineage = 1; parameters = {}; terminal_unknown = False
    started = time.perf_counter()
    def keep(item, reason=None):
        value = dict(item)
        if reason:
            value['decision'] = 'review'
            value['reasons'] = list(item.get('reasons', []))+[reason]
        retained.append(value)
    def enqueue(bounds, owner, fallback, lineage):
        if lineage in observed:
            bounds = [[min(a[0], b[0]), max(a[1], b[1])] for a, b in zip(bounds, observed[lineage])]
        if lineage in queued:
            pending = queued[lineage]
            pending['bounds'] = _bounds([[min(a[0], b[0]), max(a[1], b[1])]
                                        for a, b in zip(bounds, pending['bounds'])], shape)
        else:
            item = {'bounds': _bounds(bounds, shape), 'owner': owner, 'fallback': fallback, 'lineage': lineage}
            queued[lineage] = item; queue.append(item)
    try:
        while queue:
            check_cancel()
            work = queue.popleft(); bounds = work['bounds']; owner = work['owner']
            queued.pop(work['lineage'])
            previous = observed.get(work['lineage'])
            if previous is not None and all(a[0] >= b[0] and a[1] <= b[1] for a, b in zip(bounds, previous)):
                # The same original core already has these actual pixels; its
                # result or larger queued context owns the evidence.
                continue
            observed[work['lineage']] = deepcopy(bounds)
            with read_lock:
                raw = reader.read_region(channel, *(slice(*pair) for pair in bounds),
                                         copy=False, check_cancel=check_cancel)
            step = time.perf_counter()
            result = discover_support_bodies(raw, spacing, origin_zyx=tuple(pair[0] for pair in bounds),
                source_shape_zyx=shape, discovery_foreground_zyx=proposal['discovery_foreground_zyx'],
                discovery_grid_xy=proposal['discovery_grid_xy'], background_profile=proposal['background_profile'],
                intensity_reference=proposal['intensity_reference'])
            check_cancel()
            parameters = result['metrics'].get('parameters', parameters)
            ids = [item['local_label_id'] for item in result['objects']]
            if len(set(ids)) != len(ids) or any(type(value) is not int or value < 1 for value in ids):
                raise ValueError('Discovery object IDs must be positive and unique in each observed ROI')
            for item in result['objects']:
                if any(a < bounds[i][0] or b > bounds[i][1] or a >= b
                       for i, (a, b) in enumerate(item['bounds_zyx'])):
                    raise ValueError('Discovery returned body pixels outside the actually observed ROI')
            signal = result.get('context_expansion_request', {})
            attempt = {'bounds_zyx': bounds, 'objects': len(result['objects']), 'lineage': work['lineage'],
                       'signal_context_expansion_request': signal, 'child_bounds_zyx': []}
            attempts.append(attempt)
            calls.append({'bounds_zyx': bounds, 'seconds': time.perf_counter()-step, 'metrics': result['metrics']})
            rejected.extend(result['rejected_regions'])
            if signal.get('required'):
                expanded = deepcopy(bounds)
                for face in signal['faces']:
                    if face['boundary'] != 'roi': continue
                    a = 'zyx'.index(face['axis']); amount = max(1, (bounds[a][1]-bounds[a][0])//2)
                    if face['side'] == 'low': expanded[a][0] = max(0, bounds[a][0]-amount)
                    else: expanded[a][1] = min(shape[a], bounds[a][1]+amount)
                attempt['expanded_bounds_zyx'] = expanded
                enqueue(expanded, owner, work['fallback'], work['lineage'])
                continue
            if (result['metrics'].get('body_identity_count_known') is False and
                    signal.get('reason') == 'bright_foreground_fills_observed_roi' and
                    bounds == [[0, int(n)] for n in shape]):
                terminal_unknown = True
            owned = 0
            for item in result['objects']:
                belongs = owner is None or _core_intersection(owner, item)
                if belongs: owned += 1
                if not item.get('requires_context_expansion'):
                    # Closed neighbors are retained now; later source work uses
                    # actual-mask deduplication, never a new count per crop.
                    keep(item)
                    continue
                if not belongs:
                    coverage = _core_work_coverage(item, proposal['regions'])
                    existing = [key for key, original in owners.items() if _core_intersection(original, item)]
                    if existing:
                        focused = _object_context_bounds(item, shape, spacing, parameters)
                        for key in existing:
                            enqueue(focused, owners[key], owners[key], key)
                        reused_owners.append({'from_lineage': work['lineage'], 'existing_lineages': existing,
                            'bounds_zyx': focused, 'actual_core_envelope_overlap_checked': True,
                            'automatic_work_coverage': coverage, 'body_identities_merged': False})
                        continue
                    other_work = [r for r in coverage if r['region_id'] != region['region_id']]
                    if other_work:
                        deferred.append({'origin_zyx': item['origin_zyx'], 'bounds_zyx': item['bounds_zyx'],
                            'source_region_id': region['region_id'], 'automatic_work_coverage': other_work,
                            'actual_core_overlap_checked': True})
                        continue
                    # If this region is its only automatic owner, schedule it
                    # independently instead of discarding or growing old owners.
                    belongs = True
                    branch_owner = item
                    lineage = next_lineage; next_lineage += 1
                else:
                    branch_owner = owner if owner is not None else item
                    lineage = work['lineage'] if owner is not None else next_lineage
                    if owner is None: next_lineage += 1
                focused = _object_context_bounds(item, shape, spacing, parameters)
                attempt['child_bounds_zyx'].append(focused)
                owners.setdefault(lineage, branch_owner)
                enqueue(focused, branch_owner, branch_owner, lineage)
            if owner is not None and not owned:
                keep(work['fallback'], 'original_body_core_not_resolved_by_expanded_context')
        for i, item in enumerate(retained, 1):
            item['local_label_id'] = i
        metrics = {'method': 'object_scoped_native_support_context', 'parameters': parameters,
            'discovery_calls': calls, 'deferred_neighbor_work': deferred,
            'all_contexts_bounded': True, 'neighbors_do_not_expand_original_owner': True,
            'each_original_core_context_grows_monotonically': True,
            'computation_owner_count': len(owners), 'reused_owner_contexts': reused_owners,
            'body_identity_count_known': not terminal_unknown}
        output = {'objects': retained, 'rejected_regions': rejected, 'metrics': metrics}
        if terminal_unknown:
            output['context_expansion_request'] = {'required': False, 'faces': [],
                                                  'reason': 'bright_foreground_fills_observed_roi'}
        return {'region_id': region['region_id'], 'result': output, 'attempts': attempts,
                'seconds': time.perf_counter()-started}
    except CountingError as error:
        error.body_region_diagnostics = {'region_id': region['region_id'], 'attempts': attempts,
                                         'error_code': error.code, 'completed': False}
        raise


def _persist_objects(store, result, output):
    """Persist geometry separately; only measured positive voxels enter body labels."""
    region_id = result['region_id']
    for item in result['result']['objects']:
        mask = np.asarray(item['mask_zyx'])
        envelope = np.asarray(item['association_envelope_zyx'])
        if (mask.ndim != 3 or mask.dtype.kind != 'b' or envelope.dtype.kind != 'b' or
                envelope.shape != mask.shape or not mask.any() or np.any(mask & ~envelope)):
            raise ValueError('Body observed mask and enclosing geometry must be matching native boolean arrays')
        origin = item['origin_zyx']
        expected = [[int(origin[a]), int(origin[a])+mask.shape[a]] for a in range(3)]
        if expected != item['bounds_zyx']:
            raise ValueError('Body array origin, shape and original bounds disagree')
        _bounds(expected, store.labels.shape)
        local_id = int(item['local_label_id'])
        relative = f'body-observations/region-{region_id}-object-{local_id}.npz'
        path = output/relative
        path.parent.mkdir(exist_ok=True)
        arrays = {'association_envelope': envelope}
        if 'core_mask_zyx' in item:
            core = np.asarray(item['core_mask_zyx'])
            if core.shape != mask.shape or core.dtype.kind != 'b' or np.any(core & ~envelope):
                raise ValueError('Body core must lie within its association envelope')
            arrays['core'] = core
        np.savez_compressed(path, **arrays)
        path.chmod(0o444)
        metadata = {key: deepcopy(value) for key, value in item.items()
                    if key not in ('mask_zyx', 'association_envelope_zyx', 'core_mask_zyx')}
        metadata.update(label_id=local_id, geometry_path=relative,
                        geometry_artifact_id='support_bodies',
                        geometry_path_relative_to='label_artifact_directory',
                        flags=list(item.get('reasons', [])), open_boundary_faces=item.get('open_faces', []))
        store.add(mask.astype(np.uint32)*local_id, [metadata], origin, region_id)


def _observation(store, record):
    """Recompute anchors only inside one actual stored body, never the full stack."""
    from .segmentation import interior_anchor
    bounds = record['bounds_zyx']
    origin = [pair[0] for pair in bounds]
    mask = store.labels[tuple(slice(*pair) for pair in bounds)] == record['label_id']
    proposed = {}
    for item in record['local_objects']:
        for z, anchor in item['plane_anchors'].items():
            proposed.setdefault(z, []).append(anchor)
    anchors, representative, largest = {}, None, -1
    for z in np.flatnonzero(mask.any(axis=(1, 2))):
        source_z = int(z+origin[0])
        anchor = None
        for xy in proposed.get(str(source_z), []):
            x, y = int(xy[0])-origin[2], int(xy[1])-origin[1]
            if 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2] and mask[z, y, x]:
                anchor = list(xy)
                break
        if anchor is None:
            x, y = interior_anchor(mask[z])
            anchor = [x+origin[2], y+origin[1]]
        anchors[str(source_z)] = anchor
        size = int(mask[z].sum())
        if size > largest:
            representative, largest = [*anchor, source_z], size
    decisions = {item['decision'] for item in record['local_objects']}
    decision = ('review' if record['flags'] or 'review' in decisions else
                'accepted' if 'accepted' in decisions else 'excluded')
    reasons = sorted(set(record['flags'] + [reason for item in record['local_objects']
                                            for reason in item.get('reasons', [])]))
    return {'id': f"support-body-{record['label_id']}", 'kind': 'body', 'artifact_id': 'support_bodies',
        'label_id': record['label_id'], 'bounds_zyx': deepcopy(bounds), 'plane_anchors': anchors,
        'representative_xyz': representative, 'decision': decision, 'reasons': reasons,
        'metrics': {'source_regions': list(record['source_windows']),
                    'local_observations': deepcopy(record['local_objects']),
                    'cellpose_required': False, 'DAPI_required_for_body_discovery': False}}


def process_body_regions(reader, proposal, output, support_channel, spacing, events, *,
                         cpu_pool, read_lock):
    """Consume every retained region with at most two queued/running body jobs."""
    output = Path(output)
    regions = proposal['regions']
    ids = [r['region_id'] for r in regions]
    if len(set(ids)) != len(ids) or any(type(value) is not int or value < 1 for value in ids):
        raise ValueError('Body regions require unique positive integer IDs')
    for region in regions:
        _bounds(region['bounds_zyx'], proposal['source_shape_zyx'])
    store = NativeLabelStore(output/'body-labels.npy', tuple(proposal['source_shape_zyx']))
    pending = deque()
    started = time.perf_counter()
    diagnostics = {'regions': [], 'no_candidate_or_elapsed_time_truncation': True,
                   'maximum_pending_body_jobs': MAX_PENDING_BODY_JOBS,
                   'maximum_native_roi_voxels': MAX_BODY_ROI_VOXELS,
                   'cellpose_used': False, 'DAPI_model_success_required': False}
    def consume(future):
        try:
            item = future.result()
        except CountingError as error:
            detail = getattr(error, 'body_region_diagnostics', None)
            if detail is not None:
                with (output/'body-progress.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(detail, allow_nan=False)+'\n')
            raise
        events.check_cancel()
        _persist_objects(store, item, output)
        detail = {'region_id': item['region_id'], 'attempts': item['attempts'], 'seconds': item['seconds'],
                  'metrics': item['result']['metrics'], 'objects': len(item['result']['objects']),
                  'rejected_regions': item['result']['rejected_regions']}
        diagnostics['regions'].append(detail)
        with (output/'body-progress.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(detail, allow_nan=False)+'\n')
        signal_request = item['result'].get('context_expansion_request', {})
        if (item['result']['metrics'].get('body_identity_count_known') is False and
                signal_request.get('reason') == 'bright_foreground_fills_observed_roi' and
                not signal_request.get('required') and
                item['attempts'][-1]['bounds_zyx'] == [[0, int(n)] for n in proposal['source_shape_zyx']]):
            raise CountingError('body_identity_unresolved',
                'Support 区域到原图边界仍缺少可观察的体部分界，无法确定细胞身份；未发布不完整计数，区域证据已保留。')
    try:
        for region in regions:
            events.check_cancel()
            while len(pending) >= MAX_PENDING_BODY_JOBS:
                consume(pending[0])
                pending.popleft()
            pending.append(cpu_pool.submit(_discover_region, reader, read_lock, support_channel, spacing,
                                           region, proposal, events.check_cancel))
        while pending:
            consume(pending[0])
            pending.popleft()
        observations = [_observation(store, record) for record in store.objects.values()]
        diagnostics['label_store'] = store.finish()
        diagnostics['seconds'] = time.perf_counter()-started
        return observations, diagnostics
    finally:
        # A shared executor outlives this coordinator. Every submitted task must
        # finish before the shared TIFF reader and temporary source views close.
        for future in pending:
            future.cancel()
        for future in pending:
            if not future.cancelled():
                try:
                    future.result()
                except Exception:
                    pass  # Preserve the original failure after joining workers.
        store.close()
