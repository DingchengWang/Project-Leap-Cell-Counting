"""All-Z localization, native local model evidence, native 3D identity and Support."""
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
import math
import json
import os
import threading
import time
import numpy as np
from .errors import CountingError

CPU_WORKERS = 4
SUPPORT_WORKERS = os.cpu_count() or 1
MAX_RECOVERY_VOXELS = 64*1024*1024


def _prepare_source_and_model(reader, output, dapi, support, selected, events):
    """One background model load overlaps the independent native source pass."""
    from .segmentation import load_model
    from .source_scan import scan_source
    from .proposals import locate_windows, native_model_windows
    from .worker import atomic_json
    aborted = threading.Event()
    started = time.perf_counter()
    def load_once():
        def check():
            events.check_cancel()
            if aborted.is_set():
                from .errors import CountingCancelled
                raise CountingCancelled('图像准备已结束，停止模型加载。')
        check()
        step = time.perf_counter()
        model, information = load_model(check)
        check()
        return model, information, time.perf_counter()-step
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='counting-model-load') as pool:
        future = pool.submit(load_once)
        def check_preparation():
            events.check_cancel()
            if future.done() and future.exception() is not None:
                future.result()
        try:
            scan = scan_source(reader, output/'statistics', dapi, support, selected,
                               check_cancel=check_preparation,
                               progress=lambda n, total: events.emit('progress', 'count_projection',
                                   '读取原生统计；模型同时加载', current=n, total=total))
            step = time.perf_counter()
            computation_windows, ledger = locate_windows(scan, check_cancel=check_preparation)
            windows, model_ledger = native_model_windows(computation_windows, scan['shape_zyx'])
            ledger['model_work'] = model_ledger
            localization_seconds = time.perf_counter()-step
            atomic_json(output/'work-ledger.json', {'windows': windows, **ledger})
            events.emit('stage', 'count_model', '等待共享模型准备完成')
            step = time.perf_counter()
            model, model_info, load_seconds = future.result()
            wait_seconds = time.perf_counter()-step
        except BaseException:
            aborted.set()
            future.cancel()
            raise
    model_info['loaded_concurrently_with_source_statistics'] = True
    timings = {'source_scan_seconds': scan['timings']['total_seconds'],
               'localization_seconds': localization_seconds, 'model_load_seconds': load_seconds,
               'model_wait_seconds': wait_seconds,
               'source_and_model_preparation_seconds': time.perf_counter()-started}
    return scan, windows, ledger, model, model_info, timings


def analyse(request, events):
    from .imaging import PlaneReader, channel_colors, display_plane, save_png, selection
    from .worker import atomic_json
    from .support_discovery import propose_support_regions
    from .body_pipeline import MAX_BODY_ROI_VOXELS
    from .cell_pipeline import (run_observation_routes, collect_association_evidence,
                                resolve_observations, label_artifacts)
    from .identity import geometric_association
    started = time.perf_counter()
    source = Path(request['source_path'])
    before = source.stat()
    output = Path(request['run_dir'])/'counting'
    timings = {}
    with PlaneReader(source, request.get('series_index'), request.get('time_index')) as reader:
        info = reader.info
        count = info['channel_count']
        dapi = request.get('dapi_channel')
        if dapi is None or dapi == -1:
            raise CountingError('dapi_selection_required', '请确认 DAPI 通道后开始计数。')
        selection(dapi, count, 'dapi_channel')
        roles = request.get('channel_roles')
        if not isinstance(roles, list) or len(roles) != count or any(not isinstance(r, str) or not r.strip() for r in roles):
            raise CountingError('invalid_channel_roles', '每个源通道必须有一个显示标记名称。')
        if roles[dapi] != 'DAPI' or roles.count('DAPI') != 1:
            raise CountingError('inconsistent_dapi_channel', 'DAPI 索引必须与唯一的 DAPI 通道角色一致。')
        support = request.get('support_channel')
        if support is None:
            raise CountingError('support_selection_required', '请选择与 DAPI 不同的 Support Channel。')
        selection(support, count, 'support_channel')
        if support == dapi:
            raise CountingError('support_equals_dapi', 'Support Channel 必须与 DAPI 不同。')
        marker = request.get('support_marker')
        if not isinstance(marker, str) or not marker.strip():
            raise CountingError('support_marker_required', '请指定 Support 标记。')
        selected = request.get('display_channels', list(range(count)))
        if (not isinstance(selected, list) or any(type(c) is not int for c in selected)
                or len(set(selected)) != len(selected)):
            raise CountingError('invalid_display_channels', '显示通道必须为无重复的源索引列表。')
        for channel in selected:
            selection(channel, count, 'display_channel')
        colors = channel_colors(roles, request.get('channel_colors'))
        spacing = tuple(info['physical_size_um'][axis] for axis in 'XYZ')
        axes = spacing[:2] if info['z_count'] == 1 else spacing
        if any(v is None or not np.isfinite(v) or v <= 0 for v in axes):
            raise CountingError('physical_calibration_required', '原生体积需要实际 XYZ 标定；真实单层仅要求 XY。')
        dtype = np.dtype(reader.series.dtype)
        if dtype.kind != 'u' or dtype.itemsize not in (1, 2):
            raise CountingError('unsupported_statistics_input', '原生统计需要 uint8/uint16 源数据。')
        output.mkdir()
        events.emit('stage', 'count_projection', '一次读取全部原始 Z 层并生成定位统计')
        scan, windows, ledger, model, model_info, preparation = _prepare_source_and_model(
            reader, output, dapi, support, selected, events)
        timings.update(preparation)
        model_info.update(input='native XY event-local raw-DAPI maximum',
            tiling_version='native256_direct_v1',
            input_mode='single_plane_2d' if info['z_count'] == 1 else 'local_xy_evidence_native_3d_recovery',
            normalization='explicit per-feature P1/P99; actual limits in work diagnostics',
            calibration={'spacing_xyz_um': list(spacing), 'xy_resampling': False, 'z_resampling': False},
            parameters={**model_info.get('parameters', {}), 'do_3D': False, 'z_axis': None,
                        'channel_axis': 3, 'normalize': False, 'compute_masks': False})
        step = time.perf_counter()
        body_proposal = propose_support_regions(scan['moments'][16]['support'], scan['counts'][16],
            source_shape_zyx=scan['shape_zyx'], spacing_xyz=spacing, max_work_voxels=MAX_BODY_ROI_VOXELS)
        body_proposal['source_shape_zyx'] = list(scan['shape_zyx'])
        np.save(output/'body-discovery-foreground.npy', body_proposal['discovery_foreground_zyx'])
        body_ledger = {key: value for key, value in body_proposal.items()
                       if key not in ('discovery_foreground_zyx', 'intensity_reference')}
        reference = body_proposal['intensity_reference']
        if reference is not None:
            np.save(output/'body-reference-domains.npy', reference['domain_labels_zyx'])
            np.save(output/'body-reference-peaks.npy', reference['peak_excess_by_id'])
            body_ledger['intensity_reference'] = {key: value for key, value in reference.items()
                if key not in ('domain_labels_zyx', 'peak_excess_by_id')}
            body_ledger['intensity_reference'].update(domain_labels_path='body-reference-domains.npy',
                                                     peak_excess_path='body-reference-peaks.npy')
        atomic_json(output/'body-work-ledger.json', body_ledger)
        timings['body_localization_seconds'] = time.perf_counter()-step
        events.emit('stage', 'count_inference', '识别 DAPI 核与独立原生胞体，并恢复逐层证据',
                    model=json.loads(json.dumps(model_info)))
        observations, route_diagnostics = run_observation_routes(reader, windows, body_proposal, output,
            dapi, support, marker, spacing, model, events)
        del model
        diagnostics = route_diagnostics['nuclear_route']
        atomic_json(output/'work-ledger.json', {'windows': windows, **ledger,
            'executed_model_windows': len(diagnostics['windows']),
            'boundary_context_checks': diagnostics['boundary_context_checks']})
        diagnostics['body_route'] = route_diagnostics['body_route']
        diagnostics['body_localization'] = body_proposal['metrics']
        timings['observation_routes_wall_seconds'] = route_diagnostics['parallel_routes_wall_seconds']
        timings['nuclear_route_seconds'] = (diagnostics['local_inference_and_recovery_seconds'] +
                                            diagnostics['support_seconds'])
        timings['body_route_seconds'] = diagnostics['body_route']['seconds']
        timings['route_times_overlap_and_must_not_be_added'] = True
        step = time.perf_counter()
        measured_pairs = collect_association_evidence(observations, output, spacing,
                                                      check_cancel=events.check_cancel)
        points, relations = resolve_observations(observations, measured_pairs,
                                                 relation_decider=geometric_association)
        atomic_json(output/'observations.json', {'observations': observations, 'relations': relations,
            'association_policy': 'representative_and_entire_maximal_core_enclosure_with_actual_body_z',
            'scientific_validation': 'representative_real_mask_validation_pending'})
        timings['identity_association_seconds'] = time.perf_counter()-step
        step = time.perf_counter()
        height, width = scan['shape_zyx'][1:]
        channels = []
        composite = np.zeros((height, width, 3), np.float32)
        dapi_range = _source_range(scan['histograms']['dapi'])
        for channel in range(count):
            path, limits = None, [0., float(np.iinfo(reader.series.dtype).max)]
            if channel in scan['projections']:
                pixels, limits = display_plane(scan['projections'][channel])
                path = f'channels/C{channel+1}.png'
                save_png(output/path, pixels)
                rgb = np.array([int(colors[channel][j:j+2], 16) for j in (1, 3, 5)], np.float32)/255.
                composite += pixels[..., None]*rgb
            channels.append({'index': channel, 'name': info['channel_hints'][channel], 'role': roles[channel],
                'color': colors[channel], 'path': path, 'display_range': limits,
                'slice_display_range': dapi_range if channel == dapi else limits})
        save_png(output/'composite.png', np.clip(composite, 0, 255).astype(np.uint8))
        timings['display_artifacts_seconds'] = time.perf_counter()-step
        counts = {state: sum(p['status'] == state for p in points) for state in ('accepted', 'review', 'excluded')}
        diagnostics['source_scan'] = {k: scan[k] for k in ('timings', 'all_z_participated', 'source_channels_read', 'whole_raw_volume_copied')}
        diagnostics['localization'] = ledger
        atomic_json(output/'diagnostics.json', diagnostics)
        timings['total_seconds'] = time.perf_counter()-started
        analysis = {'schema_version': 2, 'run_id': request['run_id'], 'source_path': str(source),
            'source_signature': {'size_bytes': before.st_size, 'mtime_ns': before.st_mtime_ns},
            'source_identity': scan['source_identity'],
            'series_index': reader.series_index, 'time_index': reader.time_index, 'dapi_channel': dapi,
            'support_channel': support, 'support_marker': marker, 'display_channels': selected,
            'width': width, 'height': height, 'z_count': info['z_count'], 'source_dtype': info['dtype'],
            'physical_size_um': info['physical_size_um'], 'calibration_sources': info['calibration_sources'],
            'channel_roles': roles, 'channels': channels, 'preview_path': 'composite.png',
            'labels_path': 'labels.npy', 'labels_axes': 'ZYX', 'diagnostics_path': 'diagnostics.json',
            'auto_count': counts['accepted'], 'pending_count': counts['review'], 'excluded_count': counts['excluded'],
            'identity_model': 'multi_observation_cell_v1',
            'label_artifacts': label_artifacts(scan['shape_zyx']),
            'observations_path': 'observations.json', 'points': points,
            'default_label_size': 24, 'model': model_info, 'timings': timings,
            'normalization': {'method': 'event_local_feature_P1_P99', 'source_display_limits': dapi_range,
                              'actual_feature_limits_path': 'diagnostics.json'},
            'scientific_validation': {'status': 'development_identity_acceptance_pending',
                'all_z_maximum_display_only': True, 'local_z_projection_is_model_feature_not_3d_label': True,
                'native_boundary_uncertainty_recorded_per_object': True,
                'independent_body_discovery_full_source_validation': 'pending',
                'automatic_association_full_source_validation': 'pending'}}
        events.check_cancel()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise CountingError('source_changed', '分析过程中原图已改变，结果未发布。')
        atomic_json(output/'analysis.json', analysis)
    return {'status': 'review_ready', 'analysis_path': str(output/'analysis.json'),
            'auto_count': counts['accepted'], 'pending_count': counts['review'], 'excluded_count': counts['excluded'],
            'message': '原生候选与逐层证据已就绪，请复核并确认。'}


def _source_range(histogram):
    from .normalization import histogram_percentiles
    return histogram_percentiles(histogram)


def _read(reader, lock, channel, bounds, check_cancel):
    with lock:
        return reader.read_region(channel, *(slice(*b) for b in bounds), copy=False,
                                  check_cancel=check_cancel)


def _recover(reader, lock, dapi, spacing, window, xy_labels, check_cancel):
    from .native_recovery import recover_native_3d
    started = time.perf_counter()
    shape = (reader.info['z_count'], *reader.series.shape[-2:])
    initial = window['bounds_zyx']
    # Context is the recorded density operator's support, not a nuclear radius.
    margin = [0 if shape[0] == 1 else int(math.ceil(4*.568/spacing[2])),
              int(math.ceil(4*.568/spacing[1])), int(math.ceil(4*.568/spacing[0]))]
    bounds = [[max(0, pair[0]-margin[a]), min(shape[a], pair[1]+margin[a])]
              for a, pair in enumerate(initial)]
    attempts = []
    while True:
        check_cancel()
        if math.prod(b-a for a, b in bounds) > MAX_RECOVERY_VOXELS:
            raise CountingError('native_recovery_memory_budget',
                                '原生连通区域超出当前有界恢复容量；任务保留诊断且不发布不完整计数。')
        raw = _read(reader, lock, dapi, bounds, check_cancel)
        cp = np.zeros(raw.shape[1:], np.uint32)
        y, x = initial[1][0]-bounds[1][0], initial[2][0]-bounds[2][0]
        cp[y:y+xy_labels.shape[0], x:x+xy_labels.shape[1]] = xy_labels
        result = recover_native_3d(raw, spacing, cp, origin_zyx=tuple(b[0] for b in bounds),
                                  config={'cp_source_z_bounds_local': [initial[0][0]-bounds[0][0],
                                                                      initial[0][1]-bounds[0][0]]})
        for obj in result['objects']:
            obj['source_window_id'] = window['window_id']
            obj['model_xy_boundary'] = window.get('xy_boundary_labels', {}).get(obj['cp_xy_label_id'],
                {'internal_faces': [], 'source_faces': []})
            obj['context_expansion_of_window'] = window.get('context_expansion_of_window')
            obj['context_expansion_parent_windows'] = window.get('context_expansion_parent_windows', [])
            if obj.get('cp_xy_label_id') in window.get('weak_xy_ids', []):
                obj['flags'].append('permissive_dapi_candidate_requires_review')
        faces = {face for obj in result['objects'] for face in obj['open_boundary_faces']}
        if shape[0] > 1 and any(not p.get('endpoint_evidence_available', False)
                               for p in result.get('axial_profiles', [])):
            faces.update(('z_low', 'z_high'))
        expanded = [list(b) for b in bounds]
        for axis, name in enumerate('zyx'):
            step = max(margin[axis], (bounds[axis][1]-bounds[axis][0])//2)
            if name+'_low' in faces:
                expanded[axis][0] = max(0, bounds[axis][0]-step)
            if name+'_high' in faces:
                expanded[axis][1] = min(shape[axis], bounds[axis][1]+step)
        attempts.append({'bounds_zyx': bounds, 'open_faces': sorted(faces)})
        if expanded == bounds:
            result['origin_zyx'] = [b[0] for b in bounds]
            result['window_id'] = window['window_id']
            result['window'] = window
            result['xy_labels'] = xy_labels
            result['endpoint_expansion'] = attempts
            result['seconds'] = time.perf_counter()-started
            return result
        bounds = expanded


def _point(reader, lock, store, record, support_channel, marker, spacing, check_cancel):
    from .support import measure_support, required_padding_um
    from .segmentation import interior_anchor
    from .volume import _external_distance_um, local_dapi_evidence
    check_cancel()
    shape = store.labels.shape
    padding = [0 if shape[0] == 1 else math.ceil(required_padding_um()/spacing[2])+1,
               math.ceil(required_padding_um()/spacing[1])+1, math.ceil(required_padding_um()/spacing[0])+1]
    bounds = [[max(0, a-padding[i]), min(shape[i], b+padding[i])]
              for i, (a, b) in enumerate(record['bounds_zyx'])]
    region = tuple(slice(*b) for b in bounds)
    mask = np.asarray(store.labels[region]) == record['label_id']
    raw = _read(reader, lock, reader._counting_dapi, bounds, check_cancel)
    support_pixels = _read(reader, lock, support_channel, bounds, check_cancel)
    distance = _external_distance_um(mask, spacing)
    evidence = measure_support(mask, support_pixels, spacing, marker, _distance_zyx=distance)
    evidence['local_dapi_evidence'] = local_dapi_evidence(mask, raw, spacing, _distance_zyx=distance)
    evidence['roi_bounds_zyx'] = bounds
    anchors = {}
    representative = None
    largest = -1
    inherited = {}
    for obj in record['local_objects']:
        for z, anchor in obj.get('plane_anchors', {}).items():
            inherited.setdefault(z, []).append(anchor)
    for z in np.flatnonzero(mask.any(axis=(1, 2))):
        source_z = int(z+bounds[0][0])
        anchor = None
        for proposed in inherited.get(str(source_z), []):
            py, px = int(proposed[1])-bounds[1][0], int(proposed[0])-bounds[2][0]
            if 0 <= py < mask.shape[1] and 0 <= px < mask.shape[2] and mask[z, py, px]:
                anchor = list(proposed)
                break
        if anchor is None:
            anchor = interior_anchor(mask[z])
            anchor[0] += bounds[2][0]
            anchor[1] += bounds[1][0]
        anchors[str(source_z)] = anchor
        size = int(mask[z].sum())
        if size > largest:
            representative = [*anchor, source_z]
            largest = size
    identity_flags = sorted(set(record['flags'] + [flag for obj in record['local_objects'] for flag in obj['flags']]))
    unresolved_model_boundary = any(obj.get('model_xy_boundary', {}).get('internal_faces') and
        obj.get('source_window_id') not in record.get('resolved_model_boundary_windows', [])
        for obj in record['local_objects'])
    if unresolved_model_boundary:
        identity_flags.append('model_xy_internal_boundary_truncated')
    identity_uncertain = bool(record['flags']) or any(
        obj.get('requires_identity_review') or 'permissive_dapi_candidate_requires_review' in obj['flags'] or
        (obj.get('model_xy_boundary', {}).get('internal_faces') and
         obj.get('source_window_id') not in record.get('resolved_model_boundary_windows', []))
        for obj in record['local_objects'])
    search_boundary_limited = any(obj.get('xy_search_envelope_boundary_touched', False)
                                  for obj in record['local_objects'])
    status = evidence['decision']
    # Keep the accepted v4 automatic-count policy. A search-boundary limitation
    # remains Unrecognized even though it is recorded separately from identity.
    if status != 'excluded' and (identity_uncertain or search_boundary_limited):
        status = 'review'
    point = {'id': f"auto-{record['label_id']}", 'label_id': record['label_id'],
            'x': representative[0], 'y': representative[1], 'z': representative[2],
            'source': 'auto', 'status': status, 'plane_anchors': anchors, 'support': evidence,
            'native_identity': {'source_windows': record['source_windows'],
                                'resolved_model_boundary_windows': record.get('resolved_model_boundary_windows', []),
                                'bounds_zyx': record['bounds_zyx'], 'flags': identity_flags,
                                'local_objects': record['local_objects']}}
    if (status == 'review' and evidence['decision'] == 'accepted' and search_boundary_limited and
            not identity_uncertain and set(identity_flags) == {'xy_search_envelope_boundary_touched'}):
        point['review_hint'] = 'likely_astrocyte'
    return point


def _support_points(reader, lock, store, support, marker, spacing, check_cancel, *,
                    workers, cpu_pool=None):
    """Refill completed independent measurements, then restore record order."""
    records = list(store.objects.values())
    points = [None] * len(records)
    pending = {}
    next_record = 0
    with (nullcontext(cpu_pool) if cpu_pool is not None else
          ThreadPoolExecutor(max_workers=workers, thread_name_prefix='native-support')) as pool:
        try:
            while next_record < len(records) or pending:
                check_cancel()
                while next_record < len(records) and len(pending) < workers:
                    check_cancel()
                    future = pool.submit(_point, reader, lock, store, records[next_record],
                                         support, marker, spacing, check_cancel)
                    pending[future] = next_record
                    next_record += 1
                completed, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=pending.__getitem__):
                    points[pending[future]] = future.result()
                    del pending[future]
        finally:
            # Even an externally owned executor must release all references to
            # the reader and label mmap before this helper can return or raise.
            for future in pending:
                future.cancel()
            for future in pending:
                if not future.cancelled():
                    try:
                        future.result()
                    except BaseException:
                        pass
    return points


def process_windows(reader, windows, output, dapi, support, marker, spacing, model, events, *,
                    cpu_pool=None, read_lock=None):
    from .xy_batch import (infer_xy_batch, permissive_from_fields, infer_boundary_context,
                           observed_boundary_labels)
    from .label_store import NativeLabelStore
    support_workers = SUPPORT_WORKERS
    store = NativeLabelStore(output/'labels.npy', (reader.info['z_count'], *reader.series.shape[-2:]))
    reader._counting_dapi = dapi
    lock = read_lock if read_lock is not None else threading.Lock()
    pending = deque()
    diagnostics = {'windows': [], 'unmatched_native_families': [], 'endpoint_expansions': [],
                   'additional_axial_work': [],
                   'maximum_pending_recovery_jobs': CPU_WORKERS,
                   'maximum_pending_support_jobs': support_workers,
                   'model_input': 'native XY maximum over each source-local Z trigger',
                   'model_normalization': 'explicit per-feature P1/P99; normalize=False inside model',
                   'no_candidate_or_elapsed_time_truncation': True}
    started = time.perf_counter()
    source_shape = tuple(store.labels.shape)
    next_window_id = max((w['window_id'] for w in windows), default=0)+1
    scheduled_contexts = {}
    diagnostics['boundary_context_checks'] = []
    def consume(future):
        result = future.result()
        store.add(result['labels'], result['objects'], result['origin_zyx'], result['window_id'])
        diagnostics['unmatched_native_families'].extend(
            {'window_id': result['window_id'], **obj} for obj in result['unmatched_families'])
        diagnostics['endpoint_expansions'].append({'window_id': result['window_id'],
            'attempts': result['endpoint_expansion'], 'seconds': result['seconds'], 'metrics': result['metrics'],
            'axial_profiles': result.get('axial_profiles', [])})
        original_window = result.get('window')
        if original_window is not None:
            for profile in result.get('axial_profiles', []):
                cp_yx = np.argwhere(result['xy_labels'] == profile['cp_xy_id'])
                if not len(cp_yx):
                    continue
                cy, cx = map(int, cp_yx[len(cp_yx)//2])
                cy += original_window['bounds_zyx'][1][0]
                cx += original_window['bounds_zyx'][2][0]
                for event in profile.get('other_axial_events_for_investigation', []):
                    lo, hi = [int(v+result['origin_zyx'][0]) for v in event['bounds_z_local_half_open']]
                    seed_z = (lo+hi-1)//2
                    covering = [w['window_id'] for w in windows
                        if w['bounds_zyx'][0][0] <= seed_z < w['bounds_zyx'][0][1]
                        and w['bounds_zyx'][1][0] <= cy < w['bounds_zyx'][1][1]
                        and w['bounds_zyx'][2][0] <= cx < w['bounds_zyx'][2][1]]
                    if not covering:
                        # An axial tail-growth threshold does not establish a
                        # new spatial candidate. All qualified full-source
                        # concentration regions already have retained windows.
                        diagnostics['additional_axial_work'].append({
                            'source_window_id': original_window['window_id'],
                            'event_source_z_bounds': [lo, hi], 'scheduled_window_ids': [],
                            'decision': 'weak_axial_fluctuation_without_independent_spatial_candidate',
                            'creates_unprocessed_candidate': False})
                        continue
                    diagnostics['additional_axial_work'].append({'source_window_id': original_window['window_id'],
                        'event_source_z_bounds': [lo, hi], 'scheduled_window_ids': covering,
                        'decision': 'associated_with_retained_full_source_spatial_candidate'})
        with (output/'native-recovery-progress.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(diagnostics['endpoint_expansions'][-1], allow_nan=False)+'\n')
    try:
        with (nullcontext(cpu_pool) if cpu_pool is not None else
              ThreadPoolExecutor(max_workers=CPU_WORKERS, thread_name_prefix='native-recovery')) as pool:
            offset = 0
            while offset < len(windows) or pending:
                if offset >= len(windows):
                    consume(pending[0])
                    pending.popleft()
                    continue
                if windows[offset].get('context_expansion_of_window'):
                    batch = windows[offset:offset+1]
                else:
                    end = offset
                    while end < min(offset+8, len(windows)) and not windows[end].get('context_expansion_of_window'):
                        end += 1
                    batch = windows[offset:end]
                inputs = []
                for window in batch:
                    events.check_cancel()
                    raw = _read(reader, lock, dapi, window['bounds_zyx'], events.check_cancel)
                    feature = np.max(raw, axis=0)
                    low, high = np.percentile(feature, [1., 99.])
                    if high-low <= 1e-3:
                        raise CountingError('degenerate_local_feature',
                            '局部 DAPI 特征的第 1/99 百分位相同；已保留区域账本，不能静默跳过。')
                    inputs.append({'image': feature, 'normalization_limits': [float(low), float(high)],
                                   'source_shape_yx': list(reader.series.shape[-2:]),
                                   'window_id': window['window_id'], 'bounds_zyx': window['bounds_zyx'],
                                   'context_expansion_of_window': window.get('context_expansion_of_window'),
                                   'feature': 'event_local_native_z_max'})
                step = time.perf_counter()
                if batch[0].get('context_expansion_of_window'):
                    predictions = [infer_boundary_context(model, inputs[0], check_cancel=events.check_cancel)]
                else:
                    predictions = infer_xy_batch(model, inputs, check_cancel=events.check_cancel, include_weak=False)
                batch_seconds = time.perf_counter()-step
                if len(predictions) != len(batch):
                    raise CountingError('incomplete_xy_batch', '模型返回的局部区域数不完整。')
                for window, item, prediction in zip(batch, inputs, predictions):
                    local_labels = np.asarray(prediction['labels'], np.uint32).copy()
                    seeds = [event['localization_seed_zyx'] for event in window.get('source_events', [window])]
                    covered = True
                    for seed in seeds:
                        sy, sx = seed[1]-window['bounds_zyx'][1][0], seed[2]-window['bounds_zyx'][2][0]
                        covered = covered and bool(local_labels[max(0, sy-8):sy+8, max(0, sx-8):sx+8].any())
                    weak_ids = []
                    weak_started = time.perf_counter()
                    if not covered:
                        weak = permissive_from_fields(model, prediction)
                        if weak is not None:
                            next_id = int(local_labels.max())+1
                            for weak_id in np.unique(weak):
                                if not weak_id:
                                    continue
                                candidate = weak == weak_id
                                overlaps = local_labels[candidate]
                                if np.count_nonzero(overlaps)/int(candidate.sum()) >= .5:
                                    continue
                                candidate &= local_labels == 0
                                if candidate.any():
                                    local_labels[candidate] = next_id
                                    weak_ids.append(next_id)
                                    next_id += 1
                    weak_seconds = time.perf_counter()-weak_started if not covered else 0.
                    boundaries = observed_boundary_labels(local_labels, window['bounds_zyx'], source_shape)
                    internal_ids = [value for value, evidence in boundaries.items() if evidence['internal_faces']]
                    if internal_ids and not window.get('context_expansion_of_window'):
                        expanded = [list(window['bounds_zyx'][0])]
                        for axis in (1, 2):
                            size = min(384, source_shape[axis])
                            lo = max(0, min(source_shape[axis]-size, (sum(window['bounds_zyx'][axis])-size)//2))
                            expanded.append([lo, lo+size])
                        key = tuple(value for pair in expanded for value in pair)
                        if key not in scheduled_contexts:
                            scheduled_contexts[key] = next_window_id
                            windows.append({**window, 'window_id': next_window_id, 'bounds_zyx': expanded,
                                'context_expansion_of_window': window['window_id'],
                                'context_expansion_parent_windows': [window['window_id']],
                                'context_trigger_xy_label_ids': internal_ids,
                                'source_events': [dict(event) for event in window.get('source_events', [window])],
                                'tiling_version': 'official_native384_internal_boundary_check_v1',
                                'network_tiles': None,
                                'model_context_mode': 'official_native384_internal_boundary_check'})
                            next_window_id += 1
                        else:
                            context_window = next(w for w in windows if w['window_id'] == scheduled_contexts[key])
                            context_window['context_expansion_parent_windows'].append(window['window_id'])
                            context_window['source_events'].extend(window.get('source_events', [window]))
                        diagnostics['boundary_context_checks'].append({'source_window_id': window['window_id'],
                            'internal_xy_label_ids': internal_ids, 'source_xy_boundary_labels': boundaries,
                            'context_window_id': scheduled_contexts[key], 'bounds_zyx': expanded,
                            'original_observations_retained': True})
                    score = np.asarray(prediction.get('dapi_cellprob', []))
                    finite = score[np.isfinite(score)]
                    window['network_tiles'] = prediction.get('network_tiles')
                    diagnostics['windows'].append({**window, 'normalization_limits': item['normalization_limits'],
                        'feature': item['feature'], 'model_batch_seconds': batch_seconds,
                        'primary_instances': int(np.count_nonzero(np.unique(prediction['labels']))),
                        'permissive_triggered': not covered, 'permissive_new_xy_ids': weak_ids,
                        'permissive_seconds': weak_seconds,
                        'tiling_version': prediction.get('tiling_version'),
                        'actual_network_tiles': prediction.get('network_tiles'),
                        'computational_padding_yx': prediction.get('computational_padding_yx'),
                        'xy_boundary_labels': boundaries,
                        'permissive_trigger_reason': None if covered else 'localization_seed_not_covered_by_primary',
                        'score_percentiles': np.percentile(finite, [0, 1, 50, 99, 100]).tolist() if finite.size else [],
                        'above_primary_score_pixels': int(np.count_nonzero(score > 0))})
                    with (output/'model-progress.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(diagnostics['windows'][-1], allow_nan=False)+'\n')
                    if not local_labels.any():
                        empty = {'window_id': window['window_id'], 'attempts': [], 'seconds': 0.,
                                 'metrics': {'native_recovery_evaluated': False,
                                             'reason': 'no_primary_or_permissive_cp_seed',
                                             'nuclear_identities_created': 0}}
                        diagnostics['endpoint_expansions'].append(empty)
                        with (output/'native-recovery-progress.jsonl').open('a', encoding='utf-8') as handle:
                            handle.write(json.dumps(empty, allow_nan=False)+'\n')
                        continue
                    while len(pending) >= CPU_WORKERS:
                        consume(pending[0])
                        pending.popleft()
                    pending.append(pool.submit(_recover, reader, lock, dapi, spacing,
                                               {**window, 'weak_xy_ids': weak_ids, 'xy_boundary_labels': boundaries},
                                               local_labels, events.check_cancel))
                events.emit('progress', 'count_inference', '处理局部 DAPI 特征与原生三维归属',
                            current=offset+len(batch), total=len(windows))
                offset += len(batch)
        diagnostics['label_store'] = store.finish()
        diagnostics['total_actual_network_tiles'] = sum(item.get('actual_network_tiles') or 0
                                                         for item in diagnostics['windows'])
        diagnostics['local_inference_and_recovery_seconds'] = time.perf_counter()-started
        events.emit('stage', 'count_prepare_review', '测量每个原生核区域的 Support 与逐层证据')
        started = time.perf_counter()
        # A larger Support pool starts only after model inference and nuclear
        # recovery finish; it cannot raise concurrency during the MPS stage.
        points = _support_points(reader, lock, store, support, marker, spacing, events.check_cancel,
            workers=support_workers, cpu_pool=cpu_pool if support_workers == CPU_WORKERS else None)
        diagnostics['support_seconds'] = time.perf_counter()-started
        events.check_cancel()
        return points, diagnostics
    finally:
        # The caller may own the shared CPU pool and TIFF reader. Join our work
        # before either can be closed, retaining the initiating exception.
        for future in pending:
            future.cancel()
        for future in pending:
            if not future.cancelled():
                try:
                    future.result()
                except Exception:
                    pass
        store.close()
