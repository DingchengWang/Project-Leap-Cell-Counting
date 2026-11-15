"""Spatial concentration partitions schedule native work; they are never cells."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

LOCALIZATION = {'grid_xy': 16, 'wide_sigma_zyx': [1., 2., 2.],
                'narrow_sigma_zyx': [1., 1., 1.], 'score_threshold': 5.,
                'partition_h': 1., 'xy_context_pixels': 32,
                'maximum_computation_window_xy': 240, 'computation_core_pixels': 208}


def _scores(mean, sigma, counts):
    field = ndi.gaussian_filter(mean, sigma=sigma, mode='nearest')
    for z in range(len(field)):
        center = np.median(field[z])
        scale = max(float(1.4826*np.median(np.abs(field[z]-center))), 1./float(counts.max()))
        field[z] = (field[z]-center)/scale
    return field


def _partition(field, mask):
    floor = float(field[mask].min())-2.
    inside = np.where(mask, field, floor)
    maxima = h_maxima(np.pad(inside, 1, constant_values=floor), 1.)[1:-1, 1:-1, 1:-1].astype(bool) & mask
    markers, number = ndi.label(maxima, structure=ndi.generate_binary_structure(3, 1))
    if not number:
        raise RuntimeError('Positive computation region has no partition seed')
    pieces = watershed(-inside, markers, mask=mask, connectivity=ndi.generate_binary_structure(3, 1))
    if not np.array_equal(pieces > 0, mask):
        raise RuntimeError('Computation partition lost candidate foreground')
    return pieces


def locate_windows(scan, *, check_cancel=lambda: None):
    """Retain every foreground piece; split large work without resizing pixels.

    The measured h=1 partition only controls computation. Z bounds are triggers;
    the original-volume recovery must establish and extend actual endpoints.
    """
    mean = np.asarray(scan['moments'][16]['dapi']['sum'], np.float32)/scan['counts'][16][None]
    wide = _scores(mean, LOCALIZATION['wide_sigma_zyx'], scan['counts'][16])
    narrow = _scores(mean, LOCALIZATION['narrow_sigma_zyx'], scan['counts'][16])
    mask = wide > LOCALIZATION['score_threshold']
    del mean, wide
    labels, number = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    shape = scan['shape_zyx']
    windows, conserved = [], 0
    for parent_id, region in enumerate(ndi.find_objects(labels), 1):
        check_cancel()
        local = labels[region] == parent_id
        def native_box(box):
            return [[box[0].start, box[0].stop],
                    [max(0, box[1].start*16-32), min(shape[1], box[1].stop*16+32)],
                    [max(0, box[2].start*16-32), min(shape[2], box[2].stop*16+32)]]
        outer = native_box(region)
        small = max(outer[1][1]-outer[1][0], outer[2][1]-outer[2][0]) <= 240
        pieces = local.astype(np.uint32) if small else _partition(narrow[region], local)
        for local_id, box in enumerate(ndi.find_objects(pieces), 1):
            selected = pieces[box] == local_id
            conserved += int(selected.sum())
            absolute = tuple(slice(region[a].start+box[a].start, region[a].start+box[a].stop) for a in range(3))
            bounds = native_box(absolute)
            if max(bounds[1][1]-bounds[1][0], bounds[2][1]-bounds[2][0]) <= 240:
                boxes = [bounds]
            else:
                boxes = []
                for y0 in range(bounds[1][0], bounds[1][1], 208):
                    for x0 in range(bounds[2][0], bounds[2][1], 208):
                        y1, x1 = min(bounds[1][1], y0+208), min(bounds[2][1], x0+208)
                        a, b = max(y0//16, absolute[1].start)-absolute[1].start, min((y1+15)//16, absolute[1].stop)-absolute[1].start
                        c, d = max(x0//16, absolute[2].start)-absolute[2].start, min((x1+15)//16, absolute[2].stop)-absolute[2].start
                        if a < b and c < d and selected[:, a:b, c:d].any():
                            boxes.append([bounds[0], [max(0, y0-16), min(shape[1], y1+16)],
                                          [max(0, x0-16), min(shape[2], x1+16)]])
            for bounds in boxes:
                y0 = max(absolute[1].start, bounds[1][0]//16)
                y1 = min(absolute[1].stop, (bounds[1][1]+15)//16)
                x0 = max(absolute[2].start, bounds[2][0]//16)
                x1 = min(absolute[2].stop, (bounds[2][1]+15)//16)
                view = (slice(absolute[0].start, absolute[0].stop), slice(y0, y1), slice(x0, x1))
                own = selected[:, y0-absolute[1].start:y1-absolute[1].start,
                               x0-absolute[2].start:x1-absolute[2].start]
                peak = np.unravel_index(np.argmax(np.where(own, narrow[view], -np.inf)), narrow[view].shape)
                seed = [int(peak[0]+absolute[0].start), min(shape[1]-1, int((peak[1]+y0)*16+8)),
                        min(shape[2]-1, int((peak[2]+x0)*16+8))]
                windows.append({'window_id': len(windows)+1, 'source_component_id': parent_id,
                    'compute_partition_id': local_id, 'bounds_zyx': bounds,
                    'localization_seed_zyx': seed,
                    'available_source_z_range': [0, shape[0]],
                    'assigns_nucleus_identity': False, 'source_grid_preserved': True,
                    'needs_native_z_endpoint_expansion': True})
    if conserved != int(mask.sum()):
        raise RuntimeError('Spatial proposal conservation failed')
    # Identical complete XYZ requests reuse inference; all parent memberships remain.
    unique = {}
    for window in windows:
        key = tuple(v for pair in window['bounds_zyx'] for v in pair)
        if key in unique:
            unique[key]['reused_window_ids'].append(window['window_id'])
        else:
            unique[key] = {**window, 'reused_window_ids': [window['window_id']]}
    return list(unique.values()), {'parameters': dict(LOCALIZATION), 'wide_components': int(number),
        'foreground_grid_cells': conserved, 'windows_before_reuse': len(windows),
        'computation_window_count': len(unique),
        'all_windows_retained': True, 'summary_is_model_or_segmentation_input': False,
        'z_bounds_are_final_nuclear_bounds': False, 'threshold_status': 'development_validated_reference_coverage_only'}


def native_model_windows(computation_windows, source_shape_zyx):
    """Supply true source context independently of concentration partitions."""
    from .xy_batch import TILE_SIZE, TILING_VERSION
    unique = {}
    for window in computation_windows:
        bounds = [list(window['bounds_zyx'][0])]
        for axis in (1, 2):
            original = window['bounds_zyx'][axis]
            width = min(TILE_SIZE, source_shape_zyx[axis])
            if original[1]-original[0] > width:
                raise ValueError('A computation window exceeds one native model context')
            lower = max(0, min(source_shape_zyx[axis]-width, (sum(original)-width)//2))
            bounds.append([lower, lower+width])
        key = tuple(value for pair in bounds for value in pair)
        event = {'window_id': window['window_id'], 'bounds_zyx': window['bounds_zyx'],
                 'localization_seed_zyx': window['localization_seed_zyx']}
        if key in unique:
            unique[key]['source_events'].append(event)
        else:
            unique[key] = {**window, 'bounds_zyx': bounds, 'source_events': [event],
                'source_shape_yx': list(source_shape_zyx[1:]), 'model_context_mode': 'direct_native256',
                'tiling_version': TILING_VERSION, 'network_tiles': 1}
    return list(unique.values()), {'tiling_version': TILING_VERSION,
        'real_context_xy': [min(TILE_SIZE, n) for n in source_shape_zyx[1:]],
        'source_event_windows': len(computation_windows), 'model_windows': len(unique),
        'required_network_tiles': len(unique), 'exact_xyz_reuse_only': True,
        'synthetic_padding_only_for_genuinely_small_source_axes': True,
        'all_source_events_retained': sum(len(w['source_events']) for w in unique.values()) == len(computation_windows)}
