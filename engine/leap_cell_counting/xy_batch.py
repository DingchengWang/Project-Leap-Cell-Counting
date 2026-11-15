"""Native256 tiles with explicit true-source boundary handling.

Original 256-square features go directly to the pinned official network. Only
an actual source dimension smaller than the model tile is computationally
padded, then cropped before mask dynamics. This empirically validated tiling
choice is not numerically equivalent to eval's four tiles at native256.
"""
from __future__ import annotations
import numpy as np
from .errors import CountingError
from .segmentation import EVAL_PARAMETERS, WEAK_CELLPROB_THRESHOLD, validate_labels

TILING_VERSION = 'native256_direct_v1'
TILE_SIZE = 256


def _native_canvas(normalized, source_shape_yx):
    array = np.asarray(normalized)
    if (array.ndim != 2 or array.dtype != np.float32 or len(source_shape_yx) != 2 or
            any(type(n) not in (int, np.int32, np.int64) or n < 1 for n in source_shape_yx) or
            array.shape != tuple(min(TILE_SIZE, n) for n in source_shape_yx)):
        raise CountingError('insufficient_native_model_context',
            '模型需要完整真实 256 像素上下文；只有原图本身较小的轴允许计算补边。')
    height, width = array.shape
    top, left = (TILE_SIZE-height)//2, (TILE_SIZE-width)//2
    region = (slice(top, top+height), slice(left, left+width))
    if array.shape == (TILE_SIZE, TILE_SIZE):
        return array, region
    canvas = np.zeros((TILE_SIZE, TILE_SIZE), np.float32)
    canvas[region] = array
    return canvas, region


def _mask_settings():
    return {**{key: EVAL_PARAMETERS[key] for key in
               ('flow_threshold', 'min_size', 'max_size_fraction', 'niter')},
            'stitch_threshold': 0., 'do_3D': False}


def infer_xy_batch(model, candidates, *, check_cancel=lambda: None,
                   progress=lambda *_: None, include_weak=True):
    """One independent native tile per candidate; at most eight per forward.

    source_shape_yx distinguishes true small sources from inadequate crops.
    Explicit feature low/high normalization is applied once; mask dynamics
    receive original-size fields after any true-source computational padding.
    """
    from cellpose import core, transforms
    if len(candidates) > 8:
        raise CountingError('xy_batch_memory_budget', '单批最多八个原生窗口；全部待处理区域必须分批保留。')
    if not candidates:
        return []
    packed, regions = [], []
    for item in candidates:
        check_cancel()
        image = np.asarray(item['image'])
        if image.ndim != 2 or image.dtype.kind not in 'uif' or (image.dtype.kind == 'f' and not np.isfinite(image).all()):
            raise CountingError('invalid_dapi_pixels', '局部 DAPI 必须为有限原生二维灰度数据。')
        low, high = item['normalization_limits']
        if not np.isfinite([low, high]).all() or high <= low:
            raise CountingError('invalid_normalization_limits', '局部特征需要明确、有限、递增的归一化范围。')
        normalized = transforms.normalize_img(image[..., None].astype(np.float32, copy=True),
            normalize=True, norm3D=False, lowhigh=[float(low), float(high)])[..., 0]
        if 'source_shape_yx' not in item:
            raise CountingError('source_shape_required', '原生窗口必须记录原图 XY 尺寸。')
        canvas, region = _native_canvas(normalized, item['source_shape_yx'])
        packed.append(canvas); regions.append(region)
    inputs = np.stack(packed)[:, None]
    pre = model.net.register_forward_pre_hook(lambda *_: check_cancel())
    def after(*_):
        check_cancel(); progress(1, 1)
    post = model.net.register_forward_hook(after)
    try:
        fields, _styles = core._forward(model.net, inputs)
        check_cancel()
    finally:
        pre.remove(); post.remove()
    fields = np.asarray(fields)
    if fields.shape != (len(candidates), 3, TILE_SIZE, TILE_SIZE):
        raise CountingError('invalid_xy_batch_fields', '模型未返回独立原生窗口的流场和分数。')
    outputs = []
    settings = _mask_settings()
    for index, (item, region) in enumerate(zip(candidates, regions)):
        check_cancel()
        height, width = np.asarray(item['image']).shape
        flow = fields[index, :2, region[0], region[1]].copy()
        score = fields[index, 2, region[0], region[1]].copy()
        primary = model._compute_masks((1, height, width, 1), flow[:, None], score[None],
            cellprob_threshold=EVAL_PARAMETERS['cellprob_threshold'], **settings)
        result = {'labels': validate_labels(primary, (1, height, width))[0],
            'dapi_flow': flow, 'dapi_cellprob': score,
            'metadata': {key: value for key, value in item.items() if key != 'image'},
            'original_xy_shape': [height, width], 'tiling_version': TILING_VERSION,
            'network_tiles': 1,
            'computational_padding_yx': [[region[0].start, TILE_SIZE-region[0].stop],
                                         [region[1].start, TILE_SIZE-region[1].stop]],
            'creates_3d_identity': False, 'masks_computed_after_original_xy_crop': True}
        if include_weak:
            weak = model._compute_masks((1, height, width, 1), flow[:, None], score[None],
                cellprob_threshold=WEAK_CELLPROB_THRESHOLD, **settings)
            result['weak_labels'] = validate_labels(weak, (1, height, width))[0]
        outputs.append(result)
    return outputs


def permissive_from_fields(model, prediction):
    """Targeted permissive mask dynamics reuse this exact native feature flow."""
    flow, score = prediction['dapi_flow'], prediction['dapi_cellprob']
    height, width = score.shape
    masks = model._compute_masks((1, height, width, 1), flow[:, None], score[None],
        cellprob_threshold=WEAK_CELLPROB_THRESHOLD, **_mask_settings())
    return validate_labels(masks, (1, height, width))[0]


def infer_boundary_context(model, candidate, *, check_cancel=lambda: None):
    """One recorded real384 check for an observed internal native256 truncation."""
    from cellpose import transforms
    image = np.asarray(candidate['image'])
    if (not candidate.get('context_expansion_of_window') or image.ndim != 2 or
            image.shape != tuple(min(384, n) for n in candidate['source_shape_yx'])):
        raise CountingError('invalid_boundary_context', '扩大上下文仅用于已记录的内部模型窗口截断。')
    low, high = candidate['normalization_limits']
    normalized = transforms.normalize_img(image[..., None].astype(np.float32, copy=True),
        normalize=True, norm3D=False, lowhigh=[float(low), float(high)])
    tiles = []
    def before(_net, args):
        check_cancel(); tiles.append(int(args[0].shape[0]))
    pre = model.net.register_forward_pre_hook(before)
    post = model.net.register_forward_hook(lambda *_: check_cancel())
    try:
        labels, flows, _ = model.eval(normalized, **{**EVAL_PARAMETERS,
            'channel_axis': 2, 'normalize': False, 'compute_masks': True})
        check_cancel()
    finally:
        pre.remove(); post.remove()
    return {'labels': validate_labels(labels, (1, *image.shape))[0],
        'dapi_flow': np.asarray(flows[1]), 'dapi_cellprob': np.asarray(flows[2]),
        'metadata': {key: value for key, value in candidate.items() if key != 'image'},
        'original_xy_shape': list(image.shape), 'network_tiles': sum(tiles),
        'tiling_version': 'official_native384_internal_boundary_check_v1',
        'creates_3d_identity': False, 'masks_computed_after_original_xy_crop': True}


def observed_boundary_labels(labels, bounds_zyx, source_shape_zyx):
    """Distinguish actual source limits from computation-window truncation."""
    result = {}
    for axis, name in ((0, 'y'), (1, 'x')):
        source_axis = axis+1
        for side, at in (('low', 0), ('high', -1)):
            ids = np.unique(np.take(labels, at, axis=axis))
            source_edge = (bounds_zyx[source_axis][0] == 0 if side == 'low' else
                           bounds_zyx[source_axis][1] == source_shape_zyx[source_axis])
            for value in ids[ids > 0]:
                record = result.setdefault(int(value), {'internal_faces': [], 'source_faces': []})
                record['source_faces' if source_edge else 'internal_faces'].append(name+'_'+side)
    return result
