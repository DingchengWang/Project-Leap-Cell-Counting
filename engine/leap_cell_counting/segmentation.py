"""Pinned native-XY model loading, parameters and actual-mask anchor helpers."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path
import numpy as np
from .errors import CountingError

MODEL_NAME = "cpsam_v2"
MODEL_VERSION = "4.2.1.1"
MODEL_SHA256 = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"
EVAL_PARAMETERS = {
    "batch_size": 8, "resample": True,
    "channel_axis": 3, "z_axis": None, "do_3D": False,
    "flow_threshold": 0.4, "cellprob_threshold": 0.0, "min_size": -1,
    "max_size_fraction": 1.0, "augment": False, "tile_overlap": 0.1,
    "bsize": 256, "compute_masks": False, "niter": 200,
}
WEAK_CELLPROB_THRESHOLD = -2.0  # Development threshold: every new candidate requires review.


def load_model(check_cancel=lambda: None):
    directory = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH", "")
    if not directory or not Path(directory).is_absolute():
        raise CountingError("model_path_required", "需要安装的共享 Cellpose 模型绝对路径。")
    model_path = Path(directory).resolve(strict=True) / MODEL_NAME
    if not model_path.is_file():
        raise CountingError("model_missing", f"缺少固定模型 {MODEL_NAME}。")
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            check_cancel(); digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise CountingError("model_hash_mismatch", "共享 Cellpose 模型校验失败，请修复组件。")
    if version("cellpose") != MODEL_VERSION:
        raise CountingError("cellpose_version_mismatch", f"需要 Cellpose {MODEL_VERSION}。")
    import torch
    from cellpose import models
    np.random.seed(0); torch.manual_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    check_cancel()
    model = models.CellposeModel(pretrained_model=str(model_path), device=device,
                                 gpu=device.type != "cpu", use_bfloat16=False)
    check_cancel()
    return model, {"name": MODEL_NAME, "version": MODEL_VERSION, "sha256": MODEL_SHA256,
                   "device": str(device), "parameters": dict(EVAL_PARAMETERS),
                   "input": "native XY DAPI feature; true 3D identity is recovered from the original volume",
                   "normalization": "explicit feature low/high required; display pixels never enter inference",
                   "mask_dynamics_device": str(device)}

def validate_labels(labels, shape):
    labels = np.asarray(labels)
    # Restore Cellpose-squeezed axes to the original ZYX storage contract.
    if labels.size == math.prod(shape) and labels.shape == tuple(n for n in shape if n != 1):
        labels = labels.reshape(shape)
    unsafe_signed = labels.dtype.kind == 'i' and np.any(labels < 0)
    too_wide = labels.dtype.kind in 'iu' and labels.dtype.itemsize > 4 and labels.size and labels.max() > np.iinfo(np.uint32).max
    if labels.shape != tuple(shape) or labels.dtype.kind not in "iu" or unsafe_signed or too_wide:
        raise CountingError("invalid_instance_labels", "模型未返回原始 ZYX 尺寸的 uint32 范围非负整数实例标签。")
    return labels

def interior_anchor(mask):
    """An anchor in the largest 8-connected planar component, never its hole."""
    from scipy import ndimage
    components, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    main = components == int(sizes.argmax())
    distance = ndimage.distance_transform_edt(np.pad(main, 1))[1:-1, 1:-1]
    choices = np.argwhere(distance == distance.max())
    centroid = np.argwhere(main).mean(axis=0)
    y, x = choices[np.argmin(np.square(choices-centroid).sum(axis=1))]
    return [float(x)+.5, float(y)+.5]
