"""Original-DAPI local evidence on actual native masks; no signal-retention veto."""
import numpy as np


def _external_distance_um(mask, spacing_xyz):
    """Return the float64 ZYX exterior EDT for this exact bool mask and spacing.

    Single-plane objects use XY distances only. The caller may share this array
    between Support and DAPI measurements of the same object; neither mutates it.
    """
    from scipy import ndimage
    if mask.shape[0] == 1:
        return ndimage.distance_transform_edt(~mask[0],sampling=(spacing_xyz[1],spacing_xyz[0]))[None]
    return ndimage.distance_transform_edt(~mask,sampling=tuple(reversed(spacing_xyz)))


def local_dapi_evidence(mask, raw, spacing_xyz, *, _distance_zyx=None):
    """Describe raw local DAPI and actual Z intersections without a signal veto.

    Quantized weak nuclei can share the background median while having a higher
    mean. These measurements support review; no intensity summary decides
    whether a permissive DAPI-flow candidate is retained.
    ``_distance_zyx`` is an internal reuse contract: the unmodified float64 ZYX
    result of ``_external_distance_um`` for this exact mask and spacing.
    """
    if not mask.any(): return {"evaluable": False, "reason": "empty_dapi_instance"}
    distance = _external_distance_um(mask, spacing_xyz) if _distance_zyx is None else _distance_zyx
    ring=(distance>=1.)&(distance<=3.)&~mask
    core=raw[mask]; background=raw[ring]
    core=core[np.isfinite(core)]; background=background[np.isfinite(background)]
    occupied=np.flatnonzero(np.any(mask,axis=(1,2)))
    gaps=np.diff(occupied)
    planes=[]
    for z in occupied:
        inside=raw[z][mask[z]]; outside=raw[z][ring[z]]
        inside=inside[np.isfinite(inside)];outside=outside[np.isfinite(outside)]
        mean=float(np.mean(inside)) if len(inside) else None
        baseline=float(np.mean(outside)) if len(outside) else None
        planes.append({"z_local":int(z),"nuclear_voxels":int(mask[z].sum()),
                       "nuclear_mean":mean,"background_mean":baseline,
                       "mean_contrast":mean-baseline if mean is not None and baseline is not None else None})
    evidence={"evaluable":bool(len(core) and len(background)),
        "ring_um":[1.,3.],"background_excludes":"current nucleus only; neighboring signal remains observable", "nuclear_samples":len(core),"background_samples":len(background),
        "z_planes_local":occupied.tolist(),"z_intersections_contiguous":bool(not len(gaps) or np.all(gaps==1)),
        "z_gap_count":int(np.sum(gaps>1)),"planes":planes,
        "retention_rule":"DAPI intensity summaries are evidence only; no median or mean veto",
        "validation_status":"development_not_biologically_validated"}
    if not evidence["evaluable"]:
        evidence["reason"]="insufficient_local_dapi_background";return evidence
    inside=float(np.median(core));outside=float(np.median(background))
    nuclear_mean=float(np.mean(core));background_mean=float(np.mean(background))
    evidence.update(nuclear_median=inside,background_median=outside,
        background_mad=float(np.median(np.abs(background-outside))),
        nuclear_mean=nuclear_mean,background_mean=background_mean,
        median_contrast=inside-outside,mean_contrast=nuclear_mean-background_mean)
    return evidence
