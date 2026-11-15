"""Calibrated, local support-channel evidence for an existing DAPI object.

Inputs are matching native-grid ZYX arrays; spacing is XYZ in micrometres. No
intensities or masks are modified, and no nucleus is created. ``accepted`` means
that the development association rule passes, not validated cell-type identity.
The caller supplies at least ``required_padding_um`` around the nuclear bounding
box where the original image permits it, without padding missing image data.

Six axis-aligned outward rays start at each exposed nuclear voxel face. Their
weights are physical face areas. Contact and a consecutive positive run measure
near-nuclear cytoplasm; disconnected remote signal cannot complete a run. This
is a Cartesian surface estimator, not a reconstruction of a membrane. Nuclear
signal is recorded separately and neither vetoes nor establishes support.
Single-Z inputs use four XY boundary directions, physical perimeter weighting,
and an explicit review result because axial support is unobservable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Mapping

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class SupportConfig:
    """Development values; physical distances never encode a nuclear diameter.

    The 0.6 um contact allowance and 1.2 um sustained signal depth distinguish
    resolved adjacent cytoplasm from a thin crossing or a remote envelope in
    engineering challenges. Partial coverage is permitted; a closed ring is not
    required. A 0.30 thick-coverage threshold keeps the observed 0.237-0.269
    coarse-process challenge in review; it is a development guardrail, not a
    universal biological boundary. These values still require annotated,
    representative real data.
    """

    contact_gap_um: float = 0.6
    cytoplasm_depth_um: float = 1.2
    background_inner_um: float = 4.0
    background_outer_um: float = 8.0
    max_sampling_step_um: float = 0.6
    smoothing_sigma_um: float = 0.2
    min_background_area_um2: float = 8.0
    min_background_volume_um3: float = 8.0
    min_boundary_samples: int = 32
    min_background_samples: int = 32
    min_valid_boundary_fraction: float = 0.6
    min_finite_fraction: float = 0.995
    positive_noise_multiplier: float = 3.0
    positive_dynamic_fraction: float = 0.1
    min_dynamic_noise_ratio: float = 4.0
    min_contact_fraction: float = 0.2
    min_thick_fraction: float = 0.30
    min_plane_thick_fraction: float = 0.1
    min_z_support_fraction: float = 0.35
    max_absent_contact_fraction: float = 0.03
    max_absent_nuclear_fraction: float = 0.1


def _config(config):
    if config is None:
        cfg = SupportConfig()
    elif isinstance(config, SupportConfig):
        cfg = config
    elif isinstance(config, Mapping):
        unknown = set(config) - {field.name for field in fields(SupportConfig)}
        if unknown:
            raise ValueError("Unknown support configuration: " + ", ".join(sorted(unknown)))
        cfg = SupportConfig(**config)
    else:
        raise TypeError("config must be SupportConfig, a mapping, or None")
    for name, value in asdict(cfg).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError("Support configuration must be finite and positive: " + name)
        if "fraction" in name and value > 1:
            raise ValueError("Support fractions must be at most one: " + name)
        if name.endswith("samples") and int(value) != value:
            raise ValueError("Support sample counts must be integers: " + name)
    if cfg.background_inner_um <= cfg.contact_gap_um + cfg.cytoplasm_depth_um:
        raise ValueError("Background must start beyond the measured cytoplasm")
    if cfg.background_outer_um <= cfg.background_inner_um:
        raise ValueError("Background outer distance must exceed inner distance")
    if cfg.max_sampling_step_um > cfg.cytoplasm_depth_um / 2:
        raise ValueError("Cytoplasmic thickness needs at least two resolved samples")
    if cfg.max_absent_contact_fraction >= cfg.min_contact_fraction:
        raise ValueError("Absent-contact threshold must be below acceptance threshold")
    return cfg


def required_padding_um(config=None):
    """Minimum caller ROI padding, clipped only by the original image bounds."""
    return _config(config).background_outer_um


def _result(decision, metrics, reasons):
    return {"decision": decision, "metrics": metrics, "reasons": reasons,
            "review_hint": decision == "review"}


def _face_coordinates(mask, axis, direction):
    """Include out-of-array faces in the denominator, never as observations."""
    neighbor = np.zeros_like(mask)
    source = [slice(None)] * mask.ndim
    target = [slice(None)] * mask.ndim
    if direction > 0:
        source[axis], target[axis] = slice(1, None), slice(None, -1)
    else:
        source[axis], target[axis] = slice(None, -1), slice(1, None)
    neighbor[tuple(target)] = mask[tuple(source)]
    return np.argwhere(mask & ~neighbor)


def _surface_faces(mask, low, high):
    """Find faces in the tight nuclear box, then restore original ROI indices.

    Voxels outside this box are known to be false. Faces remain in the total
    denominator even when their outward rays leave the original image ROI.
    """
    tight = mask[tuple(slice(int(a), int(b) + 1) for a, b in zip(low, high))]
    return {(axis, direction): _face_coordinates(tight, axis, direction) + low
            for axis in range(mask.ndim) for direction in (-1, 1)}


def _smooth_support(support, finite, sigma):
    """Preserve normalized-convolution values without a constant ROI weight map."""
    if finite.all():
        smoothed = ndi.gaussian_filter(support.astype(np.float64), sigma, mode="nearest")
        # Constant-field convolution can round just below 1. Reproduce that
        # scalar rather than silently changing the existing floating result.
        weight = ndi.gaussian_filter(np.ones((1,) * support.ndim), sigma, mode="nearest").item()
        if weight != 1.0:
            smoothed /= weight
        return smoothed
    weights = ndi.gaussian_filter(finite.astype(np.float64), sigma, mode="nearest")
    smoothed = ndi.gaussian_filter(np.where(finite, support, 0).astype(np.float64), sigma, mode="nearest")
    smoothed = np.divide(smoothed, weights, out=np.full_like(smoothed, np.nan), where=weights > 0)
    smoothed[~finite] = np.nan
    return smoothed


def _boundary_evidence(mask, support, threshold, sampling, cfg, faces):
    """Area-weighted contact and consecutive outward support on valid faces."""
    total_area = valid_area = contact_area = thick_area = 0.0
    total_faces = valid_faces = 0
    is_volume = mask.ndim == 3
    per_z_valid = np.zeros(mask.shape[0] if is_volume else 1, dtype=np.float64)
    per_z_thick = np.zeros_like(per_z_valid)
    depth_values, depth_weights = [], []
    per_axis = []
    for axis, spacing in enumerate(sampling):
        face_area = float(np.prod(np.delete(sampling, axis)))
        axis_total = axis_valid = axis_thick = 0.0
        for direction in (-1, 1):
            coords = faces[axis, direction]
            count = len(coords)
            total_faces += count
            total_area += count * face_area
            axis_total += count * face_area
            if not count or spacing > cfg.max_sampling_step_um * (1 + 1e-9):
                continue
            # One full positive run must fit after the allowed initial gap.
            steps = int(np.floor(cfg.contact_gap_um / spacing + 1e-9)) + int(np.ceil(cfg.cytoplasm_depth_um / spacing - 1e-9))
            valid = np.ones(count, dtype=bool)
            first = np.full(count, -1, dtype=np.int32)
            run = np.zeros(count, dtype=np.int32)
            running = np.ones(count, dtype=bool)
            for step in range(1, steps + 1):
                indexes = coords.copy()
                indexes[:, axis] += direction * step
                inside = (indexes[:, axis] >= 0) & (indexes[:, axis] < mask.shape[axis])
                observed = np.zeros(count, dtype=bool)
                positive = np.zeros(count, dtype=bool)
                if inside.any():
                    selected = tuple(indexes[inside].T)
                    values = support[selected]
                    observed[inside] = np.isfinite(values) & ~mask[selected]
                    positive[inside] = observed[inside] & (values > threshold)
                valid &= observed
                starting = (first < 0) & positive
                first[starting] = step
                active = (first >= 0) & running
                run[active & positive] += 1
                running[active & ~positive] = False
            contact = (first >= 0) & ((first - 1) * spacing <= cfg.contact_gap_um + 1e-9)
            thickness = run * spacing
            thick = contact & (thickness >= cfg.cytoplasm_depth_um - 1e-9)
            n_valid = int(valid.sum())
            valid_faces += n_valid
            valid_area += n_valid * face_area
            axis_valid += n_valid * face_area
            contact_area += int((valid & contact).sum()) * face_area
            thick_area += int((valid & thick).sum()) * face_area
            axis_thick += int((valid & thick).sum()) * face_area
            plane_indexes = coords[:, 0] if is_volume else np.zeros(count, dtype=int)
            np.add.at(per_z_valid, plane_indexes[valid], face_area)
            np.add.at(per_z_thick, plane_indexes[valid & thick], face_area)
            if (valid & contact).any():
                depth_values.extend(thickness[valid & contact].tolist())
                depth_weights.extend([face_area] * int((valid & contact).sum()))
        per_axis.append({"axis": ("ZYX" if is_volume else "YX")[axis], "spacing_um": float(spacing),
                         "resolved": bool(spacing <= cfg.max_sampling_step_um * (1 + 1e-9)),
                         "thick_support_fraction": axis_thick / axis_valid if axis_valid else 0.0,
                         ("surface_area_um2" if is_volume else "boundary_length_um"): axis_total,
                         ("valid_surface_area_um2" if is_volume else "valid_boundary_length_um"): axis_valid})
    plane_fraction = np.divide(per_z_thick, per_z_valid, out=np.zeros_like(per_z_valid), where=per_z_valid > 0)
    present = mask.any(axis=(1, 2)) if is_volume else np.array([mask.any()])
    supported_z = present & (plane_fraction >= cfg.min_plane_thick_fraction) & (per_z_valid > 0)
    longest_run = current_run = 0
    for value in supported_z:
        current_run = current_run + 1 if value else 0
        longest_run = max(longest_run, current_run)
    depth_mean = float(np.average(depth_values, weights=depth_weights)) if depth_values else 0.0
    return {"measurement_dimension": mask.ndim, "axial_support_evaluable": is_volume,
            "surface_area_um2": total_area if is_volume else None,
            "valid_surface_area_um2": valid_area if is_volume else None,
            "boundary_length_um": total_area if not is_volume else None,
            "valid_boundary_length_um": valid_area if not is_volume else None,
            "boundary_face_samples": total_faces, "valid_boundary_face_samples": valid_faces,
            "boundary_valid_fraction": valid_area / total_area if total_area else 0.0,
            "contact_area_um2": contact_area if is_volume else None, "thick_support_area_um2": thick_area if is_volume else None,
            "contact_length_um": contact_area if not is_volume else None, "thick_support_length_um": thick_area if not is_volume else None,
            "contact_fraction": contact_area / valid_area if valid_area else 0.0,
            "thick_support_fraction": thick_area / valid_area if valid_area else 0.0,
            "mean_contact_run_depth_um": depth_mean,
            "z_support_fraction": float(longest_run) / max(int(present.sum()), 1) if is_volume else None,
            "z_supported_plane_fraction": float(supported_z.sum()) / max(int(present.sum()), 1) if is_volume else None,
            "longest_supported_z_run_planes": int(longest_run) if is_volume else None,
            "longest_supported_z_run_um": longest_run * float(sampling[0]) if is_volume else None,
            "supported_z_planes": np.flatnonzero(supported_z).tolist() if is_volume else [],
            "nuclear_z_planes": np.flatnonzero(present).tolist(),
            "per_z_thick_fraction": plane_fraction.tolist(), "sampling_by_axis": per_axis}


def _intensity_statistics(mask, support, finite, background, local, cfg):
    """The same robust local threshold definition before and after smoothing."""
    background_values = support[background & finite].astype(np.float64)
    local_values = support[local & finite].astype(np.float64)
    median = float(np.median(background_values))
    noise = float(1.4826 * np.median(np.abs(background_values - median)))
    high_signal = float(np.percentile(local_values, 99.5))
    numeric_floor = float(np.finfo(np.float64).eps * max(float(np.max(np.abs(local_values))), 1.0) * 32)
    dynamic = max(high_signal - median, 0.0)
    # The percentile still tests signal against background noise. For thickness,
    # a real peak anchors the fractional threshold: a sparse thin shell must not
    # acquire thickness from Gaussian tails below the percentile's occupancy.
    peak = float(np.max(local_values))
    threshold_dynamic = max(peak - median, 0.0)
    threshold = median + max(cfg.positive_noise_multiplier * noise, cfg.positive_dynamic_fraction * threshold_dynamic, numeric_floor)
    detectable = bool(dynamic > max(cfg.min_dynamic_noise_ratio * noise, numeric_floor))
    nuclear_values = support[mask & finite]
    return {"background_median": median, "background_robust_sigma": noise,
            "local_high_percentile": 99.5, "local_high_intensity": high_signal,
            "local_dynamic": dynamic, "local_peak_intensity": peak,
            "threshold_dynamic": threshold_dynamic,
            "threshold_dynamic_reference": "local_peak_for_thickness_percentile_for_detectability",
            "positive_threshold": float(threshold), "signal_detectable": detectable,
            "nuclear_positive_fraction": float((nuclear_values > threshold).mean()) if nuclear_values.size else 0.0,
            "finite_nuclear_fraction": float(finite[mask].mean()) if mask.any() else None}


def measure_support(nucleus_mask, support_volume, spacing_xyz, marker, config=None, *, diagnostics=False,
                    _distance_zyx=None):
    """Return ``decision``, JSON-safe ``metrics``, ``reasons`` and ``review_hint``.

    Missing calibration, nonfinite/boundary evidence,
    absent dynamic range, and ambiguous geometry produce review. Exclusion needs
    locally detectable support signal but negligible contact and nuclear signal.
    Input programming errors raise ValueError; biological uncertainty is a result.
    ``diagnostics=True`` also measures raw signal before smoothing. Production
    decisions always use the same full native surface and final-signal statistics.
    ``_distance_zyx`` is an internal reuse contract: the unmodified float64 ZYX
    result of ``volume._external_distance_um`` for this exact mask and spacing.
    """
    if not isinstance(diagnostics, bool):
        raise ValueError("diagnostics must be an explicit boolean")
    cfg = _config(config)
    mask_input = np.asarray(nucleus_mask)
    support = np.asarray(support_volume)
    if mask_input.ndim != 3 or mask_input.shape != support.shape or any(size == 0 for size in mask_input.shape):
        raise ValueError("nucleus_mask and support_volume must be matching nonempty ZYX arrays")
    if mask_input.dtype.kind not in "biuf" or not np.isfinite(mask_input).all() or not np.isin(mask_input, [0, 1]).all():
        raise ValueError("nucleus_mask must be a binary mask for exactly one DAPI object")
    if support.dtype.kind not in "biuf":
        raise ValueError("support_volume must contain real numeric intensities")
    mask = mask_input.astype(bool, copy=False)
    if not isinstance(marker, str) or not marker.strip():
        raise ValueError("A declared Support marker is required")
    marker_text = marker.strip()
    metrics = {"method": "native_local_face_rays_v1", "validation_status": "development_thresholds",
               "marker": marker_text, "marker_provenance": "caller-declared Support marker",
               "marker_name_establishes_cell_type": False,
               "thresholds": asdict(cfg), "nucleus_voxels": int(mask.sum()),
               "closed_ring_required": False, "nuclear_signal_veto": False,
               "surface_estimator": "native Cartesian faces weighted by physical area or single-plane perimeter",
               "signal_preprocessing": "Gaussian support-only smoothing in calibrated physical units; no mask smoothing",
               "before_smoothing_evaluated": False}
    if not mask.any():
        return _result("review", metrics, ["no_dapi_nucleus"])
    try:
        spacing = np.asarray(spacing_xyz, dtype=np.float64)
    except (ValueError, TypeError):
        return _result("review", metrics, ["physical_calibration_unavailable"])
    is_volume = mask.shape[0] > 1
    if spacing.shape != (3,):
        return _result("review", metrics, ["physical_calibration_unavailable"])
    required_spacing = spacing if is_volume else spacing[:2]
    if not np.isfinite(required_spacing).all() or (required_spacing <= 0).any():
        return _result("review", metrics, ["physical_calibration_unavailable"])
    metrics["spacing_xyz_um"] = [float(value) if np.isfinite(value) and value > 0 else None for value in spacing]
    if is_volume:
        sampling = spacing[::-1]
        metrics["nucleus_volume_um3"] = int(mask.sum()) * float(np.prod(sampling))
    else:
        mask, support = mask[0], support[0]
        sampling = spacing[:2][::-1]
        metrics["nucleus_area_um2"] = int(mask.sum()) * float(np.prod(sampling))
        metrics["nucleus_volume_um3"] = None
    sample_measure = float(np.prod(sampling))
    metrics.update(measurement_dimension=mask.ndim, axial_support_evaluable=is_volume)
    # Preserve every predicted voxel. Disconnection requires identity review,
    # but does not hide the independent Support measurements from that review.
    components, component_count = ndi.label(mask, ndi.generate_binary_structure(mask.ndim, mask.ndim))
    component_sizes = np.bincount(components.ravel())
    component_sizes[0] = 0
    metrics.update(nuclear_component_count=int(component_count),
                   largest_nuclear_component_fraction=float(component_sizes.max() / mask.sum()))
    del components
    identity_reasons = ["disconnected_dapi_object"] if component_count != 1 else []
    coords = np.argwhere(mask)
    low, high = coords.min(axis=0), coords.max(axis=0)
    truncated = bool(np.any(low == 0) or np.any(high == np.asarray(mask.shape) - 1))
    margins = np.r_[low * sampling, (np.asarray(mask.shape) - 1 - high) * sampling]
    metrics["nucleus_touches_roi_boundary"] = truncated
    metrics["roi_margins_zyx_low_high_um" if is_volume else "roi_margins_yx_low_high_um"] = margins.tolist()
    if _distance_zyx is None:
        distance = ndi.distance_transform_edt(~mask, sampling=sampling)
    else:
        distance = _distance_zyx if is_volume else _distance_zyx[0]
    finite = np.isfinite(support)
    local = distance <= cfg.background_outer_um
    metrics["finite_local_fraction"] = float(finite[local].mean())
    background = (distance >= cfg.background_inner_um) & (distance <= cfg.background_outer_um)
    # Apply only to support signal: physical smoothing suppresses voxel noise that
    # would otherwise fragment every cytoplasmic ray. Missing data stay missing.
    raw_support = support
    sigma = cfg.smoothing_sigma_um / sampling
    support = _smooth_support(support, finite, sigma)
    metrics["smoothing_sigma_zyx_px" if is_volume else "smoothing_sigma_yx_px"] = sigma.tolist()
    background_values = support[background & finite]
    metrics["background_samples"] = int(background_values.size)
    background_measure = int(background_values.size) * sample_measure
    metrics["background_volume_um3" if is_volume else "background_area_um2"] = background_measure
    min_background_measure = cfg.min_background_volume_um3 if is_volume else cfg.min_background_area_um2
    if background_values.size < cfg.min_background_samples or background_measure < min_background_measure:
        return _result("review", metrics, identity_reasons + ["insufficient_local_background"])
    statistics = _intensity_statistics(mask, support, finite, background, local, cfg)
    metrics.update(statistics)
    faces = _surface_faces(mask, low, high)
    metrics.update(_boundary_evidence(mask, support, statistics["positive_threshold"], sampling, cfg, faces))
    if diagnostics:
        raw_statistics = _intensity_statistics(mask, raw_support, finite, background, local, cfg)
        raw_boundary = _boundary_evidence(mask, raw_support, raw_statistics["positive_threshold"], sampling, cfg, faces)
        metrics["before_smoothing"] = dict(raw_statistics, **{key: raw_boundary[key] for key in (
            "contact_fraction", "thick_support_fraction", "mean_contact_run_depth_um",
            "z_support_fraction", "z_supported_plane_fraction", "longest_supported_z_run_um", "boundary_valid_fraction")})
        metrics["before_smoothing_evaluated"] = True
    reasons = identity_reasons + ([] if is_volume else ["single_plane_axial_support_unavailable"])
    if truncated:
        reasons.append("nucleus_truncated_by_roi")
    if metrics["finite_local_fraction"] < cfg.min_finite_fraction or metrics["finite_nuclear_fraction"] < cfg.min_finite_fraction:
        reasons.append("nonfinite_local_signal")
    if metrics["boundary_valid_fraction"] < cfg.min_valid_boundary_fraction or metrics["valid_boundary_face_samples"] < cfg.min_boundary_samples:
        reasons.append("insufficient_resolved_boundary")
    if not statistics["signal_detectable"]:
        reasons.append("local_signal_not_distinguishable")
    if reasons:
        return _result("review", metrics, reasons)
    if (metrics["contact_fraction"] >= cfg.min_contact_fraction
            and metrics["thick_support_fraction"] >= cfg.min_thick_fraction
            and metrics["z_support_fraction"] >= cfg.min_z_support_fraction
            and len(metrics["supported_z_planes"]) >= 2):
        return _result("accepted", metrics, ["adjacent_resolved_cytoplasm"])
    if (metrics["contact_fraction"] <= cfg.max_absent_contact_fraction
            and metrics["nuclear_positive_fraction"] <= cfg.max_absent_nuclear_fraction):
        return _result("excluded", metrics, ["local_signal_present_without_nuclear_association"])
    if metrics["nuclear_positive_fraction"] > cfg.max_absent_nuclear_fraction and metrics["thick_support_fraction"] < cfg.min_thick_fraction:
        reasons.append("nuclear_signal_without_resolved_cytoplasm")
    if metrics["contact_fraction"] > cfg.max_absent_contact_fraction and metrics["thick_support_fraction"] < cfg.min_thick_fraction:
        reasons.append("contact_without_sufficient_cytoplasmic_thickness")
    if metrics["z_support_fraction"] < cfg.min_z_support_fraction or len(metrics["supported_z_planes"]) < 2:
        reasons.append("limited_z_support")
    return _result("review", metrics, reasons or ["partial_or_ambiguous_association"])
