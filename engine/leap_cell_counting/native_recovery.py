"""CP-XY constrained native-DAPI 3D event recovery.

No inference, I/O, resampling, nucleus-size prior, or Support-derived boundaries
in recover_native_3d. The returned labels are provisional measured-density
regions with explicit identity and endpoint evidence. CP XY labels constrain lateral identity;
different disconnected 3D families always receive different event IDs.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


DEFAULTS = {
    "density_sigma_um": 0.568,
    "density_robust_score": 1.5,
    "gaussian_truncate": 4.0,
    "connectivity": 1,
    "cp_source_z_bounds_local": None,
    "min_cp_projected_overlap_fraction": 0.50,
    "xy_envelope_padding_um": 1.136,
    "xy_background_gap_um": 0.568,
    "xy_background_width_um": 1.136,
    "axial_background_mad_multiplier": 1.0,
}


def _gaussian_peak_weight(sigma, truncate):
    value = 1.0
    for width in sigma:
        if width <= 0:
            continue
        radius = int(truncate * width + 0.5)
        x = np.arange(-radius, radius + 1, dtype=float)
        value /= np.exp(-0.5 * (x / width) ** 2).sum()
    return value


def _region_record(mask, raw, events, density, box, origin, label_id,
                   family_id, cp_id, cp_ids, cp_z_bounds, shape):
    """Anchors are real mask pixels; per-Z raw evidence is never a veto."""
    starts = np.array([s.start for s in box], dtype=int)
    voxel_total = int(np.count_nonzero(mask))
    occupied_axes = [np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
                     for axis in range(3)]
    lower = starts + [a[0] for a in occupied_axes]
    upper = starts + [a[-1] + 1 for a in occupied_axes]
    touched = []
    for axis, name in enumerate("zyx"):
        if lower[axis] == 0:
            touched.append(name + "_low")
        if upper[axis] == shape[axis]:
            touched.append(name + "_high")
    planes = []
    anchors = {}
    source_z = []
    for local_z in np.flatnonzero(mask.any(axis=(1, 2))):
        yx = np.argwhere(mask[local_z])
        # Do not return an arithmetic centroid that may fall outside the mask.
        center = yx.mean(axis=0)
        anchor = yx[np.argmin(((yx - center) ** 2).sum(axis=1))] + starts[1:]
        z = int(local_z + starts[0])
        original_z = int(z + origin[0])
        source_z.append(original_z)
        anchors[str(original_z)] = [int(anchor[1] + origin[2]),
                                    int(anchor[0] + origin[1])]
        local_mask = mask[local_z]
        plane_raw = raw[box][local_z]
        plane_events = events[box][local_z]
        # Same-Z ROI reference is descriptive, not an independent noise sample.
        other = ~local_mask
        raw_mean = float(plane_raw[local_mask].mean())
        event_fraction = float(plane_events[local_mask].mean())
        planes.append({
            "z_local": z, "z_source": original_z,
            "voxels": int(local_mask.sum()),
            "raw_mean": raw_mean,
            "raw_event_fraction": event_fraction,
            "raw_mean_minus_bbox_complement": (
                raw_mean - float(plane_raw[other].mean()) if other.any() else None),
            "raw_event_fraction_minus_bbox_complement": (
                event_fraction - float(plane_events[other].mean()) if other.any() else None),
            "plane_removed_by_contrast": False,
        })
    maximum = np.argmax(np.where(mask, density[box], -np.inf))
    representative = np.array(np.unravel_index(maximum, mask.shape)) + starts
    bounds = [int(v) for pair in zip(lower + origin, upper + origin) for v in pair]
    extrapolated = (cp_z_bounds is not None and
                    (lower[0] < cp_z_bounds[0] or upper[0] > cp_z_bounds[1]))
    flags = []
    if touched:
        flags.append("open_native_roi_boundary")
    if len(cp_ids) > 1:
        flags.append("shared_density_family_split_by_cp_xy_instances")
    return {
        "label_id": int(label_id), "density_family_id": int(family_id),
        "cp_xy_label_id": int(cp_id), "family_cp_xy_ids": [int(x) for x in cp_ids],
        "status": "review" if touched else "ready_for_support",
        "requires_identity_review": bool(touched),
        "cp_identity_extended_by_native_evidence_beyond_projection_z": bool(extrapolated),
        "voxel_count": int(voxel_total),
        "bounds_zyx_half_open": bounds,
        "bounds_local_zyx_half_open": [int(v) for p in zip(lower, upper) for v in p],
        "representative_xyz": [int(v) for v in (representative + origin)[::-1]],
        "plane_anchors": anchors, "source_z_planes": source_z,
        "open_boundary_faces": touched,
        "raw_per_plane_evidence": planes,
        "flags": flags,
    }


def _axial_search(raw, events, cp, cp_id, spacing, z_scope, cfg):
    """A CP-shaped search envelope and neighboring-Z contrast events.

    This is a descriptive local signal/background comparison, not an IID
    significance test. External Z planes may include other cells; the robust
    reference assumes fewer than half are dominated by the same positive event.
    A crop without external Z observations remains axially unresolved.
    """
    sx, sy, sz = spacing
    distance = ndi.distance_transform_edt(cp != cp_id, sampling=(sy, sx))
    padding = cfg["xy_envelope_padding_um"]
    envelope = distance <= padding
    ring = ((distance > padding + cfg["xy_background_gap_um"]) &
            (distance <= padding + cfg["xy_background_gap_um"] + cfg["xy_background_width_um"]))
    record = {"cp_xy_id": cp_id, "envelope_area_pixels": int(envelope.sum()),
              "background_ring_area_pixels": int(ring.sum()),
              "xy_envelope_padding_um": padding,
              "xy_background_gap_um": cfg["xy_background_gap_um"],
              "xy_background_width_um": cfg["xy_background_width_um"],
              "cp_source_z_bounds_local": list(z_scope)}
    if not ring.any():
        record.update(status="insufficient_spatial_background", associated_events_local=[list(z_scope)],
                      other_axial_events_for_investigation=[], endpoint_evidence_available=False)
        return envelope, [tuple(z_scope)], record
    mean_delta = raw[:, envelope].mean(axis=1) - raw[:, ring].mean(axis=1)
    event_delta = events[:, envelope].mean(axis=1) - events[:, ring].mean(axis=1)
    sigma_z = 0.0 if raw.shape[0] == 1 else cfg["density_sigma_um"] / sz
    smooth = (event_delta if sigma_z == 0 else
              ndi.gaussian_filter1d(event_delta, sigma_z, mode="nearest", truncate=cfg["gaussian_truncate"]))
    external = np.ones(raw.shape[0], bool)
    external[slice(*z_scope)] = False
    record.update(raw_mean_contrast_per_z=mean_delta.tolist(),
                  raw_event_fraction_contrast_per_z=event_delta.tolist(),
                  neighbor_aggregated_event_contrast_per_z=smooth.tolist(),
                  axial_sigma_native_planes=sigma_z,
                  external_reference_z_count=int(external.sum()),
                  reference_status="local descriptive background; correlated observations")
    if np.count_nonzero(external) < max(3, int(np.ceil(2 * sigma_z))):
        record.update(status="insufficient_external_z_reference", associated_events_local=[list(z_scope)],
                      other_axial_events_for_investigation=[], endpoint_evidence_available=False)
        # Retain all actually observed native layers for boundary expansion;
        # this branch never claims that the crop's endpoints are complete.
        return envelope, [(0, raw.shape[0])], record
    median = float(np.median(smooth[external]))
    mad = float(np.median(np.abs(smooth[external] - median)) * 1.4826)
    quantization_floor = 1.0 / max(int(envelope.sum()), int(ring.sum()))
    scale = max(mad, quantization_floor)
    positive_reference_fraction = float(np.mean(smooth[external] > 0))
    record.update(external_contrast_median=median, external_contrast_mad=mad,
                  external_positive_contrast_fraction=positive_reference_fraction,
                  contrast_quantization_floor=quantization_floor)
    if median > 2 * scale and positive_reference_fraction > 0.8:
        record.update(status="positive_external_reference_requires_expansion",
                      associated_events_local=[[0, raw.shape[0]]],
                      other_axial_events_for_investigation=[], endpoint_evidence_available=False)
        return envelope, [(0, raw.shape[0])], record
    cutoff = max(0.0, median + cfg["axial_background_mad_multiplier"] * scale)
    runs, _ = ndi.label(smooth > cutoff)
    associated = []
    others = []
    for interval in ndi.find_objects(runs):
        lo, hi = interval[0].start, interval[0].stop
        evidence = {"bounds_z_local_half_open": [lo, hi],
                    "peak_neighbor_aggregated_contrast": float(smooth[lo:hi].max()),
                    "status": "raw_axial_contrast_fluctuation", "creates_nucleus": False,
                    "requires_independent_spatial_evidence": True,
                    "eligible_for_model_dispatch": False}
        if lo < z_scope[1] and hi > z_scope[0]:
            associated.append((lo, hi))
        else:
            others.append(evidence)
    record.update(status="observed_axial_events", external_contrast_median=median,
                  external_contrast_mad=mad, contrast_quantization_floor=quantization_floor,
                  event_contrast_cutoff=cutoff, associated_events_local=[list(x) for x in associated],
                  other_axial_events_for_investigation=others, endpoint_evidence_available=True)
    record["other_axial_events_are_independent_candidates"] = False
    return envelope, associated, record


def recover_native_3d(raw_zyx, spacing_xyz, xy_labels, origin_zyx=(0, 0, 0),
                      config=None, *, diagnostics=False):
    """Return native local labels + source anchors, family/edge evidence.

    raw_zyx: finite original DAPI values on the unmodified native grid.
    spacing_xyz: actual positive physical sampling (um); Z may be None only
      for genuine one-plane input, which remains dimension=2 and review.
    xy_labels: nonnegative integer CP labels on the same unresized XY grid.
    origin_zyx: original source origin of raw_zyx, no coordinate inference.
    config: explicitly recorded development density operator and CP source-Z
      provenance. sigma is a concentration-measurement bandwidth, never an
      expected nuclear diameter or an object-size exclusion.

    Native local density uses raw>per-plane-median events with a calibrated
    Gaussian kernel. A local density excess domain is labelled in 3D first.
    A family with one CP XY ID stays one event despite chromatin micropeaks.
    A family with several CP IDs is split using one actual density maximum
    per CP ID as watershed markers. Unmatched families are investigation
    positions, never assigned to a nearby nucleus. No raw single-plane sign
    veto, hole filling, minimum volume, or fixed-center Z endpoint rule.
    """
    cfg = dict(DEFAULTS)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError("Unknown recovery config: " + ", ".join(sorted(unknown)))
        cfg.update(config)
    raw = np.asarray(raw_zyx)
    cp = np.asarray(xy_labels)
    origin_input = np.asarray(origin_zyx)
    if (origin_input.shape != (3,) or not np.issubdtype(origin_input.dtype, np.integer)
            or np.any(origin_input < 0)):
        raise ValueError("origin_zyx must contain three nonnegative source indices")
    origin = origin_input.astype(int)
    if raw.ndim != 3 or any(n < 1 for n in raw.shape):
        raise ValueError("raw_zyx must be a nonempty native ZYX array")
    if cp.shape != raw.shape[1:] or not np.issubdtype(cp.dtype, np.integer) or np.any(cp < 0):
        raise ValueError("xy_labels must be nonnegative integer labels on matching native XY")
    if not np.isfinite(raw).all() or origin.shape != (3,):
        raise ValueError("Finite native DAPI and a ZYX source origin are required")
    if len(spacing_xyz) != 3:
        raise ValueError("Actual XYZ calibration is required")
    sx, sy, sz = spacing_xyz
    if sx is None or sy is None or min(sx, sy) <= 0 or not np.isfinite([sx, sy]).all():
        raise ValueError("Actual positive XY calibration is required")
    single = raw.shape[0] == 1
    if not single and (sz is None or not np.isfinite(sz) or sz <= 0):
        raise ValueError("Actual positive Z calibration is required for 3D")
    width = float(cfg["density_sigma_um"])
    cutoff = float(cfg["density_robust_score"])
    if width <= 0 or not np.isfinite(width) or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Positive finite development density bandwidth and score required")
    if cfg["connectivity"] not in (1, 2, 3):
        raise ValueError("Connectivity must be 1, 2, or 3")
    if not np.isfinite(cfg["gaussian_truncate"]) or cfg["gaussian_truncate"] <= 0:
        raise ValueError("Gaussian truncation must be positive and finite")
    for name in ("xy_envelope_padding_um", "xy_background_gap_um", "xy_background_width_um",
                 "axial_background_mad_multiplier"):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(name + " must be positive and finite")
    z_scope = cfg["cp_source_z_bounds_local"]
    if z_scope is None:
        z_scope = [0, raw.shape[0]]
    if (len(z_scope) != 2 or not all(isinstance(z, (int, np.integer)) for z in z_scope)
            or not 0 <= z_scope[0] < z_scope[1] <= raw.shape[0]):
        raise ValueError("CP source Z bounds must be a valid local half-open range")
    cp_ids_present = [int(i) for i in np.unique(cp) if i]
    sigma = np.array([0.0 if single else width / sz, width / sy, width / sx])
    baseline = np.median(raw, axis=(1, 2))
    events = raw > baseline[:, None, None]
    density = ndi.gaussian_filter(events.astype(np.float32), sigma,
                                  mode="nearest", truncate=cfg["gaussian_truncate"])
    med = np.median(density, axis=(1, 2))
    mad = np.median(np.abs(density - med[:, None, None]), axis=(1, 2)) * 1.4826
    floor = _gaussian_peak_weight(sigma, cfg["gaussian_truncate"])
    noise = np.maximum(mad, floor)
    domain = density > (med + cutoff * noise)[:, None, None]
    searches = {}
    allowed = np.zeros(raw.shape, bool)
    for cp_id in cp_ids_present:
        envelope, intervals, profile = _axial_search(raw, events, cp, cp_id, spacing_xyz, z_scope, cfg)
        searches[cp_id] = (envelope, intervals, profile)
        for lo, hi in intervals:
            allowed[lo:hi] |= envelope[None]
    outside_search_voxels = int(np.count_nonzero(domain & ~allowed))
    domain &= allowed
    del allowed
    structure = ndi.generate_binary_structure(3, int(cfg["connectivity"]))
    families, family_n = ndi.label(domain, structure=structure, output=np.uint32)
    boxes = ndi.find_objects(families)
    family_cp = {}
    association_candidates = {}
    minimum_overlap = float(cfg["min_cp_projected_overlap_fraction"])
    if not 0 < minimum_overlap <= 1:
        raise ValueError("CP projected-overlap fraction must be in (0,1]")
    for cp_id in cp_ids_present:
        cp_area = int(np.count_nonzero(cp == cp_id))
        for family_id in np.unique(families[z_scope[0]:z_scope[1], cp == cp_id]):
            if not family_id:
                continue
            box = boxes[int(family_id)-1]
            scoped_box = (slice(max(z_scope[0], box[0].start), min(z_scope[1], box[0].stop)),
                          box[1], box[2])
            native_region = families[scoped_box] == family_id
            projection = native_region.any(axis=0)
            cp_mask = cp[box[1:]] == cp_id
            overlap = int(np.count_nonzero(projection & cp_mask))
            projection_area = int(projection.sum())
            coverage = overlap / cp_area
            original_events = int(np.count_nonzero(native_region & cp_mask[None] & events[scoped_box]))
            matched = coverage >= minimum_overlap and original_events > 0
            evidence = {
                "family_id": int(family_id), "cp_xy_id": cp_id,
                "cp_source_z_bounds_local": list(z_scope),
                "cp_projected_area_pixels": cp_area,
                "family_projected_area_pixels_in_cp_source_z": projection_area,
                "overlap_projected_area_pixels": overlap,
                "cp_projected_overlap_fraction": float(coverage),
                "family_projected_overlap_fraction": float(overlap / projection_area),
                "original_positive_event_voxels_in_overlap": original_events,
                "matched_for_provisional_identity": bool(matched),
                "threshold_status": "development_geometric_association_not_biological_validation",
            }
            association_candidates.setdefault(int(family_id), []).append(evidence)
            if matched:
                family_cp.setdefault(int(family_id), []).append(cp_id)
    labels = np.zeros(raw.shape, dtype=np.uint32)
    objects = []
    unmatched = []
    next_id = 1
    for family_id, box in enumerate(boxes, 1):
        if box is None:
            continue
        region = families[box] == family_id
        cp_ids = family_cp.get(family_id, [])
        if not cp_ids:
            peak = np.argmax(np.where(region, density[box], -np.inf))
            point = np.array(np.unravel_index(peak, region.shape)) + [b.start for b in box]
            unmatched.append({
                "family_id": family_id, "source_zyx": (point + origin).tolist(),
                "voxel_count": int(region.sum()), "status": "investigation",
                "creates_nucleus": False,
                "cp_association_candidates": association_candidates.get(family_id, []),
            })
            continue
        if len(cp_ids) == 1:
            assigned = region.astype(np.uint32)
        else:
            markers = np.zeros(region.shape, np.uint32)
            cp_crop = cp[box[1:]]
            for marker_id, cp_id in enumerate(cp_ids, 1):
                eligible = region & (cp_crop == cp_id)[None]
                eligible[:max(0, z_scope[0]-box[0].start)] = False
                eligible[max(0, z_scope[1]-box[0].start):] = False
                peak = np.argmax(np.where(eligible, density[box], -np.inf))
                markers[np.unravel_index(peak, region.shape)] = marker_id
            assigned = watershed(-density[box], markers, mask=region,
                                 connectivity=structure).astype(np.uint32)
        for marker_id, cp_id in enumerate(cp_ids, 1):
            mask = assigned == marker_id
            envelope, intervals, profile = searches[cp_id]
            mask &= envelope[box[1:]][None]
            z_allowed = np.zeros(mask.shape[0], bool)
            for lo, hi in intervals:
                z_allowed[max(0, lo-box[0].start):max(0, hi-box[0].start)] = True
            mask &= z_allowed[:, None, None]
            if not mask.any():
                continue
            labels[box][mask] = next_id
            objects.append(_region_record(mask, raw, events, density, box, origin,
                           next_id, family_id, cp_id, cp_ids,
                           cfg["cp_source_z_bounds_local"], raw.shape))
            objects[-1]["cp_association_evidence"] = next(
                e for e in association_candidates[family_id] if e["cp_xy_id"] == cp_id)
            obj = objects[-1]
            envelope_edge = envelope & ~ndi.binary_erosion(envelope, border_value=1)
            touches_envelope = bool(np.any(mask & envelope_edge[box[1:]][None]))
            obj["xy_search_envelope_boundary_touched"] = touches_envelope
            obj["axial_endpoint_evidence_available"] = profile["endpoint_evidence_available"]
            obj["associated_axial_events_local"] = [list(i) for i in intervals]
            if touches_envelope:
                # The bounded density mask has incomplete lateral extent;
                # this alone does not invalidate its resolved CP identity.
                obj["flags"].append("xy_search_envelope_boundary_touched")
            if not profile["endpoint_evidence_available"]:
                obj["flags"].append("axial_endpoints_require_more_native_context")
                obj["requires_identity_review"] = True
            if obj["requires_identity_review"]:
                obj["status"] = "review"
            next_id += 1
    per_cp = {}
    for obj in objects:
        per_cp.setdefault(str(obj["cp_xy_label_id"]), []).append(obj["label_id"])
    for obj in objects:
        peers = per_cp[str(obj["cp_xy_label_id"])]
        if len(peers) > 1:
            obj["flags"].append("same_cp_projection_has_multiple_native_density_events")
            obj["same_cp_other_event_ids"] = [i for i in peers if i != obj["label_id"]]
        if single:
            obj["flags"].append("single_plane_axial_identity_unassessed")
            obj["requires_identity_review"] = True
            obj["status"] = "review"
    result = {
        "labels": labels, "objects": objects,
        "unmatched_families": unmatched,
        "axial_profiles": [s[2] for s in searches.values()],
        "metrics": {
            "method": "cp_xy_envelope_native_axial_events_v2",
            "measurement_dimension": 2 if single else 3,
            "spacing_xyz_um": list(spacing_xyz), "origin_zyx": origin.tolist(),
            "parameters": cfg, "density_sigma_zyx_pixels": sigma.tolist(),
            "raw_event_baseline_per_z": baseline.tolist(),
            "density_median_per_z": med.tolist(), "density_scale_per_z": noise.tolist(),
            "density_family_count_not_nuclei": int(family_n),
            "cp_associated_region_count_not_nuclei": len(objects),
            "unmatched_family_count_not_nuclei": len(unmatched),
            "unresolved_cp_xy_ids": [i for i in cp_ids_present if str(i) not in per_cp],
            "cp_association_z_scope_local": list(z_scope),
            "native_density_voxels_outside_cp_search_unassigned": outside_search_voxels,
            "threshold_status": "development_not_biologically_validated",
            "raw_single_plane_contrast_is_gate": False,
            "fixed_nuclear_diameter": None, "native_grid_preserved": True,
            "support_used_for_nuclear_identity": False,
            "limitations": [
                "Density families are segmentation hypotheses, not nuclei.",
                "Axial event intervals depend on local background and neighboring-Z contrast, with uncertain weak tails.",
                "XY search-envelope contact records incomplete lateral extent; it alone does not veto counting identity.",
                "The median/MAD score is a local contrast index, not a calibrated false-positive probability.",
            ],
        },
    }
    if diagnostics:
        result.update(density=density, families=families)
    return result
