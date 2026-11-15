"""Native nucleus/body observations and explicit, auditable cell identities.

I/O, detection, and scientific match thresholds belong to callers. These pure
functions never compare cross-kind mask IoU and never require Cellpose for a
body observation to count. Label integers are scoped by artifact_id.
"""
from copy import deepcopy
import math
import numpy as np


def _binary(mask):
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 3 or not value.size or not value.any():
        raise ValueError('An observed nonempty native ZYX mask is required')
    return value


def _origin(value):
    if len(value) != 3 or any(type(v) not in (int, np.int32, np.int64) or v < 0 for v in value):
        raise ValueError('Original nonnegative integer ZYX origin required')
    return tuple(map(int, value))


def nucleus_geometry(mask, origin_zyx, spacing_xyz, representative_xyz):
    """Compute one nucleus core once, without a fixed radius or percentile.

    The core consists of maximal physical interior-distance mask voxels. It is
    descriptive observed geometry, including any explicitly recorded truncation.
    """
    from scipy import ndimage as ndi
    mask = _binary(mask)
    origin = _origin(origin_zyx)
    sx, sy, sz = spacing_xyz
    if any(v is None or not math.isfinite(v) or v <= 0 for v in (sx, sy)):
        raise ValueError('Actual positive XY calibration required')
    if sz is None:
        if len(mask) != 1:
            raise ValueError('Unknown Z calibration is only valid for an actual single plane')
        distance = ndi.distance_transform_edt(np.pad(mask[0], 1), sampling=(sy, sx))[1:-1, 1:-1][None]
    else:
        if not math.isfinite(sz) or sz <= 0:
            raise ValueError('Actual positive Z calibration required')
        distance = ndi.distance_transform_edt(np.pad(mask, 1), sampling=(sz, sy, sx))[1:-1, 1:-1, 1:-1]
    maximum = float(distance.max())
    core = (distance == maximum) & mask
    return {'mask': mask, 'origin_zyx': origin, 'core': core,
            'core_radius_um': maximum, 'representative_xyz': list(representative_xyz),
            'z_planes': (np.flatnonzero(mask.any(axis=(1, 2)))+origin[0]).tolist()}


def enclosure_evidence(nucleus, body_mask, body_origin_zyx, body_envelope, envelope_origin_zyx):
    """Directional nucleus-in-body evidence, with no automatic match decision."""
    body = _binary(body_mask)
    envelope = _binary(body_envelope)
    body_origin = _origin(body_origin_zyx)
    envelope_origin = _origin(envelope_origin_zyx)
    mask, core, origin = nucleus['mask'], nucleus['core'], nucleus['origin_zyx']
    enclosed = np.zeros(mask.shape, bool)
    lo = tuple(max(origin[a], envelope_origin[a]) for a in range(3))
    hi = tuple(min(origin[a]+mask.shape[a], envelope_origin[a]+envelope.shape[a]) for a in range(3))
    if all(a < b for a, b in zip(lo, hi)):
        destination = tuple(slice(lo[a]-origin[a], hi[a]-origin[a]) for a in range(3))
        source = tuple(slice(lo[a]-envelope_origin[a], hi[a]-envelope_origin[a]) for a in range(3))
        enclosed[destination] = envelope[source]
    inside = enclosed & mask
    body_planes = (np.flatnonzero(body.any(axis=(1, 2)))+body_origin[0]).tolist()
    shared = sorted(set(body_planes) & set(nucleus['z_planes']))
    representative = tuple(int(math.floor(v)) for v in reversed(nucleus['representative_xyz']))
    local = tuple(representative[a]-envelope_origin[a] for a in range(3))
    point_inside = all(0 <= local[a] < envelope.shape[a] for a in range(3)) and bool(envelope[local])
    planes = []
    for z in np.flatnonzero(mask.any(axis=(1, 2))):
        total = int(mask[z].sum())
        amount = int(inside[z].sum())
        source_z = int(z+origin[0])
        planes.append({'source_z': source_z, 'nucleus_voxels': total,
                       'nucleus_voxels_in_body_envelope': amount,
                       'nucleus_fraction_in_body_envelope': amount/total,
                       'body_observed_on_this_z': source_z in body_planes})
    return {'decision': 'evidence_only', 'nucleus_voxels': int(mask.sum()),
            'nucleus_voxels_in_body_envelope': int(inside.sum()),
            'nucleus_fraction_in_body_envelope': int(inside.sum())/int(mask.sum()),
            'nucleus_representative_xyz': list(nucleus['representative_xyz']),
            'nucleus_representative_inside_body_envelope': bool(point_inside),
            'core_definition': 'maximal physical interior-distance voxels of the observed nucleus mask',
            'nucleus_core_radius_um': nucleus['core_radius_um'], 'nucleus_core_voxels': int(core.sum()),
            'nucleus_core_fraction_in_body_envelope': int((core & enclosed).sum())/int(core.sum()),
            'nucleus_z_planes': list(nucleus['z_planes']), 'body_z_planes': body_planes,
            'shared_observed_z_planes': shared, 'per_z': planes,
            'cross_kind_iou_used': False, 'centroid_distance_used_as_identity': False}


def observation_from_nucleus(point):
    value = int(point['label_id'])
    observation = {'id': f'nucleus-{value}', 'kind': 'nucleus', 'artifact_id': 'dapi_nuclei',
            'label_id': value, 'bounds_zyx': deepcopy(point['native_identity']['bounds_zyx']),
            'plane_anchors': deepcopy(point['plane_anchors']),
            'representative_xyz': [point['x'], point['y'], point['z']],
            'decision': point['status'], 'reasons': list(point['native_identity'].get('flags', [])),
            'metrics': {'support': deepcopy(point.get('support', {})),
                        'native_identity': deepcopy(point['native_identity'])}}
    if point.get('review_hint') == 'likely_astrocyte' and point['status'] == 'review':
        observation['review_hint'] = 'likely_astrocyte'
    return observation


def geometric_association(evidence):
    """Approved geometric candidate rule; representative real masks still validate it.

    Whole-mask containment fractions remain descriptive. The representative,
    maximal interior core and actual body-positive Z intersection establish a
    candidate; the graph resolver separately checks one-to-one uniqueness.
    """
    representative = evidence['nucleus_representative_inside_body_envelope']
    core_fraction = evidence['nucleus_core_fraction_in_body_envelope']
    representative_z = int(math.floor(evidence['nucleus_representative_xyz'][2]))
    body_present = representative_z in evidence['body_z_planes']
    if representative and core_fraction == 1. and body_present:
        return {'decision': 'match', 'reasons': [
            'nuclear_representative_and_entire_interior_core_inside_body_envelope',
            'actual_body_signal_present_on_nuclear_representative_z']}
    if not representative and core_fraction == 0.:
        return {'decision': 'no_match', 'reasons': ['nuclear_representative_and_core_outside_body_envelope']}
    reasons = ['partial_nuclear_representative_or_core_enclosure']
    if not body_present:
        reasons.append('no_actual_body_signal_on_nuclear_representative_z')
    return {'decision': 'ambiguous', 'reasons': reasons}


def _validate_observations(observations):
    ids, labels = set(), set()
    for item in observations:
        required = {'id', 'artifact_id', 'label_id', 'kind', 'decision', 'plane_anchors',
                    'representative_xyz', 'bounds_zyx'}
        if not isinstance(item, dict) or not required <= item.keys():
            raise ValueError('Observation is missing required native-identity fields')
        if (not isinstance(item['id'], str) or not item['id'] or
                not isinstance(item['artifact_id'], str) or not item['artifact_id'] or
                isinstance(item['label_id'], (bool, np.bool_)) or
                not isinstance(item['label_id'], (int, np.integer)) or item['label_id'] <= 0):
            raise ValueError('Observation IDs, artifact namespace and positive label ID are required')
        if item['id'] in ids or (item['artifact_id'], item['label_id']) in labels:
            raise ValueError('Observation identity and artifact-scoped label must be unique')
        if item['kind'] not in ('nucleus', 'body') or item['decision'] not in ('accepted', 'review', 'excluded'):
            raise ValueError('Invalid observation kind or decision')
        if not item['plane_anchors']:
            raise ValueError('An automatic observation needs actual native Z anchors')
        ids.add(item['id']); labels.add((item['artifact_id'], item['label_id']))


def unify_cells(observations, relations=()):
    """Resolve explicit one-to-one matches; do not invent scientific thresholds.

    relations contain nucleus_id/body_id, decision match|ambiguous|no_match,
    and evidence. Competing matches remain distinct review cells. An unmatched
    accepted body remains accepted and counts without a DAPI/Cellpose gate.
    """
    observations = [deepcopy(o) for o in observations]
    _validate_observations(observations)
    indexed = {o['id']: o for o in observations}
    neighbors = {o['id']: set() for o in observations}
    candidate_neighbors = {o['id']: set() for o in observations}
    uncertain = set()
    relations = [deepcopy(r) for r in relations]
    seen_relations = set()
    for relation in relations:
        if not isinstance(relation, dict) or not {'nucleus_id', 'body_id', 'decision'} <= relation.keys():
            raise ValueError('Association relation requires both observation IDs and an explicit decision')
        n, b = relation['nucleus_id'], relation['body_id']
        if not isinstance(n, str) or not isinstance(b, str):
            raise ValueError('Association IDs must be observation-ID strings')
        if (n, b) in seen_relations:
            raise ValueError('Duplicate or conflicting association relation pair')
        seen_relations.add((n, b))
        if n not in indexed or b not in indexed or indexed[n]['kind'] != 'nucleus' or indexed[b]['kind'] != 'body':
            raise ValueError('Relations must reference one existing nucleus and one existing body')
        if relation['decision'] == 'match':
            neighbors[n].add(b); neighbors[b].add(n)
            candidate_neighbors[n].add(b); candidate_neighbors[b].add(n)
        elif relation['decision'] == 'ambiguous':
            uncertain.update((n, b))
            candidate_neighbors[n].add(b); candidate_neighbors[b].add(n)
        elif relation['decision'] != 'no_match':
            raise ValueError('Scientific association decisions must be explicit')
    for key, candidates in neighbors.items():
        if len(candidates) > 1:
            uncertain.add(key); uncertain.update(candidates)
    pending = list(uncertain)
    while pending:
        key = pending.pop()
        for peer in candidate_neighbors[key]-uncertain:
            uncertain.add(peer); pending.append(peer)
    groups, consumed = [], set()
    for item in observations:
        key = item['id']
        if key in consumed:
            continue
        group = [item]
        if key not in uncertain and len(neighbors[key]) == 1:
            peer = next(iter(neighbors[key]))
            if peer not in uncertain and neighbors[peer] == {key}:
                group.append(indexed[peer])
        consumed.update(o['id'] for o in group)
        groups.append(group)
    cells = []
    for group in groups:
        nucleus = next((o for o in group if o['kind'] == 'nucleus'), None)
        body = next((o for o in group if o['kind'] == 'body'), None)
        primary = nucleus or body
        cell_id = 'cell-'+primary['id']
        ambiguous = any(o['id'] in uncertain for o in group)
        decisions = {o['decision'] for o in group}
        status = ('review' if ambiguous else 'accepted' if 'accepted' in decisions else
                  'review' if 'review' in decisions else 'excluded')
        anchors, sources = {}, {}
        # Prefer the actual body intersection where available, and retain
        # nuclear-only native Z intersections outside that body evidence.
        for item in sorted(group, key=lambda o: o['kind'] == 'body'):
            for z, anchor in item['plane_anchors'].items():
                anchors[z] = list(anchor)
                sources[z] = {'kind': 'observation', 'observation_id': item['id']}
        representative = body or nucleus
        x, y, z = representative['representative_xyz']
        if str(z) not in anchors:
            z = int(min(anchors, key=lambda value: abs(int(value)-int(z))))
        x, y = anchors[str(z)]
        group_ids = {o['id'] for o in group}
        relevant = [r for r in relations if r['nucleus_id'] in group_ids or r['body_id'] in group_ids]
        cell = {'id': cell_id, 'cell_id': cell_id, 'label_artifact_id': primary['artifact_id'],
            'label_id': primary['label_id'], 'source': 'dapi_support' if len(group) == 2 else
            'support_body' if body else 'auto', 'status': status, 'x': x, 'y': y, 'z': z,
            'plane_anchors': anchors, 'plane_anchor_sources': sources,
            'source_observations': group,
            'association': {'status': 'ambiguous' if ambiguous else 'one_to_one' if len(group) == 2 else 'unmatched',
                            'method': 'explicit_native_nucleus_body_evidence', 'relations': relevant,
                            'competing_observation_ids': sorted(set().union(*(candidate_neighbors[i] for i in group_ids))-group_ids)},
            'support': deepcopy(primary.get('metrics', {}).get('support', {})),
            'count_basis_observation_ids': [o['id'] for o in group if o['decision'] == 'accepted']}
        if (status == 'review' and not ambiguous and nucleus is not None and
                nucleus.get('review_hint') == 'likely_astrocyte' and
                all(o['kind'] == 'nucleus' or o['decision'] == 'accepted' for o in group)):
            cell['review_hint'] = 'likely_astrocyte'
        cells.append(cell)
    return sorted(cells, key=lambda cell: cell['id'])
