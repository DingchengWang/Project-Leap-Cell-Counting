"""Sparse native NPY writes and object-local metadata; no whole-label rescans."""
from pathlib import Path
import numpy as np


class NativeLabelStore:
    def __init__(self, path, shape):
        self.path = Path(path)
        # NPY header + sparse file allocation: untouched source voxels remain zero.
        self.labels = np.lib.format.open_memmap(self.path, mode='w+', dtype=np.uint32, shape=shape)
        self.objects = {}
        self.next_id = 1
        self.audit = []

    def add(self, local_labels, objects, origin, window_id):
        region = tuple(slice(int(origin[a]), int(origin[a])+local_labels.shape[a]) for a in range(3))
        current = self.labels[region]
        assigned = {}
        for obj in objects:
            value = obj['label_id']
            mask = local_labels == value
            size = int(mask.sum())
            overlap_ids, overlap_sizes = np.unique(current[mask], return_counts=True)
            matches = []
            touched = []
            overlaps = []
            for old, overlap in zip(overlap_ids, overlap_sizes):
                if not old:
                    continue
                old = int(old)
                touched.append(old)
                shared_old = int(np.count_nonzero(current == old))
                overlaps.append((old, overlap/size, overlap/max(1, shared_old)))
                if overlap/size >= .5 and overlap/max(1, shared_old) >= .5:
                    matches.append(old)
            if len(matches) == 1 and any(old != matches[0] and (new_fraction >= .25 or old_fraction >= .25)
                                         for old, new_fraction, old_fraction in overlaps):
                # A dominant match does not resolve a significant second
                # identity. Preserve the established IDs and expose ambiguity.
                matches = []
            if len(matches) == 1:
                label_id = matches[0]
                record = self.objects[label_id]
                context_parents = set(obj.get('context_expansion_parent_windows', []))
                if not obj.get('model_xy_boundary', {}).get('internal_faces'):
                    resolved = context_parents.intersection(record['source_windows'])
                    record.setdefault('resolved_model_boundary_windows', []).extend(sorted(resolved))
                record['source_windows'].append(window_id)
                record['local_objects'].append(obj)
            else:
                label_id = self.next_id
                self.next_id += 1
                record = {'label_id': label_id, 'source_windows': [window_id],
                          'local_objects': [obj], 'bounds_zyx': None, 'flags': []}
                self.objects[label_id] = record
                if touched:
                    record['flags'].append('overlap_identity_ambiguous')
                    for old in touched:
                        self.objects[old]['flags'].append('overlap_identity_ambiguous')
            # An overlap owned by another unresolved identity is retained once.
            writable = mask & ((current == 0) | (current == label_id))
            current[writable] = label_id
            if not writable.any():
                self.audit.append({'window_id': window_id, 'local_label_id': int(value),
                                   'reason': 'candidate_fully_overlaps_existing_identity',
                                   'existing_ids': touched, 'object_evidence': obj})
                if not matches:
                    self.objects.pop(label_id)
                assigned[int(value)] = None
                continue
            coords = np.where(writable)
            bounds = [[int(coords[a].min()+origin[a]), int(coords[a].max()+origin[a]+1)] for a in range(3)]
            if record['bounds_zyx'] is None:
                record['bounds_zyx'] = bounds
            else:
                record['bounds_zyx'] = [[min(a[0], b[0]), max(a[1], b[1])]
                                        for a, b in zip(record['bounds_zyx'], bounds)]
            assigned[int(value)] = label_id
        return assigned

    def finish(self):
        self.labels.flush()
        result = {'storage_dtype': 'uint32', 'storage_layout': 'native sparse NPY',
                  'logical_file_bytes': self.path.stat().st_size,
                  'allocated_file_bytes': self.path.stat().st_blocks*512,
                  'whole_volume_remapping_or_unique_scan': False,
                  'stored_identity_count': len(self.objects),
                  'overlap_rule': 'reciprocal actual-mask overlap >= 0.5 within shared native ROI',
                  'unresolved_overlap_candidates': self.audit}
        return result

    def close(self):
        self.labels._mmap.close()
        self.path.chmod(0o444)
