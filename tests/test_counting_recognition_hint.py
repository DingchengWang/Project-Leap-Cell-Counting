"""Presentation hints preserve the accepted v4 automatic-count decisions."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))
from leap_cell_counting.efficient import _point
from leap_cell_counting.identity import observation_from_nucleus, unify_cells


class RecognitionHintTests(unittest.TestCase):
    def point(self, support='accepted', *, geometry=True, real_identity=False, overlap=False):
        labels = np.zeros((5, 19, 21), np.uint32)
        labels[1:4, 7:12, 8:14] = 1
        raw = np.ones(labels.shape, np.uint8)*2
        raw[labels > 0] = 20
        reader = SimpleNamespace(_counting_dapi=0,
            read_region=lambda _channel, z, y, x, **kwargs: raw[z, y, x])
        flags = ['xy_search_envelope_boundary_touched'] if geometry else []
        if real_identity: flags.append('axial_endpoints_require_more_native_context')
        record = {'label_id': 1, 'source_windows': [1], 'bounds_zyx': [[1, 4], [7, 12], [8, 14]],
            'flags': ['overlap_identity_ambiguous'] if overlap else [],
            'local_objects': [{'flags': flags, 'xy_search_envelope_boundary_touched': geometry,
                               'requires_identity_review': real_identity}]}
        evidence = {'decision': support, 'metrics': {}, 'reasons': [], 'review_hint': support == 'review'}
        with patch('leap_cell_counting.support.measure_support', return_value=evidence):
            return _point(reader, threading.Lock(), SimpleNamespace(labels=labels), record,
                          1, 'Biocytin', (.2, .2, 1.), lambda: None)

    def test_search_boundary_only_stays_unrecognized_and_gets_display_hint(self):
        point = self.point()
        self.assertEqual(point['status'], 'review')
        self.assertEqual(point['review_hint'], 'likely_astrocyte')
        self.assertFalse(point['native_identity']['local_objects'][0]['requires_identity_review'])
        self.assertNotIn('cell_type', point)
        cell = unify_cells([observation_from_nucleus(point)])[0]
        self.assertEqual(cell['status'], 'review')
        self.assertEqual(cell['review_hint'], 'likely_astrocyte')

    def test_other_identity_uncertainty_or_insufficient_support_has_no_hint(self):
        for options in ({'real_identity': True}, {'overlap': True}, {'support': 'review'}, {'support': 'excluded'}):
            with self.subTest(options=options):
                point = self.point(**options)
                self.assertNotIn('review_hint', point)
                self.assertEqual(point['status'], 'excluded' if options.get('support') == 'excluded' else 'review')
        self.assertEqual(self.point(geometry=False)['status'], 'accepted')

    def test_competing_cell_association_or_unresolved_body_blocks_hint(self):
        nucleus = observation_from_nucleus(self.point())
        body = {'id': 'body-2', 'kind': 'body', 'artifact_id': 'support_bodies', 'label_id': 2,
            'decision': 'accepted', 'reasons': [], 'metrics': {}, 'bounds_zyx': [[1, 4], [7, 12], [8, 14]],
            'plane_anchors': {'2': [10.5, 9.5]}, 'representative_xyz': [10.5, 9.5, 2]}
        relation = {'nucleus_id': nucleus['id'], 'body_id': body['id'], 'decision': 'ambiguous'}
        cells = unify_cells([nucleus, body], [relation])
        self.assertTrue(all('review_hint' not in cell for cell in cells))
        body['decision'] = 'review'; body['reasons'] = ['overlap_identity_ambiguous']
        relation['decision'] = 'match'
        cell = unify_cells([nucleus, body], [relation])[0]
        self.assertEqual(cell['status'], 'review')
        self.assertNotIn('review_hint', cell)

    def test_hint_changes_no_identity_status_or_automatic_count(self):
        point = self.point(); with_hint = observation_from_nucleus(point)
        without_hint = deepcopy(with_hint); without_hint.pop('review_hint')
        a, b = unify_cells([without_hint])[0], unify_cells([with_hint])[0]
        self.assertEqual((a['id'], a['status'], a['label_id']), (b['id'], b['status'], b['label_id']))
        self.assertEqual(a['count_basis_observation_ids'], b['count_basis_observation_ids'])
        self.assertEqual(a['plane_anchors'], b['plane_anchors'])


if __name__ == '__main__': unittest.main()
