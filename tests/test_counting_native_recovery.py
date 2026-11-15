"""Native identity/coordinate regressions; synthetic tests are not biology."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from leap_cell_counting.native_recovery import recover_native_3d


def fixture(two_z=False, negative_slice=False):
    rng = np.random.default_rng(47281)
    z, y, x = np.indices((81, 81, 81))
    cp = (((y[0] - 40) ** 2 + (x[0] - 40) ** 2) < 12 ** 2).astype(np.uint16)
    xy = ((y - 40) / 10) ** 2 + ((x - 40) / 10) ** 2
    cloud = ((xy + ((z - 20) / 9) ** 2 < 1) | (xy + ((z - 60) / 9) ** 2 < 1)
             if two_z else xy + ((z - 40) / 12) ** 2 < 1)
    raw = rng.poisson(2, z.shape).astype(np.float32)
    raw[cloud] += rng.poisson(4, int(cloud.sum()))
    if negative_slice:
        raw[40, 30:51, 30:51] = 0
    return raw, cp


class NativeRecoveryTests(unittest.TestCase):
    spacing = (.2, .2, .4)

    def test_micropeaks_inside_one_cp_remain_one_object(self):
        raw, cp = fixture()
        z, y, x = np.indices(raw.shape)
        raw += 25 * np.exp(-((z-38)**2/3 + (y-40)**2/4 + (x-35)**2/4))
        raw += 25 * np.exp(-((z-42)**2/3 + (y-40)**2/4 + (x-45)**2/4))
        result = recover_native_3d(raw, self.spacing, cp)
        self.assertEqual(len(result["objects"]), 1)
        self.assertGreater(result["labels"][35, 40, 40], 0)
        self.assertEqual(result["labels"][35, 40, 40], result["labels"][45, 40, 40])

    def test_same_xy_two_axial_clouds_have_distinct_ids(self):
        raw, cp = fixture(two_z=True)
        result = recover_native_3d(raw, self.spacing, cp,
                                 config={"cp_source_z_bounds_local": [11, 70]})
        self.assertEqual(len(result["objects"]), 2)
        low, high = result["labels"][20, 40, 40], result["labels"][60, 40, 40]
        self.assertGreater(low, 0)
        self.assertGreater(high, 0)
        self.assertNotEqual(low, high)
        self.assertFalse(result["labels"][35:46].any())

    def test_remote_z_cloud_does_not_receive_old_projection_identity(self):
        raw, cp = fixture(two_z=True)
        result = recover_native_3d(raw, self.spacing, cp,
                                 config={"cp_source_z_bounds_local": [12, 30]})
        self.assertGreater(result["labels"][20, 40, 40], 0)
        self.assertEqual(result["labels"][60, 40, 40], 0)
        self.assertTrue(result["axial_profiles"][0]["other_axial_events_for_investigation"])
        for event in result["axial_profiles"][0]["other_axial_events_for_investigation"]:
            self.assertFalse(event["eligible_for_model_dispatch"])
            self.assertTrue(event["requires_independent_spatial_evidence"])

    def test_two_cp_identities_in_one_connected_density_family_stay_separate(self):
        z, y, x = np.indices((81, 81, 81))
        left = ((y-40)/12)**2 + ((x-29)/12)**2 + ((z-40)/14)**2 < 1
        right = ((y-40)/12)**2 + ((x-51)/12)**2 + ((z-40)/14)**2 < 1
        raw = np.full(z.shape, 2, np.float32)
        raw[left | right] = 6
        cp = np.zeros((81, 81), np.uint16)
        cp[(y[0]-40)**2 + (x[0]-29)**2 < 9**2] = 1
        cp[(y[0]-40)**2 + (x[0]-51)**2 < 9**2] = 2
        result = recover_native_3d(raw, self.spacing, cp,
                                 config={"cp_source_z_bounds_local": [26, 55]})
        a, b = result["labels"][40, 40, 29], result["labels"][40, 40, 51]
        self.assertGreater(a, 0)
        self.assertGreater(b, 0)
        self.assertNotEqual(a, b)
        mapping = {o["label_id"]: o["cp_xy_label_id"] for o in result["objects"]}
        self.assertEqual((mapping[a], mapping[b]), (1, 2))

    def test_one_dark_raw_plane_does_not_split_coherent_event(self):
        raw, cp = fixture(negative_slice=True)
        result = recover_native_3d(raw, self.spacing, cp,
                                 config={"cp_source_z_bounds_local": [28, 53]})
        self.assertGreater(result["labels"][40, 40, 40], 0)
        self.assertEqual(result["labels"][35, 40, 40], result["labels"][45, 40, 40])
        record = next(p for o in result["objects"] for p in o["raw_per_plane_evidence"]
                      if p["z_local"] == 40)
        self.assertFalse(record["plane_removed_by_contrast"])

    def test_source_anchors_are_actual_mask_pixels_and_inputs_unchanged(self):
        raw, cp = fixture()
        raw_before, cp_before = raw.copy(), cp.copy()
        origin = (7, 100, 200)
        result = recover_native_3d(raw, self.spacing, cp, origin,
                                 {"cp_source_z_bounds_local": [28, 53]})
        for obj in result["objects"]:
            for z, (x, y) in obj["plane_anchors"].items():
                self.assertEqual(result["labels"][int(z)-7, y-100, x-200], obj["label_id"])
        np.testing.assert_array_equal(raw, raw_before)
        np.testing.assert_array_equal(cp, cp_before)

    def test_diagnostics_do_not_change_labels_or_evidence(self):
        raw, cp = fixture()
        plain = recover_native_3d(raw, self.spacing, cp)
        detailed = recover_native_3d(raw, self.spacing, cp, diagnostics=True)
        self.assertNotIn("density", plain)
        self.assertNotIn("families", plain)
        np.testing.assert_array_equal(plain.pop("labels"), detailed.pop("labels"))
        detailed.pop("density")
        detailed.pop("families")
        self.assertEqual(plain, detailed)

    def test_single_plane_keeps_unknown_z_and_requires_review(self):
        raw, cp = fixture()
        result = recover_native_3d(raw[40:41], (.2, .2, None), cp)
        self.assertEqual(result["metrics"]["measurement_dimension"], 2)
        self.assertIsNone(result["metrics"]["spacing_xyz_um"][2])
        self.assertTrue(result["objects"])
        self.assertTrue(all(o["requires_identity_review"] for o in result["objects"]))

    def test_missing_external_z_context_does_not_claim_closed_endpoints(self):
        raw, cp = fixture()
        result = recover_native_3d(raw[30:51], self.spacing, cp)
        self.assertFalse(result["axial_profiles"][0]["endpoint_evidence_available"])
        self.assertEqual(result["axial_profiles"][0]["external_reference_z_count"], 0)
        self.assertTrue(all(o["requires_identity_review"] for o in result["objects"]))

    def test_positive_external_z_reference_requests_more_context(self):
        z, y, x = np.indices((41, 81, 81))
        cp = (((y[0]-40)**2 + (x[0]-40)**2) < 12**2).astype(np.uint16)
        raw = np.full(z.shape, 2, np.float32)
        raw[:, cp > 0] = 6
        result = recover_native_3d(raw, self.spacing, cp,
                                 config={"cp_source_z_bounds_local": [15, 26]})
        profile = result["axial_profiles"][0]
        self.assertEqual(profile["status"], "positive_external_reference_requires_expansion")
        self.assertFalse(profile["endpoint_evidence_available"])

    def test_development_provenance_alone_is_not_a_review_veto(self):
        raw, cp = fixture()
        z, y, x = np.indices(raw.shape)
        cloud = (((y-40)/10)**2 + ((x-40)/10)**2 + ((z-40)/12)**2) < 1
        raw[~cloud] = 2  # Clean bounded fixture, with no background-bridge ambiguity.
        cp = (((y[0]-40)**2 + (x[0]-40)**2) < 18**2).astype(np.uint16)
        result = recover_native_3d(raw, self.spacing, cp,
                                 config={"cp_source_z_bounds_local": [28, 53]})
        target_id = result["labels"][40, 40, 40]
        obj = next(o for o in result["objects"] if o["label_id"] == target_id)
        self.assertFalse(obj["requires_identity_review"])
        self.assertEqual(obj["status"], "ready_for_support")

    def test_search_envelope_contact_records_extent_without_identity_veto(self):
        z, y, x = np.indices((81, 81, 81))
        cp = (((y[0]-40)**2 + (x[0]-40)**2) < 8**2).astype(np.uint16)
        cloud = ((y-40)/12)**2 + ((x-40)/12)**2 + ((z-40)/12)**2 < 1
        raw = np.full(z.shape, 2, np.float32)
        raw[cloud] = 6
        result = recover_native_3d(raw, self.spacing, cp,
                                  config={"cp_source_z_bounds_local": [28, 53]})
        self.assertEqual(len(result["objects"]), 1)
        obj = result["objects"][0]
        self.assertTrue(obj["xy_search_envelope_boundary_touched"])
        self.assertEqual(obj["flags"], ["xy_search_envelope_boundary_touched"])
        self.assertEqual(obj["open_boundary_faces"], [])
        self.assertTrue(obj["axial_endpoint_evidence_available"])
        self.assertFalse(obj["requires_identity_review"])
        self.assertEqual(obj["status"], "ready_for_support")
        self.assertEqual(obj["voxel_count"], int(np.count_nonzero(result["labels"])))
        # The same XY geometry with missing original Z context still requires
        # identity review; the boundary distinction does not clear that gate.
        unresolved = recover_native_3d(raw[30:51], self.spacing, cp)
        self.assertTrue(unresolved["objects"])
        for other in unresolved["objects"]:
            self.assertTrue(other["xy_search_envelope_boundary_touched"])
            self.assertIn("axial_endpoints_require_more_native_context", other["flags"])
            self.assertTrue(other["requires_identity_review"])
            self.assertEqual(other["status"], "review")

    def test_no_cp_mask_never_promotes_density_regions(self):
        raw, cp = fixture()
        result = recover_native_3d(raw, self.spacing, np.zeros_like(cp))
        self.assertFalse(result["labels"].any())
        self.assertEqual(result["objects"], [])

    def test_invalid_calibration_origin_and_source_scope_rejected(self):
        raw, cp = fixture()
        for args in [dict(spacing_xyz=(.2, .2, None)), dict(origin_zyx=(1.5, 0, 0)),
                     dict(config={"cp_source_z_bounds_local": [-1, 15]}),
                     dict(config={"min_cp_projected_overlap_fraction": 0})]:
            kwargs = dict(raw_zyx=raw, spacing_xyz=self.spacing, xy_labels=cp)
            kwargs.update(args)
            with self.assertRaises(ValueError):
                recover_native_3d(**kwargs)


if __name__ == "__main__":
    unittest.main()
