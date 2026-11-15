"""Engineering challenges for local 3D association, not biological accuracy."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from leap_cell_counting.support import SupportConfig, measure_support, required_padding_um
from leap_cell_counting.support import _surface_faces, _smooth_support


def fixture(spacing=(0.5, 0.5, 0.5)):
    sampling = np.array(spacing[::-1])
    shape = tuple(int(2 * np.ceil(12 / step) + 1) for step in sampling)
    z, y, x = [(np.arange(size) - size // 2) * step for size, step in zip(shape, sampling)]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    radius = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
    nucleus = radius <= 3
    outside = ndi.distance_transform_edt(~nucleus, sampling=sampling)
    return nucleus, outside, (zz, yy, xx)


def signal(region):
    return 10 + region.astype(np.float64) * 80


class CountingSupportTests(unittest.TestCase):
    def setUp(self):
        self.nucleus, self.distance, self.xyz = fixture()
        self.spacing = (0.5, 0.5, 0.5)
        self.soma = (self.distance <= 2.5) & ~self.nucleus

    def measure(self, values, marker="eGFP", mask=None, spacing=None, config=None, diagnostics=True):
        return measure_support(self.nucleus if mask is None else mask, values,
                               self.spacing if spacing is None else spacing, marker, config,
                               diagnostics=diagnostics)

    def test_optional_diagnostics_do_not_change_any_decision_metric(self):
        for values in (signal(self.soma), signal(self.nucleus), signal((self.distance <= .5) & ~self.nucleus)):
            diagnostic = self.measure(values, diagnostics=True)
            production = self.measure(values, diagnostics=False)
            self.assertTrue(diagnostic["metrics"].pop("before_smoothing_evaluated"))
            self.assertFalse(production["metrics"].pop("before_smoothing_evaluated"))
            diagnostic["metrics"].pop("before_smoothing")
            self.assertNotIn("before_smoothing", production["metrics"])
            self.assertEqual(diagnostic, production)

    def test_thick_cytoplasm_accepts_and_inputs_are_unchanged(self):
        values = signal(self.soma)
        before = values.copy(), self.nucleus.copy()
        result = self.measure(values)
        self.assertEqual(result["decision"], "accepted")
        self.assertGreater(result["metrics"]["thick_support_fraction"], .5)
        self.assertEqual(result["metrics"]["validation_status"], "development_thresholds")
        json.dumps(result, allow_nan=False)
        np.testing.assert_array_equal(values, before[0])
        np.testing.assert_array_equal(self.nucleus, before[1])

    def test_nuclear_positive_egfp_does_not_veto_supported_cytoplasm(self):
        without = self.measure(signal(self.soma))
        with_nuclear = self.measure(signal(self.soma | self.nucleus))
        self.assertEqual(with_nuclear["decision"], "accepted")
        self.assertEqual(with_nuclear["metrics"]["nuclear_positive_fraction"], 1)
        self.assertAlmostEqual(without["metrics"]["thick_support_fraction"], with_nuclear["metrics"]["thick_support_fraction"])

    def test_nuclear_signal_alone_is_review(self):
        result = self.measure(signal(self.nucleus))
        self.assertEqual(result["decision"], "review")
        self.assertIn("nuclear_signal_without_resolved_cytoplasm", result["reasons"])

    def test_partial_soma_needs_no_closed_ring(self):
        region = self.soma & (self.xyz[2] >= 0)
        result = self.measure(signal(region))
        self.assertEqual(result["decision"], "accepted")
        self.assertLess(result["metrics"]["contact_fraction"], .8)
        self.assertFalse(result["metrics"]["closed_ring_required"])

    def test_thin_adjacent_sheath_cannot_count_as_thick_cytoplasm(self):
        result = self.measure(signal((self.distance <= .5) & ~self.nucleus))
        self.assertEqual(result["decision"], "review")
        self.assertIn("contact_without_sufficient_cytoplasmic_thickness", result["reasons"])

    def test_sparse_thin_shell_cannot_gain_thickness_from_gaussian_tails(self):
        for step in (.2, .25):
            q = np.arange(-40, 41) * step
            z, y, x = np.meshgrid(q, q, q, indexing="ij")
            mask = z*z + y*y + x*x <= .5**2
            distance = ndi.distance_transform_edt(~mask, sampling=(step,)*3)
            thin = (distance <= step) & ~mask
            result = self.measure(signal(thin), mask=mask, spacing=(step,)*3)
            self.assertEqual(result["decision"], "review")
            self.assertEqual(result["metrics"]["before_smoothing"]["thick_support_fraction"], 0)
            self.assertLess(result["metrics"]["thick_support_fraction"], .3)
            self.assertGreater(result["metrics"]["threshold_dynamic"], result["metrics"]["local_dynamic"])

    def test_crossing_thin_process_cannot_accept_nucleus(self):
        zz, yy, xx = self.xyz
        process = (np.abs(yy - 3) <= .25) & (np.abs(zz) <= .25)
        result = self.measure(signal(process))
        self.assertNotEqual(result["decision"], "accepted")

    def test_remote_thick_envelope_does_not_bridge_gap(self):
        result = self.measure(signal((self.distance >= 2) & (self.distance <= 3.5)))
        self.assertEqual(result["decision"], "excluded")
        self.assertEqual(result["metrics"]["contact_fraction"], 0)
        self.assertEqual(result["metrics"]["thick_support_fraction"], 0)

    def test_gap_then_distant_positive_run_cannot_inflate_thin_contact(self):
        region = ((self.distance <= .5) | ((self.distance >= 1.5) & (self.distance <= 3.5))) & ~self.nucleus
        result = self.measure(signal(region))
        self.assertNotEqual(result["decision"], "accepted")
        self.assertLess(result["metrics"]["thick_support_fraction"], .15)

    def test_thick_signal_only_above_nucleus_is_not_sufficient_z_evidence(self):
        zz, _, _ = self.xyz
        result = self.measure(signal((zz >= 3) & (self.distance <= 2.5) & ~self.nucleus))
        self.assertNotEqual(result["decision"], "accepted")

    def test_disjoint_supported_planes_do_not_establish_axial_continuity(self):
        region = self.soma.copy()
        region[::2] = False
        result = self.measure(signal(region))
        self.assertEqual(result["decision"], "review")
        self.assertIn("limited_z_support", result["reasons"])
        self.assertGreater(result["metrics"]["z_supported_plane_fraction"], .35)
        self.assertLess(result["metrics"]["z_support_fraction"], .35)

    def test_flat_channel_is_unknown_not_marker_negative(self):
        for intensity in (0.0, 20.0):
            result = self.measure(np.full(self.nucleus.shape, intensity))
            self.assertEqual(result["decision"], "review")
            self.assertIn("local_signal_not_distinguishable", result["reasons"])

    def test_noise_only_cannot_establish_local_marker_detectability(self):
        values = np.random.default_rng(91).normal(10, 1, self.nucleus.shape)
        result = self.measure(values)
        self.assertEqual(result["decision"], "review")
        self.assertFalse(result["metrics"]["signal_detectable"])

    def test_absent_nucleus_never_created_from_bright_support(self):
        result = self.measure(signal(self.soma), mask=np.zeros_like(self.nucleus))
        self.assertEqual(result["decision"], "review")
        self.assertIn("no_dapi_nucleus", result["reasons"])

    def test_declared_custom_marker_uses_the_same_geometric_evidence(self):
        result = self.measure(signal(self.soma), marker="my reporter")
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["metrics"]["marker"], "my reporter")
        self.assertFalse(result["metrics"]["marker_name_establishes_cell_type"])
        self.assertEqual(result["metrics"]["validation_status"], "development_thresholds")
        self.assertGreater(result["metrics"]["thick_support_fraction"], .5)

    def test_missing_marker_declaration_is_rejected(self):
        for marker in ("", "  ", None):
            with self.assertRaises(ValueError):
                self.measure(signal(self.soma), marker=marker)

    def test_known_cytoplasmic_markers_share_geometry_without_age_rule(self):
        for marker in ("eGFP", "Biocytin", "ALDH1L1"):
            self.assertEqual(self.measure(signal(self.soma), marker)["decision"], "accepted")

    def test_missing_physical_calibration_requires_review(self):
        for spacing in ([None, .5, .5], [.5, .5, 0], [.5, .5, float("nan")], [.5, .5]):
            result = self.measure(signal(self.soma), spacing=spacing)
            self.assertEqual(result["decision"], "review")
            json.dumps(result, allow_nan=False)

    def test_truncated_nucleus_requires_review_without_zero_padding(self):
        crop = (slice(24, None), slice(None), slice(None))
        result = self.measure(signal(self.soma)[crop], mask=self.nucleus[crop])
        self.assertEqual(result["decision"], "review")
        self.assertIn("nucleus_truncated_by_roi", result["reasons"])
        self.assertLess(result["metrics"]["boundary_valid_fraction"], 1)

    def test_nonfinite_signal_is_missing_evidence(self):
        values = signal(self.soma)
        values[:, :, 23:26] = np.nan
        result = self.measure(values)
        self.assertEqual(result["decision"], "review")
        self.assertIn("nonfinite_local_signal", result["reasons"])
        json.dumps(result, allow_nan=False)

    def test_coarse_native_resolution_never_invents_cytoplasmic_thickness(self):
        mask, distance, _ = fixture((1, 1, 1))
        result = self.measure(signal((distance <= 2.5) & ~mask), mask=mask, spacing=(1, 1, 1))
        self.assertEqual(result["decision"], "review")
        self.assertIn("insufficient_resolved_boundary", result["reasons"])

    def test_anisotropic_spacing_uses_physical_face_weights(self):
        mask, distance, _ = fixture((.25, .5, .5))
        values = signal((distance <= 2.5) & ~mask)
        result = self.measure(values, mask=mask, spacing=(.25, .5, .5))
        self.assertEqual(result["decision"], "accepted")
        moved = measure_support(mask.transpose(0, 2, 1), values.transpose(0, 2, 1), (.5, .25, .5), "eGFP")
        self.assertAlmostEqual(result["metrics"]["surface_area_um2"], moved["metrics"]["surface_area_um2"])
        self.assertAlmostEqual(result["metrics"]["thick_support_fraction"], moved["metrics"]["thick_support_fraction"])

    def test_positive_affine_intensity_change_does_not_change_decision(self):
        values = signal(self.soma)
        result = self.measure(values)
        transformed = self.measure(values * 7 + 300)
        self.assertEqual(result["decision"], transformed["decision"])
        self.assertAlmostEqual(result["metrics"]["contact_fraction"], transformed["metrics"]["contact_fraction"])

    def test_disconnected_nuclei_are_not_one_support_object(self):
        mask = self.nucleus.copy()
        mask[5, 5, 5] = True
        result = self.measure(signal(self.soma), mask=mask)
        self.assertEqual(result["decision"], "review")
        self.assertIn("disconnected_dapi_object", result["reasons"])
        self.assertEqual(result["metrics"]["nuclear_component_count"], 2)
        self.assertIn("thick_support_fraction", result["metrics"])

    def test_insufficient_background_is_not_replaced_with_whole_roi(self):
        crop = (slice(18, 31),) * 3
        result = self.measure(signal(self.soma)[crop], mask=self.nucleus[crop])
        self.assertEqual(result["decision"], "review")
        self.assertIn("insufficient_local_background", result["reasons"])

    def test_config_accepts_explicit_distances_rejects_hidden_unknown_keys(self):
        self.assertEqual(required_padding_um(), 8)
        self.assertEqual(required_padding_um({"background_outer_um": 9}), 9)
        for config in ({"nuclear_diameter": 8}, {"background_inner_um": 1},
                       {"min_thick_fraction": 2}, {"max_sampling_step_um": 2},
                       {"min_boundary_samples": 1.5}):
            with self.assertRaises(ValueError):
                self.measure(signal(self.soma), config=config)
        self.assertEqual(self.measure(signal(self.soma), config=SupportConfig())["decision"], "accepted")

    def test_observed_coarse_process_ambiguity_stays_review(self):
        mask, _, (zz, yy, xx) = fixture((.25, .25, .25))
        for direction in (np.array([0., 0., 1.]), np.array([0., 1., 1.]) / np.sqrt(2),
                          np.array([.3, .5, .8]) / np.linalg.norm([.3, .5, .8])):
            coords = np.stack((zz, yy, xx), axis=-1)
            along = coords @ direction
            perpendicular = np.cross(direction, [1., 0., 0.])
            perpendicular /= np.linalg.norm(perpendicular)
            radial = coords - along[..., None] * direction - 2 * perpendicular
            cylinder = (radial ** 2).sum(axis=-1) <= 2 ** 2
            result = self.measure(signal(cylinder), mask=mask, spacing=(.25, .25, .25))
            self.assertEqual(result["decision"], "review")

    def test_rotated_partial_soma_keeps_strong_support(self):
        zz, yy, xx = self.xyz
        for direction in ([0., 1., 1.], [.3, .5, .8]):
            projection = zz * direction[0] + yy * direction[1] + xx * direction[2]
            result = self.measure(signal(self.soma & (projection >= 0)))
            self.assertEqual(result["decision"], "accepted")

    def test_single_plane_has_xy_evidence_and_no_invented_z_calibration(self):
        plane = (slice(24, 25), slice(None), slice(None))
        result = self.measure(signal(self.soma)[plane], mask=self.nucleus[plane], spacing=(.5, .5, None))
        self.assertEqual(result["decision"], "review")
        self.assertIn("single_plane_axial_support_unavailable", result["reasons"])
        m = result["metrics"]
        self.assertEqual(m["measurement_dimension"], 2)
        self.assertEqual(m["spacing_xyz_um"], [.5, .5, None])
        self.assertFalse(m["axial_support_evaluable"])
        self.assertIsNone(m["z_support_fraction"])
        self.assertIsNone(m["surface_area_um2"])
        self.assertGreater(m["boundary_length_um"], 0)
        self.assertGreater(m["thick_support_fraction"], .5)
        self.assertFalse(m["nucleus_touches_roi_boundary"])
        json.dumps(result, allow_nan=False)

    def test_single_plane_still_requires_true_xy_calibration(self):
        plane = (slice(24, 25), slice(None), slice(None))
        result = self.measure(signal(self.soma)[plane], mask=self.nucleus[plane], spacing=(None, .5, None))
        self.assertEqual(result["decision"], "review")
        self.assertIn("physical_calibration_unavailable", result["reasons"])

    def test_native_pixel_texture_does_not_break_resolved_cytoplasm(self):
        mask, distance, _ = fixture((.25, .25, .5))
        rng = np.random.default_rng(104)
        textured = (rng.random(mask.shape) < .45) & (distance <= 2.5) & ~mask
        values = 10 + 80 * textured.astype(float) + rng.normal(0, .1, mask.shape)
        result = self.measure(values, mask=mask, spacing=(.25, .25, .5))
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["metrics"]["thresholds"]["smoothing_sigma_um"], .2)
        self.assertLess(result["metrics"]["before_smoothing"]["thick_support_fraction"], result["metrics"]["thick_support_fraction"])

    def test_shape_and_nonbinary_object_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            self.measure(np.zeros((3, 3)))
        mask = self.nucleus.astype(int) * 2
        with self.assertRaises(ValueError):
            self.measure(signal(self.soma), mask=mask)


class SupportOptimizationEquivalenceTests(unittest.TestCase):
    def test_tight_bbox_faces_match_original_grid_neighbor_definition(self):
        rng = np.random.default_rng(811)
        for shape in ((9, 11), (7, 9, 11)):
            masks = [rng.random(shape) > .7, np.ones(shape, dtype=bool)]
            interior = np.zeros(shape, dtype=bool)
            interior[(slice(2, -2),) * len(shape)] = True
            masks.append(interior)
            for mask in masks:
                coordinates = np.argwhere(mask)
                faces = _surface_faces(mask, coordinates.min(axis=0), coordinates.max(axis=0))
                for axis in range(mask.ndim):
                    for direction in (-1, 1):
                        expected = []
                        for coordinate in coordinates:
                            neighbor = coordinate.copy()
                            neighbor[axis] += direction
                            if not 0 <= neighbor[axis] < shape[axis] or not mask[tuple(neighbor)]:
                                expected.append(coordinate)
                        expected = np.asarray(expected, dtype=int).reshape(-1, mask.ndim)
                        np.testing.assert_array_equal(faces[axis, direction], expected)

    def test_optimized_smoothing_is_bit_exact_with_normalized_reference(self):
        rng = np.random.default_rng(812)
        for sampling in ((.5, .5, .5), (.3458, .07097069597069597, .07097069597069597),
                         (.2432, .09014353464878591, .09014353464878591), (.5, .5)):
            shape = (7, 9, 11) if len(sampling) == 3 else (9, 11)
            sigma = .2 / np.asarray(sampling)
            for missing in (False, True):
                values = rng.normal(12, 3, shape)
                if missing:
                    values.flat[30:35] = np.nan
                for view in (values, values[..., ::-1], values.swapaxes(-1, -2)):
                    finite = np.isfinite(view)
                    weights = ndi.gaussian_filter(finite.astype(np.float64), sigma, mode="nearest")
                    numerator = ndi.gaussian_filter(np.where(finite, view, 0).astype(np.float64), sigma, mode="nearest")
                    expected = np.divide(numerator, weights, out=np.full_like(numerator, np.nan), where=weights > 0)
                    expected[~finite] = np.nan
                    np.testing.assert_array_equal(_smooth_support(view, finite, sigma), expected)


if __name__ == "__main__":
    unittest.main()
