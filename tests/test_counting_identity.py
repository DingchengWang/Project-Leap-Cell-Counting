"""Pure cell identity contracts; supplied association decisions are test inputs.

These tests assess reference integrity, grouping and original-Z provenance. They
do not select biological association thresholds or validate detection accuracy.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from leap_cell_counting.identity import (
    enclosure_evidence,
    geometric_association,
    nucleus_geometry,
    observation_from_nucleus,
    unify_cells,
)


def observation(identifier, kind, label_id, *, artifact=None, decision="accepted", anchors=None):
    anchors = deepcopy(anchors or {"10": [31.0, 21.0], "11": [32.0, 22.0]})
    plane = int(next(iter(anchors)))
    x, y = anchors[str(plane)]
    return {
        "id": identifier,
        "kind": kind,
        "artifact_id": artifact or ("dapi_nuclei" if kind == "nucleus" else "biocytin_bodies"),
        "label_id": label_id,
        "bounds_zyx": [[min(map(int, anchors)), max(map(int, anchors)) + 1], [20, 24], [30, 34]],
        "plane_anchors": anchors,
        "representative_xyz": [x, y, plane],
        "decision": decision,
        "reasons": ["synthetic_supplied_decision"],
        "metrics": {"provenance": {"source": identifier, "stages": ["synthetic_observation"]}},
    }


def relation(nucleus_id, body_id, decision="match"):
    return {"nucleus_id": nucleus_id, "body_id": body_id, "decision": decision,
            "evidence": {"origin": "explicit_test_input", "details": {"notes": ["no_threshold_in_test"]}}}


def semantic_cells(cells):
    """Ignore list presentation order, not identity, geometry or provenance."""
    result = deepcopy(cells)
    for cell in result:
        cell["source_observations"].sort(key=lambda item: item["id"])
        cell["count_basis_observation_ids"].sort()
        cell["association"]["relations"].sort(key=lambda item: (item["nucleus_id"], item["body_id"]))
    return sorted(result, key=lambda cell: cell["id"])


class CountingIdentityTests(unittest.TestCase):
    def assert_observations_conserved(self, cells, observations):
        original_ids = [item["id"] for item in observations]
        represented_ids = [item["id"] for cell in cells for item in cell["source_observations"]]
        self.assertCountEqual(represented_ids, original_ids)
        self.assertEqual(len(represented_ids), len(set(represented_ids)))
        self.assertEqual(len(cells), len({cell["id"] for cell in cells}))
        self.assertTrue(all(cell["id"] == cell["cell_id"] for cell in cells))

    def test_accepted_body_alone_counts_once_without_nucleus_or_model_evidence(self):
        body = observation("body-independent", "body", 7)
        cells = unify_cells([body])
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual(cell["status"], "accepted")
        self.assertEqual(sum(item["status"] == "accepted" for item in cells), 1)
        self.assertEqual(cell["source"], "support_body")
        self.assertEqual(cell["source_observations"], [body])
        self.assertEqual(cell["count_basis_observation_ids"], [body["id"]])
        self.assertEqual(cell["association"]["status"], "unmatched")
        self.assert_observations_conserved(cells, [body])

    def test_unmatched_observation_decisions_remain_explicit(self):
        items = [observation(f"body-{status}", "body", index + 1, decision=status)
                 for index, status in enumerate(("accepted", "review", "excluded"))]
        cells = unify_cells(items)
        status_by_observation = {cell["source_observations"][0]["id"]: cell["status"] for cell in cells}
        self.assertEqual(status_by_observation, {item["id"]: item["decision"] for item in items})
        self.assertEqual(sum(cell["status"] == "accepted" for cell in cells), 1)

    def test_unique_match_counts_one_and_retains_both_complete_observations(self):
        nucleus = observation("nucleus-a", "nucleus", 2)
        body = observation("body-a", "body", 23)
        supplied = relation(nucleus["id"], body["id"])
        cells = unify_cells([nucleus, body], [supplied])
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual(cell["status"], "accepted")
        self.assertEqual(cell["source"], "dapi_support")
        self.assertEqual(cell["association"]["status"], "one_to_one")
        self.assertEqual(cell["association"]["relations"], [supplied])
        self.assertCountEqual(cell["source_observations"], [nucleus, body])
        self.assertCountEqual(cell["count_basis_observation_ids"], ["nucleus-a", "body-a"])
        self.assert_observations_conserved(cells, [nucleus, body])

    def test_equal_numeric_labels_in_nucleus_and_body_artifacts_do_not_imply_match(self):
        items = [observation("nucleus-one", "nucleus", 1), observation("body-one", "body", 1)]
        cells = unify_cells(items)
        self.assertEqual(len(cells), 2)
        self.assertEqual(sum(cell["status"] == "accepted" for cell in cells), 2)
        self.assertEqual({(cell["label_artifact_id"], cell["label_id"]) for cell in cells},
                         {("dapi_nuclei", 1), ("biocytin_bodies", 1)})
        self.assert_observations_conserved(cells, items)
        matched = unify_cells(items, [relation("nucleus-one", "body-one")])
        self.assertEqual(len(matched), 1)
        self.assertEqual({(item["artifact_id"], item["label_id"]) for item in matched[0]["source_observations"]},
                         {("dapi_nuclei", 1), ("biocytin_bodies", 1)})

    def test_same_kind_same_label_in_distinct_artifacts_has_unique_stable_cell_ids(self):
        for kind in ("nucleus", "body"):
            with self.subTest(kind=kind):
                items = [observation(f"{kind}-first", kind, 1, artifact="artifact-first"),
                         observation(f"{kind}-second", kind, 1, artifact="artifact-second")]
                cells = unify_cells(items)
                self.assertEqual(len(cells), 2)
                self.assert_observations_conserved(cells, items)
                self.assertEqual(semantic_cells(cells), semantic_cells(unify_cells(list(reversed(items)))))

    def test_competing_matches_only_mark_the_connected_candidate_group_for_review(self):
        items = [observation("n1", "nucleus", 1), observation("n2", "nucleus", 2),
                 *(observation(f"b{i}", "body", i) for i in range(1, 5))]
        relations = [relation("n1", "b1"), relation("n1", "b2"), relation("n2", "b3")]
        cells = unify_cells(items, relations)
        groups = {frozenset(item["id"] for item in cell["source_observations"]): cell for cell in cells}
        self.assertEqual(set(groups), {frozenset(["n1"]), frozenset(["b1"]), frozenset(["b2"]),
                                       frozenset(["n2", "b3"]), frozenset(["b4"])})
        for identifier in ("n1", "b1", "b2"):
            self.assertEqual(groups[frozenset([identifier])]["status"], "review")
            self.assertEqual(groups[frozenset([identifier])]["association"]["status"], "ambiguous")
        self.assertEqual(groups[frozenset(["n2", "b3"])]["status"], "accepted")
        self.assertEqual(groups[frozenset(["b4"])]["status"], "accepted")
        self.assertEqual(sum(cell["status"] == "accepted" for cell in cells), 2)
        self.assert_observations_conserved(cells, items)

    def test_ambiguity_propagates_along_match_edges_but_not_explicit_no_match_edges(self):
        items = [*(observation(f"n{i}", "nucleus", i) for i in range(1, 4)),
                 *(observation(f"b{i}", "body", i) for i in range(1, 5))]
        relations = [relation("n1", "b1"), relation("n1", "b2", "ambiguous"),
                     relation("n2", "b2"), relation("n3", "b3"), relation("n1", "b4", "no_match")]
        cells = unify_cells(items, relations)
        groups = {frozenset(item["id"] for item in cell["source_observations"]): cell for cell in cells}
        for identifier in ("n1", "n2", "b1", "b2"):
            self.assertEqual(groups[frozenset([identifier])]["status"], "review")
            self.assertEqual(groups[frozenset([identifier])]["association"]["status"], "ambiguous")
        self.assertEqual(groups[frozenset(["n3", "b3"])]["status"], "accepted")
        self.assertEqual(groups[frozenset(["b4"])]["status"], "accepted")
        self.assert_observations_conserved(cells, items)

    def test_explicit_no_match_never_merges_coincident_representatives(self):
        items = [observation("n", "nucleus", 1), observation("b", "body", 1)]
        cells = unify_cells(items, [relation("n", "b", "no_match")])
        self.assertEqual(len(cells), 2)
        self.assertTrue(all(cell["association"]["status"] == "unmatched" for cell in cells))
        self.assertTrue(all(cell["status"] == "accepted" for cell in cells))
        self.assert_observations_conserved(cells, items)

    def test_per_z_anchors_reference_actual_source_and_preserve_both_overlap_observations(self):
        nucleus = observation("n", "nucleus", 1, anchors={"10": [31, 21], "11": [32, 22]})
        body = observation("b", "body", 8, anchors={"11": [30.5, 20.5], "12": [33, 23]})
        body["representative_xyz"] = [33, 23, 12]
        cell = unify_cells([nucleus, body], [relation("n", "b")])[0]
        self.assertEqual(cell["plane_anchors"], {"10": [31, 21], "11": [30.5, 20.5], "12": [33, 23]})
        self.assertEqual(cell["plane_anchor_sources"], {
            "10": {"kind": "observation", "observation_id": "n"},
            "11": {"kind": "observation", "observation_id": "b"},
            "12": {"kind": "observation", "observation_id": "b"}})
        self.assertEqual((cell["x"], cell["y"], cell["z"]), (33, 23, 12))
        by_id = {item["id"]: item for item in cell["source_observations"]}
        self.assertEqual(by_id["n"]["plane_anchors"]["11"], [32, 22])
        self.assertEqual(by_id["b"]["plane_anchors"]["11"], [30.5, 20.5])

    def test_input_permutations_preserve_identity_count_and_all_provenance(self):
        items = [observation("n1", "nucleus", 1), observation("n2", "nucleus", 2),
                 observation("b1", "body", 1), observation("b2", "body", 2), observation("b3", "body", 3)]
        relations = [relation("n1", "b1"), relation("n2", "b2", "ambiguous")]
        expected = semantic_cells(unify_cells(items, relations))
        for shuffled in permutations(items):
            for ordered_relations in permutations(relations):
                cells = unify_cells(shuffled, ordered_relations)
                self.assertEqual(semantic_cells(cells), expected)
                self.assert_observations_conserved(cells, items)
        self.assertEqual(sum(cell["status"] == "accepted" for cell in expected), 2)

    def test_inputs_and_nested_evidence_are_not_mutated_or_aliased_into_outputs(self):
        items = [observation("n", "nucleus", 1), observation("b", "body", 2)]
        relations = [relation("n", "b")]
        original_items, original_relations = deepcopy(items), deepcopy(relations)
        cells = unify_cells(items, relations)
        self.assertEqual(items, original_items)
        self.assertEqual(relations, original_relations)
        cells[0]["plane_anchors"]["10"][0] = -999
        cells[0]["source_observations"][0]["metrics"]["provenance"]["stages"].append("changed_output")
        cells[0]["association"]["relations"][0]["evidence"]["details"]["notes"].append("changed_output")
        self.assertEqual(items, original_items)
        self.assertEqual(relations, original_relations)

    def test_duplicate_observation_identity_or_artifact_scoped_label_is_rejected(self):
        original = observation("n", "nucleus", 1)
        for duplicate in (observation("n", "body", 2), observation("different-id", "nucleus", 1)):
            with self.subTest(duplicate=duplicate["id"]):
                items = [original, duplicate]
                before = deepcopy(items)
                with self.assertRaises(ValueError):
                    unify_cells(items)
                self.assertEqual(items, before)

    def test_invalid_observation_references_and_missing_native_anchor_evidence_are_rejected(self):
        bad_values = [("id", ""), ("artifact_id", ""), ("label_id", 0), ("label_id", -1),
                      ("label_id", 1.5), ("label_id", True), ("kind", "cell"),
                      ("decision", "unknown"), ("plane_anchors", {})]
        for key, value in bad_values:
            with self.subTest(field=key, value=value):
                invalid = observation("n", "nucleus", 1)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    unify_cells([invalid])

    def test_missing_wrong_kind_or_unknown_relation_references_are_rejected(self):
        items = [observation("n", "nucleus", 1), observation("b", "body", 1)]
        invalid = [relation("absent", "b"), relation("n", "absent"), relation("b", "n"),
                   relation("n", "n"), relation("b", "b"), relation("n", "b", "guess")]
        for supplied in invalid:
            with self.subTest(relation=supplied):
                before = deepcopy(supplied)
                with self.assertRaises(ValueError):
                    unify_cells(items, [supplied])
                self.assertEqual(supplied, before)

    def test_duplicate_or_conflicting_relation_pair_is_rejected(self):
        items = [observation("n", "nucleus", 1), observation("b", "body", 1)]
        for first, second in (("match", "match"), ("ambiguous", "ambiguous"),
                              ("no_match", "no_match"), ("match", "no_match"), ("match", "ambiguous")):
            with self.subTest(first=first, second=second):
                with self.assertRaises(ValueError):
                    unify_cells(items, [relation("n", "b", first), relation("n", "b", second)])

    def test_empty_observations_are_valid_only_without_references(self):
        self.assertEqual(unify_cells([]), [])
        with self.assertRaises(ValueError):
            unify_cells([], [relation("missing-n", "missing-b")])

    def test_nucleus_adapter_retains_native_geometry_and_deep_copies_evidence(self):
        point = {"label_id": 9, "x": 31.5, "y": 21.5, "z": 11, "status": "review",
                 "plane_anchors": {"10": [31, 21], "11": [31.5, 21.5]},
                 "native_identity": {"bounds_zyx": [[10, 12], [20, 24], [30, 34]], "flags": ["source_flag"]},
                 "support": {"metrics": {"observed": ["evidence"]}}}
        original = deepcopy(point)
        item = observation_from_nucleus(point)
        self.assertEqual(item["kind"], "nucleus")
        self.assertEqual((item["artifact_id"], item["label_id"], item["decision"]), ("dapi_nuclei", 9, "review"))
        self.assertEqual(item["bounds_zyx"], original["native_identity"]["bounds_zyx"])
        self.assertEqual(item["plane_anchors"], original["plane_anchors"])
        item["plane_anchors"]["10"][0] = 0
        item["metrics"]["support"]["metrics"]["observed"].append("output_change")
        item["reasons"].append("output_change")
        self.assertEqual(point, original)


class GeometricAssociationContractTests(unittest.TestCase):
    def evidence(self, *, representative_inside=True, core_fraction=1.0, body_planes=(10, 11),
                 whole_fraction=0.4):
        return {
            "decision": "evidence_only",
            "nucleus_representative_inside_body_envelope": representative_inside,
            "nucleus_core_fraction_in_body_envelope": core_fraction,
            "nucleus_representative_xyz": [31.0, 21.0, 10],
            "body_z_planes": list(body_planes),
            "nucleus_fraction_in_body_envelope": whole_fraction,
        }

    def test_full_core_and_representative_with_actual_body_on_same_z_are_match_candidate(self):
        evidence = self.evidence(whole_fraction=0.4)
        original = deepcopy(evidence)
        result = geometric_association(evidence)
        self.assertEqual(result["decision"], "match")
        self.assertIn("actual_body_signal_present_on_nuclear_representative_z", result["reasons"])
        self.assertEqual(evidence, original)

    def test_partially_enclosed_core_remains_ambiguous_even_with_representative_inside(self):
        evidence = self.evidence(core_fraction=0.5)
        original = deepcopy(evidence)
        result = geometric_association(evidence)
        self.assertEqual(result["decision"], "ambiguous")
        self.assertIn("partial_nuclear_representative_or_core_enclosure", result["reasons"])
        self.assertEqual(evidence, original)

    def test_envelope_enclosure_without_actual_body_on_representative_z_is_ambiguous(self):
        evidence = self.evidence(body_planes=(9, 11))
        result = geometric_association(evidence)
        self.assertEqual(result["decision"], "ambiguous")
        self.assertIn("no_actual_body_signal_on_nuclear_representative_z", result["reasons"])

    def test_edge_overlap_without_core_or_representative_enclosure_is_no_match(self):
        evidence = self.evidence(representative_inside=False, core_fraction=0.0, whole_fraction=0.1)
        self.assertGreater(evidence["nucleus_fraction_in_body_envelope"], 0)
        result = geometric_association(evidence)
        self.assertEqual(result["decision"], "no_match")
        self.assertEqual(result["reasons"], ["nuclear_representative_and_core_outside_body_envelope"])

    def test_two_geometric_match_candidates_for_one_body_require_identity_review(self):
        items = [observation("n1", "nucleus", 1), observation("n2", "nucleus", 2),
                 observation("b1", "body", 1), observation("b2", "body", 2)]
        relations = []
        for identifier in ("n1", "n2"):
            evidence = self.evidence()
            candidate = geometric_association(evidence)
            self.assertEqual(candidate["decision"], "match")
            relations.append({"nucleus_id": identifier, "body_id": "b1", **candidate, "evidence": evidence})
        cells = unify_cells(items, relations)
        self.assertEqual(len(cells), 4)
        by_observation = {cell["source_observations"][0]["id"]: cell for cell in cells}
        for identifier in ("n1", "n2", "b1"):
            self.assertEqual(len(by_observation[identifier]["source_observations"]), 1)
            self.assertEqual(by_observation[identifier]["status"], "review")
            self.assertEqual(by_observation[identifier]["association"]["status"], "ambiguous")
        self.assertEqual(by_observation["b2"]["status"], "accepted")
        self.assertEqual(sum(cell["status"] == "accepted" for cell in cells), 1)


class NativeAssociationEvidenceTests(unittest.TestCase):
    def test_enclosure_keeps_actual_observed_z_separate_from_body_envelope(self):
        nucleus_mask = np.ones((2, 2, 2), dtype=bool)
        body_mask = np.zeros((3, 4, 4), dtype=bool)
        body_mask[1, 1:3, 1:3] = True
        envelope = np.ones_like(body_mask)
        originals = [value.copy() for value in (nucleus_mask, body_mask, envelope)]
        nucleus = nucleus_geometry(nucleus_mask, (10, 20, 30), (0.25, 0.5, 1.0), [30.5, 20.5, 10])
        evidence = enclosure_evidence(nucleus, body_mask, (9, 19, 29), envelope, (9, 19, 29))
        self.assertEqual(evidence["decision"], "evidence_only")
        self.assertEqual(evidence["nucleus_voxels"], 8)
        self.assertEqual(evidence["nucleus_voxels_in_body_envelope"], 8)
        self.assertEqual(evidence["nucleus_z_planes"], [10, 11])
        self.assertEqual(evidence["body_z_planes"], [10])
        self.assertEqual(evidence["shared_observed_z_planes"], [10])
        self.assertEqual([row["source_z"] for row in evidence["per_z"]], [10, 11])
        self.assertEqual([row["body_observed_on_this_z"] for row in evidence["per_z"]], [True, False])
        self.assertFalse(evidence["cross_kind_iou_used"])
        self.assertFalse(evidence["centroid_distance_used_as_identity"])
        json.dumps(evidence, allow_nan=False)
        for observed, original in zip((nucleus_mask, body_mask, envelope), originals):
            np.testing.assert_array_equal(observed, original)

    def test_same_xy_with_disjoint_original_z_does_not_gain_enclosure(self):
        mask = np.ones((2, 2, 2), dtype=bool)
        nucleus = nucleus_geometry(mask, (10, 20, 30), (0.25, 0.5, 1.0), [30.5, 20.5, 10])
        evidence = enclosure_evidence(nucleus, mask, (14, 20, 30), mask, (14, 20, 30))
        self.assertEqual(evidence["decision"], "evidence_only")
        self.assertEqual(evidence["nucleus_voxels_in_body_envelope"], 0)
        self.assertEqual(evidence["nucleus_core_fraction_in_body_envelope"], 0)
        self.assertFalse(evidence["nucleus_representative_inside_body_envelope"])
        self.assertEqual(evidence["shared_observed_z_planes"], [])


if __name__ == "__main__":
    unittest.main()
