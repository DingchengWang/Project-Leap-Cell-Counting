"""Counting data/instance/protocol invariants; no real model or GPU is loaded."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr, nullcontext
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import tifffile as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from leap_cell_counting.errors import CountingCancelled, CountingError
from leap_cell_counting.imaging import (PlaneReader, channel_colors, create_preview, create_slice,
                                       display_plane, inspect_image, SliceSession, serve_slices)
from leap_cell_counting.source_scan import scan_source
from leap_cell_counting.segmentation import EVAL_PARAMETERS, interior_anchor, validate_labels
from leap_cell_counting.worker import Events, component_usage_lock, run_request


class CountingEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="leap-counting-test-", dir=str(ROOT / "artifacts/prototype-d/engine"))
        self.root = Path(self.temp.name)
        self.inputs = self.root / "source"
        self.inputs.mkdir()
        self.source = self.inputs / "中文 input.tif"
        self.data = np.arange(4 * 3 * 21 * 23, dtype=np.uint16).reshape(4, 3, 21, 23)
        self.lock = self.root / "components.lock"
        self.lock.touch()
        environment = patch.dict(os.environ, {"LEAP_COUNTING_STATS_LIBRARY":
            str(ROOT/'artifacts/prototype-d/native-counting/libLeapCountingStats.dylib')})
        environment.start()
        self.addCleanup(environment.stop)

    def scan(self, reader, **kwargs):
        return scan_source(reader, self.root/'statistics', 1, 0,
                           list(range(reader.info['channel_count'])), **kwargs)

    def tearDown(self):
        self.temp.cleanup()

    def write_imagej(self, data=None, metadata=None, **kwargs):
        payload = self.data if data is None else data
        info = {"axes": "ZCYX" if payload.ndim == 4 else "CYX", "unit": "um", "spacing": .3, "Info": json.dumps({"pixel_um": [.1, .1, .3]})}
        if metadata:
            info.update(metadata)
        tf.imwrite(self.source, payload, imagej=True, photometric="minisblack", metadata=info, **kwargs)

    def test_info_labels_detect_export_markers_and_preserve_names(self):
        names = ["DAPI", "Biocytin (Cy2)", "SR101"]
        info = {"labels": names, "c": 3, "z": 4, "x": 23, "y": 21, "t": 1,
                "pixel_um": [.071, .072, .3458], "z_spacing_status": "known"}
        labels = [f"{name} | Z={z + 1}/4 | source S4C{c + 1}" for z in range(4) for c, name in enumerate(names)]
        self.write_imagej(metadata={"Info": json.dumps(info), "Labels": labels})
        with patch.object(tf.TiffPage, "asarray", side_effect=AssertionError("headers only")):
            result = inspect_image(self.source)
        self.assertEqual(result["dapi_channel"], 0)
        self.assertEqual(result["channel_hints"], names)
        self.assertEqual(result["channel_roles"], ["DAPI", "Biocytin", "SR101"])
        self.assertEqual(result["physical_size_um"], {"X": .071, "Y": .072, "Z": .3458})
        self.assertNotIn("nuclear_diameter_px", result)
        self.assertNotIn("nuclear_diameter_source", result)
        self.assertEqual(channel_colors(result["channel_roles"]), ["#0000FF", "#FFFFFF", "#FF0000"])

    def test_imagej_per_plane_labels_are_aligned_with_channel_axes(self):
        names = ["SR101", "DAPI", "Biocytin (Cy2)"]
        self.write_imagej(metadata={"Labels": [f"{name} | Z={z}" for z in range(4) for name in names]})
        result = inspect_image(self.source)
        self.assertEqual(result["dapi_channel"], 1)
        self.assertEqual(result["channel_hints"], names)

    def test_shared_title_filename_lut_and_unmapped_info_never_identify_dapi(self):
        self.write_imagej(metadata={"Labels": [f"c:{c + 1}/3 z:{z + 1}/4 - DAPI_eGFP_GFAP" for z in range(4) for c in range(3)],
                                   "Info": "Image1 ChannelDescription #0|LUTName = Blue\nDAPI eGFP GFAP"})
        result = inspect_image(self.source)
        self.assertEqual(result["dapi_channel"], -1)
        self.assertEqual(result["channel_roles"], ["Unknown"] * 3)
        with self.assertRaisesRegex(CountingError, "DAPI"):
            create_preview(self.source, self.root / "preview.png")

    def test_conflicting_and_duplicate_dapi_metadata_require_manual_choice(self):
        self.write_imagej(metadata={"Info": json.dumps({"labels": ["DAPI", "SR101", "Biocytin"]}),
                                   "Labels": ["SR101", "DAPI", "Biocytin"]})
        result = inspect_image(self.source)
        self.assertEqual(result["dapi_channel"], -1)
        self.write_imagej(metadata={"Labels": ["DAPI", "DAPI", "Biocytin"]})
        self.assertEqual(inspect_image(self.source)["dapi_channel"], -1)

    def test_stale_info_dimensions_do_not_supply_channel_identities(self):
        self.write_imagej(metadata={"Info": json.dumps({"labels": ["DAPI", "SR101", "Biocytin"], "c": 4})})
        self.assertEqual(inspect_image(self.source)["dapi_channel"], -1)

    def test_single_z_is_supported_and_unknown_z_spacing_remains_unknown(self):
        tf.imwrite(self.source, self.data[0], imagej=True, metadata={"axes": "CYX", "unit": "um",
            "Info": json.dumps({"labels": ["SR101", "DAPI", "Biocytin"], "pixel_um": [.1, .1, None],
                                "z_spacing_status": "unknown"})})
        info = inspect_image(self.source)
        self.assertTrue(info["supported_layout"])
        self.assertEqual(info["z_count"], 1)
        self.assertIsNone(info["physical_size_um"]["Z"])
        with PlaneReader(self.source) as reader:
            np.testing.assert_array_equal(self.scan(reader)["projections"][1], self.data[0, 1])

    def test_streaming_projection_covers_all_z_without_stack_read_or_source_mutation(self):
        self.write_imagej(compression="deflate")
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        calls = []
        with patch.object(tf.TiffPageSeries, "asarray", side_effect=AssertionError("full stack read")):
            with PlaneReader(self.source) as reader:
                projections = self.scan(reader, progress=lambda n, total: calls.append((n, total)))["projections"]
        for c in range(3):
            self.assertEqual(projections[c].dtype, np.uint16)
            np.testing.assert_array_equal(projections[c], self.data[:, c].max(axis=0))
        self.assertEqual(calls, [(n, 4) for n in range(1, 5)])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), before)

    def test_single_ifd_contiguous_imagej_reads_last_plane_and_full_projection(self):
        self.write_imagej(truncate=True)
        with tf.TiffFile(self.source) as tif:
            self.assertEqual(len(tif.series[0].pages), 1)
        with PlaneReader(self.source) as reader:
            self.assertFalse(reader.mapping.flags.writeable)
            self.assertEqual(reader.info["plane_layout"], "imagej_contiguous")
            np.testing.assert_array_equal(reader.plane(2, 3), self.data[3, 2])
            np.testing.assert_array_equal(self.scan(reader)["projections"][0], self.data[:, 0].max(axis=0))

    def test_native_region_slices_before_copy_and_readonly_view(self):
        self.write_imagej(truncate=True)
        with PlaneReader(self.source) as reader, patch.object(reader,"plane",side_effect=AssertionError("full plane copy")):
            view=reader.read_region(2,slice(1,4),slice(5,15),slice(6,17),copy=False)
            self.assertFalse(view.flags.writeable)
            self.assertTrue(np.shares_memory(view,reader.mapping))
            np.testing.assert_array_equal(view,self.data[1:4,2,5:15,6:17])
            owned=reader.read_region(2,slice(1,4),slice(5,15),slice(6,17),copy=True)
            self.assertFalse(np.shares_memory(owned,reader.mapping))
            owned[:]=0
            np.testing.assert_array_equal(view,self.data[1:4,2,5:15,6:17])
            for bounds in [(slice(-1,2),slice(None),slice(None)),(slice(0,4,2),slice(None),slice(None))]:
                with self.assertRaises(CountingError):reader.read_region(0,*bounds)

    def test_compressed_regions_reuse_one_immutable_decoded_plane(self):
        self.write_imagej(compression="deflate")
        with PlaneReader(self.source) as reader:
            first=reader.read_region(0,slice(0,1),slice(2,7),slice(3,8),copy=False)
            self.assertFalse(first.flags.writeable)
            with patch.object(reader.series.pages[0],"asarray",side_effect=AssertionError("duplicate decode")):
                second=reader.read_region(0,slice(0,1),slice(5,10),slice(7,12),copy=True)
            np.testing.assert_array_equal(first,self.data[0:1,0,2:7,3:8])
            np.testing.assert_array_equal(second,self.data[0:1,0,5:10,7:12])
            self.assertLessEqual(reader._page_cache_bytes,reader._page_cache_limit)

    def test_big_endian_single_ifd_keeps_exact_uint16_values(self):
        self.write_imagej(truncate=True, byteorder=">")
        with PlaneReader(self.source) as reader:
            self.assertFalse(reader.mapping.dtype.isnative)
            np.testing.assert_array_equal(reader.plane(2, 3), self.data[3, 2])

    def test_ome_reordered_axes_units_and_multiple_time_require_selection(self):
        data = self.data.reshape(2, 3, 2, 21, 23)
        tf.imwrite(self.source, data, ome=True, photometric="minisblack", metadata={"axes": "TCZYX",
            "PhysicalSizeX": 91, "PhysicalSizeXUnit": "nm", "PhysicalSizeY": .12,
            "PhysicalSizeYUnit": "um", "Channel": {"Name": ["SR101", "DAPI 405", "Biocytin"]}})
        info = inspect_image(self.source)
        self.assertEqual(info["dapi_channel"], 1)
        self.assertAlmostEqual(info["physical_size_um"]["X"], .091)
        with self.assertRaises(CountingError) as cm:
            PlaneReader(self.source)
        self.assertEqual(cm.exception.code, "time_selection_required")
        with PlaneReader(self.source, time_index=1) as reader:
            np.testing.assert_array_equal(self.scan(reader)["projections"][1], data[1, 1].max(axis=0))

    def test_multiple_series_requires_selection(self):
        with tf.TiffWriter(self.source, ome=True) as writer:
            for _ in range(2):
                writer.write(self.data, photometric="minisblack", metadata={"axes": "ZCYX"})
        self.assertIsNone(inspect_image(self.source)["dapi_channel"])
        with self.assertRaises(CountingError) as cm:
            PlaneReader(self.source)
        self.assertEqual(cm.exception.code, "series_selection_required")

    def test_ambiguous_axes_rgb_and_invalid_indices_are_rejected(self):
        tf.imwrite(self.source, self.data, photometric="minisblack", metadata={"axes": "QCYX"})
        self.assertFalse(inspect_image(self.source)["supported_layout"])
        tf.imwrite(self.source, np.zeros((21, 23, 3), dtype=np.uint8), photometric="rgb")
        self.assertFalse(inspect_image(self.source)["supported_layout"])
        self.write_imagej()
        with PlaneReader(self.source) as reader:
            for c, z in [(-1, 0), (3, 0), (0, 4), (True, 0)]:
                with self.assertRaises(CountingError):
                    reader.plane(c, z)

    def test_projection_cancel_does_not_read_remaining_planes(self):
        self.write_imagej()
        calls = []
        def cancel():
            if len(calls) == 2:
                raise CountingCancelled()
        with PlaneReader(self.source) as reader, self.assertRaises(CountingCancelled):
            self.scan(reader, check_cancel=cancel, progress=lambda *_: calls.append(1))
        self.assertEqual(len(calls), 2)

    def test_anchors_keep_sparse_touching_ids_and_largest_component_interior(self):
        labels = np.zeros((24, 28), dtype=np.uint32)
        labels[1:12, 1:12] = 9
        labels[4:9, 4:9] = 0  # A ring whose centroid is outside its own pixels.
        labels[17:19, 1:3] = 9  # A disconnected fragment of the same instance.
        labels[1:12, 12:20] = 4000000000  # Touching, distinct, sparse ID.
        before = labels.copy()
        for value in (9,4000000000):
            x,y=interior_anchor(labels==value)
            self.assertEqual(labels[int(y),int(x)],value)
            self.assertLess(y,12)
        np.testing.assert_array_equal(labels, before)

    def test_empty_labels_and_illegal_label_types(self):
        self.assertFalse(validate_labels(np.zeros((3,3),dtype=np.int32),(1,3,3)).any())
        for labels in [np.ones((3, 3), dtype=bool), np.ones((3, 3), dtype=float), -np.ones((3, 3), dtype=np.int32)]:
            with self.assertRaises(CountingError):
                validate_labels(labels,(1,3,3))



    def test_display_colors_and_conversion_do_not_modify_source_values(self):
        raw = np.array([[1000, 1500, 2000]], dtype=np.uint16)
        display, limits = display_plane(raw)
        np.testing.assert_array_equal(raw, [[1000, 1500, 2000]])
        np.testing.assert_array_equal(display, [[0, 127, 255]])
        self.assertEqual(limits, [1000., 2000.])
        self.assertEqual(channel_colors(["DAPI", "GFAP", "KCNN2"]), ["#0000FF", "#00FF00", "#FF0000"])
        self.assertEqual(channel_colors(["DAPI", "eGFP", "GFAP", "KCNN2"]), ["#0000FF", "#00FF00", "#FFFF00", "#FF0000"])
        self.assertEqual(channel_colors(["Unknown"], ["#abcdEF"]), ["#ABCDEF"])

    def test_host_optional_and_sparse_colors_use_role_defaults(self):
        roles = ["DAPI", "eGFP", "GFAP", "Target", "custom marker"]
        self.assertEqual(channel_colors(roles, []), ["#0000FF", "#00FF00", "#FFFF00", "#FF0000", "#00FFFF"])
        self.assertEqual(channel_colors(roles, ["", "#abcdEF", "", "", ""]),
                         ["#0000FF", "#ABCDEF", "#FFFF00", "#FF0000", "#00FFFF"])
        self.assertEqual(channel_colors(["DAPI", "GFAP", "Target"], ["", "", ""]),
                         ["#0000FF", "#00FF00", "#FF0000"])
        for supplied in [[""], ["", "#bad", ""], ["", None, ""]]:
            with self.assertRaises(CountingError):
                channel_colors(["DAPI", "GFAP", "Target"], supplied)

    def test_aldh1l1_default_color_and_explicit_override(self):
        roles = ["ALDH1L1", "DAPI", "SR101"]
        self.assertEqual(channel_colors(roles), ["#00FF00", "#0000FF", "#FF0000"])
        self.assertEqual(channel_colors(roles, []), ["#00FF00", "#0000FF", "#FF0000"])
        self.assertEqual(channel_colors(roles, ["#ff0000", "", ""]),
                         ["#FF0000", "#0000FF", "#FF0000"])

    def check_aldh1l1_display_paths(self, supplied, rgb_index):
        self.write_imagej()
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        path = self.request(channel_roles=["ALDH1L1", "DAPI", "SR101"],
                            support_marker="ALDH1L1", display_channels=[0], channel_colors=supplied)
        with patch('leap_cell_counting.proposals.locate_windows', return_value=([], {'fixture_only': True})):
            self.assertEqual(self.execute(path, model_loader=lambda _: (None, {})), 0)
        analysis_path = path.parent/'counting/analysis.json'
        analysis = json.loads(analysis_path.read_text())
        expected_color = "#00FF00" if rgb_index == 1 else "#FF0000"
        self.assertEqual(analysis['channels'][0]['color'], expected_color)
        mip, _ = display_plane(self.data[:, 0].max(axis=0))
        expected = np.zeros((*mip.shape, 3), np.uint8)
        expected[..., rgb_index] = mip
        with Image.open(analysis_path.parent/'composite.png') as image:
            np.testing.assert_array_equal(image, expected)
        with SliceSession(analysis_path) as slices:
            result = slices.render({'z': 1, 'channels': [0], 'output': str(self.root/'aldh-slice.png')})
        plane, _ = display_plane(self.data[1, 0], analysis['channels'][0]['slice_display_range'])
        expected[..., rgb_index] = plane
        with Image.open(result['path']) as image:
            np.testing.assert_array_equal(image, expected)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), before)

    def test_aldh1l1_default_is_green_in_analysis_mip_and_z_slice(self):
        self.check_aldh1l1_display_paths([], 1)

    def test_aldh1l1_explicit_color_is_preserved_in_analysis_mip_and_z_slice(self):
        self.check_aldh1l1_display_paths(["#ff0000", "", ""], 0)

    def request(self, **updates):
        run = self.root / "run-1"
        run.mkdir(exist_ok=True)
        request = {"schema_version": 1, "run_id": "run-1", "source_path": str(self.source),
                   "series_index": 0, "time_index": 0, "dapi_channel": 1,
                   "support_channel": 0, "support_marker": "Biocytin", "display_channels": [0,1,2],
                   "channel_roles": ["Biocytin", "DAPI", "SR101"], "run_dir": str(run),
                   "events_path": str(run / "events.jsonl"), "outcome_path": str(run / "outcome.json"),
                   "cancel_path": str(run / "cancel.request")}
        request.update(updates)
        path = run / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        return path

    def execute(self, path, *, support_proposer=None, support_discover=None, model_loader=None, inference=None):
        def no_body_proposals(_summary, _counts, *, source_shape_zyx, **_kwargs):
            z, y, x = source_shape_zyx
            return {'regions': [], 'discovery_foreground_zyx': np.zeros((z, (y+15)//16, (x+15)//16), bool),
                    'discovery_grid_xy': 16, 'background_profile': None, 'intensity_reference': None,
                    'metrics': {'fixture_only': True}}
        with patch.dict(os.environ, {"PROJECT_LEAP_COMPONENT_LOCK": str(self.lock),
                "LEAP_COUNTING_STATS_LIBRARY": str(ROOT/'artifacts/prototype-d/native-counting/libLeapCountingStats.dylib')}), \
             patch("leap_cell_counting.support.measure_support", return_value={"decision":"accepted","metrics":{},"reasons":[],"review_hint":False}), \
             patch('leap_cell_counting.support_discovery.propose_support_regions', side_effect=support_proposer or no_body_proposals), \
             (patch('leap_cell_counting.support_discovery.discover_support_bodies', side_effect=support_discover)
              if support_discover is not None else nullcontext()), \
             (patch('leap_cell_counting.segmentation.load_model', side_effect=model_loader)
              if model_loader is not None else nullcontext()), \
             (patch('leap_cell_counting.xy_batch.infer_xy_batch', side_effect=inference)
              if inference is not None else nullcontext()), \
             (patch('leap_cell_counting.xy_batch.infer_boundary_context',
                    side_effect=lambda model, item, **kw: inference(model, [item], **kw)[0])
              if inference is not None else nullcontext()), \
             (patch('leap_cell_counting.xy_batch.permissive_from_fields',
                    side_effect=lambda model, prediction: prediction.get('weak_labels'))
              if inference is not None else nullcontext()), \
             patch.object(tempfile, "tempdir", tempfile.tempdir), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_request(path)

    def test_worker_body_only_is_one_counted_cell_with_real_body_anchors(self):
        self.write_imagej()
        reference = {'domain_labels_zyx': np.ones((4, 2, 2), np.uint32),
            'peak_excess_by_id': np.array([0., 75.], np.float32), 'grid_xy': 16,
            'amplitude_source': 'synthetic fixed original event', 'domain_is_cell_id': False}
        def propose(_summary, _counts, **kwargs):
            return {'regions': [{'region_id': 1, 'bounds_zyx': [[0, 4], [0, 21], [0, 23]]}],
                'discovery_foreground_zyx': np.ones((4, 2, 2), bool), 'discovery_grid_xy': 16,
                'background_profile': None, 'intensity_reference': reference, 'metrics': {'fixture_only': True}}
        def discover(_raw, _spacing, **kwargs):
            self.assertIs(kwargs['intensity_reference'], reference)
            mask = np.ones((3, 5, 5), bool)
            obj = {'local_label_id': 8, 'origin_zyx': [1, 2, 3], 'bounds_zyx': [[1, 4], [2, 7], [3, 8]],
                   'mask_zyx': mask, 'association_envelope_zyx': mask.copy(),
                   'plane_anchors': {str(z): [5, 4] for z in (1, 2, 3)}, 'representative_xyz': [5, 4, 2],
                   'decision': 'accepted', 'reasons': ['synthetic_body'], 'metrics': {'fixture_only': True},
                   'open_faces': [], 'requires_context_expansion': False}
            return {'objects': [obj], 'metrics': {}, 'rejected_regions': []}
        path = self.request()
        with patch('leap_cell_counting.proposals.locate_windows', return_value=([], {'fixture_only': True})):
            code = self.execute(path, model_loader=lambda _: (None, {}),
                inference=lambda *_a, **_k: self.fail('Body-only work must not invoke a nuclear model'),
                support_proposer=propose, support_discover=discover)
        self.assertEqual(code, 0)
        analysis = json.loads((path.parent/'counting/analysis.json').read_text())
        self.assertEqual(analysis['auto_count'], 1)
        self.assertEqual(analysis['pending_count'], 0)
        point = analysis['points'][0]
        self.assertEqual(point['source'], 'support_body')
        self.assertEqual(point['label_artifact_id'], 'support_bodies')
        self.assertEqual(len(point['source_observations']), 1)
        ledger = json.loads((path.parent/'counting/body-work-ledger.json').read_text())
        np.testing.assert_array_equal(np.load(path.parent/'counting'/ledger['intensity_reference']['domain_labels_path']),
                                      reference['domain_labels_zyx'])
        np.testing.assert_array_equal(np.load(path.parent/'counting'/ledger['intensity_reference']['peak_excess_path']),
                                      reference['peak_excess_by_id'])
        nuclei = np.load(path.parent/'counting/labels.npy', mmap_mode='r')
        bodies = np.load(path.parent/'counting/body-labels.npy', mmap_mode='r')
        self.assertFalse(nuclei.any())
        for z, (x, y) in point['plane_anchors'].items():
            self.assertEqual(bodies[int(z), int(y), int(x)], point['label_id'])
            self.assertEqual(point['plane_anchor_sources'][z]['observation_id'], point['source_observations'][0]['id'])
        nuclei._mmap.close(); bodies._mmap.close()

    def test_worker_raw_native_volume_labels_and_true_plane_anchors(self):
        self.write_imagej()
        observed = []
        labels = np.zeros((4,21,23), dtype=np.uint32)
        labels[0:2,2:6,2:7] = 7
        labels[2:4,10:14,11:15] = 88
        def inference(_model, candidates, **kwargs):
            observed.append(candidates[0]['image'].copy())
            self.assertFalse(kwargs['include_weak'])
            return [{'labels': labels.max(axis=0), 'dapi_cellprob': np.zeros(labels.shape[1:])}]
        def recover(*args):
            return {'labels': labels, 'origin_zyx': [0,0,0], 'window_id': 1,
                'objects': [{'label_id': i, 'flags': [], 'open_boundary_faces': []} for i in (7,88)],
                'unmatched_families': [], 'endpoint_expansion': [], 'seconds': 0., 'metrics': {}}
        window={'window_id':1,'bounds_zyx':[[0,4],[0,21],[0,23]],'localization_seed_zyx':[0,3,3]}
        path = self.request()
        with patch('leap_cell_counting.proposals.locate_windows',return_value=([window],{'fixture_only':True})), \
             patch('leap_cell_counting.efficient._recover',side_effect=recover):
            code = self.execute(path, model_loader=lambda _: (None, {"name":"test-only-stub"}), inference=inference)
        self.assertEqual(code,0)
        np.testing.assert_array_equal(observed[0],self.data[:,1].max(axis=0))
        outcome=json.loads((path.parent/"outcome.json").read_text())
        self.assertEqual(outcome["status"],"review_ready")
        analysis_path=Path(outcome["analysis_path"])
        analysis=json.loads(analysis_path.read_text())
        self.assertEqual(analysis["schema_version"],2)
        self.assertEqual(analysis["auto_count"],2)
        self.assertEqual(analysis["pending_count"],0)
        self.assertEqual(analysis['identity_model'], 'multi_observation_cell_v1')
        self.assertEqual(set(analysis['label_artifacts']), {'dapi_nuclei', 'support_bodies'})
        self.assertEqual(analysis["support_channel"],0)
        stored=np.load(analysis_path.parent/analysis["labels_path"],mmap_mode="r")
        self.assertFalse(stored.flags.writeable)
        self.assertEqual(stored.shape,(4,21,23))
        self.assertEqual([set(p["plane_anchors"]) for p in analysis["points"]],[{"0","1"},{"2","3"}])
        for point in analysis["points"]:
            for z,(x,y) in point["plane_anchors"].items():
                self.assertEqual(stored[int(z),int(y),int(x)],point["label_id"])
        stored._mmap.close()
        self.assertFalse((analysis_path.parent/".dapi-work.npy").exists())
        for value in [analysis["preview_path"],analysis["labels_path"],*(c["path"] for c in analysis["channels"])]:
            self.assertFalse(Path(value).is_absolute())
            self.assertTrue((analysis_path.parent/value).is_file())
        analysis["source_path"]=os.path.relpath(self.source,analysis_path.parent)
        analysis_path.write_text(json.dumps(analysis))
        destination=self.root/"slice.png"
        create_slice(analysis_path,1,destination)
        with Image.open(destination) as image:
            expected,_=display_plane(self.data[1,1],analysis["channels"][1]["slice_display_range"])
            np.testing.assert_array_equal(image,expected)

    def test_inference_stage_records_actual_model_snapshot(self):
        self.write_imagej()
        path = self.request()
        model_info = {'name': 'fixture-model', 'device': 'fixture-device',
                      'version': 'fixture-version', 'parameters': dict(EVAL_PARAMETERS)}
        original_emit = Events.emit
        snapshots = []
        def emit(events, event_type, stage, message, **details):
            if stage == 'count_inference' and event_type == 'stage':
                self.assertFalse((path.parent/'counting/analysis.json').exists())
                snapshot = details['model']
                self.assertEqual(snapshot, model_info)
                self.assertIsNot(snapshot, model_info)
                self.assertIsNot(snapshot['parameters'], model_info['parameters'])
                self.assertIsNot(snapshot['calibration'], model_info['calibration'])
                snapshots.append(snapshot)
            return original_emit(events, event_type, stage, message, **details)
        def no_inference(*_args, **_kwargs):
            self.fail('Empty fixture proposals must not invoke the model')
        with patch('leap_cell_counting.proposals.locate_windows', return_value=([], {'fixture_only': True})), \
             patch.object(Events, 'emit', emit):
            self.assertEqual(self.execute(path, model_loader=lambda _: (None, model_info), inference=no_inference), 0)
        analysis = json.loads((path.parent/'counting/analysis.json').read_text())
        events = [json.loads(line) for line in (path.parent/'events.jsonl').read_text().splitlines()]
        model_events = [event for event in events if 'model' in event]
        self.assertEqual(len(model_events), 1)
        self.assertEqual(model_events[0]['type'], 'stage')
        self.assertEqual(model_events[0]['stage'], 'count_inference')
        self.assertEqual(model_events[0]['model'], analysis['model'])
        self.assertEqual(snapshots, [analysis['model']])
        self.assertEqual({event['type'] for event in events}, {'stage', 'progress', 'review_ready'})
        self.assertEqual(analysis['model']['name'], 'fixture-model')
        self.assertEqual(analysis['model']['device'], 'fixture-device')
        self.assertEqual(analysis['model']['parameters'], {**EVAL_PARAMETERS, 'normalize': False})
        self.assertTrue(analysis['timings']['route_times_overlap_and_must_not_be_added'])
        self.assertGreaterEqual(analysis['timings']['total_seconds'], 0)

    def test_worker_requires_distinct_support_and_explicit_marker(self):
        self.write_imagej()
        for index,values,expected in [(0,{"support_channel":None},"support_selection_required"),
                (1,{"support_channel":1},"support_equals_dapi"),
                (2,{"support_marker":""},"support_marker_required")]:
            run=self.root/f"invalid-support-{index}";run.mkdir()
            path=run/"request.json"
            base=json.loads(self.request().read_text())
            base.update(run_dir=str(run),events_path=str(run/"events.jsonl"),
                        outcome_path=str(run/"outcome.json"),cancel_path=str(run/"cancel.request"),**values)
            path.write_text(json.dumps(base))
            self.assertEqual(self.execute(path,model_loader=lambda _: self.fail("invalid support loaded model")),1)
            self.assertEqual(json.loads((run/"outcome.json").read_text())["error_code"],expected)

    def test_no_cp_seed_does_not_reread_or_reconstruct_native_volume(self):
        self.write_imagej()
        window={'window_id':1,'bounds_zyx':[[0,4],[0,21],[0,23]],'localization_seed_zyx':[1,10,11]}
        def inference(_model,candidates,**kwargs):
            shape=candidates[0]['image'].shape
            return [{'labels':np.zeros(shape,np.uint32),'weak_labels':np.zeros(shape,np.uint32),
                     'dapi_cellprob':np.full(shape,-4.,np.float32)}]
        path=self.request()
        with patch('leap_cell_counting.proposals.locate_windows',return_value=([window],{'fixture_only':True})), \
             patch('leap_cell_counting.efficient._recover',side_effect=AssertionError('zero seed must not read/recover native ROI')):
            self.assertEqual(self.execute(path,model_loader=lambda _: (None,{}),inference=inference),0)
        diagnostics=json.loads((path.parent/'counting/diagnostics.json').read_text())
        self.assertEqual(diagnostics['endpoint_expansions'][0]['metrics']['reason'],'no_primary_or_permissive_cp_seed')
        self.assertFalse(diagnostics['endpoint_expansions'][0]['metrics']['native_recovery_evaluated'])

    def test_worker_single_plane_preserves_unknown_z_and_explicit_input_mode(self):
        self.write_imagej(self.data[0],metadata={"Info":json.dumps({"pixel_um":[.1,.1,None],"z_spacing_status":"unknown"}),
                                               "spacing":None})
        def inference(_model,candidates,**kwargs):
            self.fail('Empty fixture proposals should not call the model')
        path=self.request()
        with patch('leap_cell_counting.proposals.locate_windows',return_value=([],{'fixture_only':True})):
            self.assertEqual(self.execute(path,model_loader=lambda _: (None,{}),inference=inference),0)
        analysis=json.loads((path.parent/"counting/analysis.json").read_text())
        self.assertEqual(analysis["model"]["input_mode"],"single_plane_2d")
        self.assertIsNone(analysis["physical_size_um"]["Z"])
        self.assertFalse(analysis["model"]["parameters"]["do_3D"])

    def test_worker_cancellation_and_model_failure_do_not_publish_review(self):
        self.write_imagej()
        path = self.request()
        (path.parent / "cancel.request").touch()
        code = self.execute(path, model_loader=lambda _: self.fail("cancelled run loaded model"))
        self.assertEqual(code, 130)
        self.assertEqual(json.loads((path.parent / "outcome.json").read_text())["status"], "cancelled")
        self.assertFalse((path.parent / "counting/analysis.json").exists())

    def test_model_failure_has_terminal_failed_outcome_without_analysis(self):
        self.write_imagej()
        path = self.request()
        def failed_loader(_check):
            raise CountingError("model_missing", "test-only unavailable model")
        self.assertEqual(self.execute(path, model_loader=failed_loader), 1)
        outcome = json.loads((path.parent / "outcome.json").read_text())
        self.assertEqual(outcome["status"], "failed")
        self.assertEqual(outcome["error_code"], "model_missing")
        self.assertFalse((path.parent / "counting/analysis.json").exists())

    def test_worker_requires_consistent_explicit_dapi_and_rejects_prior_run(self):
        self.write_imagej()
        path = self.request(dapi_channel=-1)
        self.assertEqual(self.execute(path), 1)
        self.assertEqual(json.loads((path.parent / "outcome.json").read_text())["error_code"], "dapi_selection_required")
        with self.assertRaises(CountingError) as cm:
            self.execute(path)
        self.assertEqual(cm.exception.code, "existing_run")

    def test_component_lock_rejects_exclusive_maintenance_without_replacing_inode(self):
        inode = self.lock.stat().st_ino
        with self.lock.open("r") as handle, patch.dict(os.environ, {"PROJECT_LEAP_COMPONENT_LOCK": str(self.lock)}):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(CountingError) as cm:
                with component_usage_lock(required=True):
                    self.fail("exclusive lock was ignored")
            self.assertEqual(cm.exception.code, "component_busy")
        self.assertEqual(self.lock.stat().st_ino, inode)

    def test_inspect_cli_and_slice_do_not_import_torch(self):
        self.write_imagej(metadata={"Labels": ["SR101", "DAPI", "Biocytin"]})
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "engine") + os.pathsep + str(ROOT / "runtime/seed/site-packages"))
        environment.pop("PROJECT_LEAP_COMPONENT_LOCK", None)
        code = "import sys; from leap_cell_counting.__main__ import main; rc=main(sys.argv[1:]); assert 'torch' not in sys.modules; raise SystemExit(rc)"
        result = subprocess.run([sys.executable, "-B", "-c", code, "--inspect", str(self.source),
                                 "--preview-path", str(self.root / "preview.png")], env=environment,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["dapi_channel"], 1)
        self.assertTrue((self.root / "preview.png").is_file())
        analysis = self.root / "analysis.json"
        analysis.write_text(json.dumps({"source_path": "source/中文 input.tif", "series_index": 0,
            "time_index": 0, "dapi_channel": 1, "height": 21, "width": 23,
            "channels": [{}, {"display_range": [0., 6000.]}, {}]}))
        result = subprocess.run([sys.executable, "-B", "-c", code, "--slice", str(analysis),
                                 "--z", "3", "--output", str(self.root / "slice.png")], env=environment,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["z_index"], 3)
        self.assertTrue((self.root / "slice.png").is_file())
        requests=[{"id":1,"z":0,"channels":[1],"output":str(self.root/"server-0.png")},
                  {"id":2,"z":3,"channels":[0,2],"output":str(self.root/"server-3.png")}]
        result=subprocess.run([sys.executable,"-B","-c",code,"--slice-server",str(analysis)],
            input="".join(json.dumps(item)+"\n" for item in requests),env=environment,capture_output=True,text=True,check=True)
        rows=[json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r["id"] for r in rows],[1,2])
        self.assertEqual([r["z"] for r in rows],[0,3])
        self.assertTrue(all(Path(r["path"]).is_file() for r in rows))


if __name__ == "__main__":
    unittest.main()
