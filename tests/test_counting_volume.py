"""Bounded volume and protocol invariants; oracle masks do not validate biology."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch
import numpy as np
from scipy import ndimage as ndi
import tifffile
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine'))
from leap_cell_counting.volume import _external_distance_um, local_dapi_evidence
from leap_cell_counting.support import measure_support
from leap_cell_counting.segmentation import interior_anchor
from leap_cell_counting.imaging import SliceSession,serve_slices
from leap_cell_counting.errors import CountingCancelled,CountingError
from leap_cell_counting.normalization import new_histogram,histogram_percentiles


class SharedDistanceTests(unittest.TestCase):
    def test_one_readonly_distance_preserves_both_measurements_exactly(self):
        for depth, spacing in ((1, (.3, .5, None)), (9, (.3, .5, 1.25))):
            with self.subTest(depth=depth):
                mask = np.zeros((depth, 35, 41), dtype=bool)
                mask[0 if depth == 1 else 3:1 if depth == 1 else 6, 14:21, 17:24] = True
                raw = np.arange(mask.size, dtype=np.float32).reshape(mask.shape) % 17
                raw[mask] += 30
                support = raw * np.float32(2)
                if depth > 1:
                    raw[4, 16, 20] = np.nan
                    support[4, 16, 20] = np.nan
                    support[2, 3, 4] = np.inf
                original = mask.copy(), raw.copy(), support.copy()
                expected_support = measure_support(mask, support, spacing, "eGFP", diagnostics=True)
                expected_dapi = local_dapi_evidence(mask, raw, spacing)
                expected_distance = (ndi.distance_transform_edt(~mask[0], sampling=spacing[:2][::-1])[None]
                                     if depth == 1 else ndi.distance_transform_edt(~mask, sampling=spacing[::-1]))
                with patch.object(ndi, "distance_transform_edt", wraps=ndi.distance_transform_edt) as transform:
                    distance = _external_distance_um(mask, spacing)
                    distance.flags.writeable = False
                    actual_support = measure_support(mask, support, spacing, "eGFP", diagnostics=True,
                                                     _distance_zyx=distance)
                    actual_dapi = local_dapi_evidence(mask, raw, spacing, _distance_zyx=distance)
                self.assertEqual(transform.call_count, 1)
                self.assertEqual(distance.shape, mask.shape)
                self.assertEqual(distance.dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(distance, expected_distance)
                self.assertEqual(actual_support, expected_support)
                self.assertEqual(actual_dapi, expected_dapi)
                json.dumps([actual_support, actual_dapi], allow_nan=False)
                for actual, before in zip((mask, raw, support), original):
                    np.testing.assert_array_equal(actual, before)

    def test_reuse_preserves_empty_and_missing_calibration_results(self):
        mask = np.zeros((3, 9, 9), dtype=bool)
        raw = np.zeros(mask.shape, dtype=np.uint8)
        distance = _external_distance_um(mask, (1., 1., 1.))
        self.assertEqual(local_dapi_evidence(mask, raw, None, _distance_zyx=distance),
                         local_dapi_evidence(mask, raw, None))
        self.assertEqual(measure_support(mask, raw, None, "eGFP", _distance_zyx=distance),
                         measure_support(mask, raw, None, "eGFP"))
        mask[1, 4, 4] = True
        distance = _external_distance_um(mask, (1., 1., 1.))
        for spacing in ((1., 1., None), (1., 1., 0), (1., 1.)):
            self.assertEqual(measure_support(mask, raw, spacing, "eGFP", _distance_zyx=distance),
                             measure_support(mask, raw, spacing, "eGFP"))
        with self.assertRaises(ValueError):
            measure_support(mask, raw, (1., 1., 1.), "", _distance_zyx=distance)


class VolumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts/prototype-d/engine',prefix='volume-')
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_quantized_weak_nucleus_equal_median_preserves_mean_and_candidate(self):
        raw=np.full((7,35,35),2,np.uint8)
        mask=np.zeros_like(raw,bool);mask[2:5,12:23,12:23]=True
        core=raw[mask].copy();core[::3]=14;raw[mask]=core
        evidence=local_dapi_evidence(mask,raw,(.2,.2,.3))
        self.assertEqual(evidence['nuclear_median'],2.)
        self.assertEqual(evidence['background_median'],2.)
        self.assertGreater(evidence['nuclear_mean'],evidence['background_mean'])
        self.assertEqual(evidence['z_planes_local'],[2,3,4])
        self.assertTrue(evidence['z_intersections_contiguous'])
    def test_local_dapi_single_plane_has_no_invented_z_spacing(self):
        raw=np.zeros((1,41,41),np.uint8);raw[:,18:23,18:23]=100
        evidence=local_dapi_evidence(raw>0,raw,(.2,.2,None))
        self.assertTrue(evidence['evaluable'])
    def test_common_histogram_limits_match_exact_full_source_percentiles(self):
        for dtype,maximum in [(np.uint8,255),(np.uint16,65535)]:
            data=np.linspace(0,maximum,3*43*59,dtype=dtype).reshape(3,43,59)
            histogram=new_histogram(dtype)
            histogram[:]=np.bincount(data.ravel(),minlength=len(histogram))
            np.testing.assert_allclose(histogram_percentiles(histogram),np.percentile(data,[1,99]))
            self.assertEqual(int(histogram.sum()),data.size)
        with self.assertRaises(CountingError):new_histogram(np.float32)
        histogram=new_histogram(np.uint8);histogram[2]=1000
        self.assertEqual(histogram_percentiles(histogram),[2.,2.])

    def test_persistent_multichannel_same_z_and_error_recovery(self):
        data=np.zeros((4,3,11,13),np.uint8)
        for z in range(4):
            for c in range(3):data[z,c]=z*20+c*7
        source=self.root/'source/input.tif';source.parent.mkdir()
        tifffile.imwrite(source,data,imagej=True,metadata={'axes':'ZCYX'})
        analysis={'schema_version':2,'source_path':str(source),'series_index':0,'time_index':0,
            'dapi_channel':0,'width':13,'height':11,'display_channels':[0,1,2],
            'channels':[{'color':color,'display_range':[0,255], 'slice_display_range':[0,100] if i==0 else [0,255]}
                        for i,color in enumerate(['#FF0000','#00FF00','#0000FF'])],
            'points':[{'id':'nucleus','plane_anchors':{'2':[1.5,2.5]}}]}
        path=self.root/'analysis.json';path.write_text(json.dumps(analysis))
        requests=[{'id':1,'z':2,'channels':[0,2],'output':str(self.root/'two.png')},
                  {'id':2,'z':99,'channels':[0],'output':str(self.root/'bad.png')},
                  {'id':3,'z':1,'channels':[],'output':str(self.root/'empty.png')}]
        stream=io.StringIO();serve_slices(path,io.StringIO(''.join(json.dumps(r)+'\n' for r in requests)),stream)
        results=[json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(results[0]['plane_anchors'],{'nucleus':[1.5,2.5]})
        self.assertIn('error',results[1]);self.assertNotIn('error',results[2])
        with Image.open(self.root/'two.png') as image:
            pixels=np.array(image);np.testing.assert_array_equal(pixels[0,0],[102,0,54])
        with Image.open(self.root/'empty.png') as image:self.assertFalse(np.array(image).any())
    def test_slice_source_change_is_detected_before_frame(self):
        data=np.zeros((2,1,11,13),np.uint8);source=self.root/'source/input.tif';source.parent.mkdir()
        tifffile.imwrite(source,data,imagej=True,metadata={'axes':'ZCYX'})
        stat=source.stat();path=self.root/'analysis.json'
        path.write_text(json.dumps({'source_path':str(source),'series_index':0,'time_index':0,'dapi_channel':0,
            'width':13,'height':11,'channels':[{'display_range':[0,255]}],
            'source_signature':{'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}}))
        with SliceSession(path) as session:
            import os
            os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000))
            with self.assertRaises(CountingError):session.render({'z':1,'output':str(self.root/'changed.png')})
        self.assertFalse((self.root/'changed.png').exists())




if __name__=='__main__':unittest.main()
