"""Exact native statistics ABI and integer/dtype invariants, no segmentation."""
import ctypes
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine'))
from leap_cell_counting.statistics import summarize_plane,combine_2x2,library_path
from leap_cell_counting.errors import CountingError
LIBRARY=ROOT/'artifacts/prototype-d/native-counting/libLeapCountingStats.dylib'

class StatisticsTests(unittest.TestCase):
    def setUp(self):
        if not LIBRARY.is_file():self.skipTest('Build the signed native statistics test component first')
        self.environment=patch.dict(os.environ,{'LEAP_COUNTING_STATS_LIBRARY':str(LIBRARY)})
        self.environment.start()
    def tearDown(self):
        if hasattr(self,'environment'):self.environment.stop()
    def check(self,data,workers):
        before=data.copy();result=summarize_plane(data,workers=workers)
        for gy in range(result['sum'].shape[0]):
            for gx in range(result['sum'].shape[1]):
                part=data[gy*8:(gy+1)*8,gx*8:(gx+1)*8].astype(np.uint64)
                self.assertEqual(int(result['sum'][gy,gx]),int(part.sum()))
                self.assertEqual(int(result['sum_sq'][gy,gx]),int((part*part).sum()))
                self.assertEqual(int(result['max'][gy,gx]),int(part.max()))
                self.assertEqual(int(result['count'][gy,gx]),part.size)
        np.testing.assert_array_equal(result['histogram'],np.bincount(data.astype(np.uint64).ravel().astype(np.int64),
             minlength=256 if data.dtype.itemsize==1 else 65536).astype(np.uint64))
        np.testing.assert_array_equal(before,data)
        self.assertFalse(result['summary_is_segmentation_input'])
        return result
    def test_uint8_random_extreme_partial_edges_and_worker_partitions(self):
        rng=np.random.default_rng(15)
        for workers in (1,4,8):
            for data in (rng.integers(0,256,(19,25),dtype=np.uint8),np.full((17,20),255,np.uint8)):
                result=self.check(data,workers)
                self.assertEqual(result['sum'].dtype,np.uint16);self.assertEqual(result['sum_sq'].dtype,np.uint32)
    def test_uint16_both_byte_orders_exact_large_square_sums(self):
        rng=np.random.default_rng(16)
        for dtype in ('<u2','>u2'):
            for workers in (1,4,8):
                for data in (rng.integers(0,65536,(19,25),dtype=np.uint16).astype(dtype),np.full((17,20),65535,dtype=dtype)):
                    result=self.check(data,workers)
                    self.assertEqual(result['sum'].dtype,np.uint32);self.assertEqual(result['sum_sq'].dtype,np.uint64)
        result=self.check(np.full((8,8),65535,np.uint16),4)
        self.assertEqual(int(result['sum'][0,0]),4194240)
        self.assertEqual(int(result['sum_sq'][0,0]),274869518400)
    def test_row_strided_readonly_roi_and_invalid_pixel_strides(self):
        raw=np.arange(31*43,dtype=np.uint16).reshape(31,43);raw.flags.writeable=False
        self.check(raw[2:24,3:29],4)
        for data in (raw.T,raw[::-1],raw[:,::2],np.zeros((4,4),np.float32)):
            with self.assertRaises(CountingError):summarize_plane(data)
        for workers in (0,33,True,2.5):
            with self.assertRaises(CountingError):summarize_plane(raw,workers=workers)
    def test_exact_16x16_aggregation_preserves_all_original_moments(self):
        for dtype in (np.uint8,np.uint16):
            top=np.iinfo(dtype).max;data=np.full((35,41),top,dtype);data[1::3,2::4]-=1
            result=combine_2x2(self.check(data,4))
            for y in range(result['sum'].shape[0]):
                for x in range(result['sum'].shape[1]):
                    part=data[y*16:(y+1)*16,x*16:(x+1)*16].astype(np.uint64)
                    self.assertEqual(int(result['sum'][y,x]),int(part.sum()))
                    self.assertEqual(int(result['sum_sq'][y,x]),int((part*part).sum()))
                    self.assertEqual(int(result['max'][y,x]),int(part.max()))
                    self.assertEqual(int(result['count'][y,x]),part.size)
            with self.assertRaises(CountingError):combine_2x2(result)
    def test_missing_library_never_runs_a_fallback(self):
        with patch.dict(os.environ,{'LEAP_COUNTING_STATS_LIBRARY':str(ROOT/'artifacts/prototype-d/missing-library.dylib')}):
            with self.assertRaises(CountingError):library_path()
        with patch.dict(os.environ,{'LEAP_COUNTING_STATS_LIBRARY':'relative.dylib'}):
            with self.assertRaises(CountingError):library_path()

if __name__=='__main__':unittest.main()
