"""Exact native-pixel statistics for localization; these arrays are never labels.

The signed component contains the prebuilt library. Development may select an
explicit absolute LEAP_COUNTING_STATS_LIBRARY path; no runtime compiler or
alternate numerical implementation is used when the library is unavailable.
"""
from __future__ import annotations
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import sys
import numpy as np
from .errors import CountingError

ABI_VERSION=1
BLOCK_SIZE=8


@lru_cache(maxsize=2)
def _load_library(path):
    try:library=ctypes.CDLL(path)
    except OSError as exc:
        raise CountingError('statistics_library_unavailable','计数原生统计组件不可用，请修复组件。') from exc
    library.leap_counting_stats_abi_version.argtypes=[]
    library.leap_counting_stats_abi_version.restype=ctypes.c_int
    if library.leap_counting_stats_abi_version()!=ABI_VERSION:
        raise CountingError('statistics_library_incompatible','计数原生统计组件 ABI 版本不匹配。')
    prefix=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t]
    outputs=[ctypes.c_void_p]*4+[ctypes.c_size_t]
    library.leap_counting_stats_u8.argtypes=prefix+outputs
    library.leap_counting_stats_u16.argtypes=prefix+[ctypes.c_int]+outputs
    library.leap_counting_stats_u8.restype=ctypes.c_int
    library.leap_counting_stats_u16.restype=ctypes.c_int
    return library


def library_path():
    supplied=os.environ.get('LEAP_COUNTING_STATS_LIBRARY')
    path=Path(supplied) if supplied else Path(__file__).with_name('libLeapCountingStats.dylib')
    if not path.is_absolute():
        raise CountingError('statistics_library_path','原生统计组件路径必须为绝对路径。')
    if not path.is_file():
        raise CountingError('statistics_library_unavailable','缺少随计数组件预编译的原生统计库。')
    return path.resolve()


def summarize_plane(plane,*,workers=4):
    """Return exact 8x8 sum, sum_sq, max, counts and a full-plane histogram.

    Input is a two-dimensional uint8/uint16 view. Rows may be strided; X pixels
    must be consecutive and row strides positive. uint16 byte order is explicit
    to the native kernel. Each call owns its buffers and is thread-safe.
    """
    data=np.asarray(plane)
    if data.ndim!=2 or data.dtype.kind!='u' or data.dtype.itemsize not in (1,2) or not data.size:
        raise CountingError('unsupported_statistics_input','原生统计需要非空 uint8/uint16 二维平面。')
    if data.strides[1]!=data.dtype.itemsize or data.strides[0]<data.shape[1]*data.dtype.itemsize:
        raise CountingError('unsupported_statistics_layout','原生统计需要连续 X 像素与正向 Y 行视图。')
    if type(workers) is not int or not 1<=workers<=32:
        raise CountingError('invalid_statistics_workers','原生统计 worker 数必须为 1 至 32。')
    height,width=data.shape;shape=((height+7)//8,(width+7)//8)
    is_u8=data.dtype.itemsize==1
    sums=np.empty(shape,np.uint16 if is_u8 else np.uint32)
    squares=np.empty(shape,np.uint32 if is_u8 else np.uint64)
    peaks=np.empty(shape,np.uint8 if is_u8 else np.uint16)
    bins=256 if is_u8 else 65536
    histogram_rows=np.empty((workers,bins),np.uint64)
    library=_load_library(str(library_path()))
    prefix=[data.ctypes.data,height,width,data.strides[0]]
    outputs=[sums.ctypes.data,squares.ctypes.data,peaks.ctypes.data,histogram_rows.ctypes.data,workers]
    if is_u8:status=library.leap_counting_stats_u8(*prefix,*outputs)
    else:
        endian=data.dtype.byteorder
        little=int(endian=='<' or (endian in ('=','|') and sys.byteorder=='little'))
        status=library.leap_counting_stats_u16(*prefix,little,*outputs)
    if status:raise CountingError('statistics_kernel_failure',f'原生统计输入校验失败：{status}。')
    rows=np.minimum(8,height-np.arange(shape[0])*8)
    columns=np.minimum(8,width-np.arange(shape[1])*8)
    return {'sum':sums,'sum_sq':squares,'max':peaks,
        'count':(rows[:,None]*columns[None,:]).astype(np.uint16),
        'histogram':histogram_rows.sum(axis=0,dtype=np.uint64),'block_size':8,
        'source_shape_yx':[height,width],'source_dtype':str(data.dtype),
        'summary_is_segmentation_input':False}


def combine_2x2(summary):
    """Derive exact 16x16 moments from 8x8; never average maxima or variances."""
    if summary.get("block_size")!=8:
        raise CountingError("invalid_statistics_pool","16×16 摘要必须由原始 8×8 摘要合成。")
    result={}
    for name in ('sum','sum_sq','max','count'):
        values=np.asarray(summary[name])
        padded=np.pad(values,((0,values.shape[0]%2),(0,values.shape[1]%2)))
        cells=padded.reshape(padded.shape[0]//2,2,padded.shape[1]//2,2)
        result[name]=(cells.max(axis=(1,3)) if name=='max' else cells.sum(axis=(1,3),dtype=np.uint64)).astype(values.dtype)
    result.update(block_size=16,source_shape_yx=list(summary['source_shape_yx']),source_dtype=summary['source_dtype'],
                  summary_is_segmentation_input=False)
    return result
