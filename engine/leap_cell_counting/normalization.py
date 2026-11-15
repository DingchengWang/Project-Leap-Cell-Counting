"""Exact source histogram/display statistics; inference uses per-feature limits."""
import numpy as np
from .errors import CountingError


def new_histogram(dtype):
    dtype=np.dtype(dtype)
    if dtype.kind!='u' or dtype.itemsize not in (1,2):
        raise CountingError('unsupported_normalization_dtype',
            '当前原生统计支持原始 uint8/uint16 TIFF；其他位深尚未验证。')
    return np.zeros(1 << (8*dtype.itemsize),dtype=np.uint64)


def histogram_percentiles(histogram, percentiles=(1.,99.)):
    """Match NumPy's linear percentiles, including genuinely constant sources."""
    cumulative=np.cumsum(histogram);total=int(cumulative[-1])
    if not total:raise CountingError('empty_dapi_volume','DAPI 体积没有像素。')
    def percentile(percent):
        position=(total-1)*percent/100.;lower=int(np.floor(position));upper=int(np.ceil(position))
        low=int(np.searchsorted(cumulative,lower+1));high=int(np.searchsorted(cumulative,upper+1))
        return float(low+(high-low)*(position-lower))
    return [percentile(p) for p in percentiles]
