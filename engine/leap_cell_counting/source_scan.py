"""One native source pass for exact localization statistics and display MIPs."""
from pathlib import Path
import time
import numpy as np
from .statistics import summarize_plane, combine_2x2
from .normalization import new_histogram


def scan_source(reader, output, dapi, support, display_channels, *, check_cancel=lambda: None,
                progress=lambda *_: None):
    """Read every Z and every pixel of each necessary channel exactly once.

    Only the selected display channels receive MIPs. Raw DAPI is never copied
    to a second whole-volume file. The exact moments remain localization data.
    """
    from .imaging import native_plane_digest, source_metadata
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    z_count = reader.info['z_count']
    height, width = reader.series.shape[-2:]
    dtype = np.dtype(reader.series.dtype)
    names = {'dapi': dapi, 'support': support}
    histograms = {name: new_histogram(dtype) for name in names}
    shape = (z_count, (height + 7)//8, (width + 7)//8)
    coarse_shape = (z_count, (height + 15)//16, (width + 15)//16)
    small = dtype.itemsize == 1
    dtypes = {'sum': np.uint16 if small else np.uint32,
              'sum_sq': np.uint32 if small else np.uint64,
              'max': np.uint8 if small else np.uint16}
    arrays = {}
    for grid, dims in ((8, shape), (16, coarse_shape)):
        folder = output / ('summary' if grid == 8 else 'summary16')
        folder.mkdir()
        arrays[grid] = {name: {key: np.lib.format.open_memmap(folder/f'{name}-{key}.npy',
                        mode='w+', dtype=value, shape=dims) for key, value in dtypes.items()}
                        for name in names}
    projections = {c: np.zeros((height, width), dtype=dtype) for c in display_channels}
    channels = sorted(set(names.values()) | set(display_channels))
    plane_digests = []
    started = time.perf_counter()
    timings = {'statistics_seconds': 0., 'display_projection_seconds': 0.}
    counts = {}
    for z in range(z_count):
        for channel in channels:
            check_cancel()
            plane = reader.read_region(channel, slice(z, z+1), slice(0, height),
                                       slice(0, width), copy=False, check_cancel=check_cancel)[0]
            # Reuse the native scan: no second TIFF read, no raw-volume copy.
            plane_digests.append({'channel': channel, 'z': z, 'sha256': native_plane_digest(plane)})
            for name, index in names.items():
                if channel != index:
                    continue
                step = time.perf_counter()
                fine = summarize_plane(plane, workers=4)
                coarse = combine_2x2(fine)
                for grid, values in ((8, fine), (16, coarse)):
                    for key in dtypes:
                        arrays[grid][name][key][z] = values[key]
                    counts[grid] = values['count']
                histograms[name] += fine['histogram']
                timings['statistics_seconds'] += time.perf_counter()-step
            if channel in projections:
                step = time.perf_counter()
                np.maximum(projections[channel], plane, out=projections[channel])
                timings['display_projection_seconds'] += time.perf_counter()-step
        progress(z+1, z_count)
    for grid in arrays:
        folder = output / ('summary' if grid == 8 else 'summary16')
        np.save(folder/'count.npy', counts[grid])
        for name in names:
            for value in arrays[grid][name].values():
                value.flush()
    for name, histogram in histograms.items():
        if int(histogram.sum()) != z_count*height*width:
            raise RuntimeError('Native scan pixel-accounting mismatch')
        np.save(output/f'{name}-histogram.npy', histogram)
    timings['total_seconds'] = time.perf_counter()-started
    return {'moments': arrays, 'counts': counts, 'histograms': histograms,
            'source_identity': {'method': 'native_plane_sha256_v1',
                                'metadata': source_metadata(reader), 'channels': channels,
                                'planes': plane_digests},
            'projections': projections, 'shape_zyx': (z_count, height, width),
            'timings': timings, 'all_z_participated': True,
            'source_channels_read': channels, 'whole_raw_volume_copied': False,
            'summary_is_model_or_segmentation_input': False}
