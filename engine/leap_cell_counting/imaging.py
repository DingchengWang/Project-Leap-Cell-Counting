"""Read-only TIFF metadata, plane streaming, and display conversion.

Only explicit TCZYX axes and scalar YX pages are accepted. Contiguous single-IFD
ImageJ hyperstacks use a read-only mapping, never a stack-sized allocation.
Maximum projections and display stretching are display-only; inference reads original ZYX pixels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import tifffile as tf

from .errors import CountingError
from .paths import output_path


def selection(value, count, name):
    if value is None:
        if count != 1:
            raise CountingError(name + "_selection_required", f"文件含 {count} 个 {name}，请明确选择。")
        return 0
    if type(value) is not int or not 0 <= value < count:
        raise CountingError("invalid_" + name, f"{name} 必须在 0 到 {count - 1} 之间。")
    return value


def _unit_factor(unit):
    value = str(unit).replace("\\u00B5", "u").replace("µ", "u").replace("μ", "u").strip().lower()
    return {"um": 1., "micron": 1., "microns": 1., "micrometer": 1.,
            "micrometers": 1., "micrometre": 1., "micrometres": 1.,
            "nm": .001, "mm": 1000., "cm": 10000., "m": 1000000.}.get(value)


def _scale(value, unit):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    factor = _unit_factor(unit)
    return number * factor if factor is not None and math.isfinite(number) and number > 0 else None


def _ome_pixels(tif, index):
    if not tif.is_ome:
        return None
    try:
        pixels = ET.fromstring(tif.ome_metadata).findall("{*}Image")[index].find("{*}Pixels")
    except (ET.ParseError, IndexError, TypeError) as exc:
        raise CountingError("invalid_ome_metadata", "无法读取 OME Pixels 元数据。") from exc
    if pixels is None:
        raise CountingError("invalid_ome_metadata", "OME 图像缺少 Pixels 元数据。")
    return pixels


def _info_json(tif, series):
    if not tif.is_imagej or len(tif.series) != 1:
        return {}
    try:
        data = json.loads((tif.imagej_metadata or {}).get("Info", ""))
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    dimensions = dict(zip(series.axes, series.shape))
    for axis in "XYZCT":
        if axis.lower() in data and data[axis.lower()] != dimensions.get(axis, 1):
            return {}
    return data


def _calibration(tif, series, index):
    sizes = dict.fromkeys("XYZ")
    sources = dict.fromkeys("XYZ")
    pixels = _ome_pixels(tif, index)
    if pixels is not None:
        for axis in "XYZ":
            sizes[axis] = _scale(pixels.get("PhysicalSize" + axis), pixels.get("PhysicalSize" + axis + "Unit", "um"))
            if sizes[axis] is not None:
                sources[axis] = "OME PhysicalSize" + axis
    ij = tif.imagej_metadata or {}
    data = _info_json(tif, series)
    explicit = data.get("pixel_um")
    if isinstance(explicit, list) and len(explicit) == 3:
        for i, axis in enumerate("XYZ"):
            if axis == "Z" and data.get("z_spacing_status") == "unknown":
                continue
            if sizes[axis] is None:
                sizes[axis] = _scale(explicit[i], "um")
                if sizes[axis] is not None:
                    sources[axis] = "ImageJ Info pixel_um"
    if sizes["Z"] is None:
        sizes["Z"] = _scale(ij.get("spacing"), ij.get("unit"))
        if sizes["Z"] is not None:
            sources["Z"] = "ImageJ spacing/unit"
    axes = [a.strip().upper() for a in str(ij.get("axes", "")).split(",")]
    scales, units = str(ij.get("scales", "")).split(","), str(ij.get("units", "")).split(",")
    for axis in "XYZ":
        if sizes[axis] is None and axis in axes and len(axes) == len(scales):
            pos = axes.index(axis)
            sizes[axis] = _scale(scales[pos], units[pos] if len(units) == len(axes) else ij.get("unit"))
            if sizes[axis] is not None:
                sources[axis] = "SCIFIO axes/scales/units"
    tags = series.pages[0].tags
    resolution_unit = int(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else 1
    factor = {2: 25400., 3: 10000.}.get(resolution_unit)
    if factor is None and resolution_unit == 1 and tif.is_imagej:
        factor = _unit_factor(ij.get("unit"))
    for axis in "XY":
        tag = tags.get(axis + "Resolution")
        if sizes[axis] is None and factor is not None and tag is not None:
            numerator, denominator = tag.value
            if numerator > 0 and denominator > 0:
                sizes[axis] = factor * denominator / numerator
                sources[axis] = "TIFF resolution"
    return sizes, sources


def marker_role(name):
    """A name with multiple markers cannot establish one channel's identity."""
    value = str(name).lower()
    aliases = {"DAPI": ("dapi",), "eGFP": ("egfp", "gfp"), "GFAP": ("gfap",),
               "Biocytin": ("biocytin",), "ALDH1L1": ("aldh1l1",), "SR101": ("sr101",),
               "KCNN1": ("kcnn1", "sk1"), "KCNN2": ("kcnn2", "sk2"),
               "KCNN3": ("kcnn3", "sk3"), "KCNJ10": ("kcnj10", "kir4.1")}
    found = [role for role, names in aliases.items() if any(
        re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", value) for token in names)]
    return found[0] if len(found) == 1 else "Unknown"


def _label_names(tif, series, count):
    labels = (tif.imagej_metadata or {}).get("Labels")
    if not isinstance(labels, (tuple, list)):
        return None
    planes = math.prod(series.shape[:-2])
    if len(labels) not in (count, planes):
        return None
    names = [set() for _ in range(count)]
    c_position = series.axes[:-2].find("C")
    for i, label in enumerate(labels):
        c = i if len(labels) == count else (
            np.unravel_index(i, series.shape[:-2])[c_position] if c_position >= 0 else 0)
        # ImageJ exports put the channel name before per-plane location fields.
        name = re.split(r"\s*\|\s*", str(label), maxsplit=1)[0].strip()
        name = re.sub(r"^(?:[czt]:\d+(?:/\d+)?\s*)+[-:]?\s*", "", name, flags=re.I)
        names[c].add(name)
    if any(len(items) != 1 for items in names):
        return None
    result = [next(iter(items)) for items in names]
    if count > 1 and len(set(result)) == 1:
        return None
    return result


def _channel_metadata(tif, series, index, count):
    candidates = []
    pixels = _ome_pixels(tif, index)
    if pixels is not None:
        channels = pixels.findall("{*}Channel")
        if len(channels) == count:
            candidates.append(("OME Channel Name", [c.get("Name") or "" for c in channels]))
    if tif.is_imagej and len(tif.series) == 1:
        data = _info_json(tif, series)
        for key in ("labels", "channel_names"):
            names = data.get(key)
            if isinstance(names, list) and len(names) == count and all(isinstance(n, str) for n in names):
                candidates.append(("ImageJ Info " + key, names))
        labels = _label_names(tif, series, count)
        if labels is not None:
            candidates.append(("ImageJ Labels", labels))
    hints, roles, evidence = [], [], []
    for c in range(count):
        names = [(kind, values[c].strip()) for kind, values in candidates if values[c].strip()]
        recognized = {marker_role(name) for _, name in names} - {"Unknown"}
        hints.append(names[0][1] if names else f"Channel {c + 1}")
        roles.append(next(iter(recognized)) if len(recognized) == 1 else "Unknown")
        evidence.append([{"source": kind, "name": name} for kind, name in names])
    found = [i for i, role in enumerate(roles) if role == "DAPI"]
    return hints, roles, found[0] if len(found) == 1 else -1, evidence


def validate_layout(tif, series):
    axes, shape = str(series.axes), tuple(series.shape)
    if (len(axes) != len(shape) or len(set(axes)) != len(axes) or not axes.endswith("YX")
            or any(a not in "TCZYX" for a in axes) or any(n <= 0 for n in shape)):
        raise CountingError("unsupported_axes", f"需要明确的 YX 平面及可选 T/C/Z 轴；实际为 {axes} {shape}。")
    if np.dtype(series.dtype).kind not in "uif" or np.dtype(series.dtype).itemsize > 8:
        raise CountingError("unsupported_dtype", f"不支持标量像素类型 {series.dtype}。")
    page = series.pages[0]
    if int(page.samplesperpixel) != 1 or int(page.photometric) not in (1, 3):
        raise CountingError("unsupported_samples", "需要每像素一个灰度样本的 TIFF。")
    if tuple(page.shape) != shape[-2:] or page.axes != "YX":
        raise CountingError("unsupported_page_layout", "TIFF 页必须为完整的 YX 平面。")
    expected = math.prod(shape[:-2])
    if len(series.pages) == expected:
        return "pages"
    if tif.is_imagej and len(series.pages) == 1 and page.is_contiguous and series.dataoffset is not None:
        end = series.dataoffset + math.prod(shape) * np.dtype(series.dtype).itemsize
        if end <= tif.filehandle.size:
            return "imagej_contiguous"
    raise CountingError("incomplete_planes", f"元数据声明 {expected} 个平面，但无法确认完整像素布局。")


def series_info(tif, series, index):
    dimensions = dict(zip(series.axes, series.shape))
    count = int(dimensions.get("C", 1))
    physical, calibration_sources = _calibration(tif, series, index)
    names, roles, dapi, evidence = _channel_metadata(tif, series, index, count)
    info = {"series_index": index, "name": str(series.name or f"Image {index + 1}"),
            "axes": str(series.axes), "shape": [int(n) for n in series.shape], "dtype": str(series.dtype),
            "channel_count": count, "time_count": int(dimensions.get("T", 1)), "z_count": int(dimensions.get("Z", 1)),
            "physical_size_um": physical, "calibration_sources": calibration_sources,
            "channel_hints": names, "channel_roles": roles, "dapi_channel": dapi,
            "channel_metadata_evidence": evidence, "photometric": series.pages[0].photometric.name}
    supports = [i for i, role in enumerate(roles) if role in ("eGFP", "Biocytin", "ALDH1L1")]
    info["support_channel"] = supports[0] if len(supports) == 1 else None
    info["support_marker"] = roles[supports[0]] if len(supports) == 1 else ""
    try:
        info["plane_layout"] = validate_layout(tif, series)
        info.update(supported_layout=True, layout_error=None)
    except CountingError as exc:
        info.update(supported_layout=False, layout_error={"code": exc.code, "message": str(exc)})
    return info


def inspect_image(path):
    source = Path(path).expanduser().resolve(strict=True)
    with tf.TiffFile(source) as tif:
        entries = [series_info(tif, series, i) for i, series in enumerate(tif.series)]
        if not entries:
            raise CountingError("missing_series", "TIFF 中没有可读取的图像。")
        result = {"path": str(source), "name": source.name, "size_bytes": source.stat().st_size,
                  "series_count": len(entries), "series": entries, "requires_series_selection": len(entries) > 1,
                  "is_imagej": bool(tif.is_imagej), "is_ome": bool(tif.is_ome)}
        result.update({key: entries[0][key] if len(entries) == 1 else None
                       for key in entries[0] if key not in ("name", "series_index")})
        return result


class PlaneReader:
    """Own the TIFF handle and optional read-only memory map for one series/time."""
    def __init__(self, source, series_index=None, time_index=None):
        from collections import OrderedDict
        self.source = Path(source).expanduser().resolve(strict=True)
        self.tif = tf.TiffFile(self.source)
        self.mapping = None
        self._page_cache = OrderedDict()
        self._page_cache_bytes = 0
        self._page_cache_limit = 64 * 1024 * 1024
        try:
            self.series_index = selection(series_index, len(self.tif.series), "series")
            self.series = self.tif.series[self.series_index]
            layout = validate_layout(self.tif, self.series)
            self.dimensions = dict(zip(self.series.axes, self.series.shape))
            self.time_index = selection(time_index, self.dimensions.get("T", 1), "time")
            self.info = series_info(self.tif, self.series, self.series_index)
            if layout == "imagej_contiguous":
                self.mapping = tf.memmap(self.source, series=self.series_index, mode="r")
        except BaseException:
            self.close()
            raise

    def _plane_view(self, channel, z):
        """Return immutable pixels; compressed pages use a bounded decode cache."""
        selection(channel, self.info["channel_count"], "channel")
        selection(z, self.info["z_count"], "z")
        coordinates = {"C": channel, "Z": z, "T": self.time_index}
        indices = tuple(coordinates[axis] for axis in self.series.axes[:-2])
        if self.mapping is not None:
            plane = self.mapping[indices]
        else:
            key = (channel, z)
            if key in self._page_cache:
                self._page_cache.move_to_end(key)
                return self._page_cache[key]
            index = int(np.ravel_multi_index(indices, self.series.shape[:-2])) if indices else 0
            page = self.series.pages[index]
            if page is None:
                raise CountingError("missing_plane", f"缺少图像平面 {index}。")
            plane = page.asarray()
        if (tuple(plane.shape) != tuple(self.series.shape[-2:])
                or plane.dtype.newbyteorder("=") != np.dtype(self.series.dtype).newbyteorder("=")):
            raise CountingError("inconsistent_plane", "图像平面的尺寸或位深与元数据不一致。")
        plane.flags.writeable = False
        if self.mapping is None and plane.nbytes <= self._page_cache_limit:
            while self._page_cache and self._page_cache_bytes + plane.nbytes > self._page_cache_limit:
                _, expired = self._page_cache.popitem(last=False)
                self._page_cache_bytes -= expired.nbytes
            self._page_cache[(channel, z)] = plane
            self._page_cache_bytes += plane.nbytes
        return plane

    def plane(self, channel, z):
        """Legacy owned full plane. Local consumers must use read_region."""
        return np.array(self._plane_view(channel, z), dtype=self.series.dtype, copy=True)

    @staticmethod
    def _region_slice(value, size):
        if not isinstance(value, slice) or value.step not in (None, 1):
            raise CountingError("invalid_region", "局部读取必须使用不跳过像素的半开范围。")
        start, stop = 0 if value.start is None else value.start, size if value.stop is None else value.stop
        if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (start, stop))
                or not 0 <= start < stop <= size):
            raise CountingError("invalid_region", "局部读取范围超出原始图像。")
        return slice(int(start), int(stop))

    def read_region(self, channel, zslice, yslice, xslice, *, copy=True, check_cancel=lambda: None):
        """Read native ZYX pixels after cropping, preserving byte order.

        With copy=False, continuous TIFFs return a read-only view valid while
        this reader is open. Page layouts decode each cached plane once; a
        multi-Z request assembles only its bounded ROI, never a full-source copy.
        """
        selection(channel, self.info["channel_count"], "channel")
        bounds = tuple(self._region_slice(value, size) for value, size in zip(
            (zslice, yslice, xslice), (self.info["z_count"], *self.series.shape[-2:])))
        shape = tuple(s.stop - s.start for s in bounds)
        check_cancel()
        if self.mapping is not None:
            coordinates = {"C": channel, "T": self.time_index, "Z": bounds[0], "Y": bounds[1], "X": bounds[2]}
            view = self.mapping[tuple(coordinates[axis] for axis in self.series.axes)]
            if "Z" not in self.series.axes:
                view = view[None]
            view.flags.writeable = False
            if not copy:
                return view
        elif shape[0] == 1 and not copy:
            return self._plane_view(channel, bounds[0].start)[bounds[1], bounds[2]][None]
        if math.prod(shape) * np.dtype(self.series.dtype).itemsize > 256 * 1024 * 1024:
            raise CountingError("region_memory_budget", "局部读取超过 256 MiB；请按原生区域分批读取。")
        output = np.empty(shape, dtype=self.series.dtype)
        for i, z in enumerate(range(bounds[0].start, bounds[0].stop)):
            check_cancel()
            output[i] = self._plane_view(channel, z)[bounds[1], bounds[2]]
        if not copy:
            output.flags.writeable = False
        return output

    def close(self):
        self._page_cache.clear()
        self._page_cache_bytes = 0
        if self.mapping is not None:
            self.mapping._mmap.close()
            self.mapping = None
        self.tif.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def display_plane(plane, limits=None):
    """Display-only linear conversion; uint8 retains the original 0..255 scale."""
    finite = np.isfinite(plane)
    if limits is None:
        if plane.dtype == np.uint8:
            limits = (0., 255.)
        elif finite.any():
            limits = (float(plane[finite].min()), float(plane[finite].max()))
        else:
            limits = (0., 0.)
    low, high = limits
    image = np.zeros(plane.shape, dtype=np.uint8)
    if high > low:
        image[finite] = np.clip((plane[finite].astype(np.float64) - low) * (255. / (high - low)), 0, 255).astype(np.uint8)
    return image, [float(low), float(high)]


def channel_colors(roles, supplied=None):
    colors = {"DAPI": "#0000FF", "eGFP": "#00FF00", "GFAP": "#FFFF00" if "eGFP" in roles else "#00FF00",
              "Biocytin": "#FFFFFF", "ALDH1L1": "#00FF00", "SR101": "#FF0000", "KCNN1": "#FF0000", "KCNN2": "#FF0000",
              "KCNN3": "#FF0000", "KCNJ10": "#FF0000", "Target": "#FF0000",
              "Ignore": "#FFFFFF", "Unknown": "#00FFFF"}
    defaults = [colors.get(role, "#00FFFF") for role in roles]
    if supplied is None or supplied == []:
        return defaults
    if (not isinstance(supplied, (list, tuple)) or len(supplied) != len(roles)
            or any(not isinstance(v, str) or (v and not re.fullmatch(r"#[0-9a-fA-F]{6}", v)) for v in supplied)):
        raise CountingError("invalid_channel_colors", "每个通道的颜色必须为 #RRGGBB，或留空使用默认色。")
    return [value.upper() if value else defaults[i] for i, value in enumerate(supplied)]


def save_png(path, pixels, *, compression=None):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if compression is None else {"compress_level": compression}
    Image.fromarray(pixels).save(path, format="PNG", **options)


def create_preview(source, destination):
    from PIL import Image
    target = output_path(destination, source)
    with PlaneReader(source) as reader:
        c = reader.info["dapi_channel"]
        if c < 0:
            raise CountingError("dapi_selection_required", "DAPI 通道尚未确定，请手动选择。")
        z = reader.info["z_count"] // 2
        display, _ = display_plane(reader.plane(c, z))
    image = Image.fromarray(display)
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")
    return {"preview_path": str(target), "preview_channel_index": c, "preview_z_index": z}


def native_plane_digest(plane):
    """Hash exact native pixels in a portable, explicitly little-endian order."""
    import hashlib
    canonical = np.asarray(plane, dtype=plane.dtype.newbyteorder('<'), order='C')
    return hashlib.sha256(memoryview(canonical).cast('B')).hexdigest()


def source_metadata(reader):
    keys = ('axes', 'shape', 'dtype', 'channel_count', 'time_count', 'z_count',
            'physical_size_um', 'channel_hints', 'channel_roles', 'photometric')
    return {**{key: reader.info[key] for key in keys},
            'series_index': reader.series_index, 'time_index': reader.time_index}


def _validate_source_metadata(reader, analysis):
    info = reader.info
    if (tuple(reader.series.shape[-2:]) != (analysis['height'], analysis['width'])
            or info['z_count'] != analysis['z_count']
            or info['channel_count'] != len(analysis['channels'])
            or str(reader.series.dtype) != analysis.get('source_dtype', str(reader.series.dtype))):
        raise CountingError('source_changed', '原图的尺寸、位深或通道数与计数任务不一致。')
    expected_sizes = analysis.get('physical_size_um', {})
    for axis in 'XYZ':
        if expected_sizes.get(axis, expected_sizes.get(axis.lower())) != info['physical_size_um'][axis]:
            raise CountingError('source_changed', '原图的物理标定与计数任务不一致。')
    if any(channel.get('name') != info['channel_hints'][index]
           for index, channel in enumerate(analysis['channels']) if channel.get('name')):
        raise CountingError('source_changed', '原图的通道元数据与计数任务不一致。')
    identity = analysis.get('source_identity')
    if identity and identity.get('metadata') != source_metadata(reader):
        raise CountingError('source_changed', '原图的系列、时间点、通道或标定与计数任务不一致。')


def relocate_source(analysis_path, candidate):
    """Validate an existing source in place, then atomically change only its binding.

    Modern analyses prove the native pixel identity of every channel used by
    analysis or display. Legacy analyses retain their existing size/mtime
    identity contract, with the saved geometry, channels and calibration checked.
    """
    from .worker import atomic_json
    path = Path(analysis_path).resolve(strict=True)
    original = path.read_bytes()
    analysis = json.loads(original)
    source = Path(candidate).expanduser().resolve(strict=True)
    output_path(path, source)
    before = source.stat()
    identity = analysis.get('source_identity')
    signature = analysis.get('source_signature')
    if identity is None and (not signature or
            (before.st_size, before.st_mtime_ns) != (signature.get('size_bytes'), signature.get('mtime_ns'))):
        raise CountingError('source_identity_unverified', '此旧任务没有内容指纹；所选原图的大小和修改时间必须与原记录完全一致。')
    with PlaneReader(source, analysis['series_index'], analysis['time_index']) as reader:
        _validate_source_metadata(reader, analysis)
        if identity is not None:
            if identity.get('method') != 'native_plane_sha256_v1':
                raise CountingError('source_identity_unverified', '此任务的原图内容指纹格式不受支持。')
            channels = identity.get('channels', [])
            required = set(analysis.get('display_channels', [])) | {analysis['dapi_channel'], analysis['support_channel']}
            planes = identity.get('planes', [])
            expected = {(c, z) for c in channels for z in range(reader.info['z_count'])}
            if (set(channels) != required or len(planes) != len(expected)
                    or {(p['channel'], p['z']) for p in planes} != expected):
                raise CountingError('source_identity_unverified', '原图内容指纹未覆盖此任务使用的全部通道和 Z 层。')
            for entry in planes:
                if native_plane_digest(reader._plane_view(entry['channel'], entry['z'])) != entry['sha256']:
                    raise CountingError('source_changed', '所选原图的原始像素与计数任务不一致，未更改原图路径。')
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise CountingError('source_changed', '验证过程中原图已改变，未更改原图路径。')
    if path.read_bytes() != original:
        raise CountingError('analysis_changed', '验证期间任务记录已改变，请重新打开任务后重试。')
    analysis['source_path'] = str(source)
    analysis['source_signature'] = {'size_bytes': after.st_size, 'mtime_ns': after.st_mtime_ns}
    atomic_json(path, analysis)
    return {'source_path': str(source), 'source_signature': analysis['source_signature'],
            'identity_method': identity['method'] if identity else 'legacy_size_mtime_metadata'}


def create_slice(analysis_path, z, destination):
    analysis_path = Path(analysis_path).resolve(strict=True)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    source = Path(analysis["source_path"])
    if not source.is_absolute():
        source = analysis_path.parent / source
    source = source.resolve(strict=True)
    target = output_path(destination, source)
    signature = analysis.get("source_signature")
    if signature and (source.stat().st_size != signature["size_bytes"] or source.stat().st_mtime_ns != signature["mtime_ns"]):
        raise CountingError("source_changed", "原图自分析后已改变，无法显示对应 Z 层。")
    c = analysis["dapi_channel"]
    with PlaneReader(source, analysis["series_index"], analysis["time_index"]) as reader:
        if reader.series.shape[-2:] != (analysis["height"], analysis["width"]):
            raise CountingError("source_changed", "原图尺寸已改变，无法显示对应 Z 层。")
        display, _ = display_plane(reader.plane(c, z), analysis["channels"][c].get("slice_display_range", analysis["channels"][c].get("display_range")))
    save_png(target, display)
    return {"status": "ready", "output_path": str(target), "z_index": z}


class SliceSession:
    """One TIFF handle for synchronous requests; independent of all model imports."""
    def __init__(self, analysis_path):
        self.path = Path(analysis_path).resolve(strict=True)
        self.analysis = json.loads(self.path.read_text(encoding="utf-8"))
        source = Path(self.analysis["source_path"])
        self.source = (source if source.is_absolute() else self.path.parent / source).resolve(strict=True)
        self.reader = PlaneReader(self.source, self.analysis["series_index"], self.analysis["time_index"])
        try:
            self._check_source()
            if tuple(self.reader.series.shape[-2:]) != (self.analysis["height"], self.analysis["width"]):
                raise CountingError("source_changed", "原图尺寸已改变。")
        except BaseException:
            self.reader.close()
            raise

    def _check_source(self):
        signature = self.analysis.get("source_signature")
        stat = self.source.stat()
        if signature and (stat.st_size != signature["size_bytes"] or stat.st_mtime_ns != signature["mtime_ns"]):
            raise CountingError("source_changed", "原图自分析后已改变，无法显示对应 Z 层。")

    def render(self, request):
        self._check_source()
        z = selection(request.get("z"), self.reader.info["z_count"], "z")
        channels = request.get("channels", self.analysis.get("display_channels", [self.analysis["dapi_channel"]]))
        if not isinstance(channels, list) or len(set(channels)) != len(channels):
            raise CountingError("invalid_display_channels", "显示通道必须是无重复的索引列表。")
        permitted = self.analysis.get("display_channels", list(range(self.reader.info["channel_count"])))
        for channel in channels:
            selection(channel, self.reader.info["channel_count"], "channel")
            if channel not in permitted:
                raise CountingError("display_channel_not_preselected", "复核只能显示运行前预选的源通道。")
        target = output_path(request.get("output"), self.source)
        max_edge = request.get("max_edge", 0)
        if type(max_edge) is not int or max_edge not in (0, 1024, 1536):
            raise CountingError("invalid_display_proxy", "显示代理边长必须为 1024、1536 或 0（原分辨率）。")
        native_height, native_width = self.analysis["height"], self.analysis["width"]
        width, height = native_width, native_height
        rows = columns = None
        if max_edge and max(native_width, native_height) > max_edge:
            ratio = max_edge / max(native_width, native_height)
            width, height = max(1, round(native_width * ratio)), max(1, round(native_height * ratio))
            # Display only: each proxy pixel takes one original pixel center. Fixed
            # native display ranges and the exact original channel-composition order
            # are retained. No proxy array is used for segmentation or measurement.
            rows = np.minimum(((np.arange(height) + .5) * native_height / height).astype(np.int64), native_height - 1)
            columns = np.minimum(((np.arange(width) + .5) * native_width / width).astype(np.int64), native_width - 1)
        else:
            max_edge = 0
        composite = np.zeros((height, width, 3), dtype=np.float32)
        for channel in channels:
            settings = self.analysis["channels"][channel]
            plane = self.reader.plane(channel, z)
            limits = settings.get("slice_display_range", settings.get("display_range"))
            if rows is not None and limits is None:
                # Historical analyses without saved ranges must use the same full-plane
                # window as their native frame, never a proxy-dependent contrast window.
                native_pixels, _ = display_plane(plane)
                pixels = native_pixels[np.ix_(rows, columns)]
            else:
                if rows is not None:
                    plane = plane[np.ix_(rows, columns)]
                pixels, _ = display_plane(plane, limits)
            color = settings.get("color", "#FFFFFF")
            rgb = np.array([int(color[j:j+2], 16) for j in (1, 3, 5)], dtype=np.float32) / 255.
            composite += pixels[..., None] * rgb
        np.clip(composite, 0, 255, out=composite)
        save_png(target, composite.astype(np.uint8), compression=1)
        anchors = {p["id"]: p["plane_anchors"][str(z)] for p in self.analysis.get("points", [])
                   if str(z) in p.get("plane_anchors", {})}
        return {"id": request.get("id"), "z": z, "path": str(target), "plane_anchors": anchors,
                "width": width, "height": height, "native_width": native_width, "native_height": native_height,
                "max_edge": max_edge, "quality": "proxy" if max_edge else "native",
                "sampling": "nearest_pixel_centers" if max_edge else "native"}

    def close(self): self.reader.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()


def serve_slices(analysis_path, input_stream, output_stream):
    with SliceSession(analysis_path) as session:
        for line in input_stream:
            request = {}
            try:
                request = json.loads(line)
                if not isinstance(request,dict):
                    raise CountingError("invalid_slice_request", "切片请求必须为 JSON 对象。")
                result = session.render(request)
            except Exception as exc:
                result = {"id":request.get("id") if isinstance(request,dict) else None,
                          "z":request.get("z") if isinstance(request,dict) else None,
                          "error":{"code":getattr(exc,"code","slice_failed"),"message":str(exc)}}
            output_stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n")
            output_stream.flush()
    return 0
