"""Native Support-body observations, independent of Cellpose and nuclear gates.

Work components and thickness basins are not counted. A compact, resolved
thickness core establishes a body observation after adjacent microbasins merge.
The returned fluorescence mask and closed-cavity geometry are separate arrays.
No original pixels are modified, no biological diameter or top-K is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


@dataclass(frozen=True)
class BodyDiscoveryConfig:
    smoothing_sigma_um: float = 0.20
    background_noise_multiplier: float = 3.0
    positive_dynamic_fraction: float = 0.10
    core_geometry_dynamic_fraction: float = 0.50
    merge_saddle_to_smaller_peak_ratio: float = 0.70
    core_to_peak_thickness_ratio: float = 0.50
    accepted_core_r95_to_peak_radius: float = 1.50
    excluded_core_r95_to_peak_radius: float = 3.0
    resolution_diagonal_multiplier: float = 1.0


def _configuration(value):
    cfg = value if isinstance(value, BodyDiscoveryConfig) else BodyDiscoveryConfig(**(value or {}))
    for name, number in asdict(cfg).items():
        if not np.isfinite(number) or number <= 0:
            raise ValueError(name + " must be finite and positive")
    for name in ("positive_dynamic_fraction", "core_geometry_dynamic_fraction", "merge_saddle_to_smaller_peak_ratio", "core_to_peak_thickness_ratio"):
        if not 0 < getattr(cfg, name) < 1:
            raise ValueError(name + " must lie strictly between zero and one")
    if cfg.excluded_core_r95_to_peak_radius <= cfg.accepted_core_r95_to_peak_radius:
        raise ValueError("Extended-ridge rejection must exceed compact-core acceptance")
    if cfg.core_geometry_dynamic_fraction < cfg.positive_dynamic_fraction:
        raise ValueError("Dense geometry threshold cannot be weaker than external Support extent")
    return cfg


def background_from_support_summary(summary, counts, *, source_z_start=0):
    """Observed raw-pixel moments in per-Z low-mean source blocks.

    These are descriptive global low-signal samples, not guaranteed pure noise
    and not independent observations. No original TIFF reread is required.
    """
    sums,squares=np.asarray(summary["sum"]),np.asarray(summary["sum_sq"])
    count=np.asarray(counts)
    if sums.ndim!=3 or squares.shape!=sums.shape or count.shape!=sums.shape[1:] or np.any(count<=0):
        raise ValueError("Matching ZYX sum/sum_sq and positive YX counts required")
    means=[];scales=[];samples=[];cutoffs=[]
    for z in range(len(sums)):
        block_mean=sums[z].astype(np.float64)/count
        cutoff=float(np.median(block_mean));chosen=block_mean<=cutoff
        n=int(count[chosen].sum(dtype=np.uint64))
        mean=float(sums[z][chosen].sum(dtype=np.float64)/n)
        second=float(squares[z][chosen].sum(dtype=np.float64)/n)
        means.append(mean);scales.append(float(np.sqrt(max(second-mean*mean,0))))
        samples.append(n);cutoffs.append(cutoff)
    return {"source_z_start":int(source_z_start),"mean_per_z":means,"std_per_z":scales,
        "sampled_original_pixels_per_z":samples,"selected_block_mean_cutoff_per_z":cutoffs,
        "sampling_source":"original Support pixels in source-summary blocks with mean <= per-Z median block mean",
        "is_proven_pure_background":False,"scale_is_standard_error":False}


def propose_support_regions(summary16, counts, *, source_shape_zyx, spacing_xyz,
                            max_work_voxels=32_000_000):
    """Complete fixed-field work ledger, not body/cell observations.

    The established summary16 concentration/partition operator is reused on
    Support. Overlapping work from one source component is coalesced when its
    native bounding volume fits the declared compute budget. No candidate is
    truncated or discarded to satisfy that budget. A too-large atomic request
    raises an explicit capacity error before dispatch.
    """
    from .proposals import locate_windows, _scores, LOCALIZATION
    from .errors import CountingError
    shape=tuple(int(x) for x in source_shape_zyx)
    if len(shape)!=3 or any(x<=0 for x in shape):
        raise ValueError("Actual native source shape is required")
    if type(max_work_voxels) is not int or max_work_voxels<=0:
        raise ValueError("Work-ROI voxel budget must be a positive integer")
    sums=np.asarray(summary16['sum']);count=np.asarray(counts)
    expected=(shape[0],(shape[1]+15)//16,(shape[2]+15)//16)
    if sums.shape!=expected or count.shape!=expected[1:] or np.any(count<=0):
        raise ValueError("Source-global summary16 moments/counts do not match the native source")
    profile=background_from_support_summary(summary16,count)
    scan={'moments':{16:{'dapi':{'sum':sums}}},'counts':{16:count},'shape_zyx':shape}
    windows,audit=locate_windows(scan)
    def volume(box):return int(np.prod([b-a for a,b in box],dtype=np.int64))
    for w in windows:
        if volume(w['bounds_zyx'])>max_work_voxels:
            raise CountingError('support_discovery_capacity',
                'Support原生工作区域超出已声明内存预算；全部候选已保留，须扩大容量或明确拆分后再运行。')
    regions=[]
    for w in windows:
        parent=int(w['source_component_id']);bounds=[list(p) for p in w['bounds_zyx']]
        selected=None;best_growth=None
        for i,r in enumerate(regions):
            if r['source_component_ids']!=[parent]:continue
            if not all(max(a[0],b[0])<min(a[1],b[1]) for a,b in zip(r['bounds_zyx'],bounds)):continue
            union=[[min(a[0],b[0]),max(a[1],b[1])] for a,b in zip(r['bounds_zyx'],bounds)]
            amount=volume(union)
            if amount<=max_work_voxels and (best_growth is None or amount-volume(r['bounds_zyx'])<best_growth):
                selected=i;best_growth=amount-volume(r['bounds_zyx'])
        if selected is None:
            regions.append({'region_id':len(regions)+1,'bounds_zyx':bounds,
                'source_component_ids':[parent],'source_window_ids':[int(w['window_id'])],
                'creates_body_or_cell_id':False})
        else:
            r=regions[selected]
            r['bounds_zyx']=[[min(a[0],b[0]),max(a[1],b[1])] for a,b in zip(r['bounds_zyx'],bounds)]
            r['source_window_ids'].append(int(w['window_id']))
    conserved=sum(len(r['source_window_ids']) for r in regions)
    if conserved!=len(windows):raise RuntimeError("Support work coalescing lost a source window")
    mean=sums.astype(np.float32)/count[None]
    foreground=_scores(mean,LOCALIZATION['wide_sigma_zyx'],count)>LOCALIZATION['score_threshold']
    # A signal-reference domain is independent of memory/h=1 work partitions.
    # Weak independent peaks keep their own amplitude; low-prominence halo
    # peaks inherit their parent amplitude. Neither kind creates a cell ID.
    event_labels,event_peaks,reference_audit=_intensity_reference_domains(
        mean,foreground,profile,LOCALIZATION['wide_sigma_zyx'],sums.dtype.kind in 'iu')
    foreground.setflags(write=False)
    event_labels.setflags(write=False);event_peaks.setflags(write=False)
    return {'regions':regions,'background_profile':profile,
        'discovery_foreground_zyx':foreground,'discovery_grid_xy':16,
        'intensity_reference':{'domain_labels_zyx':event_labels,'peak_excess_by_id':event_peaks,
            'grid_xy':16,'domain_is_cell_id':False,
            'amplitude_source':'maximum exact original Support block mean minus same-Z low-signal source background mean within a retained 3D peak-saddle reference domain',
            'same_event_amplitude_shared_across_work_partitions':True,
            'peak_saddle_reference_evidence':reference_audit},
        'metrics':{'source_shape_zyx':list(shape),'spacing_xyz_um':list(spacing_xyz),
            'fixed_localization_parameters':dict(LOCALIZATION),
            'source_components_not_cells':audit['wide_components'],
            'source_windows_before_work_coalescing':len(windows),'coalesced_work_regions':len(regions),
            'source_compute_partitions':len({(w['source_component_id'],w['compute_partition_id']) for w in windows}),
            'all_source_windows_retained':True,'max_work_voxels':max_work_voxels,
            'largest_work_region_voxels':max((volume(r['bounds_zyx']) for r in regions),default=0),
            'voxel_budget_is_memory_constraint_not_biological_size':True,
            'candidate_count_cap':None,'summary_used_as_final_body_mask':False}}


def _intensity_reference_domains(mean, foreground, profile, sigma, integer_source):
    """Reconcile low-prominence peaks before assigning native intensity scales.

    The existing wide concentration kernel is applied to background-corrected
    intensity, without a separate normalization on each Z. The empirical noise
    scale is measured in low-signal source blocks of this same filtered field.
    It is descriptive, not a null-model probability or independent-pixel SEM.
    """
    excess=mean-np.asarray(profile['mean_per_z'],np.float32)[:,None,None]
    field=ndi.gaussian_filter(excess,sigma=sigma,mode='nearest')
    floor=1. if integer_source else float(np.finfo(np.float32).eps*max(1.,np.max(np.abs(field)))*32)
    scales=[]
    for z in range(len(mean)):
        low=mean[z]<=profile['selected_block_mean_cutoff_per_z'][z]
        sample=field[z][low];center=float(np.median(sample))
        scales.append(max(float(1.4826*np.median(np.abs(sample-center))),floor))
    scales=np.asarray(scales,np.float32)
    inside=np.where(foreground,field,-np.inf)
    maxima=foreground & (inside==ndi.maximum_filter(inside,size=3,mode='nearest'))
    markers,number=ndi.label(maxima,structure=ndi.generate_binary_structure(3,1))
    if not number:
        return np.zeros(mean.shape,np.uint32),np.zeros(1,np.float32),{'retained_reference_domains_not_cells':0}
    basins=watershed(-inside,markers,mask=foreground,connectivity=1).astype(np.uint32)
    peaks=np.r_[0.,ndi.maximum(field,markers,np.arange(1,number+1))]
    positions=ndi.maximum_position(field,markers,np.arange(1,number+1))
    noise=np.r_[0.,[scales[p[0]] for p in positions]]
    parent=np.arange(number+1);height=peaks.copy();root_noise=noise.copy()
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    merges=[]
    for (left,right),saddle in sorted(_saddle_edges(basins,field).items(),key=lambda row:(-row[1],row[0])):
        a,b=root(left),root(right)
        if a==b:continue
        lower,higher=(a,b) if height[a]<=height[b] else (b,a)
        prominence=float(height[lower]-saddle);limit=float(3.*root_noise[lower])
        if prominence<limit:
            parent[lower]=higher
            merges.append({'lower_peak_id':int(lower),'parent_peak_id':int(higher),
                'peak_minus_connecting_saddle':prominence,'three_times_local_scale':limit})
    mapping=np.zeros(number+1,np.uint32);roots={}
    for i in range(1,number+1):
        owner=root(i)
        if owner not in roots:roots[owner]=len(roots)+1
        mapping[i]=roots[owner]
    domains=mapping[basins]
    amplitudes=np.r_[0.,np.maximum(ndi.maximum(excess,domains,np.arange(1,len(roots)+1)),0.)].astype(np.float32)
    return domains,amplitudes,{'initial_signal_peaks_not_cells':int(number),
        'retained_reference_domains_not_cells':len(roots),'low_prominence_merges':merges,
        'noise_scale_per_z':scales.tolist(),'integer_signal_resolution_floor_adu':floor if integer_source else None,
        'noise_source':'MAD of the same filtered excess field in source blocks below the per-Z median raw mean',
        'prominence_rule':'lower current root peak minus highest connecting saddle < 3 * local measured scale',
        'noise_is_pure_background_or_p_value':False,'work_partition_parameters_used':False}


def _reference_core_foreground(planes, baseline, noise, numeric_floor, low_thresholds, origin, source_shape,
                               spacing_xyz, reference, cfg, single):
    """Apply event-fixed amplitudes to native samples, with per-Z background."""
    labels=np.asarray(reference['domain_labels_zyx'])
    peaks=np.asarray(reference['peak_excess_by_id'],np.float32)
    grid=reference['grid_xy']
    if type(grid) is not int or grid<=0:raise ValueError('Invalid intensity-reference grid')
    expected=(int(source_shape[0]),int((source_shape[1]+grid-1)//grid),int((source_shape[2]+grid-1)//grid))
    if (labels.shape!=expected or labels.dtype.kind not in 'iu'
            or peaks.ndim!=1 or not len(peaks) or not np.isfinite(peaks).all() or np.any(peaks<0)):
        raise ValueError('Invalid source-event intensity reference')
    z0=max(0,int(origin[0])-4);z1=min(labels.shape[0],int(origin[0])+planes.shape[0]+4)
    y0=max(0,int(origin[1])//grid-8);y1=min(labels.shape[1],(int(origin[1])+planes.shape[1]+grid-1)//grid+8)
    x0=max(0,int(origin[2])//grid-8);x1=min(labels.shape[2],(int(origin[2])+planes.shape[2]+grid-1)//grid+8)
    local=np.asarray(labels[z0:z1,y0:y1,x0:x1])
    if not local.any():
        raise ValueError('No source-event intensity reference covers the observed work region')
    if local.max()>=len(peaks):raise ValueError('Reference event ID has no measured amplitude')
    if single:
        indices=ndi.distance_transform_edt(local[0]==0,sampling=(spacing_xyz[1]*grid,spacing_xyz[0]*grid),
                                           return_distances=False,return_indices=True)
        nearest=local[0][tuple(indices)][None]
    else:
        indices=ndi.distance_transform_edt(local==0,sampling=(spacing_xyz[2],spacing_xyz[1]*grid,spacing_xyz[0]*grid),
                                           return_distances=False,return_indices=True)
        nearest=local[tuple(indices)]
    yy=(int(origin[1])+np.arange(planes.shape[1]))//grid-y0
    xx=(int(origin[2])+np.arange(planes.shape[2]))//grid-x0
    core=np.zeros(planes.shape,bool);threshold_ranges=[];used=set()
    for z in range(len(planes)):
        ids=nearest[z+int(origin[0])-z0][yy[:,None],xx[None,:]]
        used.update(map(int,np.unique(ids)))
        threshold=baseline[z]+np.maximum(cfg.core_geometry_dynamic_fraction*peaks[ids],
                         max(cfg.background_noise_multiplier*noise[z],numeric_floor[z]))
        threshold=np.maximum(threshold,low_thresholds[z])
        core[z]=planes[z]>threshold
        threshold_ranges.append([float(threshold.min()),float(threshold.max())])
    return core,{'reference_event_ids_used':sorted(used),
        'reference_event_peak_excess':[{'event_id':i,'peak_excess':float(peaks[i])} for i in sorted(used)],
        'amplitude_source':reference.get('amplitude_source','caller-declared source-event peak excess'),
        'same_event_amplitude_shared_across_work_partitions':True,
        'reference_interpolation':'nearest source-event amplitude on physical summary coordinates; never a body mask',
        'core_geometry_threshold_range_per_z':threshold_ranges}


def _tight_box(mask):
    positions = [np.flatnonzero(mask.any(axis=tuple(b for b in range(mask.ndim) if b != a)))
                 for a in range(mask.ndim)]
    if any(len(x) == 0 for x in positions):
        return None
    return tuple(slice(int(x[0]), int(x[-1])+1) for x in positions)


def _saddle_edges(labels, distance):
    """Highest observed face-sharing saddle for each pair of native basins."""
    base = int(labels.max()) + 1
    edges = {}
    for axis in range(labels.ndim):
        a = [slice(None)] * labels.ndim
        b = list(a)
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        la, lb = labels[tuple(a)], labels[tuple(b)]
        boundary = (la > 0) & (lb > 0) & (la != lb)
        if not boundary.any():
            continue
        left, right = la[boundary].astype(np.int64), lb[boundary].astype(np.int64)
        keys = np.minimum(left, right) * base + np.maximum(left, right)
        levels = np.minimum(distance[tuple(a)][boundary], distance[tuple(b)][boundary])
        unique, inverse = np.unique(keys, return_inverse=True)
        maxima = np.zeros(len(unique), np.float32)
        np.maximum.at(maxima, inverse, levels)
        for key, level in zip(unique, maxima):
            pair = (int(key // base), int(key % base))
            edges[pair] = max(edges.get(pair, 0.0), float(level))
    return edges


def _merge_basins(labels, peaks, edges, ratio):
    parents = np.arange(len(peaks), dtype=np.uint32)
    root_peak = np.asarray(peaks, dtype=float).copy()
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i
    merged = []
    for (left, right), level in sorted(edges.items(), key=lambda row: (-row[1], row[0])):
        a, b = root(left), root(right)
        if a == b:
            continue
        # A small bridge already absorbed by a large body must inherit that
        # body's peak. Otherwise a chain of tiny peaks joins two large bodies.
        relative = level / min(root_peak[a], root_peak[b])
        if relative >= ratio:
            kept, absorbed = min(a,b), max(a,b)
            parents[absorbed] = kept
            root_peak[kept] = max(root_peak[a], root_peak[b])
            merged.append({"peak_ids": [left,right], "saddle_radius_um": level,
                           "saddle_to_smaller_component_peak_ratio": float(relative)})
    mapping = np.zeros(len(peaks), np.uint32)
    groups = {}
    for i in range(1, len(peaks)):
        groups.setdefault(root(i), []).append(i)
    ordered = list(groups.values())
    for group_id, ids in enumerate(ordered, 1):
        mapping[ids] = group_id
    return mapping[labels], ordered, merged


def _anchors(observed, local_origin, density):
    anchors = {}
    for z in np.flatnonzero(observed.any(axis=(1,2))):
        yy, xx = np.nonzero(observed[z])
        center = np.array([yy.mean(), xx.mean()])
        chosen = int(np.argmin((yy-center[0])**2 + (xx-center[1])**2))
        anchors[str(int(z+local_origin[0]))] = [int(xx[chosen]+local_origin[2]),
                                               int(yy[chosen]+local_origin[1])]
    point = np.unravel_index(np.argmax(np.where(observed, density, -np.inf)), observed.shape)
    return anchors, [int(v) for v in (np.array(point)+local_origin)[::-1]]


def discover_support_bodies(support_zyx, spacing_xyz, *, origin_zyx, source_shape_zyx,
                            discovery_foreground_zyx, intensity_reference, discovery_grid_xy=16,
                            background_profile=None, dapi_zyx=None, existing_nuclear_labels=None, config=None):
    """Return body observations as bounded per-object native arrays.

    ``objects`` records contain ``local_label_id``, source ``origin_zyx`` and
    half-open ``bounds_zyx``, plus matching ``mask_zyx`` (actual raw-positive
    Support) and ``association_envelope_zyx`` (geometry, including closed 3D
    cavities). These arrays share the object's tight native box. Source-Z
    ``plane_anchors`` always lie in actual positive Support, including when the
    representative body geometry contains a dark nucleus. Integration assigns
    global counting IDs; rejected processing basins receive no cell ID.

    Optional DAPI/old nuclear labels supply descriptive correspondence only.
    Neither changes body discovery or requires a DAPI Cellpose success.
    The function is single-worker CPU computation; the caller bounds concurrent
    ROI jobs to four and owns source reads and global body/nucleus association.
    """
    cfg = _configuration(config)
    raw = np.asarray(support_zyx)
    origin = np.asarray(origin_zyx)
    source_shape = np.asarray(source_shape_zyx)
    if raw.ndim != 3 or not raw.size or raw.dtype.kind not in "buif" or not np.isfinite(raw).all():
        raise ValueError("Support must be a finite nonempty native ZYX array")
    for value, name in ((origin,"origin_zyx"),(source_shape,"source_shape_zyx")):
        if value.shape != (3,) or value.dtype.kind not in "iu":
            raise ValueError(name + " must contain three native integer coordinates")
    if np.any(origin < 0) or np.any(source_shape <= 0) or np.any(origin+raw.shape > source_shape):
        raise ValueError("Native ROI must fit within its declared original source")
    foreground=np.asarray(discovery_foreground_zyx)
    if type(discovery_grid_xy) is not int or discovery_grid_xy<=0:
        raise ValueError("Discovery grid must have positive integer XY sampling")
    expected=(int(source_shape[0]),int((source_shape[1]+discovery_grid_xy-1)//discovery_grid_xy),
              int((source_shape[2]+discovery_grid_xy-1)//discovery_grid_xy))
    if foreground.shape!=expected or foreground.dtype.kind!='b':
        raise ValueError("Discovery foreground must cover source-global Z and declared XY summary grid")
    if len(spacing_xyz) != 3:
        raise ValueError("Actual XYZ calibration is required")
    sx, sy, sz = spacing_xyz
    single = source_shape[0] == 1
    if sx is None or sy is None or not np.isfinite([sx,sy]).all() or min(sx,sy) <= 0:
        raise ValueError("Actual positive XY calibration is required")
    if not single and (sz is None or not np.isfinite(sz) or sz <= 0):
        raise ValueError("Actual positive Z calibration is required for a volume")
    if dapi_zyx is not None and np.asarray(dapi_zyx).shape != raw.shape:
        raise ValueError("Optional raw DAPI must share this native ROI")
    if existing_nuclear_labels is not None:
        old = np.asarray(existing_nuclear_labels)
        if old.shape != raw.shape or old.dtype.kind not in "iu" or np.any(old < 0):
            raise ValueError("Existing nuclear labels must be matching nonnegative integers")
    trigger_box=(slice(int(origin[0]),int(origin[0]+raw.shape[0])),
        slice(int(origin[1]//discovery_grid_xy),int((origin[1]+raw.shape[1]+discovery_grid_xy-1)//discovery_grid_xy)),
        slice(int(origin[2]//discovery_grid_xy),int((origin[2]+raw.shape[2]+discovery_grid_xy-1)//discovery_grid_xy)))
    if not foreground[trigger_box].any():
        return {'objects':[],'rejected_regions':[],
            'metrics':{'method':'native_support_thickness_core_v1','parameters':asdict(cfg),
                'measurement_dimension':2 if single else 3,'spacing_xyz_um':list(spacing_xyz),
                'source_origin_zyx':origin.tolist(),'input_shape_zyx':list(raw.shape),
                'status':'no_source_concentration_trigger_in_observed_region',
                'counted_bodies':0,'review_bodies':0,'candidate_count_cap':None,
                'source_field_used_as_final_mask':False}}
    sampling = np.array([sy,sx] if single else [sz,sy,sx], dtype=float)
    data = raw[0] if single else raw
    smooth = ndi.gaussian_filter(data.astype(np.float32), cfg.smoothing_sigma_um/sampling, mode="nearest")
    planes = smooth[None] if single else smooth
    # A bright body may occupy most of a local crop. Only actual peripheral
    # samples, or an explicitly sourced global profile, estimate background.
    by=max(1,int(np.ceil(4*cfg.smoothing_sigma_um/sy)))
    bx=max(1,int(np.ceil(4*cfg.smoothing_sigma_um/sx)))
    peripheral=np.zeros(raw.shape[1:],bool)
    peripheral[:by]=True;peripheral[-by:]=True;peripheral[:,:bx]=True;peripheral[:,-bx:]=True
    local_samples=planes[:,peripheral]
    local_baseline=np.median(local_samples,axis=1)
    local_noise=1.4826*np.median(np.abs(local_samples-local_baseline[:,None]),axis=1)
    if background_profile is None:
        baseline,noise=local_baseline,local_noise
        background_source={"sampling_source":"observed XY peripheral band, width=4*smoothing_sigma in physical units",
            "sampled_pixels_per_plane":int(peripheral.sum()),"is_proven_pure_background":False}
    else:
        start=int(origin[0])-int(background_profile["source_z_start"])
        stop=start+raw.shape[0]
        if start<0 or stop>len(background_profile["mean_per_z"]):
            raise ValueError("External background profile does not cover this source Z range")
        baseline=np.asarray(background_profile["mean_per_z"][start:stop],np.float32)
        noise=np.asarray(background_profile["std_per_z"][start:stop],np.float32)
        if not np.isfinite(baseline).all() or not np.isfinite(noise).all() or np.any(noise<0):
            raise ValueError("External background measurements must be finite with nonnegative scale")
        background_source={"sampling_source":background_profile["sampling_source"],
            "source_z_range_half_open":[int(origin[0]),int(origin[0]+raw.shape[0])],
            "sampled_original_pixels_per_z":background_profile["sampled_original_pixels_per_z"][start:stop],
            "is_proven_pure_background":False,"scale_is_standard_error":False}
    high = planes.max(axis=(1,2))
    numeric_floor = np.finfo(np.float32).eps*np.maximum(np.abs(high),1)*32
    thresholds = baseline + np.maximum.reduce([cfg.background_noise_multiplier*noise,
                    cfg.positive_dynamic_fraction*np.maximum(high-baseline,0),numeric_floor])
    threshold = thresholds[0] if single else thresholds[:,None,None]
    positive = smooth > threshold
    core_planes,reference_evidence=_reference_core_foreground(planes,baseline,noise,numeric_floor,thresholds,
                                  origin,source_shape,spacing_xyz,intensity_reference,cfg,single)
    core_positive=core_planes[0] if single else core_planes
    observed_positive = data > threshold
    # Enclosed 3D cavities are a geometry construction. The measured/raw-positive
    # mask is retained separately and is never filled or written back to source.
    geometry = ndi.binary_fill_holes(core_positive) if not single else core_positive.copy()
    metrics = {"method":"native_support_thickness_core_v1", "parameters":asdict(cfg),
        "measurement_dimension":2 if single else 3,"spacing_xyz_um":list(spacing_xyz),
        "source_origin_zyx":origin.tolist(),"input_shape_zyx":list(raw.shape),
        "background_location_per_z":baseline.tolist(),"background_scale_per_z":noise.tolist(),
        "background_sample_provenance":background_source,
        "local_peripheral_median_diagnostic_per_z":local_baseline.tolist(),
        "local_peripheral_robust_sigma_diagnostic_per_z":local_noise.tolist(),
        "positive_threshold_per_z":thresholds.tolist(),"positive_geometry_voxels":int(positive.sum()),
        "intensity_reference_evidence":reference_evidence,
        "closed_cavity_voxels_added_for_geometry_only":int(np.count_nonzero(geometry & ~core_positive)),
        "discovery_trigger":"physical centroid representative of maximum-thickness plateau must lie in actual source foreground",
        "discovery_grid_xy":discovery_grid_xy,
        "threshold_status":"development_geometry_not_biological_validation",
        "cellpose_used":False,"DAPI_used_as_discovery_gate":False,"fixed_biological_diameter":None,
        "candidate_count_cap":None,"single_worker_cpu":True}
    result = {"objects":[],"rejected_regions":[],"metrics":metrics}
    if geometry.all():
        faces=[]
        for axis,name in enumerate('zyx'):
            if single and axis==0:continue
            faces.append({'axis':name,'side':'low','boundary':'source' if origin[axis]==0 else 'roi'})
            faces.append({'axis':name,'side':'high','boundary':'source' if origin[axis]+raw.shape[axis]==source_shape[axis] else 'roi'})
        required=any(f['boundary']=='roi' for f in faces)
        result['context_expansion_request']={'required':required,'faces':faces,
            'reason':'bright_foreground_fills_observed_roi'}
        metrics.update(status='needs_context_expansion' if required else 'source_fills_frame_identity_undetermined',
                       body_identity_count_known=False,counted_bodies=0,review_bodies=0)
        return result
    if not geometry.any():
        metrics.update(status="no_resolved_signal_core",body_identity_count_known=False,counted_bodies=0,review_bodies=0)
        return result
    # No artificial zero border: a thick core reaching the observed ROI edge
    # must be reported as open, not made falsely complete by zero padding.
    distance = ndi.distance_transform_edt(geometry,sampling=sampling).astype(np.float32)
    min_radius = cfg.resolution_diagonal_multiplier*float(np.linalg.norm(sampling))
    local_max = (distance == ndi.maximum_filter(distance,size=3,mode="nearest")) & (distance >= min_radius)
    markers, number = ndi.label(local_max,structure=ndi.generate_binary_structure(data.ndim,1))
    metrics.update(minimum_resolved_radius_um=min_radius,initial_thickness_peaks_not_cells=int(number))
    if not number:
        metrics.update(status="no_spatially_resolved_thickness_core",counted_bodies=0,review_bodies=0)
        return result
    peaks = np.r_[0.,ndi.maximum(distance,markers,np.arange(1,number+1))]
    basins = watershed(-distance,markers,mask=geometry,connectivity=1).astype(np.uint32)
    edges = _saddle_edges(basins,distance)
    groups, memberships, merges = _merge_basins(basins,peaks,edges,cfg.merge_saddle_to_smaller_peak_ratio)
    edge_neighbors = {}
    for (left,right),level in edges.items():
        edge_neighbors.setdefault(left,[]).append((right,level))
        edge_neighbors.setdefault(right,[]).append((left,level))
    del basins,markers,local_max
    if np.any(positive & ~geometry):
        # The dense field defines peaks/identity. Only measured low-threshold
        # Support and enclosed dense-core cavities can extend their territory.
        groups=watershed(-distance,groups,mask=(positive | geometry),connectivity=1).astype(np.uint32)
    metrics.update(merged_micropeak_connections=merges,merged_thickness_groups_not_cells=len(memberships))
    for group_id, box in enumerate(ndi.find_objects(groups),1):
        if box is None:
            continue
        basin = groups[box] == group_id
        own_distance = distance[box]
        radius = float(own_distance[basin].max())
        plateau=np.argwhere(basin & (own_distance==radius))
        center=plateau.mean(axis=0)
        peak_local=plateau[np.argmin(np.sum(((plateau-center)*sampling)**2,axis=1))]
        peak_source=peak_local+np.array([s.start for s in box])
        if single:peak_source=np.r_[0,peak_source]
        peak_source=peak_source+origin
        if not foreground[int(peak_source[0]),int(peak_source[1]//discovery_grid_xy),int(peak_source[2]//discovery_grid_xy)]:
            result['rejected_regions'].append({'decision':'excluded',
                'reason':'thickness_peak_has_no_source_concentration_trigger',
                'metrics':{'geometry_group_id_not_cell_id':group_id,
                           'thickness_peak_source_xyz':peak_source[::-1].astype(int).tolist(),
                           'thickness_peak_radius_um':radius}})
            continue
        core = basin & (own_distance >= cfg.core_to_peak_thickness_ratio*radius)
        coords = np.nonzero(core)
        centroid = np.array([v.mean() for v in coords])
        squared = np.zeros(len(coords[0]),np.float64)
        for a in range(core.ndim):
            squared += ((coords[a]-centroid[a])*sampling[a])**2
        r95 = float(np.sqrt(np.percentile(squared,95)))
        compactness = r95/radius
        initial_ids = memberships[group_id-1]
        neighbors=[]
        own_set=set(initial_ids)
        for a in initial_ids:
            for other,saddle in edge_neighbors.get(a,[]):
                if other not in own_set:
                    neighbors.append({"neighbor_peak_id":int(other),
                                      "saddle_radius_um":saddle,"saddle_to_this_peak_ratio":saddle/radius})
        geometry_metrics={"thickness_peak_radius_um":radius,"thickness_peak_diameter_um":2*radius,
            "core_r95_um":r95,"core_r95_to_peak_radius":compactness,
            "core_voxels":int(core.sum()),"merged_thickness_peak_ids":initial_ids,
            "neighbor_saddles":neighbors,"geometry_group_id_not_cell_id":group_id}
        geometry_metrics["thickness_peak_source_xyz"]=peak_source[::-1].astype(int).tolist()
        geometry_metrics["thickness_peak_in_discovery_foreground"]=True
        if compactness > cfg.excluded_core_r95_to_peak_radius:
            result["rejected_regions"].append({"decision":"excluded",
                "reason":"extended_thickness_ridge_not_a_compact_body","metrics":geometry_metrics})
            continue
        # The measured peak radius provides a local expansion distance. This
        # includes the observed outer body while stopping remote process arms;
        # it is not a fixed biological size, nor a whole-network component.
        distance_from_core = ndi.distance_transform_edt(~core,sampling=sampling)
        envelope = basin & (distance_from_core <= radius)
        # Growth retains structure only when connected to this body's own
        # identity core. A detached outer island cannot create or inherit an ID.
        connected,_=ndi.label(envelope,structure=ndi.generate_binary_structure(core.ndim,1))
        owned=np.unique(connected[core]);owned=owned[owned>0]
        retained=np.isin(connected,owned)
        geometry_metrics['detached_outer_voxels_removed']=int(np.count_nonzero(envelope & ~retained))
        envelope &= retained
        observed = envelope & observed_positive[box]
        if not observed.any():
            result["rejected_regions"].append({"decision":"excluded",
                "reason":"geometry_without_original_positive_support","metrics":geometry_metrics})
            continue
        tight = _tight_box(envelope)
        starts = np.array([s.start for s in box])+np.array([s.start for s in tight])
        observed = observed[tight].copy()
        envelope = envelope[tight].copy()
        core_tight = core[tight]
        density = smooth[box][tight]
        if single:
            starts = np.r_[0,starts]
            observed,envelope,core_tight,density = [v[None] for v in (observed,envelope,core_tight,density)]
        global_origin = origin+starts
        global_stop = global_origin+np.array(envelope.shape)
        core_open=[];signal_open=[]
        for axis,name in enumerate("zyx"):
            if single and axis == 0:
                continue
            for side,local_face,at_roi in (("low",0,starts[axis]==0),
                                          ("high",-1,starts[axis]+envelope.shape[axis]==raw.shape[axis])):
                if not at_roi:
                    continue
                at_source = (origin[axis]==0 if side=="low" else origin[axis]+raw.shape[axis]==source_shape[axis])
                face={"axis":name,"side":side,"boundary":"source" if at_source else "roi"}
                if np.take(observed,local_face,axis=axis).any():signal_open.append(face)
                if np.take(core_tight,local_face,axis=axis).any():core_open.append(face)
        reasons=[]
        decision="accepted"
        if compactness > cfg.accepted_core_r95_to_peak_radius:
            reasons.append("intermediate_core_compactness");decision="review"
        if any(f["boundary"]=="roi" for f in core_open):
            reasons.append("thick_core_truncated_at_observed_boundary");decision="review"
        if any(f["boundary"]=="source" for f in core_open):
            reasons.append("visible_compact_body_partial_at_original_source_boundary")
        peak_source_faces=[]
        for a,name in enumerate('zyx'):
            if single and a==0:continue
            if peak_source[a]==0:peak_source_faces.append(name+'_low')
            if peak_source[a]==source_shape[a]-1:peak_source_faces.append(name+'_high')
        geometry_metrics['thickness_peak_censored_source_faces']=peak_source_faces
        if len(peak_source_faces)>=2 and neighbors:
            reasons.append('source_corner_peak_with_unclosed_continuation_is_ambiguous');decision='review'
        z_present=np.flatnonzero(observed.any(axis=(1,2)))
        if not single and (len(z_present)<2 or np.any(np.diff(z_present)>1)):
            reasons.append("incomplete_continuous_original_z_evidence");decision="review"
        if single:
            reasons.append("genuine_2d_source_body_geometry_no_axial_volume_claim")
        anchors, representative = _anchors(observed,global_origin,density)
        box3=tuple(slice(int(starts[a]),int(starts[a]+envelope.shape[a])) for a in range(3))
        local_id=len(result["objects"])+1
        geometry_metrics.update({"observed_positive_voxels":int(observed.sum()),
            "association_envelope_voxels":int(envelope.sum()),
            "geometry_only_voxels":int(np.count_nonzero(envelope & ~observed)),
            "body_extent_rule":"positive basin within one measured peak radius of its compact thickness core",
            "continuous_source_z_planes":(z_present+global_origin[0]).astype(int).tolist(),
            "original_support_mean":float(raw[box3][observed].mean()),
            "fluorescence_modified":False})
        if dapi_zyx is not None:
            geometry_metrics["raw_dapi_mean_in_body_envelope"] = float(np.asarray(dapi_zyx)[box3][envelope].mean())
        if existing_nuclear_labels is not None:
            old = np.asarray(existing_nuclear_labels)[box3]
            ids,counts=np.unique(old[envelope],return_counts=True)
            geometry_metrics["overlapping_existing_nuclear_voxels"] = [
                {"nuclear_label_id":int(i),"intersection_voxels":int(n),"identity_match_confirmed":False}
                for i,n in zip(ids,counts) if i]
        result["objects"].append({"local_label_id":local_id,"kind":"support_body",
            "source":"native_support_spatial_discovery","origin_zyx":global_origin.astype(int).tolist(),
            "bounds_zyx":[[int(a),int(b)] for a,b in zip(global_origin,global_stop)],
            "mask_zyx":observed,"association_envelope_zyx":envelope,
            "core_mask_zyx":core_tight.copy(),
            "plane_anchors":anchors,"representative_xyz":representative,
            "decision":decision,"reasons":reasons or ["resolved_compact_native_support_body"],
            "open_faces":core_open,"observed_signal_open_faces":signal_open,
            "requires_context_expansion":any(f["boundary"]=="roi" for f in core_open),
            "source_partial":any(f["boundary"]=="source" for f in core_open),"metrics":geometry_metrics})
    metrics.update(status="complete",counted_bodies=sum(o["decision"]=="accepted" for o in result["objects"]),
                   review_bodies=sum(o["decision"]=="review" for o in result["objects"]),
                   rejected_geometry_groups=len(result["rejected_regions"]))
    return result
