"""Geometry counts are test-fixture units, not empirical biological accuracy."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'engine'))
from leap_cell_counting.support_discovery import discover_support_bodies as _discover_support_bodies
from leap_cell_counting.support_discovery import _merge_basins,background_from_support_summary,propose_support_regions


def discover_support_bodies(raw,spacing,**kwargs):
    """Small geometry fixtures explicitly declare one intensity-reference event.

    The full producer/reference regression below passes its actual generated
    reference instead; it cannot silently use this fixture declaration.
    """
    if 'intensity_reference' not in kwargs:
        foreground=kwargs['discovery_foreground_zyx']
        background=(min(kwargs['background_profile']['mean_per_z']) if kwargs.get('background_profile') else float(np.min(raw)))
        kwargs['intensity_reference']={'domain_labels_zyx':foreground.astype(np.uint16),
            'peak_excess_by_id':np.array([0.,float(np.max(raw))-background],np.float32),
            'grid_xy':kwargs['discovery_grid_xy'],'domain_is_cell_id':False,
            'amplitude_source':'explicit synthetic fixture event, not a measured biological cell'}
    return _discover_support_bodies(raw,spacing,**kwargs)


class SupportDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        q=(np.arange(73)-36)*.3
        cls.z,cls.y,cls.x=np.meshgrid(q,q,q,indexing='ij')
        cls.spacing=(.3,.3,.3)

    def discover(self,positive,**kwargs):
        raw=np.where(positive,110,10).astype(np.float32)
        args=dict(origin_zyx=(0,0,0),source_shape_zyx=raw.shape)
        args.update(kwargs)
        args.update(discovery_foreground_zyx=np.ones(args['source_shape_zyx'],bool),discovery_grid_xy=1)
        return discover_support_bodies(raw,self.spacing,**args)

    def test_one_solid_body_is_one_observation(self):
        r=self.discover(self.x**2+self.y**2+self.z**2<3**2)
        self.assertEqual(len(r['objects']),1)
        self.assertEqual(r['objects'][0]['decision'],'accepted')

    def test_closed_nuclear_cavity_is_geometry_only_and_not_two_bodies(self):
        radius=self.x**2+self.y**2+self.z**2
        r=self.discover((radius<3**2)&(radius>1.4**2))
        self.assertEqual(len(r['objects']),1)
        o=r['objects'][0];point=tuple(36-v for v in o['origin_zyx'])
        self.assertFalse(o['mask_zyx'][point]);self.assertTrue(o['association_envelope_zyx'][point])
        self.assertGreater(r['metrics']['closed_cavity_voxels_added_for_geometry_only'],0)

    def test_uniform_tube_and_crossing_are_not_compact_body_counts(self):
        tube=self.y**2+self.z**2<.7**2
        cross=tube|(self.x**2+self.z**2<.7**2)
        for signal in (tube,cross):
            r=self.discover(signal)
            self.assertEqual(len(r['objects']),0)
            self.assertTrue(r['rejected_regions'])

    def test_thin_bridge_connects_two_bodies_but_count_remains_two(self):
        positive=(((self.x-5)**2+self.y**2+self.z**2<3**2)|
                  ((self.x+5)**2+self.y**2+self.z**2<3**2)|
                  ((self.y**2+self.z**2<.5**2)&(np.abs(self.x)<5)))
        r=self.discover(positive)
        self.assertEqual(len(r['objects']),2)

    def test_rotated_body_pairs_and_crossings_keep_counting_units(self):
        xyz=np.stack([self.x,self.y,self.z],axis=-1)
        for vector in (np.array([1.,1.,0.]),np.array([.3,.5,.8]),np.array([.8,-.4,.3])):
            u=vector/np.linalg.norm(vector);v=np.cross(u,[0.,0.,1.]);v/=np.linalg.norm(v)
            t=xyz@u;w=xyz@v;r2=np.sum(xyz*xyz,axis=-1)
            bridge=(r2-t*t<.5**2)&(np.abs(t)<5)
            pair=(np.sum((xyz-5*u)**2,axis=-1)<3**2)|(np.sum((xyz+5*u)**2,axis=-1)<3**2)|bridge
            cross=(r2-t*t<.7**2)|(r2-w*w<.7**2)
            self.assertEqual(len(self.discover(pair)['objects']),2)
            crossing=self.discover(cross)
            self.assertEqual(crossing['metrics']['counted_bodies'],0)
            self.assertTrue(all('source_corner_peak_with_unclosed_continuation_is_ambiguous' in o['reasons'] for o in crossing['objects']))

    def test_bumpy_single_body_and_saturated_core_are_not_micropeak_counts(self):
        shape=(((self.x-.6)**2+self.y**2+self.z**2<3**2)|
               ((self.x+.6)**2+self.y**2+self.z**2<3**2))
        raw=np.where(shape,255,10).astype(np.uint8)
        r=discover_support_bodies(raw,self.spacing,origin_zyx=(0,0,0),source_shape_zyx=raw.shape,discovery_foreground_zyx=np.ones(raw.shape,bool),discovery_grid_xy=1)
        self.assertEqual(len(r['objects']),1)

    def test_more_than_32_independent_bodies_are_not_capped(self):
        z,y,x=np.indices((21,192,192));positive=np.zeros(z.shape,bool)
        for cy in (20,50,80,110,140,170):
            for cx in (20,50,80,110,140,170):
                positive|=(z-10)**2+(y-cy)**2+(x-cx)**2<4**2
        raw=np.where(positive,110,10).astype(np.uint8)
        r=discover_support_bodies(raw,self.spacing,origin_zyx=(0,0,0),source_shape_zyx=raw.shape,discovery_foreground_zyx=np.ones(raw.shape,bool),discovery_grid_xy=1)
        self.assertEqual(len(r['objects']),36)

    def test_bridge_micropeak_inherits_merged_component_peak(self):
        labels=np.array([1,2,3],np.uint32);peaks=np.array([0,8,3,8],float)
        merged,groups,records=_merge_basins(labels,peaks,{(1,2):2.2,(2,3):2.1},.7)
        self.assertEqual(groups,[[1,2],[3]])
        self.assertEqual(merged.tolist(),[1,1,2])
        self.assertEqual(len(records),1)

    def test_same_xy_different_z_bodies_remain_two(self):
        r=self.discover(((self.z-5)**2+self.y**2+self.x**2<3**2)|
                        ((self.z+5)**2+self.y**2+self.x**2<3**2))
        self.assertEqual(len(r['objects']),2)
        a,b=r['objects'];self.assertLess(a['bounds_zyx'][0][1],b['bounds_zyx'][0][0])

    def test_process_exiting_roi_does_not_open_complete_body_core(self):
        positive=(self.x**2+self.y**2+self.z**2<3**2)|(self.y**2+self.z**2<.6**2)
        r=self.discover(positive,origin_zyx=(10,10,10),source_shape_zyx=(100,100,100))
        self.assertEqual(len(r['objects']),1)
        self.assertFalse(r['objects'][0]['requires_context_expansion'])

    def test_visible_source_z0_body_can_count_without_invented_negative_z(self):
        positive=self.x**2+self.y**2+(self.z+10.8)**2<3**2
        r=self.discover(positive)
        self.assertEqual(len(r['objects']),1)
        o=r['objects'][0];self.assertTrue(o['source_partial'])
        self.assertFalse(o['requires_context_expansion']);self.assertEqual(o['decision'],'accepted')
        self.assertIn('0',o['plane_anchors'])

    def test_true_single_z_unknown_calibration_can_count_2d_body(self):
        yy,xx=np.indices((81,81));raw=np.where((yy-40)**2+(xx-40)**2<12**2,100,10)[None].astype(np.uint8)
        r=discover_support_bodies(raw,(.2,.2,None),origin_zyx=(0,0,0),source_shape_zyx=raw.shape,discovery_foreground_zyx=np.ones(raw.shape,bool),discovery_grid_xy=1)
        self.assertEqual(r['metrics']['measurement_dimension'],2)
        self.assertIsNone(r['metrics']['spacing_xyz_um'][2])
        self.assertEqual(len(r['objects']),1);self.assertEqual(r['objects'][0]['decision'],'accepted')
        self.assertEqual(set(r['objects'][0]['plane_anchors']),{'0'})

    def test_one_slice_of_volume_is_not_misreported_as_true_2d(self):
        yy,xx=np.indices((81,81));raw=np.where((yy-40)**2+(xx-40)**2<12**2,100,10)[None].astype(np.uint8)
        with self.assertRaises(ValueError):
            discover_support_bodies(raw,(.2,.2,None),origin_zyx=(10,0,0),source_shape_zyx=(50,81,81),discovery_foreground_zyx=np.ones((50,81,81),bool),discovery_grid_xy=1)
        r=discover_support_bodies(raw,(.2,.2,.5),origin_zyx=(10,0,0),source_shape_zyx=(50,81,81),discovery_foreground_zyx=np.ones((50,81,81),bool),discovery_grid_xy=1)
        self.assertEqual(r['metrics']['measurement_dimension'],3)
        self.assertTrue(r['objects']);self.assertTrue(r['objects'][0]['requires_context_expansion'])

    def test_raw_dapi_and_existing_cp_ids_do_not_gate_discovery(self):
        positive=self.x**2+self.y**2+self.z**2<3**2
        raw=np.where(positive,110,10).astype(np.float32);before=raw.copy()
        r=discover_support_bodies(raw,self.spacing,origin_zyx=(0,0,0),source_shape_zyx=raw.shape,
                discovery_foreground_zyx=np.ones(raw.shape,bool),discovery_grid_xy=1,
                dapi_zyx=np.zeros_like(raw),existing_nuclear_labels=np.zeros(raw.shape,np.uint16))
        self.assertEqual(len(r['objects']),1)
        np.testing.assert_array_equal(raw,before)
        o=r['objects'][0]
        for z,(x,y) in o['plane_anchors'].items():
            self.assertTrue(o['mask_zyx'][int(z)-o['origin_zyx'][0],y-o['origin_zyx'][1],x-o['origin_zyx'][2]])

    def test_background_source_is_measured_raw_moments_not_sem(self):
        count=np.full((2,2),4,np.uint16)
        sums=np.array([[[8,8],[40,40]]],np.uint16)
        squares=np.array([[[16,16],[400,400]]],np.uint32)
        p=background_from_support_summary({'sum':sums,'sum_sq':squares},count)
        self.assertEqual(p['mean_per_z'],[2.]);self.assertEqual(p['std_per_z'],[0.])
        self.assertEqual(p['sampled_original_pixels_per_z'],[8])
        self.assertFalse(p['scale_is_standard_error']);self.assertFalse(p['is_proven_pure_background'])

    def test_actual_peak_trigger_is_required_not_bbox_or_core_edge_touch(self):
        positive=self.x**2+self.y**2+self.z**2<3**2
        raw=np.where(positive,110,10).astype(np.float32)
        foreground=np.zeros(raw.shape,bool);foreground[36,36,43]=True
        args=dict(origin_zyx=(0,0,0),source_shape_zyx=raw.shape,
                  discovery_foreground_zyx=foreground,discovery_grid_xy=1)
        missing=discover_support_bodies(raw,self.spacing,**args)
        self.assertEqual(missing['objects'],[])
        foreground[36,36,36]=True
        present=discover_support_bodies(raw,self.spacing,**args)
        self.assertEqual(len(present['objects']),1)
        self.assertGreater(present['objects'][0]['mask_zyx'].sum(),foreground.sum())

    def test_half_signal_geometry_does_not_use_broad_low_intensity_halo_as_thickness(self):
        radius=self.x**2+self.y**2+self.z**2
        raw=np.full(radius.shape,10,np.float32);raw[radius<8**2]=35;raw[radius<3**2]=200
        args=dict(origin_zyx=(0,0,0),source_shape_zyx=raw.shape,
                  discovery_foreground_zyx=np.ones(raw.shape,bool),discovery_grid_xy=1)
        low=discover_support_bodies(raw,self.spacing,config={'core_geometry_dynamic_fraction':.1},**args)
        high=discover_support_bodies(raw,self.spacing,config={'core_geometry_dynamic_fraction':.5},**args)
        self.assertEqual(len(low['objects']),1);self.assertEqual(len(high['objects']),1)
        self.assertLess(high['objects'][0]['metrics']['thickness_peak_radius_um'],
                        low['objects'][0]['metrics']['thickness_peak_radius_um'])
        self.assertFalse(high['objects'][0]['metrics']['fluorescence_modified'])

    def test_summary_work_regions_conserve_foreground_without_creating_cell_ids(self):
        means=np.full((9,32,32),2,np.uint16);means[2:7,8:12,8:12]=50;means[2:7,16:20,16:20]=40
        count=np.full((32,32),256,np.uint16)
        sums=(means.astype(np.uint32)*256).astype(np.uint16)
        squares=means.astype(np.uint32)**2*256
        out=propose_support_regions({'sum':sums,'sum_sq':squares},count,
                source_shape_zyx=(9,512,512),spacing_xyz=(.1,.1,.3),max_work_voxels=1_000_000)
        coverage=np.zeros(means.shape,bool)
        for r in out['regions']:
            z,y,x=r['bounds_zyx'];coverage[z[0]:z[1],y[0]//16:(y[1]+15)//16,x[0]//16:(x[1]+15)//16]=True
            self.assertFalse(r['creates_body_or_cell_id'])
        self.assertTrue(np.all(coverage[out['discovery_foreground_zyx']]))
        self.assertTrue(out['metrics']['all_source_windows_retained'])

    def test_internal_all_bright_crop_requests_context_without_claiming_zero_bodies(self):
        raw=np.full((5,12,12),100,np.uint8);shape=(20,40,40)
        profile={'source_z_start':0,'mean_per_z':[10.]*20,'std_per_z':[0.]*20,
                 'sampled_original_pixels_per_z':[1000]*20,'sampling_source':'synthetic observed external background'}
        r=discover_support_bodies(raw,(.2,.2,.4),origin_zyx=(5,10,10),source_shape_zyx=shape,
            discovery_foreground_zyx=np.ones(shape,bool),discovery_grid_xy=1,background_profile=profile)
        self.assertFalse(r['metrics']['body_identity_count_known'])
        self.assertTrue(r['context_expansion_request']['required'])
        self.assertEqual(r['objects'],[])

    def test_actual_producer_shares_one_event_amplitude_across_halo_work_partitions(self):
        # One bright body, diffuse same-event halo and two sub-half-peak halo
        # bumps. Coordinates/sizes describe this fixture, never model priors.
        z=(np.arange(73)-36)*.3;y=(np.arange(512)-256)*.1;x=y
        zz,yy,xx=np.meshgrid(z,y,x,indexing='ij');r2=zz*zz+yy*yy+xx*xx
        raw=np.full(r2.shape,10,np.uint8);raw[r2<8**2]=35;raw[r2<3**2]=200
        bump=(((zz-5)**2+yy*yy+xx*xx<1.5**2)|((zz+5)**2+yy*yy+xx*xx<1.5**2))
        raw[bump]=75
        blocks=raw.reshape(73,32,16,32,16)
        sums=blocks.sum(axis=(2,4),dtype=np.uint32).astype(np.uint16)
        squares=(blocks.astype(np.uint32)**2).sum(axis=(2,4),dtype=np.uint32)
        counts=np.full((32,32),256,np.uint16)
        proposed=propose_support_regions({'sum':sums,'sum_sq':squares},counts,
            source_shape_zyx=raw.shape,spacing_xyz=(.1,.1,.3))
        reference=proposed['intensity_reference']
        self.assertEqual(len(reference['peak_excess_by_id']),2)
        self.assertGreater(proposed['metrics']['source_compute_partitions'],1)
        other_budget=propose_support_regions({'sum':sums,'sum_sq':squares},counts,
            source_shape_zyx=raw.shape,spacing_xyz=(.1,.1,.3),max_work_voxels=8_388_608)
        np.testing.assert_array_equal(reference['domain_labels_zyx'],other_budget['intensity_reference']['domain_labels_zyx'])
        np.testing.assert_array_equal(reference['peak_excess_by_id'],other_budget['intensity_reference']['peak_excess_by_id'])
        result=discover_support_bodies(raw,(.1,.1,.3),origin_zyx=(0,0,0),source_shape_zyx=raw.shape,
            discovery_foreground_zyx=proposed['discovery_foreground_zyx'],discovery_grid_xy=16,
            intensity_reference=reference,background_profile=proposed['background_profile'],
            config={'core_geometry_dynamic_fraction':.5})
        self.assertEqual(len(result['objects']),1)
        self.assertEqual(result['objects'][0]['decision'],'accepted')

    def test_actual_producer_preserves_weak_independent_body_across_a_low_bridge(self):
        z=(np.arange(41)-20)*.3;q=(np.arange(512)-256)*.1
        zz,yy,xx=np.meshgrid(z,q,q,indexing='ij');raw=np.full(zz.shape,10,np.uint8)
        raw[(yy*yy+zz*zz<.7**2)&(np.abs(xx)<6)]=35
        raw[(xx+6)**2+yy*yy+zz*zz<3**2]=200
        raw[(xx-6)**2+yy*yy+zz*zz<3**2]=70
        blocks=raw.reshape(41,32,16,32,16)
        summary={'sum':blocks.sum((2,4),dtype=np.uint32).astype(np.uint16),
                 'sum_sq':(blocks.astype(np.uint32)**2).sum((2,4),dtype=np.uint32)}
        proposed=propose_support_regions(summary,np.full((32,32),256,np.uint16),
            source_shape_zyx=raw.shape,spacing_xyz=(.1,.1,.3))
        fg=proposed['discovery_foreground_zyx']
        self.assertTrue(fg[20,16,12]);self.assertTrue(fg[20,16,19])
        reference=proposed['intensity_reference']
        self.assertNotEqual(reference['domain_labels_zyx'][20,16,12],reference['domain_labels_zyx'][20,16,19])
        result=discover_support_bodies(raw,(.1,.1,.3),origin_zyx=(0,0,0),source_shape_zyx=raw.shape,
            discovery_foreground_zyx=fg,intensity_reference=reference,background_profile=proposed['background_profile'])
        self.assertEqual(len(result['objects']),2)
        self.assertTrue(all(o['decision']=='accepted' for o in result['objects']))
        for o in result['objects']:
            from scipy import ndimage as ndi
            parts,_=ndi.label(o['association_envelope_zyx'])
            self.assertTrue(set(np.unique(parts[o['association_envelope_zyx']])) <= set(np.unique(parts[o['core_mask_zyx']])))


if __name__=='__main__':unittest.main()
