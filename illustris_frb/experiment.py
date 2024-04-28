import os
import numpy as np
from numpy.linalg import norm
import healpy as hp
import pandas as pd
from .illustris_frb import frb_simulation
from .utils import Rodrigues, get_box_crossings
import multiprocessing as mp

class region:
    """
    Defines the square sky region where FRBs are placed. To ensure that the
    region is not distorted, a square region is generated near (np.pi/2, 0),
    then rotated about a pole k by an angle dtheta, then rotated azimuthally
    by an angle dphi.
    """
    
    def __init__(self, k, dtheta, dphi, size=0.10, res=0.0008):
        """
        Parameters
        ----------
        k: (N,2) or (N,3) arr
            The coordinates of the axis of rotation. May be given in angular or
            cartesian coordinates.
        dtheta: float
            The angle about the axis to rotate the region
        dphi: float
            The azimuthal angle that the region is rotated after transformation
        size: float, optional
            The size of the region in radians. Default: 0.10 rad
        res: float, optional
            The size of a pixel in radians. Default: 0.0008 rad
        """

        self.dtheta = dtheta
        if len(k) == 2:
            self.k = hp.ang2vec(*k)
        elif len(k)==3:
            self.k = np.asarray(k)
        else:
            raise ValueError('k must be in angular or cartesian coordinates.')
        self.dphi = dphi
        self.size = size
        self.res = res

        self.rotmat = Rodrigues(self.k, self.dtheta)
        self.inverse_rotmat = np.linalg.inv(self.rotmat)

        if np.isclose(size%res, 0, rtol=1e-14):
            self.nside = int(size // res)
        else:
            raise ValueError('The pixel size `res` must divide `size`.')

        self.center = self.rotate(np.pi/2, 0)

        self.gridtheta_ = np.pi/2 + np.arange(-size/2+res/2, size/2, res)
        self.gridphi_ = np.arange(-size/2+res/2, size/2, res)
    
    def rotate(self, theta_, phi_):
        theta, phi = hp.vec2ang(
            np.matmul(self.rotmat, np.atleast_2d(hp.ang2vec(theta_, phi_)).T).T
        )
        return np.squeeze(theta), np.squeeze(np.mod(phi+self.dphi, 2*np.pi))
    
    def inverse_rotate(self, theta, phi):
        theta_, phi_ = hp.vec2ang(
            np.matmul(self.inverse_rotmat, np.atleast_2d(
                hp.ang2vec(theta, phi-self.dphi)).T).T
        )
        return np.squeeze(theta_), np.squeeze(phi_)

    def checkpoints(self, x, boxsize):
        """
        Returns representative coordinates in the region whose rays span all
        periodic boxes that the region's cone (of length x) intersects.
        """

        sep = boxsize / x
        n = int(np.ceil(self.size/sep))+1
        thetas_, phis_ = np.meshgrid(
                np.pi/2+np.linspace(-self.size/2+self.res/2, self.size/2-self.res/2, n),
                np.linspace(-self.size/2+self.res/2, self.size/2-self.res/2, n)
            )
        thetas, phis = self.rotate(thetas_.flatten(), phis_.flatten())
        return np.vstack((thetas, phis)).T
    
    def is_inside(self, theta_, phi_):
        
        return (np.abs(theta_ - np.pi/2) < self.size/2) & \
               (np.abs(phi_) < self.size/2)

class exp_simulation(frb_simulation):
    """
    A class for running the experiment: ray tracing through galaxy catalogs
    and FRBs in a specified region.
    """
    
    def __init__(self, origin, region, region_name, 
                 name='L205n2500TNG', max_z=0.4, suffix='',
                 gcat_dir='/data/submit/submit-illustris/april/data/g_cats',
                 scratch_dir='/work/submit/aqc',
                 results_dir='/data/submit/submit-illustris/april/data/results',
                 **kwargs):
        
        """
        The galaxy catalog is saved to {gcat_dir}/{region_name}.
        The scratch directory, used for parallel FRB ray tracing, is 
        {gcat_dir}/{region_name}{suffix}.
        Results (DM and galaxy field) are saved to 
        {results_dir}/{region_name}{suffix}.

        Parameters
        ----------
        origin: (3,) array
            The coordinates of the observer in the simulation
        region: region
            The region on which FRBs will be placed
        region_name: str
            Label or name for the region and all subdirectories 
        max_z: float
            Sets maximum redshift at which galaxies will be retrieved. FRBs
            will be placed by default at this redshift.
        """

        super().__init__(origin, name=name, **kwargs)
        self.region = region
        self.gcat_path = os.path.join(gcat_dir, region_name)
        self.scratch_path = os.path.join(scratch_dir, region_name+'_'+suffix)
        self.results_path = os.path.join(results_dir, region_name+'_'+suffix)
        self.max_z = max_z
    
    def get_shell_path(self, snap):
        return os.path.join(self.gcat_path, f'{snap}_shell.hdf5')
    
    def get_shell_boxes(self, r_min, r_max):
        """
        Gets all boxes that intersect with the part of the cone going from r_min to r_max.
        """
        
        boxlist=[]
        for theta, phi in self.region.checkpoints(r_max, self.boxsize):
            vec = hp.ang2vec(theta, phi)
            _, _, boxes = get_box_crossings(self.origin+r_min*vec, 
                                            self.origin+r_max*vec, self.boxsize)
            boxlist.append(boxes)
        
        return np.unique(np.vstack(boxlist), axis=0)
    
    def get_shell_galaxies(self, snap, min_z=None, max_z=None, columns=[],
                           metadata='metadata.txt'):
        """
        Retrieves all galaxies within the shell that the snapshot spans. May
        specify a narrower range of redshifts if needed. Writes results to
        self.gcat_path/{snap}_shell.hdf5.
        """

        if os.path.exists(self.get_shell_path(snap)):
            print(f'Skipping snapshot {snap}')
            return

        print(f'Retrieving galaxies from snapshot {snap}')
        
        if min_z is None:
            min_x = self.snap_x_lims[snap]
        else:
            min_x = max(self.snap_x_lims[snap], self.comoving_distance(min_z))
        if max_z is None:
            max_x = min(self.snap_x_lims[snap-1], 
                        self.comoving_distance(self.max_z))
        else:
            max_x = min(self.snap_x_lims[snap-1], self.comoving_distance(max_z),
                        self.comoving_distance(self.max_z))
        
        min_z = self.z_from_dist(min_x)
        max_z = self.z_from_dist(max_x)

        ## get all periodic boxes that intersect with the shell
        
        boxes = self.get_shell_boxes(min_x, max_x)
        nbox = len(boxes)
        print(f'{"":<8}{nbox} periodic box(es) found')
        
        columns = ['x', 'y', 'z'] + columns
        data = np.array(pd.read_hdf(self.get_gmap_path(snap))[ columns ])
        
        res = []
        thetas_ = []
        phis_ = []
        for box in boxes:
                
            rel_coords = data[:,:3] + box*self.boxsize - self.origin
            theta_, phi_ = self.region.inverse_rotate(*hp.vec2ang(rel_coords))
            
            dists = norm(rel_coords, axis=1) 
            mask = (dists > min_x) & (dists < max_x) & \
                   self.region.is_inside(theta_, phi_)
            
            res.append( data[mask] )
            thetas_.append( theta_[mask] )
            phis_.append( phi_[mask] )
        
        #save and write metadata
        df = pd.DataFrame( np.vstack(res), columns=columns )
        df['theta_'], df['phi_'] = np.concatenate(thetas_), np.concatenate(phis_)
        
        df.to_hdf(self.get_shell_path(snap), key='data')

        metadata_path = os.path.join(self.gcat_path, metadata)
        if not os.path.isfile( metadata_path ):
            with open(metadata_path, 'w+') as f:
                f.write(f'origin: {list(self.origin)}\n\n')
                f.write(f"{'snap':<6}{'min_x':<16}{'max_x':<16}{'min_z':<10}{'max_z':<10}{'nbox':<6}ngal\n")
        with open(os.path.join(self.gcat_path, metadata), 'a+') as f:
            f.write(f'{snap:<6}{min_x:<16.6f}{max_x:<16.6f}{min_z:<10.6f}{max_z:<10.6f}{nbox:<6}{len(df)}\n')
        
        df = None #clear
    
    def gcat_from_zrange(self, min_z=None, max_z=None):
        """
        Gets all shells from min_z to max_z.
        """
        
        if min_z is None:
            min_z = 0
        if max_z is None:
            max_z = self.max_z
        
        first_snap = self.closest_snap(self.comoving_distance(min_z))
        final_snap = self.closest_snap(self.comoving_distance(max_z))

        if not os.path.isdir(self.gcat_path):
            os.makedirs(self.gcat_path)

        for snap in range(first_snap, final_snap-1, -1):
            self.get_shell_galaxies(snap, min_z=min_z, max_z=max_z)
    
    def gcat_from_snaps(self, snaps='all'):
        """
        Gets all shell(s) from a list of snapshots. Gets all shells by default.
        """
        if not os.path.isdir(self.gcat_path):
            os.makedirs(self.gcat_path)
        
        if snaps == 'all':
            self.gcat_from_zrange()
        else:   
            for snap in np.asarray(snaps):
                self.get_shell_galaxies(snap)
    
    def count_galaxies(self, snaps):
        """
        Histograms all galaxies into the grid specified by the region. 
        """

        g_counts = np.zeros((self.region.nside, self.region.nside))
        for snap in snaps:
            df = pd.read_hdf(self.get_shell_path(snap))
            H, _, _ = np.histogram2d(df['theta'], df['phi'], 
                                    (self.region.gridtheta_, 
                                     self.region.gridphi_))
            g_counts += H
    
    # now, write the code for the FRBs
    # def place_FRBs()