import os
import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import norm
import healpy as hp
import pandas as pd
from .illustris_frb import frb_simulation
from .utils import Rodrigues, get_box_crossings
from .regions import regions
import multiprocessing as mp
try:
    import _pickle as pickle
except:
    import pickle

from functools import cached_property

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

        self.nside = int(size / res)
        if self.nside != size/res:
            raise ValueError('The pixel size `res` must divide `size`.')

        self.center = self.rotate(np.pi/2, 0)
    
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
        phi_ = np.mod(phi_, 2*np.pi)
        return (np.abs(theta_ - np.pi/2) < self.size/2) & \
               ((phi_ < self.size/2) | (phi_ > 2*np.pi - self.size/2))
    
    def pix2ang_(self, ipix):
        theta_ = -self.size/2 + self.res/2 + (ipix // self.nside)*self.res + np.pi/2
        phi_ = -self.size/2 + self.res/2 + (ipix % self.nside)*self.res 
        return theta_, phi_

class exp_simulation(frb_simulation):
    """
    A class for running the experiment: ray tracing through galaxy catalogs
    and FRBs in a specified region.
    """
    
    def __init__(self, origin, reg_name, reg=None,
                 name='L205n2500TNG', max_z=0.4, suffix='',
                 gcat_dir='/ceph/submit/data/group/submit-illustris/april/data/g_cats',
                 scratch_dir='/work/submit/aqc',
                 results_dir='/ceph/submit/data/group/submit-illustris/april/data/results',
                 **kwargs):
        
        """
        The galaxy catalog is saved to {gcat_dir}/{reg_name}.
        The scratch directory, used for parallel FRB ray tracing, is 
        {gcat_dir}/{reg_name}{suffix}.
        Results (DM and galaxy field) are saved to 
        {results_dir}/{reg_name}{suffix}.

        Parameters
        ----------
        origin: (3,) array
            The coordinates of the observer in the simulation
        
        reg_name: str
            Label or name for the region and all subdirectories 
        reg: region
            The region on which FRBs will be placed. Optional if reg_name is
            specified and is a named region in regions.py
        max_z: float
            Sets maximum redshift at which galaxies will be retrieved. FRBs
            will be placed by default at this redshift.
        """

        super().__init__(origin, name=name, **kwargs)

        if reg is None:
            reg = region(**regions[reg_name])
        self.region = reg
        self.gcat_path = os.path.join(gcat_dir, reg_name)
        self.scratch_path = os.path.join(scratch_dir, reg_name+suffix)
        self.results_path = os.path.join(results_dir, reg_name+suffix+'_cumulative_DM.npy')
        self.interp_path = os.path.join(results_dir, reg_name+suffix+'_interp.pkl')
        self.xticks_path = os.path.join(results_dir, 'xticks.npy')
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
    
    def get_shell_galaxies(self, snap, min_z=None, max_z=None, 
                           columns='all',
                           metadata='metadata.txt'):
        """
        Retrieves all galaxies within the shell that the snapshot spans. May
        specify a narrower range of redshifts if needed. Writes results to
        self.gcat_path/{snap}_shell.hdf5.

        columns: list of all columns besides 'x', 'y', and 'z'. By default,
        retrieves all columns (columns='all').
        """

        if os.path.exists(self.get_shell_path(snap)):
            print(f'Skipping snapshot {snap}')
            return

        print(f'Retrieving galaxies from snapshot {snap}')
        
        if min_z is None:
            min_x = self.snap_to_xlims(snap)[0]
        else:
            min_x = max(self.snap_to_xlims(snap)[0], self.comoving_distance(min_z))
        if max_z is None:
            max_x = min(self.snap_to_xlims(snap)[1], 
                        self.comoving_distance(self.max_z))
        else:
            max_x = min(self.snap_to_xlims(snap)[1], self.comoving_distance(max_z),
                        self.comoving_distance(self.max_z))
        
        min_z = self.z_from_dist(min_x)
        max_z = self.z_from_dist(max_x)

        ## get all periodic boxes that intersect with the shell
        
        boxes = self.get_shell_boxes(min_x, max_x)
        nbox = len(boxes)
        print(f'{"":<8}{nbox} periodic box(es) found')
        
        if type(columns) is list:
            data = np.array(pd.read_hdf(self.get_gmap_path(snap))[ ['x', 'y', 'z'] + columns ])
        else:
            if columns == 'all':
                df = pd.read_hdf(self.get_gmap_path(snap))
                columns = df.columns[3:]
                data = np.array(df)
            else:
                raise ValueError("`columns` must be either a list or 'all'")
        
        res = []
        xs = []
        thetas_ = []
        phis_ = []
        for box in boxes:
                
            rel_coords = data[:,:3] + box*self.boxsize - self.origin
            theta_, phi_ = self.region.inverse_rotate(*hp.vec2ang(rel_coords))
            
            x = norm(rel_coords, axis=1) 
            mask = (x > min_x) & (x < max_x) & \
                   self.region.is_inside(theta_, phi_)
            
            res.append( data[mask][:, 3:] )
            xs.append( x[mask] )
            thetas_.append( np.atleast_1d(theta_)[mask] )
            phis_.append( np.atleast_1d(phi_)[mask] )
        
        # save data
        df = pd.DataFrame( np.vstack(res), columns=columns )
        df['x'] = np.concatenate(xs)
        df['theta_'], df['phi_'] = np.concatenate(thetas_), np.concatenate(phis_)
        df.loc[df['phi_'] >= np.pi, 'phi_'] -= 2*np.pi

        # assign pixel number
        ipix_x = (df['theta_'] + self.region.size/2 - np.pi/2) // self.region.res
        ipix_y = (df['phi_'] + self.region.size/2) // self.region.res
        ipix_x[ipix_x >= self.region.nside] = self.region.nside-1
        ipix_y[ipix_y >= self.region.nside] = self.region.nside-1
        df['ipix'] = (ipix_x*self.region.nside + ipix_y).astype(int)
        
        df.to_hdf(self.get_shell_path(snap), key='data')

        # write metadata
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
    
    def read_shell_galaxies(self, zrange=None):
        """
        Returns all galaxies within a redshift range in a pandas Dataframe.
        """
        if zrange is None:
            zrange = (0, self.max_z)
        xrange = self.comoving_distance(zrange)
        snaps = np.arange(self.closest_snap(xrange[0]),
                          self.closest_snap(xrange[1]+1), -1)
        dfs = []
        for i, snap in enumerate(np.asarray(snaps)):
            df = pd.read_hdf(self.get_shell_path(snap))
            if i==0:
                df = df[ df['x'] >= xrange[0] ]
            if i==len(snaps)-1:
                df = df[ df['x'] <= xrange[1] ]
            dfs.append(df.copy(deep=True))
        return pd.concat(dfs, ignore_index=True)
    
    def Ngal_grid(self, zrange=None, df=None, mass_cutoff=None, m_g_cutoff=None):
        """
        Histograms all galaxies within xrange into the grid specified by the
        region. Gets all galaxies by default.
        """

        if df is None:

            g_counts = np.zeros(self.region.nside**2)

            if zrange is None:
                zrange = (0, self.max_z)
            xrange = self.comoving_distance(zrange)
            snaps = np.arange(self.closest_snap(xrange[0]),
                            self.closest_snap(xrange[1]+1), -1)
            
            for i, snap in enumerate(np.asarray(snaps)):
                df = pd.read_hdf(self.get_shell_path(snap))
                if i==0:
                    df = df[ df['x'] >= xrange[0] ]
                if i==len(snaps)-1:
                    df = df[ df['x'] <= xrange[1] ]
                if mass_cutoff is not None:
                    df = df[ df['Mass'] > mass_cutoff ]
                if m_g_cutoff is not None:
                    if 'm_g' not in df.columns:
                        df['m_g'] = 5*np.log10(df['x'] * 1000 / self.h) - 5 + df['M_g']
                    df = df[ df['m_g'] < m_g_cutoff ]
                
                g_counts += np.bincount(df['ipix'], minlength=self.region.nside**2)

            return g_counts.reshape((self.region.nside, self.region.nside))
        
        df_ = df.copy(deep=True)
        if mass_cutoff is not None:
            df_ = df_[ df['Mass'] > mass_cutoff ]
        if m_g_cutoff is not None:
            if 'm_g' not in df.columns:
                df_['m_g'] = 5*np.log10(df_['x'] * 1000 / self.h) - 5 + df_['M_g']
            df_= df_[ df_['m_g'] < m_g_cutoff ]
    
        return np.bincount(df_['ipix'], minlength=self.region.nside**2)
    
    def compute_DM_grid_partition(self, N=1, n=0, nproc=16, z=None):
        """
        Partitions the grid and computes the cumulative DM field (i.e. DM as
        a function of comoving distance, up to redshift z) of the given 
        partition. Saves to the scratch directory as cumulative_DM_{n}.npy

        Parameters
        ----------
        N: int
            Number of partitions. Must divide self.region.nside. Default: 
            no partition (N=1)
        n: int
            Partition number, where $0 \leq n \leq N-1$
        nproc: int
            Number of parallel processes (should allocate ~1.5 GB per FRB for
            z=0.4). Default: 16
        z: float
            The redshift at which FRBs are placed. Defaults to self.max_z
        """
        
        if z is None: #where FRBs are placed
            z = self.max_z
        x = self.comoving_distance(z)

        nrows = int(self.region.nside/N)
        ncols = self.region.nside
        start = nrows * n
        end = nrows * (n+1) # how to slice the theta grid

        # place FRBs at pixel centers
        centers = np.arange(
            -self.region.size/2+self.region.res/2,
            self.region.size/2, self.region.res) 
        thetas_ = np.pi/2 + centers[start:end]
        phis_ = np.array(centers)
        
        thetas_, phis_ = np.meshgrid(thetas_, phis_)
        thetas, phis = self.region.rotate(thetas_.T.flatten(), phis_.T.flatten())
        pix_vecs = self.origin + x * hp.ang2vec(thetas, phis)
        
        pool = mp.Pool(processes=nproc)
        DM_arr = pool.map(self.get_frb_DM, pix_vecs) 

        if not os.path.isdir(self.scratch_path):
            os.makedirs(self.scratch_path)
        
        nticks = 10*int(x//self.binsize)+1
        xticks = np.linspace(10000, x, nticks)
        
        if not os.path.exists(self.xticks_path):
            np.save(self.xticks_path, xticks)

        res = np.zeros((nrows, ncols, len(xticks)))
        for i, (xs, cum_DMs) in enumerate(DM_arr):
            res[i//ncols][i%ncols] = np.interp(xticks, xs, cum_DMs.value)
        np.save(os.path.join(self.scratch_path,
                             f'cumulative_DM_{n:03}.npy'), res)
    
    def DM_result(self):
        """
        Return the DM results array and corresponding comoving distances. If a 
        DM grid results file does not exist yet, aggregates the DM field from 
        the scratch directory and saves it.
        """
        prefix='cumulative_DM'

        if os.path.exists(self.results_path):
            res = np.load(self.results_path)
        else:
            DMs = []
            fns = sorted([fn for fn in os.listdir(self.scratch_path) if fn.startswith(prefix)])
            for fn in fns:
                DMs.append(np.load(os.path.join(self.scratch_path, fn)))
            res = np.concatenate(DMs, axis=0)
            np.save(self.results_path, res)
        
        xticks = np.load(self.xticks_path)

        return xticks[1:], res[:,:,1:] - res[:,:,0:1] # subtract the "Milky Way DM"
    
    @cached_property
    def interp_DM_grid(self):
        if os.path.exists(self.interp_path):
            return pickle.load(open(self.interp_path, 'rb'))
        xticks, res = self.DM_result()
        res = interp1d(xticks, res, fill_value=(0, np.nan))
        with open(self.interp_path, 'wb') as f:
            pickle.dump(res, f)
        return res

    def get_DMs(self, pixs, xs):
        """
        Given two arrays of of pixels and distances, gets the DM of an FRB
        at that distance in that pixel.
        """
        pixs_, xs_ = np.array(pixs).astype(int), np.array(xs).flatten()
        Nfrbs = len(xs_) # Nfrbs = len(pixs_.flatten()) = len(xs_.flatten())

        DMs = self.interp_DM_grid(xs_).reshape(self.region.nside**2, Nfrbs)
        return np.array([DMs[pix,i] for i, pix in enumerate(pixs_.flatten())]).reshape(pixs_.shape)

    def DM_grid(self, x_max=None):
        """
        Returns the DM for FRBs located at comoving distances specified by
        x_max. Retrieves the total DM by default (i.e. all FRBs at 
        z=self.max_z).

        Parameters
        ----------
        x_max: float, (N,N) arr
            Comoving distance in ckpc/h to where the FRBs are located. Defaults
            to max_z (gets total DM).
        x_min: float
            Where to start integrating the DM. Defaults to the first slice in
            the array, at 10000 ckpc/h.
        """

        if x_max is None:
            xticks, res = self.DM_result()
            return res[:,:,-1]
        
        # else, interpolate to get result
        x_max_ = np.array(x_max)
        N = self.region.nside
        if x_max_.ndim == 0:
            x_max_ = x_max_ * np.ones((N,N))
        pixs = np.arange(N**2).reshape((N,N))
        return self.get_DMs(pixs, x_max_)

    def bin_DM_array(self, DMs, ipix):
        """
        Puts an a list of DMs into a grid of their corresponding pixels, taking
        averages where needed. Returns a DM and FRB count grid.
        """
        N = self.region.nside
        DMgrid = np.zeros(N**2)
        np.add.at(DMgrid, ipix, DMs)
        FRBs_per_pix = np.bincount(ipix, minlength=N**2)
        DMgrid[ FRBs_per_pix > 1 ] /= FRBs_per_pix[ FRBs_per_pix > 1 ] 
        return DMgrid.reshape((N,N)), FRBs_per_pix.reshape((N,N))

    def sim_DM_grid(self, sampled_df=None, zrange=None, host_df=None, N=3000, weights=None, 
                    DM_host_func=None, DM_sfunc=None, g_sfunc=None, replace=True):
        """
        Returns the DM grid, placing FRBs in galaxies located within the
        redshift range zrange. 

        Parameters
        ----------
        sampled_df: pandas.DataFrame
            A pandas DataFrame of a randomly drawn sample of FRB host galaxies.
            If given, the other parameters (aside from the selection functions) 
            are ignored. Default: None
        zrange: (2,) tuple
            Range of redshifts from which to draw the FRBs. Defaults to None,
            which draws from the entire redshift range (0, self.z_max).
        host_df: pandas.DataFrame
            A pandas DataFrame containing the host galaxy catalog. Defaults to None,
            which retrieves all galaxies within zrange.
        N: int
            The number of FRBs to draw from g_df. Default: 3000
        weights: (N,) arr or str
            Weighting of the galaxies to draw FRBs: either an array of weights,
            or the name of a column in g_df. Default: None
        DM_host_func: callable
            A function which takes a shape as its argument and returns an array of that shape
            of randomly drawn host DM values
        DM_sfunc: callable
            A DM-dependent selection function. It should take as its first argument 
            an array of DMs and return an array of detection probabilities. Default: None
        g_sfunc: callable
            A selection function dependent on properties of the host galaxy. Should take
            the sampled host galaxy dataframe and return array of probabilities. Default: None
        """
        nside = self.region.nside

        if sampled_df is None:
            if host_df is None:
                host_df = self.read_shell_galaxies(zrange)
            sampled_df = host_df.sample(N, replace=replace, ignore_index=True, weights=weights)

        DMs = self.get_DMs(sampled_df['ipix'], sampled_df['x'])

        res = self.bin_DM_array(DMs, sampled_df['ipix'])

        if callable(DM_host_func):
            DMs += DM_host_func(DMs.shape)

        if callable(g_sfunc) or callable(DM_sfunc):
            Ps = np.ones_like(DMs)
            if callable(g_sfunc):
                Ps *= g_sfunc(sampled_df)
            if callable(DM_sfunc):
                Ps *= DM_sfunc(DMs)
            mask = np.random.rand(*Ps.shape) < Ps
            res_s = self.bin_DM_array(DMs[mask], sampled_df['ipix'][mask])
            return res_s, res

        return res