import os
import numpy as np
import h5py
import healpy as hp

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.cosmology.units as cu

from scipy.interpolate import interp1d
from functools import cached_property
from .utils import get_box_crossings

class simulation:
    """
    An object for the attributes of an IllustrisTNG simulation, including the 
    simulation boxsize, paths to the electron density maps and simulation 
    files, and cosmology.
    """
    
    def __init__(self, name='L205n2500TNG', binsize=500, sim_dir=None, 
                 header_file='/data/submit/submit-illustris/april/data/header_info/L205n2500TNG_header.hdf5',
                 snap_zs_file = '/data/submit/submit-illustris/april/data/header_info/snap_zs.npy', 
                 emap_dir='/data/submit/submit-illustris/april/data/n_e_maps', 
                 gmap_dir='/data/submit/submit-illustris/april/data/g_maps', 
                 max_z=2):
        """
        Parameters
        ----------
        name: str
            The name of the simulation, e.g. 'TNG205n2500'. Used for setting
            the default simulation and map directory paths.
        binsize: int, optional
            The size of the bin for the electron density map, in ckpc/h. Must
            divide the simulation boxsize. Default: 500
        sim_dir: str, optional
            Directory to simulation snapshot files. Defaults to the path on
            Illustris JupyterHub.
        header_file: str, optional
            Path to a .hdf5 file that has the simulation attributes. To read
            the header directly from the simulation, use header_file=None.
        snap_zs_file: str, optional
            Path to a .npy file that contains the simulation redshifts. If 
            None, snapshot redshifts will be read in from the simulation
            files.
        emap_dir: str, optional
            Directory to electron density map. Defaults to './data/n_e_maps'
        gmap_dir: str, optional
            Directory to galaxy map. Defaults to './data/g_maps'
        max_z: float, optional
            The maximum redshift out to which FRBs will be simulated. Used
            only for creating the z(x) interpolant. Default: 2
        """
        
        self.name = name
        self.snap_zs_file = snap_zs_file
        self.max_z = max_z
        
        if sim_dir is None:
            self.sim_dir = f'/home/tnguser/sims.TNG/{name}/output'
        else:
            self.sim_dir = sim_dir
        self.emap_dir = emap_dir
        self.gmap_dir = gmap_dir
        
        if (header_file is None) or (not os.path.exists(str(header_file))):
            header_file = self.get_snap_chunk_path(99)
        with h5py.File(header_file) as f:
            self.header = dict(f['Header'].attrs)

        self.boxsize = int(self.header['BoxSize'])
        self.binsize = binsize
        self.n_bins = int(self.boxsize / self.binsize)
        
        self.h = self.header['HubbleParam']
        self.Omega0 = self.header['Omega0']

        self.cosmo = FlatLambdaCDM(H0=100*self.h, Om0=self.Omega0)

    def get_snapdir_path(self, snap):
        return os.path.join(self.sim_dir, f'snapdir_{snap:03}')
    
    def get_groupdir_path(self, snap):
        return os.path.join(self.sim_dir, f'groups_{snap:03}')
                            
    def get_snap_chunk_path(self, snap, chunk=0):
        return os.path.join(self.sim_dir, f'snapdir_{snap:03}', 
                            f'snap_{snap:03}.{chunk}.hdf5')
    
    def get_group_chunk_path(self, snap, chunk=0):
        return os.path.join(self.sim_dir, f'groups_{snap:03}', 
                            f'fof_subhalo_tab_{snap:03}.{chunk}.hdf5')
    
    def get_emap_path(self, snap):
        return os.path.join(self.emap_dir, f'{snap}.npy')
    
    def get_gmap_path(self, snap):
        return os.path.join(self.gmap_dir, f'{snap}_galaxies.hdf5')
    
    def comoving_distance(self, z):
        return self.cosmo.comoving_distance(z).to(u.kpc).value * self.h
    
    def get_max_dist(self, thetas, phis): 
    
        dvecs = np.sort(np.abs(np.atleast_2d(hp.ang2vec(thetas,phis))))
        scaled_dvecs = (dvecs.T / dvecs[:, -1]).T #(N, 2) arr

        periods = []
        for col in range(2):
            intersections = np.tensordot(np.arange(1, self.n_bins+1), 
                                         scaled_dvecs[:, col], 0) % 1 
                                         #(n_bins, N) arr
            periods.append( np.argmax(
                (intersections < 1/self.n_bins) | \
                (intersections > 1-1/self.n_bins), axis=0
            ) + 1 )

        period = np.lcm(*periods)

        return period * self.boxsize / dvecs[:,-1]
    
    @cached_property
    def snap_xs(self):
        """
        Retrieve and compute the redshifts and corresponding comoving distances
        of each simulation snapshot. Save both as class attributes.
        
        Returns
        -------
        snap_xs: (100,) array
            The comoving distances corresponding to the redshift of each
            snapshot. snap_xs[i] corresponds to snapshot i.
        snap_zs: (100,) array
            The redshift of each snapshot. snap_zs[i] corresponds to snapshot
            i.
        """
        snap_zs = []
        if os.path.exists(str(self.snap_zs_file)):
            snap_zs = np.load(self.snap_zs_file)
        else:
            for snap in range(100):
                with h5py.File(self.get_snap_chunk_path(snap)) as f:
                    snap_zs.append(f['Header'].attrs['Redshift'])
                    snap_zs = np.array(snap_zs)
        self.snap_zs = snap_zs
        return self.comoving_distance(snap_zs)
    
    @cached_property
    def snap_x_lims(self):
        """
        Compute the upper limit of comoving distances of each snapshot via 
        linear interpolation of each snapshot's redshift.
        """
        return np.insert(np.convolve(self.snap_xs, [0.5,0.5], 'valid'), 99, 0) 
    
    @cached_property
    def z_from_dist(self):
        
        # create interpolant for determining redshifts from comoving distances
        redshift_vals = np.linspace(0,self.max_z,100000)
        dist_vals = self.comoving_distance(redshift_vals)
        return interp1d(dist_vals, redshift_vals)
    
    
class frb_simulation(simulation):
    """
    A class for ray tracing to FRBs in IllustrisTNG from a specified origin.
    Inherits from the simulation class. 
    """
    
    def __init__(
        self, 
        name='L205n2500TNG', 
        origin=np.array([0,0,0]), 
        **kwargs
    ):
        """
        Parameters
        ----------
        origin: (3,) array
            The coordinates of the observer in the simulation.
        kwargs:
            Arguments passed to the child class
        """
        
        super().__init__(name, **kwargs)
        self.origin=np.asarray(origin)

    def check_validity(self, dests, ang=False): 
        """
        Currently depreciated.
        """

        if ang:
            dests = self.ang2loc(dests) # convert to location coordinates
        dvecs = np.sort(np.abs(np.atleast_2d(dests) - self.origin))
        scaled_dvecs = dvecs / dvecs[:, -1] #(N, 2) arr

        periods = []
        for col in range(2):
            intersections = np.tensordot(np.arange(1,self.n_bins+1), scaled_dvecs[:, col], 0) % 1 #(n_bins, N) arr
            periods.append( np.argmax((intersections < 1/self.n_bins) | (intersections > 1-1/self.n_bins) , axis=0) + 1 )

        period = np.lcm(*periods)

        if ang: #return max distance if input given in angular coordinates
            return period * self.boxsize / dvecs[:,-1]

        res = (dvecs[:, -1] <= period*self.boxsize)
        if len(res) == 1:
            return res[0]
        return res
    
    def ray_trace(self, dest):
        """
        Ray traces from observer to FRB through the electron map to return 
        $n_e(x)$.
        
        Returns
        -------
        x_edge_dists: (N+1,) array
            An array of the distances from the observer to all bin crossings
            plus end points, in ckpc/h.
        xs: (N,) array
            An array of the distances from the observer to the midpoints of the
            ray in each bin, in ckpc/h.
        nes: (N,) array
            An array containing the electron number density $n_e$ in each bin,
            in (ckpc/h)^{-3}.
        """
        dest = np.asarray(dest)
                         
        x_edge_dists = [0] #edge distances: for riemann integration
        xs = [] #bin midpoints, for calculating redshift, n_e(x), etc.
        nes = [] #electron density in (ckpc/h)**-3

        Vcell = self.binsize**3

        sim_box_edges, sim_box_edge_dists, sim_box_coords = get_box_crossings(
            dest, self.origin, self.boxsize)

        current_pos = self.origin
        traveled_dist = 0

        open_snap = 99
        open_map = np.load(self.get_emap_path(99))

        for box_idx in range(len(sim_box_edges)):

            offset = self.boxsize * sim_box_coords[box_idx] #offset sim box coordinates
            next_pos = sim_box_edges[box_idx]

            bin_edges, bin_edge_dists, bin_coords = get_box_crossings(
                next_pos-offset, current_pos-offset, self.binsize)

            bin_coords = np.mod(bin_coords, self.n_bins)
            bin_indices = bin_coords[:, 0] * self.n_bins**2 + \
                          bin_coords[:, 1] * self.n_bins + \
                          bin_coords[:, 2]
            
            bin_edge_dists += traveled_dist
            bin_mid_dists = np.convolve(
                np.insert(bin_edge_dists, 0, traveled_dist), [0.5, 0.5], 
                'valid')
            bin_snaps = np.argmin(np.abs(np.repeat(
                np.atleast_2d(bin_mid_dists), 100, axis=0
            ).T - self.snap_xs), axis=1) #find closest snapshot for each bin
            
            for bidx in range(len(bin_edges)): #loop through bins
                snap = bin_snaps[bidx]
                #if corresponding snapshot is not open, open it
                if snap != open_snap: 
                    open_snap = snap
                    open_map = np.load(self.get_emap_path(snap))

                x_edge_dists.append(bin_edge_dists[bidx])
                xs.append(bin_mid_dists[bidx])
                nes.append(open_map[bin_indices[bidx]] / Vcell)

            current_pos = next_pos
            traveled_dist = sim_box_edge_dists[box_idx]
            

        return np.array(x_edge_dists), np.array(xs), np.array(nes) 
        
    def compute_DM(self, x_edge_dists, xs, nes, cumulative=False):
        """
        Computes the DM given the output of ray_trace.
        $\mathrm{DM} = \int n_e(x)(1+z(x))\,dx$
        """
    
        nes = (nes * (u.kpc/self.h)**(-3)).to(u.cm**(-3))
        zs = self.z_from_dist(xs)

        y = nes * (1 + zs) 
        dx = np.diff((x_edge_dists/self.h * u.kpc).to(u.pc))

        if cumulative:
            return np.flip(x_edge_dists[-1]-xs)*u.kpc/cu.littleh, \
                   np.cumsum(np.flip(y * dx)) #integrate from FRB to observer
        else:
            return np.sum(y * dx)
    
    def get_frb_DM(self, dest, cumulative=False):
        """
        Given an FRB location, ray traces through binned simulation snapshots 
        to compute the DM via Riemann integration.
        
        If cumulative, then `dist_from_FRB` and `cumulative_DM` is returned,
        which can be plotted to track the accumulated DM as the ray travels
        from FRB to observer.
        
        If not cumulative, then the total DM is returned.
        
        Parameters
        ----------
        dest: (3,) array
            The coordinates of the FRB in the simulation.
        cumulative: bool, optional
            Whether or not to return the cumulative or total DM.
        
        Returns
        -------
        dist_from_FRB: (N,) array
            Bin midpoint distances from the FRB with astropy units.
        cumulative_DM: (N,) array
            The accumulated DM along the ray from the FRB with astropy units.
        """
        
        x_edge_dists, xs, nes = self.ray_trace(dest)
        return self.compute_DM(x_edge_dists, xs, nes, cumulative)

