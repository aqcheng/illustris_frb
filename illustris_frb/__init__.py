import os
import numpy as np
import h5py
import pandas as pd

from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const

from scipy.interpolate import interp1d

from .utils import get_box_crossings

class simulation:
    """
    An object for the attributes of an IllustrisTNG simulation, including the 
    simulation boxsize, paths to the electron density maps and simulation 
    files, and cosmology.
    """
    
    def __init__(self, name, binsize=500, sim_dir=None, map_dir=None):
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
        map_dir: str, optional
            Directory to electron density map. Defaults to './n_e_maps/{name}'
        """
        
        
        self.name = name
        
        if sim_dir is None:
            self.sim_dir = f'/home/tnguser/sims.TNG/{name}/output'
        else:
            self.sim_dir = sim_dir
        if map_dir is None:
            self.map_dir = f'n_e_maps/{name}'
        else:
            self.map_dir = map_dir
        
        #get simulation attributes
        with h5py.File(self.get_chunk_path(99)) as f:
            header = dict(f['Header'].attrs)

        self.boxsize = int(header['BoxSize'])
        self.binsize=binsize
        self.n_bins = int(self.boxsize / self.binsize)
        
        self.h = header['HubbleParam']
        self.Omega0 = header['Omega0']

        self.cosmo = FlatLambdaCDM(H0=100*self.h, Om0=self.Omega0)
        self.h_equiv = cu.with_H0(self.cosmo.H0)
    
    def get_snapdir_path(self, snap):
        return os.path.join(self.sim_dir, f'snapdir_{snap:03}')

    def get_chunk_path(self, snap, chunk=0):
        return os.path.join(self.get_snapdir_path(snap), 
                            f'snap_{snap:03}.{chunk}.hdf5')
    
    def get_map_path(self, snap):
        return os.path.join(self.map_dir, f'{snap}.npy')
    
    def get_snapshot_dists(self):
        """
        Retrieve and compute the redshifts and corresponding comoving distances
        of each simulation snapshot.
        
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
        for snap in range(100):
            with h5py.File(self.get_chunk_path(snap)) as f:
                snap_zs.append(f['Header'].attrs['Redshift'])
        snap_zs = np.array(snap_zs) * cu.redshift
        snap_xs = self.cosmo.comoving_distance(snap_zs).to(u.kpc / cu.littleh, 
                                                           self.h_equiv).value
        return snap_xs, snap_zs
    
class frb_simulation(simulation):
    """
    A class for ray tracing to FRBs in IllustrisTNG from a specified origin. 
    Inherits from the simulation class. 
    """
    
    def __init__(
        self, 
        name, 
        binsize, 
        origin=np.array([0,0,0]), 
        sim_dir=None, 
        map_dir=None, 
        max_z=2
    ):
        """
        Parameters
        ----------
        origin: (3,) array
            The coordinates of the observer in the simulation.
        max_z: float, optional
            The maximum redshift out to which FRBs will be simulated. Used
            only for creating the z(x) interpolant. Default: 2
        """
        
        simulation.__init__(self, name, binsize, sim_dir, map_dir)
        self.origin=origin
        
        self.snap_xs, self.snap_zs = self.get_snapshot_dists()
        
        # create interpolant for determining redshifts from comoving distances
        redshift_vals = np.linspace(0,max_z,100000)
        dist_vals = self.cosmo.comoving_distance(redshift_vals).to(
            u.kpc/cu.littleh, self.h_equiv).value
        self.z_from_dist = interp1d(dist_vals, redshift_vals)
    
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
    
        x_edge_dists = [0] #edge distances: for riemann integration
        xs = [] #bin midpoints, for calculating redshift, n_e(x), etc.
        nes = [] #electron density in (ckpc/h)**-3

        Vcell = self.binsize**3

        sim_box_edges, sim_box_edge_dists, sim_box_coords = get_box_crossings(
            dest, self.origin, self.boxsize)

        current_pos = self.origin
        traveled_dist = 0

        open_snap = 99
        open_map = np.load(self.get_map_path(99))

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
                    open_map = np.load(self.get_map_path(snap))

                x_edge_dists.append(bin_edge_dists[bidx])
                xs.append(bin_mid_dists[bidx])
                nes.append(open_map[bin_indices[bidx]] / Vcell)

            current_pos = next_pos
            traveled_dist = sim_box_edge_dists[box_idx]
            

        return np.array(x_edge_dists), np.array(xs), np.array(nes) 
        
    def compute_DM(self, x_edge_dists, xs, nes, cumulative=False):
        """
        Computes the DM given the output of ray_trace.
        """
    
        nes = (nes * (u.kpc/cu.littleh)**(-3)).to(u.cm**(-3), self.h_equiv)
        zs = self.z_from_dist(xs)

        y = nes * (1 + zs) 
        dx = np.diff((x_edge_dists * u.kpc/cu.littleh).to(u.pc, self.h_equiv))

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

