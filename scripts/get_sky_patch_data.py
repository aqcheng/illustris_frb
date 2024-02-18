import os
import numpy as np
import healpy as hp
import pandas as pd
from illustris_frb import frb_simulation
import multiprocessing as mp

"""
This script takes a sky region, determined by a list of healpy pixels, and puts
one FRB per pixel at a fixed z. It also gets the galaxy number count per pixel 
from the pre-computed galaxy catalog of the cone (see get_g_catalog.py).

The ray tracing is parallelized using the multiprocessing module.

Output: A pandas dataframe will be saved to `outpath` that contains an array
for the DM and galaxy count, indexed by pixel number.
"""

## INPUTS

mem_per_FRB = 1.5 #Gb
total_mem = 10 #Gb
nproc = int(total_mem // mem_per_FRB)

nside=128
gcat_path = '/home/tnguser/frb_project/data/g_cats/test' #where
outpath = f'/home/tnguser/frb_project/data/test_nside{nside}.hdf5'

binsize = 500
origin = binsize * np.array([50, 70, 23]) # same origin as in DM_redshift.ipynb
z = 0.4 # place galaxies at z=0.4
sim = frb_simulation('L205n2500TNG', origin=origin, max_z=z)
x = sim.comoving_distance(z)

# see get_good_pixels.ipynb
pixels = np.load(f'/home/tnguser/frb_project/data/g_cats/test_pixels_nside{nside}.npy')
checkpixels = np.array([90370, 90894, 105730, 105742])
N = len(pixels)

# ## get FRB DMs, one FRB per pixel
# pixel_coords = x * np.vstack(hp.pix2vec(nside, pixels)).T
# pool = mp.Pool(processes=nproc)
# DM_arr = pool.map(sim.get_frb_DM, pixel_coords)

# df = pd.DataFrame({'DM': [dm.value for dm in DM_arr]}, index=pixels)
df = pd.read_hdf(outpath)

## get foreground galaxy count
df['N_g']=0
for fn in os.listdir(gcat_path):
    if '.hdf5' in fn:
        tmp = pd.read_hdf(os.path.join(gcat_path, fn))
        unique, counts = np.unique(tmp['ipix'], return_counts=True)
        df.loc[unique, 'N_g'] += counts #idk if this actually works

# save results
os.remove(outpath)
df.to_hdf(outpath, key='data')

#archive:
# def get_fg_counts(self, nside, max_z=None):
#         """
#         Gets the foreground galaxy count for each healpy pixel using the
#         galaxy catalog specified by gcat_dir.
        
#         Parameters
#         ----------
#         nside: int
#             Determines angular resolution of healpy array.
#         z_max: float
#             Applies a cutoff to the last snapshot. If none, counts all galaxies
#             in the catalog.
            
#         Returns
#         -------
#         g_counts: (N,) arr
#             g_counts[i] is the number of foreground galaxies in pixel i
#         """
#         N = hp.nside2npix(nside)
#         g_counts = np.zeros(N, dtype=int)
        
#         def get(fn, max_z=None):
#             df = pd.read_hdf(os.path.join(self.gcat_dir, fn))
#             if max_z is not None:
#                 max_x = self.comoving_distance(max_z)
#                 mask = np.linalg.norm(
#                     np.array(df[['x','y','z']]), axis=1
#                 ) < max_x
#                 df = df[ mask ]
#             tmp = np.bincount(hp.ang2pix(nside, df['theta'], df['phi']))
#             return np.pad(tmp, (0,N-tmp.shape[0]))
        
#         if max_z is None:
#             fns = sorted([fn for fn in os.listdir(self.gcat_dir) 
#                           if ('.hdf5' in fn)], reverse=True)
#         else:
#             meta_df = pd.read_table(os.path.join(self.gcat_dir, 'metadata.txt'),
#                                     delimiter='\s+', header=1)
#             last_snap = int(
#                 meta_df['snap'][(meta_df['min_z'] < max_z) & 
#                                 (max_z < meta_df['max_z'])].iloc[0]
#             )
#             fns = [str(i)+'_shell.hdf5' for i in range(99, last_snap, -1)]
        
#         for fn in fns:
#             g_counts += get(fn)
#         if max_z is not None: #get last snap, where we need to apply a z cut
#             g_counts += get(f'{last_snap}_shell.hdf5', max_z)
        
#         return g_counts