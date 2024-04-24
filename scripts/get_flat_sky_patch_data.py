import os
import numpy as np
import healpy as hp
import pandas as pd
from illustris_frb import frb_simulation
import multiprocessing as mp

"""
This script takes a rectangular equatorial sky region, given by bounds in
(theta, phi), and puts one FRB per pixel at a fixed z. The pixel array is given
by res, assuming that res evenly divides the region. It also gets the galaxy
number count per pixel from the pre-computed galaxy catalog of the cone (see
get_g_catalog_flatsky.py).

The ray tracing is parallelized using the multiprocessing module.

Output: A pandas dataframe will be saved to `outpath` that contains an array
for the DM and galaxy count. arr.reshape((n_x, n_y)) recovers the 2d array,
such that arr_2d[i_x][i_y] gets the number at pixel (i_x, i_y).
arr[i_x*n_y + i_y] = arr_2d[i_x][i_y]
"""

## --- INPUTS ---

mem_per_FRB = 1.5 #Gb
total_mem = 75 #Gb
nproc = int(total_mem // mem_per_FRB)

gcat_path = '/home/tnguser/frb_project/data/g_cats/test_flat' #where
outpath = '/home/tnguser/frb_project/data/results/test_flat_res001.hdf5'

name = 'L205n2500TNG'
binsize = 500
origin = binsize * np.array([50, 70, 23]) # same origin as in DM_redshift.ipynb
z = 0.4 # place FRBs at z=0.4

theta_min = np.pi/2 - 0.09
theta_max = np.pi/2 + 0.09
phi_min = 0.01
phi_max = 0.19 #see get_good_pixels.ipynb
#.18 x .18 rad region

res = 0.001 #32400 FRBs

## --- END OF INPUTS ---

sim = frb_simulation(name, binsize=binsize, origin=origin, max_z=z)
x = sim.comoving_distance(z)

theta_grid = np.arange(theta_min, theta_max+res/2, res)
phi_grid = np.arange(phi_min, phi_max+res/2, res) #bin edges
n_x = len(theta_grid)-1
n_y = len(phi_grid)-1
N = n_x * n_y

pix_angs = np.array(np.meshgrid(theta_grid[:-1]+res/2, 
                                phi_grid[:-1]+res/2)).T.reshape(N,2) #bin centers
pix_vecs = x * hp.ang2vec(pix_angs[:,0], pix_angs[:,1])


## get FRB DMs, one FRB per pixel
pool = mp.Pool(processes=nproc)
DM_arr = pool.map(sim.get_frb_DM, pix_vecs) #(N,) arr

## get foreground galaxy count
g_counts = np.zeros(N)
for fn in os.listdir(gcat_path):
    if '.hdf5' in fn:
        df = pd.read_hdf(os.path.join(gcat_path, fn))
        H, _, _ = np.histogram2d(df['theta'], df['phi'], 
                                 (theta_grid, phi_grid))
        g_counts += H.flatten()

## save results
df = pd.DataFrame({'DM': [dm.value for dm in DM_arr],
                   'N_g': g_counts.astype(int)})
df.to_hdf(outpath, key='data')
