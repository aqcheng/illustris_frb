import sys
sys.path.append('/home/submit/aqc/frb_project')

import os
import numpy as np
import healpy as hp
import pandas as pd
from illustris_frb import frb_simulation
import multiprocessing as mp

"""
This script is the same as get_flat_sky_patch_data.py, but is modified to
be further parallelized in different jobs. The pixels are partitioned into
args.N tasks, and which task is specified by args.n.

After the jobs complete, run aggregrate_flat_sky_patch_data for the galaxy
number counts and to concatenate the DMs together.
"""

## --- INPUTS ---

mem_per_FRB = 1.5 #Gb
total_mem = 10 #Gb
nproc = int(total_mem // mem_per_FRB)

gcat_path = '/data/submit/submit-illustris/april/data/g_cats/test_flat' 
outdir = '/work/submit/aqc/test_flat_res0001' #these will be concatenated later

name = 'L205n2500TNG'
binsize = 500
origin = binsize * np.array([50, 70, 23]) # same origin as in DM_redshift.ipynb
z = 0.4 # place galaxies at z=0.4

theta_min = np.pi/2 - 0.09
theta_max = np.pi/2 + 0.09
phi_min = 0.01
phi_max = 0.19 #see get_good_pixels.ipynb
#.18 x .18 rad region

res = 0.001 #32400 FRBs; 180x180


## --- COMMAND LINE INPUTS ##
import argparse
argp = argparse.ArgumentParser()
argp.add_argument("-N", type=int, required=True, help="Total # of partitions. Should divide the # of theta pixels on a side.")
argp.add_argument("-n", type=int, required=True, help="Partition number; must be between 1 and N. Required if N specified.")
args = argp.parse_args()



if not os.path.isdir(outdir):
    os.makedirs(outdir)

sim = frb_simulation(name, binsize=binsize, origin=origin, max_z=z)
x = sim.comoving_distance(z)

nrows_tot = int((theta_max - theta_min)/res)
nrows = nrows_tot // args.N
start = nrows * (args.n-1) 
end = nrows * args.n #how to slice the theta grid

theta_grid = np.arange(theta_min+res/2, theta_max, res)[start:end] #pixel centers
phi_grid = np.arange(phi_min+res/2, phi_max, res)
n_x = len(theta_grid)
n_y = len(phi_grid)
N = n_x * n_y

pix_angs = np.array(np.meshgrid(theta_grid, phi_grid)).T.reshape(N,2)
pix_vecs = x * hp.ang2vec(pix_angs[:,0], pix_angs[:,1])

## get FRB DMs, one FRB per pixel
pool = mp.Pool(processes=nproc)
DM_arr = pool.map(sim.get_frb_DM, pix_vecs) 

## save results
outf = os.path.join(outdir, f'{args.n}.npy')
np.save(outf, np.array([dm.value for dm in DM_arr]))
