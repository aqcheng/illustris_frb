import numpy as np
import os
import pandas as pd

## ---- INPUTS ----

tmpdir = '/work/submit/aqc/test_flat_res001'
gcat_path = '/data/submit/submit-illustris/april/data/g_cats/test_flat' 
# outpath = '/data/submit/submit-illustris/april/data/results/test_flat_res001.hdf5'
outpath = '/data/submit/submit-illustris/april/data/results/test_flat_res001_shell.hdf5'

theta_min = np.pi/2 - 0.09
theta_max = np.pi/2 + 0.09
phi_min = 0.01
phi_max = 0.19 #see get_good_pixels.ipynb
#.18 x .18 rad region

res = 0.001 #32400 FRBs
npix = 18

## -- END OF INPUTS --

# -- test_flat_res001 -- 
"""
## aggregate results (FRB DMs) from parallel_get_flat_sky_patch_data.py
DMs = []
for i in range(npix):
    DMs.append(np.load(os.path.join(tmpdir, f'{i+1}.npy')))

DMs = np.concatenate(DMs)

## get foreground galaxy count
theta_grid = np.arange(theta_min, theta_max+res/2, res)
phi_grid = np.arange(phi_min, phi_max+res/2, res) #bin edges
N = (len(theta_grid)-1)*(len(phi_grid)-1)

g_counts = np.zeros(N)
for fn in os.listdir(gcat_path):
    if '.hdf5' in fn:
        df = pd.read_hdf(os.path.join(gcat_path, fn))
        H, _, _ = np.histogram2d(df['theta'], df['phi'], 
                                 (theta_grid, phi_grid))
        g_counts += H.flatten()

## save data
df = pd.DataFrame({'DM': DMs,
                   'N_g': g_counts.astype(int)})
df.to_hdf(outpath, key='data')
"""

# -- test_flat_res001_shell -- 
df = pd.read_hdf('/data/submit/submit-illustris/april/data/results/test_flat_res001.hdf5') # get DMs

## get foreground galaxy count
theta_grid = np.arange(theta_min, theta_max+res/2, res)
phi_grid = np.arange(phi_min, phi_max+res/2, res) #bin edges
N = (len(theta_grid)-1)*(len(phi_grid)-1)

target = 84 #target shell, z=0.2
data = pd.read_hdf(os.path.join(gcat_path, f'{target}_shell.hdf5'))
H, _, _ = np.histogram2d(data['theta'], data['phi'], (theta_grid, phi_grid))
df['N_g'] = H.astype(int).flatten()

df.to_hdf(outpath, key='data')


