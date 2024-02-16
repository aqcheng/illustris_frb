import os
import numpy as np
from numpy.linalg import norm
import pandas as pd
from healpy import ang2vec, vec2ang

from illustris_frb import simulation
from illustris_frb.utils import get_box_crossings, coord2flatpix
import time

"""
Given a patch of sky, retrieves all the galaxies within that cone from z=0
to z=z_max, stacking periodic boxes. Getting galaxies from the entire sky
becomes extremely expensive (>10Gb of memory around z=0.1).

Generally, the patch of sky is specified by `checkpoints` and `mask`. 
`checkpoints` is a list of representative sky coordinates in the region to
get all periodic boxes. More than 1 checkpoint is necessary if boxsize/r_max
is greater than the angular size of the region. `mask` is a function that,
given arrays theta and phi (and any kwargs), returns a boolean array indicating
if the coordinates are within the patch of sky.

Here, the patch of sky that is implemented is specified by simple bounds of
theta and phi. A general rectangular region would rotate the system.
"""

#inputs
sim = simulation('L205n2500TNG')
galaxy_mapdir = '/data/submit/submit-illustris/april/data/g_maps'
outpath = '/data/submit/submit-illustris/april/data/g_cats/test_flat'
if not os.path.isdir(outpath):
    os.makedirs(outpath)
origin= sim.binsize * np.array([50, 70, 23]) # same origin as in DM_redshift.ipynb
max_z = 0.4 #get galaxies up to z=0.4

#square region; see get_good_pixels.ipynb
theta_min = np.pi/2 - 0.09
theta_max = np.pi/2 + 0.09
phi_min = 0.01
phi_max = 0.19 
#0.18 rad x 0.18 rad

def simple_mask(theta, phi, theta_min, theta_max, phi_min, phi_max):
    #rotation not implemented
    return (theta > theta_min) & (theta < theta_max) & (phi > phi_min) & (phi < phi_max)

def get_galaxy_map_path(snap):
    return os.path.join(galaxy_mapdir, f'{snap}_galaxies.hdf5')

def get_shell_boxes(origin, r_min, r_max, checkpoints, boxsize):
    
    # gets all boxes that intersect with rays specified by checkpoints extending from r_min to r_max
    # checkpoints is a list of (theta, phi) coordinates to check; they should be spaced at most boxsize/r_max apart
    
    boxlist=[]
    for theta, phi in checkpoints:
        vec = ang2vec(theta, phi)
        _, _, boxes = get_box_crossings(origin+r_min*vec, origin+r_max*vec, boxsize)
        boxlist.append(boxes)
    
    return np.unique(np.vstack(boxlist), axis=0)

def get_cone_shell(sim, snap, origin, outpath, checkpoints, in_region, kwargs=None,
                   min_z=None, max_z=None, columns=['Mass', 'SFR', 'Radius'], 
                   metadata='metadata.txt'):
    
    # columns: columns of dataframe to retrieve, besides the coordinates
    # checkpoints: list of (theta, phi) coordinates to check; they should be spaced at most boxsize/r_max apart
    # in_region: a function that takes (thetas, phis, **kwargs) as arguments and returns a boolean array
    
    if min_z is None:
        min_x = x_lims[snap]
        min_z = sim.z_from_dist(min_x)
    else:
        min_x = min(x_lims[snap], sim.comoving_distance(min_z))
        
    if max_z is None:
        max_x = x_lims[snap-1]
        max_z = sim.z_from_dist(max_x)
    else:
        max_x = max(x_lims[snap-1], sim.comoving_distance(max_z))
    
    ## get all periodic boxes that intersect with the shell
    
    boxes = get_shell_boxes(origin, min_x, max_x, checkpoints, sim.boxsize)
    nbox = len(boxes)
    print(f'{"":<8}{nbox} periodic box(es) found')
    
    columns = ['x', 'y', 'z'] + columns
    data = np.array(pd.read_hdf(get_galaxy_map_path(snap))[ columns ])
    
    res = []
    thetas = []
    phis = []
    for box in boxes:
              
        rel_coords = data[:,:3] + box*sim.boxsize - origin
        theta, phi = vec2ang(rel_coords)
           
        dists = norm(rel_coords, axis=1) 
        mask = (dists > min_x) & (dists < max_x) & \
               in_region(theta, phi, **kwargs)
        
        res.append( data[mask] )
        thetas.append( theta[mask] )
        phis.append( phi[mask] )
    
    #save and write metadata
    df = pd.DataFrame( np.vstack(res), columns=columns )
    df['theta'] = np.concatenate(thetas)
    df['phi'] = np.concatenate(phis)
    df.to_hdf(os.path.join(outpath, f'{snap}_shell.hdf5'), key='data')

    metadata_path = os.path.join(outpath, metadata)
    if not os.path.isfile( metadata_path ):
        with open(metadata_path, 'w+') as f:
            f.write(f'origin: {list(origin)}\n\n')
            f.write(f"{'snap':<6}{'min_x':<16}{'max_x':<16}{'min_z':<10}{'max_z':<10}{'nbox':<6}ngal\n")
    with open(os.path.join(outpath, metadata), 'a+') as f:
        f.write(f'{snap:<6}{min_x:<16.6f}{max_x:<16.6f}{min_z:<10.6f}{max_z:<10.6f}{nbox:<6}{len(df)}\n')
    
    df = None #clear

x_lims = sim.snap_x_lims
final_snap = np.argwhere(x_lims < sim.comoving_distance(max_z))[0][0]

mask_kwargs = {'theta_min': theta_min,
               'theta_max': theta_max,
               'phi_min': phi_min,
               'phi_max': phi_max}
checkpoints = ((theta_min, phi_min), (theta_max, phi_min),
               (theta_min, phi_max), (theta_max, phi_max),
               (np.pi/2, phi_min), (np.pi/2, phi_max))

start = time.time()
for snap in range(99, final_snap, -1):
    print(f'{time.time()-start:<6.0f}: Working on snapshot {snap}')
    get_cone_shell(sim, snap, origin, outpath, checkpoints, simple_mask, kwargs=mask_kwargs)
print(f'{time.time()-start:<6.0f}: Working on snapshot 72')
get_cone_shell(sim, final_snap, origin, outpath, checkpoints, simple_mask, kwargs=mask_kwargs, max_z=max_z)
print(f'{time.time()-start:<6.0f}: Done')