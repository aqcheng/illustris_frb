import os
import numpy as np
from numpy.linalg import norm
import pandas as pd
import healpy as hp

from illustris_frb import simulation
from illustris_frb.utils import get_box_crossings
import time

"""
Given a patch of sky, retrieves all the galaxies within that cone from z=0
to z=z_max, stacking periodic boxes. Getting galaxies from the entire sky
becomes extremely expensive (>10Gb of memory around z=0.1).

The patch of sky is specified by a list of pixels. `checkpixels` is a list of
representative sky pixels in the region to get all periodic boxes. More than 1 checkpoint 
is necessary if boxsize/r_max is greater than the angular size of the region.
"""

#inputs
sim = simulation('L205n2500TNG', 
                 gmap_dir = '/home/tnguser/frb_project/data/g_maps')
outpath = '/home/tnguser/frb_project/data/g_cats/test'
if not os.path.isdir(outpath):
    os.makedirs(outpath)
origin= sim.binsize * np.array([50, 70, 23]) # same origin as in DM_redshift.ipynb
max_z = 0.4 #get galaxies up to z=0.4
nside = 128

#see get_good_pixels.ipynb
pixels = np.load(f'/home/tnguser/frb_project/data/g_cats/test_pixels_nside{nside}.npy')
checkpixels = np.array([90370, 90894, 105730, 105742])

def get_shell_boxes(origin, r_min, r_max, checkpixels, boxsize):
    
    # gets all boxes that intersect with rays specified by checkpixels extending from r_min to r_max
    # checkpixels is a list of (theta, phi) coordinates to check; they should be spaced at most boxsize/r_max apart
    
    
    boxlist=[]
    for pix in checkpixels:
        vec = np.array(hp.pix2vec(nside, pix))
        _, _, boxes = get_box_crossings(origin+r_min*vec, origin+r_max*vec, boxsize)
        boxlist.append(boxes)
    
    return np.unique(np.vstack(boxlist), axis=0)

def get_cone_shell(sim, snap, origin, outpath, pixels, checkpixels,
                   min_z=None, max_z=None, columns=['Mass', 'SFR', 'Radius'], 
                   metadata='metadata.txt'):
    
    # columns: columns of dataframe to retrieve, besides the coordinates
    # checkpixels: list of (theta, phi) coordinates to check; they should be spaced at most boxsize/r_max apart
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
    
    boxes = get_shell_boxes(origin, min_x, max_x, checkpixels, sim.boxsize)
    nbox = len(boxes)
    print(f'{"":<8}{nbox} periodic box(es) found')
    
    columns = ['x', 'y', 'z'] + columns
    data = np.array(pd.read_hdf(sim.get_gmap_path(snap))[ columns ])
    
    res = []
    ipixs = []
    for box in boxes:
              
        rel_coords = data[:,:3] + box*sim.boxsize - origin
        ipix = hp.vec2pix(nside, rel_coords[:,0], rel_coords[:,1], rel_coords[:,2])
           
        dists = norm(rel_coords, axis=1) 
        
        mask = (dists > min_x) & (dists < max_x) & \
               np.isin(ipix, pixels)
        
        res.append( data[mask] )
        ipixs.append( ipix[mask] )
    
    #save and write metadata
    df = pd.DataFrame( np.vstack(res), columns=columns )
    df['ipix'] = np.concatenate(ipixs)

    df.to_hdf(os.path.join(outpath, f'{snap}_shell.hdf5'), key='data')

    metadata_path = os.path.join(outpath, metadata)
    if not os.path.isfile( metadata_path ):
        with open(metadata_path, 'w+') as f:
            f.write(f'origin: {list(origin)}\t\t\t nside:{nside}\n\n')
            f.write(f"{'snap':<6}{'min_x':<16}{'max_x':<16}{'min_z':<10}{'max_z':<10}{'nbox':<6}ngal\n")
    with open(os.path.join(outpath, metadata), 'a+') as f:
        f.write(f'{snap:<6}{min_x:<16.6f}{max_x:<16.6f}{min_z:<10.6f}{max_z:<10.6f}{nbox:<6}{len(df)}\n')
    
    df = None #clear

x_lims = sim.snap_x_lims
final_snap = np.argwhere(x_lims < sim.comoving_distance(max_z))[0][0]

start = time.time()
for snap in range(99,final_snap, -1):
    print(f'{time.time()-start:<6.0f}: Working on snapshot {snap}')
    get_cone_shell(sim, snap, origin, outpath, pixels, checkpixels)
print(f'{time.time()-start:<6.0f}: Working on snapshot {final_snap}')
get_cone_shell(sim, snap, origin, outpath, pixels, checkpixels, max_z=max_z)
print(f'{time.time()-start:<6.0f}: Done')