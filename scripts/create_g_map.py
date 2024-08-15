import os
import numpy as np
import h5py
import pandas as pd

import time

from illustris_frb import simulation

"""
This script retrieves all the galaxies (i.e. subgroups marked as SubhaloFlag=0)
from all the group files of the simulation, for each snapshot. The galaxy's 
center of mass coordinates within the simulation box, as well as the mass, SFR,
and half mass radius, are stored in '{outpath}/{snap}_galaxies.hdf5'.
"""


#inputs
sim = simulation('L205n2500TNG')
outpath = '/home/tnguser/frb_project/data/g_maps'
if not os.path.exists(outpath):
    os.makedirs(outpath)

def get_all_galaxies(snap):
    
    with h5py.File(sim.get_group_chunk_path(99)) as f:
        header = dict(f['Header'].attrs)
    N_subgroups = header['Nsubgroups_Total']
    N_chunks = header['NumFiles']
    
    mask = []
    res = np.empty((2*N_subgroups, 9))
    i = 0 #row
    
    for chunk in range(N_chunks):
        
        with h5py.File(sim.get_group_chunk_path(snap, chunk)) as f:
            n_subgroups = f['Header'].attrs['Nsubgroups_ThisFile']
            if n_subgroups == 0:
                print(f'Skipping chunk {chunk}')
                continue
            
            mask += list(f['Subhalo/SubhaloFlag'])
            res[i:i+n_subgroups, 0:3] = np.array(f['Subhalo/SubhaloCM'])
            res[i:i+n_subgroups, 3] = np.array(f['Subhalo/SubhaloMass'])
            res[i:i+n_subgroups, 4] = np.array(f['Subhalo/SubhaloMassInMaxRadType'][:,4]) #stellar mass
            res[i:i+n_subgroups, 5] = np.array(f['Subhalo/SubhaloSFR'])
            res[i:i+n_subgroups, 6] = np.array(f['Subhalo/SubhaloHalfmassRad'])
            res[i:i+n_subgroups, 7] = np.array(f['Subhalo/SubhaloStellarPhotometrics'][:,2]) #visual magnitude
            res[i:i+n_subgroups, 8] = np.array(f['Subhalo/SubhaloStellarPhotometrics'][:,4]) #g magnitude
            
        
        i += n_subgroups
    
    res = res[:i]
    df = pd.DataFrame(res[mask], columns=['x', 'y', 'z', 'Mass', 'Stellar Mass', 'SFR', 'Half mass radius', 'M_V', 'M_g'])
    df.to_hdf(os.path.join(outpath, f'{snap}_galaxies.hdf5'), key='data')
    
    return df

# for snap in range(99, 31, -1):
for snap in range(99, 70, -1):
    
    print(f'{time.time():<5.1f}: getting galaxies from snapshot {snap}')
    get_all_galaxies(snap)