import sys
sys.path.append('../')

import argparse

import os
import numpy as np
import h5py
import pandas as pd

import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const

from illustris_frb import simulation

import time
# import cProfile

"""
Gets the electron density map of the simulation box. Inputs are given on the
command line; see python create_n_e_map.py --help. Example usages are also 
given in the bottom of this file.
"""

# DEFINING FUNCTIONS

full_snaps = (2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99)

def sort_chunks_to_bins(sim, snap):
    
    full_snap = snap in full_snaps
    res = np.zeros(sim.n_bins**3)
    
    for chunk in range(len(os.listdir(sim.get_snapdir_path(snap)))):

        #process and save the electron number counts + coordinates of all particles

        chunk_path = sim.get_snap_chunk_path(snap, chunk)

        with h5py.File(chunk_path) as f:

            coords = np.array(f['PartType0/Coordinates'])

            #calculate electron number count; N_e = m_g eta_e X_H / m_p

            m_g = (np.array(f['PartType0/Masses'], dtype=np.float64) * 1e10 * u.solMass / cu.littleh).to(u.kg, sim.h_equiv)
            eta_e = np.array(f['PartType0/ElectronAbundance'])

            #X_H only given in full snaps
            # if snap in fullsnaps:
            if full_snap:
                X_H = np.array(f['PartType0/GFM_Metals'][:,0])
            else:
                X_H = (1-np.array(f['PartType0/GFM_Metallicity']))*0.76

        N_e = m_g * eta_e * X_H / const.m_p

        bin_index = (coords[:,0] // sim.binsize)*sim.n_bins**2 + \
                    (coords[:,1] // sim.binsize)*sim.n_bins + \
                    (coords[:,2]  // sim.binsize)

        res += pd.DataFrame({'i': bin_index, 'N_e': N_e}).groupby(by='i').sum().reindex(range(sim.n_bins**3), fill_value=0).to_numpy()[:,0]
        #creates a {n_bins}^3 long array. each slot has N_e for each corresponding bin
        
        # mem = process.memory_info().rss/1024**3
        # print(f'{time.time()-start_time:<6.2f}: Done with chunk {chunk}. Current memory: {mem:.1f} GB')

    np.save(os.path.join(sim.map_dir, f'{snap}.npy'), res)


#set argparse
argp = argparse.ArgumentParser()
argp.add_argument("-s", "--sim", type=str, required=True, choices=os.listdir('/home/tnguser/sims.TNG'), 
                  help="Name of simulation as given in the path, e.g. L205n2500TNG")
argp.add_argument("--binsize", type=int, default=500, help="The size of a bin in ckpc/h. Default=500")
argp.add_argument("--snaps", type=int, default=[99], nargs='+', help="Which snapshots to process. Default=99")
argp.add_argument("--snap-range", type=int, nargs=2, help="Range of snapshots to process, inclusive. Ignores --snaps if specified. ")
argp.add_argument("--outpath", type=str, default=None, help="Path to where the output electron density map will go. If unspecified, will go to ./n_e_maps/{sim}")
args = argp.parse_args()

sim = simulation(args.sim, args.binsize, map_dir=args.outpath)

if args.snap_range is None:
    snaps_list = args.snaps
else:
    a, b = args.snap_range
    if a > b:
        snaps_list = range(a, b-1, -1)
    else:
        snaps_list = range(a, b+1)

#run
print(f'{n_bins}^3 = {n_bins**3} bins of size {binsize} ckpc/h')

start_time = time.time()
for snap in snaps_list:
    sort_chunks_to_bins(sim, snap)
    print(f'{time.time()-start_time:<6.2f}: Done processing snapshot {snap}')


# start_time = time.time()
# cProfile.run("sort_chunks_to_bins(sim, 99)", filename=os.path.join(sim.map_dir, 'runtime_stats'))


# EXAMPLE USAGE
# python create_n_e_map.py -s L35n270TNG
# nohup python create_n_e_map.py -s L35n540TNG > process_L35n270TNG.log &
# nohup python create_n_e_map.py -s L205n2500TNG > time_L205n2500TNG.log &
# nohup python create_n_e_map.py --sim L205n2500TNG --snap-range 98 32  > process_all_L205n2500TNG.log &
