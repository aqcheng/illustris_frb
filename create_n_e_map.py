import argparse

import os
import numpy as np
import h5py
import pandas as pd

from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const

import time
import cProfile

import psutil #for tracking memory usage
process = psutil.Process()

# DEFINING FUNCTIONS

full_snaps = (2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99)

def get_snapdir_path(snap):
    return os.path.join(basepath, f'snapdir_{snap:03}')

def get_chunk_path(snap, chunk=0):
    return os.path.join(get_snapdir_path(snap), f'snap_{snap:03}.{chunk}.hdf5')

def sort_chunks_to_bins(snap, outfn):
    
    full_snap = snap in full_snaps
    res = np.zeros(n_bins**3)
    
    for chunk in range(NChunks):

        #process and save the electron number counts + coordinates of all particles

        chunk_path = get_chunk_path(snap, chunk)

        with h5py.File(chunk_path) as f:

            coords = np.array(f['PartType0/Coordinates'])

            #calculate electron number count; N_e = m_g eta_e X_H / m_p

            m_g = (np.array(f['PartType0/Masses'], dtype=np.float64) * 1e10 * u.solMass / cu.littleh).to(u.kg, h_equiv)
            eta_e = np.array(f['PartType0/ElectronAbundance'])

            #X_H only given in full snaps
            # if snap in fullsnaps:
            if full_snap:
                X_H = np.array(f['PartType0/GFM_Metals'][:,0])
            else:
                X_H = (1-np.array(f['PartType0/GFM_Metallicity']))*0.76

        N_e = m_g * eta_e * X_H / const.m_p

        bin_index = (coords[:,0] // bin_size)*n_bins**2 + \
                    (coords[:,1] // bin_size)*n_bins + \
                    (coords[:,2]  // bin_size)

        res += pd.DataFrame({'i': bin_index, 'N_e': N_e}).groupby(by='i').sum().reindex(range(n_bins**3), fill_value=0).to_numpy()[:,0]
        mem = process.memory_info().rss/1024**3
        print(f'{time.time()-start_time:<6.2f}: Done with chunk {chunk}. Current memory: {mem:.1f} GB')

    np.save(outfn, res)


#set argparse
argp = argparse.ArgumentParser()
argp.add_argument("-s", "--sim", type=str, required=True, choices=os.listdir('/home/tnguser/sims.TNG'), 
                  help="Name of simulation as given in the path, e.g. L205n1250TNG")
argp.add_argument("-v", "--verbose", type=bool, default=False, help="Whether or not to print out detailed progress and timestamps. Default=False")
argp.add_argument("--bin-size", type=int, default=500, help="The size of a bin in ckpc/h. Default=500")
argp.add_argument("--snaps", type=int, default=[99], nargs='+', help="Which snapshots to process. Default=99")
argp.add_argument("--outpath", type=str, default=None, help="Path to where the output electron density map will go. If unspecified, will go to ./n_e_maps/{sim}")
# argp.add_argument("--mem", type=float, default=8, help="Max memory for this script, in GB. Default=8") 
args = argp.parse_args()

if args.outpath is None:
    outpath = f'n_e_maps/{args.sim}'
else:
    outpath = args.outpath

if not os.path.exists(outpath):
    os.makedirs(outpath)

basepath = f'/home/tnguser/sims.TNG/{args.sim}/output'

#Retrieve relevant cosmological parameters
with h5py.File(get_chunk_path(99)) as f:
    header = dict(f['Header'].attrs)

BoxSize = int(header['BoxSize'])
h = header['HubbleParam']
Omega0 = header['Omega0']
NChunks = header['NumFilesPerSnapshot']

cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega0)
h_equiv = cu.with_H0(cosmo.H0)

bin_size = args.bin_size
n_bins = int(BoxSize / bin_size)

"""
#run
print(f'{n_bins}^3 = {n_bins**3} bins of size {bin_size} ckpc/h')

start_time = time.time()
for snap in args.snaps:
    outfn = os.path.join(outpath, f'{snap}.npy') #n_e_maps/{sim}/{snap}.hdf5
    print(f'{time.time()-start_time:<6.2f}: Creating electron density map for {snap}')
    sort_chunks_to_bins(snap, outfn)
    print(f'{time.time()-start_time:<6.2f}: Done processing snapshot {snap}')
    """

start_time = time.time()
cProfile.run("sort_chunks_to_bins(99, os.path.join(outpath, '99.npy'))", filename=os.path.join(outpath, 'runtime_stats'))


# EXAMPLE USAGE
# python create_n_e_map.py -s L35n270TNG
# nohup python create_n_e_map.py -s L35n540TNG > process_L35n270TNG.log &
# nohup python create_n_e_map.py -s L205n2500TNG > time_L205n2500TNG.log &
