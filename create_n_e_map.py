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



# DEFINING FUNCTIONS

def get_snapdir_path(snap):
    return os.path.join(basepath, f'snapdir_{snap:03}')

def get_chunk_path(snap, chunk=0):
    return os.path.join(get_snapdir_path(snap), f'snap_{snap:03}.{chunk}.hdf5')

def sort_chunks_to_coarse_bins(snap, print_time=False, start_chunk=0, start_bin=0):
    
    #start_chunk and start_bin is so that we can restart without having to repeat calculations
    
    #process and save the electron number counts + coordinates of all particles
    for chunk in range(NChunks):
        if chunk < start_chunk:
            continue
        
        chunk_path = get_chunk_path(snap, chunk)
    
        with h5py.File(chunk_path) as f:
        
            coords = np.array(f['PartType0/Coordinates'])

            #calculate electron number count; N_e = m_g eta_e X_H / m_p

            m_g = (np.array(f['PartType0/Masses'], dtype=np.float64) * 1e10 * u.solMass / cu.littleh).to(u.kg, h_equiv)
            eta_e = np.array(f['PartType0/ElectronAbundance'])

            #X_H only given in full snaps
            # if snap in fullsnaps:
            if 'GFM_Metals' in f['PartType0'].keys():
                X_H = np.array(f['PartType0/GFM_Metals'][:,0])
            else:
                X_H = (1-np.array(f['PartType0/GFM_Metallicity']))*0.76
                
        N_e = m_g * eta_e * X_H / const.m_p

        df = pd.DataFrame(data=coords, columns=['x', 'y', 'z'])
        df['N_e'] = N_e.value
        
        df['coarse_bin_index'] = (coords[:,0] // coarse_bin_size)*n_coarse_bins**2 + \
                                 (coords[:,1] // coarse_bin_size)*n_coarse_bins + \
                                 (coords[:,2]  // coarse_bin_size)
        
        for i, sub_df in df.groupby(by='coarse_bin_index'):
            i = int(i)
            if chunk == start_chunk and i < start_bin:
                continue
            fn = os.path.join(snap_outpath, f'{i}_particles.hdf5')
            sub_df.to_hdf(fn, key='data', format='table', append=True)
        
            if print_time:
                print(f'{time.time()-start_time:<6.2f}: Done getting N_e for Chunk {chunk}, coarse bin {i}')

def histogram_to_fine_bins(snap, print_time=False, start_hist_bin=0):
    
    for i in range(start_hist_bin, n_coarse_bins**3): #iterate over coarse bins
        
        #coarsebin (i_x, i_y, i_z) corresponds to the box coarse_bin_size * (i_x, i_y, i_z) to coarse_bin_size * (i_x+1, i_y+1, i_z+1)
        #i = n**2 * i_x + n * i_y + i_z
        
        fn = os.path.join(snap_outpath, f'{i}_particles.hdf5')
        if os.path.isfile(fn):
            particles_in_box = pd.read_hdf(fn)
        else:
            print(f'      No particles in coarse bin {i}; skipping')
            continue
        
        H, edges = np.histogramdd(np.array(particles_in_box[['x', 'y', 'z']]), bins=n_fine_bins, 
                                  range=[[coarse_bin_size*i, coarse_bin_size*(i+1)]]*3, weights=np.array(particles_in_box['N_e']))
        
        np.save(os.path.join(snap_outpath, f'{i}_hist.npy'), H/fine_bin_size**3) #save electron density in (ckpc/h)^(-3)
        # np.save(os.path.join(snap_outpath, f'{i}_edges.npy'), edges)
        
        if print_time:
            print(f'{time.time()-start_time:<6.2f}: Done histogramming N_e for coarse bin {i}')


            
            
#set argparse
argp = argparse.ArgumentParser()
argp.add_argument("-s", "--sim", type=str, required=True, choices=os.listdir('/home/tnguser/sims.TNG'), 
                  help="Name of simulation as given in the path, e.g. L205n1250TNG")
argp.add_argument("-v", "--verbose", type=bool, default=False, help="Whether or not to print out detailed progress and timestamps. Default=False")
argp.add_argument("--coarse-bin-size", type=int, default=1000, help="The size of a coarse bin in ckpc/h. Default=1000")
argp.add_argument("--fine-bin-size", type=int, default=10, help="The size of the histogram bin in ckpc/h. Default=10")
argp.add_argument("--start-snap", type=int, default=99, help="Which snapshot to start on. Note that it starts from 99 and descends. Default=99")
argp.add_argument("--end-snap", type=int, default=33, help="Which snapshot to end on, inclusive. Default=33.")
argp.add_argument("--start-chunk", type=int, default=0, help="Which chunk to start retrieving data from. Default=0")
argp.add_argument("--start-bin", type=int, default=0, help="Which coarse bin in start-chunk to start retrieving data from (should be last bin done + 1). Default=0")
argp.add_argument("--start-hist-bin", type=int, default=0, help="Which coarse bin to start histogramming (should be last bin done + 1). If not 0, then assumes the data retrieval is done. Default=0")
argp.add_argument("--outpath", type=str, default=None, help="Path to where the output electron density map will go. If unspecified, will go to ./n_e_maps/{sim}")
args = argp.parse_args()

if args.outpath is None:
    outpath = f'n_e_maps/{args.sim}'
else:
    outpath = args.outpath

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

coarse_bin_size = args.coarse_bin_size
fine_bin_size = args.fine_bin_size
n_coarse_bins = int(BoxSize / coarse_bin_size)
n_fine_bins = int(coarse_bin_size / fine_bin_size)

print(f'{n_coarse_bins}^3 = {n_coarse_bins**3} coarse bins of size {coarse_bin_size}')
print(f'{n_fine_bins}^3 = {n_fine_bins**3} fine bins per coarse bin')

#run
for snap in range(args.start_snap, args.end_snap-1, -1):
    snap_outpath = os.path.join(outpath, str(snap)) #n_e_maps/{sim}/{snap}
    if not os.path.exists(snap_outpath):
        os.makedirs(snap_outpath)

    print(f'\nProcessing snapshot {snap}')
    start_time = time.time()
    
    if args.start_hist_bin <= 0:
        sort_chunks_to_coarse_bins(snap, args.verbose, args.start_chunk, args.start_bin)
        print(f'{time.time()-start_time:<6.2f}: Done sorting chunks to coarse bins')
    histogram_to_fine_bins(snap, args.verbose, args.start_hist_bin)
    print(f'{time.time()-start_time:<6.2f}: Done histogramming all coarse bins into fine bins')


# EXAMPLE USAGE
# python create_n_e_map.py -s L35n270TNG -v True --coarse-bin-size 1000 --fine-bin-size 10 --start-snap 99 --start-chunk 0 --start-bin 0
# nohup python create_n_e_map.py -s L35n270TNG  > process_L35n270TNG.log & 
# nohup python create_n_e_map.py -s L35n270TNG -v True --coarse-bin-size 1000 --fine-bin-size 10 --start-snap 99 --start-chunk 0 --start-bin 0 > process_L35n270TNG.log & 
# nohup python create_n_e_map.py -s L35n270TNG -v True --coarse-bin-size 5000 --fine-bin-size 10 --start-hist-bin 24 >> process_L35n270TNG.log & 
# nohup python create_n_e_map.py -s L35n270TNG --start-snap 97 --start-chunk 2 --start-bin 5702 > process_L35n270TNG.log & 
# nohup python create_n_e_map.py -s L35n270TNG --start-snap 91 --start-hist-bin 668 > process_L35n270TNG.log &
# nohup python create_n_e_map.py -s L35n270TNG --end-snap 98 > process_L35n270TNG.log &