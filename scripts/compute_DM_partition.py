import sys
sys.path.append('/home/submit/aqc/frb_project')
import os
import numpy as np
from illustris_frb import exp_simulation

## --- INPUTS ---

origin = 500 * np.array([50, 70, 23])
nproc = 32

## --- COMMAND LINE INPUTS ##

import argparse
argp = argparse.ArgumentParser()
argp.add_argument("-N", type=int, required=True, help="Total # of partitions. Should divide the # of rows on a side.")
argp.add_argument("-n", type=int, required=True, help="Partition number; must be between 0 and N-1.")
argp.add_argument("--reg", type=str, required=True, help="The name of the region")
args = argp.parse_args()

## --- END OF INPUTS ---

sim = exp_simulation(origin, reg_name=args.reg)
if not os.path.isfile(os.path.join(sim.scratch_path, f'cumulative_DM_{args.n:03}.npy')):
    sim.compute_DM_grid_partition(N=args.N, n=args.n, nproc=nproc)

if len(os.listdir(sim.scratch_path)) > args.N:
    if os.path.exists(sim.results_path):
        os.remove(sim.results_path)
    sim.DM_grid()
    from shutil import rmtree
    rmtree(sim.scratch_path)
