import sys
sys.path.append('/home/submit/aqc/frb_project')

import numpy as np
from illustris_frb import exp_simulation, region


## --- INPUTS ---

origin = 500 * np.array([50, 70, 23])
reg = region((1.18, -np.pi/2), 1.11, 1.88)
region_name = 'A1'
suffix = 'res0008'
nproc = 32

## --- COMMAND LINE INPUTS ##

import argparse
argp = argparse.ArgumentParser()
argp.add_argument("-N", type=int, required=True, help="Total # of partitions. Should divide the # of rows on a side.")
argp.add_argument("-n", type=int, required=True, help="Partition number; must be between 0 and N-1.")
args = argp.parse_args()

## --- END OF INPUTS ---

sim = exp_simulation(origin, reg, region_name=region_name, suffix=suffix)
sim.compute_DM_grid_partition(N=args.N, n=args.n, nproc=nproc)