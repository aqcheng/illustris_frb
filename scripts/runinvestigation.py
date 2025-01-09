# rerunning experiments

import sys
sys.path.append('/home/submit/aqc/frb_project')
from illustris_frb import exp_simulation
from illustris_frb.regions import regions

import numpy as np
import pandas as pd
import os
try:
    import _pickle as pickle
except:
    import pickle

outpath = '/ceph/submit/data/group/submit-illustris/april/data/C_ells/trial_investigation_DMs.pkl'

g_zrange = (0.2, 0.3)
g_zranges = ((0.05, 0.4), (0.05, 0.2), (0.05, 0.1), g_zrange)

frb_zranges = ((0.3, 0.4), (0.2, 0.4), (0.1, 0.4), (0.05, 0.4), g_zrange)

if os.path.exists(outpath):
    res = pickle.load(open(outpath, 'rb'))
else:
    res = {
        'frb_zranges': {zrange: {'DM_midslice': [], 'DM_trial': [], 
                                'x1z_trial': [], 'mult_trial': []} for zrange in frb_zranges}, 
        'g_zranges': {zrange: [] for zrange in g_zranges}
    }

origin = 500 * np.array([50, 70, 23])
ntrials = 5 #per region
n_frbs = 2000 #FRBs
nbins = 10

def get_sfrweight(df):
    return df['SFR'] / (1 + sim.z_from_dist(df['x']))

for reg_name in sorted(regions.keys()):

    print(reg_name)
    sim = exp_simulation(origin, reg_name)

    # save galaxy field
    delta_gs = {}
    for g_zrange_ in g_zranges:
        N_g = sim.Ngal_grid(zrange=g_zrange_)
        delta_g = (N_g - np.mean(N_g)) / np.mean(N_g)
        res['g_zranges'][g_zrange_].append(delta_g)
    
    # save DM grid and distances to FRBs for each region, ntrials=5 per region
    full_host_df = sim.read_shell_galaxies()
    full_host_df.loc[:, 'sfr_weight'] = get_sfrweight(full_host_df)
    host_dfs = {}
    for frb_zrange_ in frb_zranges:
        xmin, xmax = sim.comoving_distance(frb_zrange_)
        host_dfs[frb_zrange_] = pd.DataFrame(full_host_df.loc[ (full_host_df['x'] > xmin) & (full_host_df['x'] <= xmax) ])

    for frb_zrange_ in frb_zranges:
        subdict = res['frb_zranges'][frb_zrange_]
        frb_mean_x = np.mean(sim.comoving_distance(frb_zrange_))
        midslice_DM = sim.DM_grid(x_max=frb_mean_x)
        subdict['DM_midslice'].append(midslice_DM)
        for _ in range(ntrials):
            sample_df = host_dfs[frb_zrange_].sample(n_frbs, replace=True, ignore_index=True, weights='sfr_weight')
            grids = sim.sim_DM_grid(sample_df, return_x1z=True)
            for key, grid in zip(('DM_trial', 'x1z_trial', 'mult_trial'), grids):
                subdict[key].append(grid)

if os.path.exists(outpath):
    os.remove(outpath)
with open(outpath, 'wb') as f:
    pickle.dump(res, f)