# rerunning experiments

import sys
sys.path.append('/home/submit/aqc/frb_project')
from illustris_frb import exp_simulation
from illustris_frb.xcorr import cross_power_estimator, get_Clerr, cross_oqe
from illustris_frb.regions import regions
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
try:
    import _pickle as pickle
except:
    import pickle

outpath = '/ceph/submit/data/group/submit-illustris/april/data/C_ells/trial_experiments.pkl'
if os.path.exists(outpath):
    res = pickle.load(open(outpath, 'rb'))
else:
    res = {'regions': []}

keys = ['scatter_DM', 'scatter_ClDgs', 'DM_sfunc'] # keys relevant to experiment
for key in keys:
    _ = res.pop(key, None)

subkeys = {'trial_DM_fields': ('DM_fid', 'mult_fid', 'DM_scatter', 'mult_scatter'),
           'n_frbs_sfrweight': (2000,)}
for key in subkeys.keys():
    for subkey in subkeys[key]:
        _ = res[key].pop(subkey, None)

origin = 500 * np.array([50, 70, 23])
sim = exp_simulation(origin, 'A1')

frb_zrange = (0.3, 0.4)
frb_xrange = sim.comoving_distance(frb_zrange)
frb_mean_x = np.mean(frb_xrange)

g_zrange = (0.2, 0.3)

ntrials = 5 #per region
n_frbs = 2000 #FRBs
nbins = 10

def get_sfrweight(df):
    return df['SFR'] / (1 + sim.z_from_dist(df['x']))

# DM-dependent selection effects
def DM_sfunc(DMs, a=1): # fiducial selection function
    # a is how many factors to squish the selection function
    return np.exp( -(2/3)*(np.log10(DMs*a)-3)**2 )

# galaxy selection effects - NOW DEPRECATED
def P_scattering(fg_galaxy_bs, r50 = 15*sim.h): #50% probability at 15 kpc, 1 - 2**((-r/r50)**2)
    Ps = 1 - np.power(2, -(fg_galaxy_bs/r50)**2)
    return Ps
def scattering_sfunc(host_g_df, fg_g_df_groups, P_scatter_func=P_scattering, **kwargs):
    if 'scatter P' in host_g_df.columns:
        return np.array(host_g_df['scatter P'])
    host_g_df['scatter P'] = 1.
    host_g_df_groups = host_g_df.groupby('ipix', sort=False)[['theta_', 'phi_', 'x']]
    for host_ipix in host_g_df_groups.groups.keys():
        if host_ipix in fg_g_df_groups.groups.keys():
            host = host_g_df_groups.get_group(host_ipix)
            fg = fg_g_df_groups.get_group(host_ipix)
            fg = fg[ fg['x'] < host['x'].max() ]
            cdists = np.array(fg['x'] / (1 + sim.z_from_dist(fg['x']))) * np.sin(np.array(fg['theta_'])) * \
                     cdist(np.array(host[['theta_', 'phi_']]), np.array(fg[['theta_', 'phi_']]))
            nhost, nfg = len(host), len(fg)
            Ps = np.where(np.tile(np.array(host['x']), (nfg, 1)).T > np.tile(np.array(fg['x']), (nhost, 1)), 
                          P_scatter_func(cdists, **kwargs), 1)
            host_g_df.loc[host.index, 'scatter P'] = np.prod(Ps, axis=1)
    return host_g_df['scatter P']

def savedata(key, data, res=res):
    if key not in res.keys():
        res[key] = []
    res[key].append(np.array(data))

def savetosubdict(key, data, dictkey, res=res):
    if dictkey not in res.keys():
        res[dictkey] = {}
    savedata(key, data, res[dictkey])

for reg_name in sorted(regions.keys()):

    print(reg_name)
    sim = exp_simulation(origin, reg_name)

    N_g = sim.Ngal_grid(zrange=g_zrange)
    delta_g = (N_g - np.mean(N_g)) / np.mean(N_g)
    
    ## for overlap experiments
    full_host_df = sim.read_shell_galaxies()
    full_host_df.loc[:, 'sfr_weight'] = get_sfrweight(full_host_df)
    host_df = pd.DataFrame(full_host_df.loc[ (full_host_df['x'] > frb_xrange[0]) & (full_host_df['x'] <= frb_xrange[1]) ])

    ## for impact parameter selection effects
    fg_g_df_groups = full_host_df.groupby('ipix', sort=False)[['theta_', 'phi_', 'x']]
    scatterPs = scattering_sfunc(host_df, fg_g_df_groups)

    # EXPERIMENTS
    for _ in range(ntrials):

        sfr_sample_df = host_df.sample(n_frbs, replace=True, ignore_index=True, weights='sfr_weight')

        DM_fid, mult_fid = sim.sim_DM_grid(sfr_sample_df, N=n_frbs)
        savetosubdict('DM_fid', DM_fid, 'trial_DM_fields')
        savetosubdict('mult_fid', mult_fid, 'trial_DM_fields')

        ells, ClDg = cross_oqe(DM_fid, delta_g, mult_fid, nbins=nbins)
        savetosubdict(n_frbs, ClDg, 'n_frbs_sfrweight')

        ## for DM selection effects
        for a in (1, 2, 5):
            (DM, mult), _ = sim.sim_DM_grid(sampled_df=sfr_sample_df, DM_sfunc=lambda x: DM_sfunc(x, a=a))
            ells, ClDg = cross_oqe(DM, delta_g, mult, nbins=nbins)
            savetosubdict(a, ClDg, 'DM_sfunc')
            
            savetosubdict(f'DM_sfunc_{a}', DM, 'trial_DM_fields')
            savetosubdict(f'mult_sfunc_{a}', mult, 'trial_DM_fields')

        ## scattering 
        (DM_s, mult_s), _ = sim.sim_DM_grid(sampled_df=sfr_sample_df, 
                                            g_sfunc=lambda x: np.array(x['scatter P']))
        savetosubdict('DM_scatter', DM_s, 'trial_DM_fields')
        savetosubdict('mult_scatter', mult_s, 'trial_DM_fields')
        ells, ClDg = cross_oqe(DM_s, delta_g, mult_s, nbins=nbins)
        savedata('scatter_ClDgs', ClDg)
        
with open(outpath, 'wb') as f:
    pickle.dump(res, f)