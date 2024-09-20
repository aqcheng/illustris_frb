"""
This is the script for the run with all effects combined, all 48 regions, 5 trials per region. 
Total list of effects:
    - Vary total number of FRBs
    - Test two FRB redshift ranges (a proper background slice, and the whole cone)
    - Putting FRBs in galaxies (weighted by SFR)
    - Injecting a host DM
    - DM-dependent selection effects
    - Scattering selection effects
    - Using an apparent magnitude cutoff in the galaxy catalog
"""

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

outpath = '/ceph/submit/data/group/submit-illustris/april/data/C_ells/combined_experiment.pkl'
if os.path.exists(outpath):
    res = pickle.load(open(outpath, 'rb'))
else:
    res = {'regions': [], 'slice_exp': {}, 'full_exp': {}}

origin = 500 * np.array([50, 70, 23])
sim = exp_simulation(origin, 'A1')

frb_zrange = (0.3, 0.4)
frb_xrange = sim.comoving_distance(frb_zrange)
frb_mean_x = np.mean(sim.comoving_distance(frb_zrange))

g_zrange = (0.2, 0.3)

ntrials = 5 #per region
nbins = 10

def get_sfrweight(df):
    return df['SFR'] / (1 + sim.z_from_dist(df['x']))

# host DM log normal, from https://arxiv.org/pdf/2207.14316
mu, sigma = 1.93 / np.log10(np.e), 0.41 / np.log10(np.e)

# DM-dependent selection effects
def DM_sfunc(DMs, a=1): # fiducial selection function
    # a is how many factors to squish the selection function
    return np.exp( -(2/3)*(np.log10(DMs*a)-3)**2 )

# galaxy selection effects
def P_scattering(fg_galaxy_bs, r50 = 8*sim.h): #50% probability at 8 kpc, 1 - 2**((-r/r50)**2)
    Ps = 1 - np.power(2, -(fg_galaxy_bs/r50)**2)
    return Ps
def scattering_sfunc(host_g_df, fg_g_df_groups, P_scatter_func=P_scattering):
    if 'scatter P' in host_g_df.columns:
        return np.array(host_g_df['scatter P'])
    host_g_df['scatter P'] = 1.
    host_g_df_groups = host_g_df.groupby('ipix', sort=False)[['theta_', 'phi_']]
    for host_ipix in host_g_df_groups.groups.keys():
        if host_ipix in fg_g_df_groups.groups.keys():
            host = host_g_df_groups.get_group(host_ipix)
            fg = fg_g_df_groups.get_group(host_ipix)
            cdists = np.array(fg['x']) * np.sin(np.array(fg['theta_'])) * cdist(np.array(host), np.array(fg[['theta_', 'phi_']]))
            Ps = P_scatter_func(cdists)
            host_g_df.loc[host.index, 'scatter P'] = np.prod(Ps, axis=1)
    return host_g_df['scatter P']

# apparent magnitude cuts
def M_to_m(M, x):
    x_pc = x * 1000 / sim.h
    return 5*np.log10(x_pc) - 5 + M
m_g_cutoff = 25.4

def savedata(key, data, res=res):
    if key not in res.keys():
        res[key] = []
    res[key].append(np.array(data))

def savetosubdict(key, data, dictkey, res=res):
    if dictkey not in res.keys():
        res[dictkey] = {}
    savedata(key, data, res[dictkey])

for reg_name in sorted(regions.keys()):
    if reg_name in res['regions']:
        continue

    print(reg_name)
    sim = exp_simulation(origin, reg_name)

    N_g = sim.Ngal_grid(zrange=g_zrange, m_g_cutoff=m_g_cutoff)
    delta_g = (N_g - np.mean(N_g)) / np.mean(N_g)
    
    N_g_nocutoff = sim.Ngal_grid(zrange=g_zrange)
    delta_g_nocutoff = (N_g_nocutoff - np.mean(N_g_nocutoff)) / np.mean(N_g_nocutoff)
    
    full_host_df = sim.read_shell_galaxies()
    full_host_df['sfr_weight'] = get_sfrweight(full_host_df)
    slice_host_df = pd.DataFrame(full_host_df.loc[ (full_host_df['x'] > frb_xrange[0]) & (full_host_df['x'] <= frb_xrange[1]) ])

    midslice_DM = sim.DM_grid(x_max=frb_mean_x)
    full_DM = sim.DM_grid()
    savetosubdict('DMs', midslice_DM, 'midslice')
    savetosubdict('DMs', full_DM, 'full')

    ells, midslice_ClDg = cross_power_estimator(midslice_DM, delta_g, nbins=nbins)
    midslice_DeltaC = get_Clerr(midslice_DM, N_g, nbins=nbins)
    
    ells, full_ClDg = cross_power_estimator(full_DM, delta_g, nbins=nbins)
    full_DeltaC = get_Clerr(full_DM, N_g, nbins=nbins)
    
    savetosubdict('ClDgs', midslice_ClDg, 'midslice')
    savetosubdict('DeltaCs', midslice_DeltaC, 'midslice')
    savetosubdict('ClDgs', full_ClDg, 'full')
    savetosubdict('DeltaCs', full_DeltaC, 'full')
    res['l'] = ells

    ## for impact parameter selection effects
    fg_g_df = sim.read_shell_galaxies((sim.z_from_dist(0.25*frb_mean_x), sim.z_from_dist(0.75*frb_mean_x)))
    fg_g_df_groups = fg_g_df.groupby('ipix', sort=False)[['theta_', 'phi_', 'x']]

    # experiments - combined effects

    for host_df, res_subdict in zip([slice_host_df, full_host_df], [res['slice_exp'], res['full_exp']]):

        for _ in range(ntrials):

            for n_frbs_ in (50, 500, 1000, 2000, 3000):

                (DM_exp, mult_exp), (DM, mult) = sim.sim_DM_grid(
                    N=n_frbs_, host_df=host_df, weights='sfr_weight',
                    DM_host_func=lambda x: np.random.lognormal(mu, sigma, x),
                    DM_sfunc=lambda x: DM_sfunc(x, a=2),
                    g_sfunc=lambda x: scattering_sfunc(x, fg_g_df_groups)
                )

                savetosubdict(n_frbs_, DM_exp, 'DM_exp', res=res_subdict)
                savetosubdict(n_frbs_, mult_exp, 'mult_exp', res=res_subdict)
                savetosubdict(n_frbs_, DM, 'DM', res=res_subdict)
                savetosubdict(n_frbs_, mult, 'mult', res=res_subdict)
                
                ells, ClDg = cross_oqe(DM_exp, delta_g, mult_exp, nbins=nbins)
                savetosubdict(n_frbs_, ClDg, 'ClDgs_exp', res=res_subdict)
                ells, ClDg = cross_oqe(DM, delta_g_nocutoff, mult, nbins=nbins)
                savetosubdict(n_frbs_, ClDg, 'ClDgs', res=res_subdict)
                
    res['regions'].append(reg_name)
    with open(outpath, 'wb') as f:
        pickle.dump(res, f)