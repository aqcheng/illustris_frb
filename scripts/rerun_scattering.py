# rerunning scattering, trying optical follow-up selection effect

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
import astropy.units as u

outpath = '/ceph/submit/data/group/submit-illustris/april/data/C_ells/trial_experiments.pkl'
if os.path.exists(outpath):
    res = pickle.load(open(outpath, 'rb'))
else:
    res = {'regions': []}

# remove previous runs
keys = ['scatter_ClDgs', 'n_frbs_optweight', 'host_m_g_cutoffs'] # keys relevant to experiment to rerun
for key in keys:
    if key in res.keys():
        _ = res.pop(key, None)
subkeys = {'trial_DM_fields': ('DM_fid', 'mult_fid', 'DM_scatter', 'mult_scatter', 
                               'P_scatter', 'DM_opt', 'mult_opt'),
           'n_frbs_sfrweight': (2000,)}
for key in subkeys.keys():
    for subkey in subkeys[key]:
        if subkey in res[key].keys():
            _ = res[key].pop(subkey, None)

# settings
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
def get_optical_loc_weight(df, cutoff=20.7):
    ### for an abrupt magnitude cut
    return get_sfrweight(df) * (df['m_g'] < cutoff).astype(int)
host_mcutoffs = (25.4, 20.7, 18)

# NEW GALAXY SELECTION EFFECTS
def scattering_timescale(bs, fg_zs, frb_zs, f=600, r50 = 4, L=1*u.kpc): 
    # observing frequency of 600 MHz
    # characteristic impact parameter for scattering 4 kpc
    # thickness of scattering medium 1 kpc
    d_go = sim.cosmo.lookback_distance(fg_zs) # galaxy observer distance
    d_fo = np.atleast_2d(sim.cosmo.lookback_distance(frb_zs)).T # frb observer distance
    d_fg = d_fo - d_go #nhost, nfg
    G = 2 * d_fg * d_go / (d_fo * L)
    taus = 0.3 * G * np.power(2, -(bs/r50)**2) / \
           ((1 + fg_zs**3) * ((f/1000)**4)) # in ms
    taus = np.where(taus < 0, 0, taus) # no scattering for galaxies in background
    # impact parameters are given pairwise in (nhost, nfg)
    return np.sum(taus, axis=1) # total scattering over all intervening galaxies for each host galaxy

def scattering_sfunc(host_g_df, fg_g_df_groups):
    host_g_df['scatter P'] = 1.
    host_g_df_groups = host_g_df.groupby('ipix', sort=False)[['theta_', 'phi_', 'x']]
    for host_ipix in host_g_df_groups.groups.keys(): # only look at foreground galaxies with same pixels as FRB
        if host_ipix in fg_g_df_groups.groups.keys():
            host = host_g_df_groups.get_group(host_ipix)
            fg = fg_g_df_groups.get_group(host_ipix)
            fg = fg[ fg['x'] < host['x'].max() ]
            #pairwise impact parameters, (nhost, nfg) for this pixel
            cdists = np.sin(np.array(fg['theta_'])) * \
                     cdist(np.array(host[['theta_', 'phi_']]), np.array(fg[['theta_', 'phi_']])) / \
                     (sim.cosmo.arcsec_per_kpc_proper(sim.z_from_dist(np.array(fg['x']))).to(u.rad/u.kpc).value) #arcsec to radians
            taus = scattering_timescale(cdists, sim.z_from_dist(fg['x']), sim.z_from_dist(host['x']))
            Ps = np.power(2, -taus**2)
            host_g_df.loc[host.index, 'scatter P'] = Ps
    return host_g_df['scatter P']

def savedata(key, data, res=res):
    if key not in res.keys():
        res[key] = []
    res[key].append(np.array(data))

def savetosubdict(key, data, dictkey, res=res):
    if dictkey not in res.keys():
        res[dictkey] = {}
    savedata(key, data, res[dictkey])

scatterPs_list = []
for reg_name in sorted(regions.keys()):

    print(reg_name)
    sim = exp_simulation(origin, reg_name)

    N_g = sim.Ngal_grid(zrange=g_zrange)
    delta_g = (N_g - np.mean(N_g)) / np.mean(N_g)

    host_df = sim.read_shell_galaxies(frb_zrange)
    host_df.loc[:, 'sfr_weight'] = get_sfrweight(host_df)

    ## for impact parameter selection effects
    fg_g_df_groups = host_df.groupby('ipix', sort=False)[['theta_', 'phi_', 'x']]
    scatterPs_list.append(scattering_sfunc(host_df, fg_g_df_groups)) 

    # EXPERIMENTS
    for _ in range(ntrials):

        sfr_sample_df = host_df.sample(n_frbs, replace=True, ignore_index=True, weights='sfr_weight')

        DM_fid, mult_fid = sim.sim_DM_grid(sfr_sample_df, N=n_frbs)
        savetosubdict('DM_fid', DM_fid, 'trial_DM_fields')
        savetosubdict('mult_fid', mult_fid, 'trial_DM_fields')

        ells, ClDg = cross_oqe(DM_fid, delta_g, mult_fid, nbins=nbins)
        savetosubdict(n_frbs, ClDg, 'n_frbs_sfrweight')

        ## scattering 
        (DM_s, mult_s), _ = sim.sim_DM_grid(sampled_df=sfr_sample_df, 
                                            g_sfunc=lambda x: np.array(x['scatter P']))
        savetosubdict('DM_scatter', DM_s, 'trial_DM_fields')
        savetosubdict('mult_scatter', mult_s, 'trial_DM_fields')
        savetosubdict('P_scatter', np.array(sfr_sample_df['scatter P']), 'trial_DM_fields')
        ells, ClDg = cross_oqe(DM_s, delta_g, mult_s, nbins=nbins)
        savedata('scatter_ClDgs', ClDg)

        ## optical follow-up selection effect

        for cutoff in host_mcutoffs:
            weights = get_optical_loc_weight(host_df, cutoff=cutoff)
            optical_loc_sample_df = host_df.sample(n_frbs, replace=True, ignore_index=True, weights=weights/np.sum(weights))
            DM_opt, mult_opt = sim.sim_DM_grid(optical_loc_sample_df, N=n_frbs)
            ells, ClDg = cross_oqe(DM_opt, delta_g, mult_opt, nbins=nbins)
            savetosubdict(cutoff, ClDg, 'host_m_g_cutoffs')
            
with open(outpath, 'wb') as f:
    pickle.dump(res, f)

# np.save(np.concatenate(scatterPs_list, axis=0), '/ceph/submit/data/group/submit-illustris/april/data/C_ells/scatter_Ps.npy')