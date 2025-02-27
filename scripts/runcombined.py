"""
This is the script for the run with all effects combined, all 48 regions, 5 trials per region. 
Total list of effects:
    - Test two FRB redshift ranges (a proper background slice, and the whole cone)
    - Putting FRBs in galaxies (weighted by SFR) with a host apparent magnitude cut
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
import astropy.units as u
try:
    import _pickle as pickle
except:
    import pickle

outpath = '/ceph/submit/data/group/submit-illustris/april/data/C_ells/combined_experiment.pkl'
if os.path.exists(outpath):
    # res = pickle.load(open(outpath, 'rb'))
    os.remove(outpath)  
res = {'regions': [], 'slice_exp': {}, 'full_exp': {}}

origin = 500 * np.array([50, 70, 23])
sim = exp_simulation(origin, 'A1')

frb_zrange = (0.3, 0.4)
frb_xrange = sim.comoving_distance(frb_zrange)
frb_mean_x = np.mean(sim.comoving_distance(frb_zrange))

g_zrange = (0.2, 0.3)

ntrials = 5 #per region
nbins = 10

n_frbs = 2000

def get_sfrweight(df):
    return df['SFR'] / (1 + sim.z_from_dist(df['x']))
def get_optical_loc_weight(df, cutoff=20.7):
    ### for an abrupt magnitude cut
    return get_sfrweight(df) * (df['m_g'] < cutoff).astype(int)

# host DM log normal, from https://arxiv.org/pdf/2207.14316
mu, sigma = 1.93 / np.log10(np.e), 0.41 / np.log10(np.e)

# DM-dependent selection effects
def DM_sfunc(DMs, a=1): # fiducial selection function
    # a is how many factors to squish the selection function
    return np.exp( -(2/3)*(np.log10(DMs*a)-3)**2 )

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
    full_host_df['weights'] = get_optical_loc_weight(full_host_df)

    ## for impact parameter selection effects
    fg_g_df_groups = full_host_df.groupby('ipix', sort=False)[['theta_', 'phi_', 'x']]
    scatterPs = scattering_sfunc(full_host_df, fg_g_df_groups)
    savedata('scatterPs_full', np.array(scatterPs))

    slice_host_df = pd.DataFrame(full_host_df.loc[ (full_host_df['x'] > frb_xrange[0]) & (full_host_df['x'] <= frb_xrange[1]) ])
    savedata('scatterPs_slice', np.array(slice_host_df['scatter P']))

    midslice_DM = sim.DM_grid(x_max=frb_mean_x)
    full_DM = sim.DM_grid()
    savetosubdict('DMs', midslice_DM, 'midslice')
    savetosubdict('DMs', full_DM, 'full')

    ells, midslice_ClDg = cross_power_estimator(midslice_DM, delta_g_nocutoff, nbins=nbins)
    midslice_DeltaC = get_Clerr(midslice_DM, N_g, nbins=nbins)
    
    ells, full_ClDg = cross_power_estimator(full_DM, delta_g_nocutoff, nbins=nbins)
    full_DeltaC = get_Clerr(full_DM, N_g, nbins=nbins)
    
    savetosubdict('ClDgs', midslice_ClDg, 'midslice')
    savetosubdict('DeltaCs', midslice_DeltaC, 'midslice')
    savetosubdict('ClDgs', full_ClDg, 'full')
    savetosubdict('DeltaCs', full_DeltaC, 'full')
    res['l'] = ells

    # experiments - combined effects

    for host_df, res_key in zip([slice_host_df, full_host_df], ['slice_exp', 'full_exp']):

        for _ in range(ntrials):

            (DM_exp, mult_exp), (DM, mult) = sim.sim_DM_grid(
                N=n_frbs, host_df=host_df, weights='weights',
                DM_host_func=lambda x: np.random.lognormal(mu, sigma, x),
                DM_sfunc=lambda x: DM_sfunc(x, a=2),
                g_sfunc=lambda x: np.array(x['scatter P'])
            )

            savetosubdict('DM_exp', DM_exp, res_key)
            savetosubdict('mult_exp', mult_exp, res_key)
            savetosubdict('DM', DM, res_key)
            savetosubdict('mult', mult, res_key)
            
            ells, ClDg = cross_oqe(DM_exp, delta_g, mult_exp, nbins=nbins)
            savetosubdict('ClDgs_exp', ClDg, res_key)
            ells, ClDg = cross_oqe(DM, delta_g_nocutoff, mult, nbins=nbins)
            savetosubdict('ClDgs', ClDg, res_key)
                
    res['regions'].append(reg_name)

with open(outpath, 'wb') as f:
    pickle.dump(res, f)