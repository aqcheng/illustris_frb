"""
This is the script to run all of the experiments, all 48 regions, 5 trials per region. For more
details on what the experiments are (as well as the results), see `notebooks/experiments.ipynb`.
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

outpath = '/data/submit/submit-illustris/april/data/C_ells/trial_experiments.pkl'
if os.path.exists(outpath):
    res = pickle.load(open(outpath, 'rb'))
else:
    res = {'regions': []}

origin = 500 * np.array([50, 70, 23])
sim = exp_simulation(origin, 'A1')

frb_zrange = (0.3, 0.4)
frb_mean_x = np.mean(sim.comoving_distance(frb_zrange))

g_zrange = (0, 0.2)

ntrials = 5 #per region
n_frbs = 3000 #FRBs
nbins = 10

def get_sfrweight(df):
    return df['SFR'] / (1 + sim.z_from_dist(df['x']))
def get_massweight(df):
    return df['Mass'] / (1 + sim.z_from_dist(df['x']))

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
m_g_tightcutoff = 20.7

# overlap experiment
full_zrange = (0,0.4)
full_meanx = np.mean(sim.comoving_distance(full_zrange))
g_zranges = ((0, 0.1), (0, 0.2), (0, 0.3), (0, 0.4))
frb_zranges = ((0.3, 0.4), (0.2, 0.4), (0.1, 0.4))

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

    delta_gs, N_gs = {}, {}
    for m_g_cutoff_ in (m_g_cutoff, m_g_tightcutoff, None):
        N_g = sim.Ngal_grid(zrange=g_zrange, m_g_cutoff=m_g_cutoff_)
        N_gs[m_g_cutoff_] = np.array(N_g)
        delta_gs[m_g_cutoff_] = (N_g - np.mean(N_g)) / np.mean(N_g)
    
    ## for overlap experiments
    for g_zrange_ in g_zranges:
        N_g_ = sim.Ngal_grid(zrange=g_zrange_)
        N_gs[g_zrange_] = np.array(N_g_)
        delta_gs[g_zrange_] = (N_g_ - np.mean(N_g_)) / np.mean(N_g_)
    full_host_df = sim.read_shell_galaxies()
    full_host_df['sfr_weight'] = get_sfrweight(full_host_df)
    full_host_df['mass_weight'] = get_massweight(full_host_df)
    host_dfs = {}
    for frb_zrange_ in frb_zranges:
        xmin, xmax = sim.comoving_distance(frb_zrange_)
        host_dfs[frb_zrange_] = pd.DataFrame(full_host_df.loc[ (full_host_df['x'] > xmin) & (full_host_df['x'] <= xmax) ])
    midslice_DM_overlap = sim.DM_grid(x_max=full_meanx)
    full_DM = sim.DM_grid()
    
    if frb_zrange in frb_zranges:
        host_df = host_dfs[frb_zrange]
    else:
        host_df = sim.read_shell_galaxies(frb_zrange)
        host_df['sfr_weight'] = get_sfrweight(host_df)
        host_df['mass_weight'] = get_massweight(host_df)

    ## midslice DM
    delta_g = delta_gs[None] #fiducial delta_g
    midslice_DM = sim.DM_grid(x_max=frb_mean_x)
    savetosubdict('DMs', midslice_DM, 'midslice')
    ells, midslice_ClDg = cross_power_estimator(midslice_DM, delta_g, nbins=nbins)
    midslice_DeltaCs = get_Clerr(midslice_DM, N_gs[None], nbins=nbins)
    res['l'] = ells
    savetosubdict('ClDg', midslice_ClDg, 'midslice')
    savetosubdict('DeltaCs', midslice_DeltaCs, 'midslice')

    ## for impact parameter selection effects
    fg_g_df = sim.read_shell_galaxies((sim.z_from_dist(0.25*frb_mean_x), sim.z_from_dist(0.75*frb_mean_x)))
    fg_g_df_groups = fg_g_df.groupby('ipix', sort=False)[['theta_', 'phi_', 'x']]
    scatterPs = scattering_sfunc(host_df, fg_g_df_groups)
    savedata('scatterPs', scatterPs)

    for key in N_gs.keys():
        savetosubdict(key, N_gs[key], 'N_gs')

    for i, g_zrange_ in enumerate(g_zranges):
        ells, ClDg = cross_power_estimator(midslice_DM_overlap, delta_gs[g_zrange_], nbins=nbins)
        DeltaCs = get_Clerr(midslice_DM_overlap, N_gs[g_zrange_], nbins=nbins)
        savetosubdict(g_zrange_, ClDg, 'overlap_midslice')
        savetosubdict(g_zrange_, DeltaCs, 'overlap_midslice_DeltaCs')

        ells, ClDg = cross_power_estimator(full_DM, delta_gs[g_zrange_], nbins=nbins)
        DeltaCs = get_Clerr(midslice_DM_overlap, N_gs[g_zrange_], nbins=nbins)
        savetosubdict(g_zrange_, ClDg, 'overlap_fullz')
        savetosubdict(g_zrange_, DeltaCs, 'overlap_fullz_DeltaCs')

    # EXPERIMENTS
    for _ in range(ntrials):

        sfr_sample_df = host_df.sample(n_frbs, replace=True, ignore_index=True, weights='sfr_weight')
        full_sfr_sample_df = full_host_df.sample(n_frbs, replace=True, ignore_index=True, weights='sfr_weight')
        mass_sample_df = host_df.sample(n_frbs, replace=True, ignore_index=True, weights='mass_weight')

        DM_fid, mult_fid = sim.sim_DM_grid(sfr_sample_df, N=n_frbs)
        DM_mass, mult_mass = sim.sim_DM_grid(mass_sample_df, N=n_frbs)
        savetosubdict('DM_fid', DM_fid, 'trial_DM_fields')
        savetosubdict('mult_fid', mult_fid, 'trial_DM_fields')
        savetosubdict('DM_mass', DM_mass, 'trial_DM_fields')
        savetosubdict('mult_mass', mult_mass, 'trial_DM_fields')

        # save DM vs x
        DM_arr = sim.get_DMs(full_sfr_sample_df['ipix'], full_sfr_sample_df['x'])
        savetosubdict('DM', DM_arr, 'fullz_FRB_properties')
        savetosubdict('ipix', full_sfr_sample_df['ipix'], 'fullz_FRB_properties')
        savetosubdict('x', full_sfr_sample_df['x'], 'fullz_FRB_properties')

        ells, ClDg_fid = cross_oqe(DM_fid, delta_g, mult_fid, nbins=nbins)
        ells, ClDg_mass = cross_oqe(DM_mass, delta_g, mult_mass, nbins=nbins)

        ## signal as a function of number of FRBs
        for n_frbs_ in (50, 100, 500, 1000, 2000, 3000):

            if n_frbs_ == n_frbs:
                savetosubdict(n_frbs_, ClDg_fid, 'n_frbs_sfrweight')
                savetosubdict(n_frbs_, ClDg_mass, 'n_frbs_massweight')
            else:
                DM_fid_, mult_fid_ = sim.sim_DM_grid(N=n_frbs_, host_df=host_df, weights='sfr_weight')
                DM_mass_, mult_mass_ = sim.sim_DM_grid(N=n_frbs_, host_df=host_df, weights='mass_weight')
                
                ells, ClDg = cross_oqe(DM_fid_, delta_g, mult_fid_, nbins=nbins)
                savetosubdict(n_frbs_, ClDg, 'n_frbs_sfrweight')
                ells, ClDg = cross_oqe(DM_mass_, delta_g, mult_mass_, nbins=nbins)
                savetosubdict(n_frbs_, ClDg, 'n_frbs_massweight')

            mult_inds = np.random.choice(sim.region.nside**2, n_frbs_)
            mult = np.bincount(mult_inds, minlength=sim.region.nside**2).reshape((sim.region.nside, sim.region.nside))
            DM = np.where(mult > 0, midslice_DM, 0)
            ells, ClDg = cross_oqe(DM, delta_g, mult, nbins=nbins)
            savetosubdict(n_frbs_, ClDg, 'n_frbs_randpixels')

        ## DM-dependent selection effects
        for a in (1, 2, 5):
            (DM, mult), _ = sim.sim_DM_grid(sampled_df=sfr_sample_df, DM_sfunc=lambda x: DM_sfunc(x, a=a))
            ells, ClDg = cross_oqe(DM, delta_g, mult, nbins=nbins)
            savetosubdict(a, ClDg, 'DM_sfunc')
        
        for DM_cutoff in (600, 800, 1000):
            (DM, mult), _ = sim.sim_DM_grid(sampled_df=sfr_sample_df, DM_sfunc=lambda x: x < DM_cutoff)
            ells, ClDg = cross_oqe(DM, delta_g, mult, nbins=nbins)
            savetosubdict(DM_cutoff, ClDg, 'DM_cutoffs')
        
        ## apparent magnitude cutoffs
        for cutoff in (m_g_cutoff, m_g_tightcutoff):
            ells, ClDg = cross_oqe(DM_fid, delta_gs[cutoff], mult_fid, nbins=nbins)
            savetosubdict(cutoff, ClDg, 'm_g_cutoffs')

        ## isolating host DM, isolating FRB placements
        mu, sigma = np.log(43), 1.26
        hostDM = np.random.lognormal(mu, sigma, DM_fid.shape)
        hostDM[mult_fid > 0] /= mult_fid[mult_fid > 0]
        hostDM[mult_fid == 0] = 0
        hostDM[mult_fid > 0] -= np.mean(hostDM[mult_fid > 0]) #normalize to 0 mean
        ells, ClDg = cross_power_estimator(midslice_DM+hostDM, delta_g, nbins=nbins)
        savedata('midslice_inj_hostDM_ClDgs', ClDg)
        ells, ClDg = cross_oqe(DM_fid + (hostDM*(mult_fid > 0)), delta_g, mult_fid, nbins=nbins)
        savedata('fid_inj_hostDM_ClDgs', ClDg)

        midslice_DM_proj = midslice_DM * (mult_fid > 0)
        ells, ClDg = cross_oqe(midslice_DM_proj, delta_g, mult_fid, nbins=nbins)
        savedata('proj_sfrweight_ClDgs', ClDg)

        midslice_DM_proj_mass = midslice_DM * (mult_mass > 0)
        ells, ClDg = cross_oqe(midslice_DM_proj_mass, delta_g, mult_fid, nbins=nbins)
        savedata('proj_massweight_ClDgs', ClDg)

        ## scattering 
        (DM_s, mult_s), _ = sim.sim_DM_grid(sampled_df=sfr_sample_df, 
                                            g_sfunc=lambda x: scattering_sfunc(x, fg_g_df_groups))
        ells, ClDg = cross_oqe(DM_s, delta_g, mult_s, nbins=nbins)
        savedata('scatter_ClDgs', ClDg)
        
        ## overlap experiments
        ### different foreground galaxy catalog ranges with 0 < z_frb < 0.4
        DM_full_sfrweight, mult_full_sfrweight = sim.sim_DM_grid(full_sfr_sample_df)
        for g_zrange_ in g_zranges:
            ells, ClDg = cross_oqe(DM_full_sfrweight, delta_gs[g_zrange_], mult_full_sfrweight, nbins=nbins)
            savetosubdict(g_zrange_, ClDg, 'overlap_g_zranges')
        ### 0 < z_g < 0.2 with different FRB ranges
        for frb_zrange_ in frb_zranges:
            if frb_zrange_ == frb_zrange:
                savetosubdict(frb_zrange_, ClDg_fid, 'overlap_frb_zranges')
                continue
            DM, mult = sim.sim_DM_grid(N=n_frbs_, host_df=host_dfs[frb_zrange_], weights='sfr_weight')
            ells, ClDg = cross_oqe(DM, delta_g, mult, nbins=nbins)
            savetosubdict(frb_zrange_, ClDg, 'overlap_frb_zranges')
        
    res['regions'].append(reg_name)
    with open(outpath, 'wb') as f:
        pickle.dump(res, f)