import sys
sys.path.append('/home/submit/aqc/frb_project/')
from illustris_frb import exp_simulation

from illustris_frb.regions import regions
import numpy as np

origin = 500 * np.array([50, 70, 23])

for reg_name in regions.keys():
    print(reg_name)
    sim = exp_simulation(reg_name=reg_name, origin=origin)
    sim.interp_DM_grid