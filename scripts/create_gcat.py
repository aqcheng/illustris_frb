import sys
sys.path.append('/home/submit/aqc/frb_project')

import numpy as np
import os
from illustris_frb import exp_simulation
from illustris_frb.regions import regions

origin = 500 * np.array([50, 70, 23])
gcat_path = '/data/submit/submit-illustris/april/data/g_cats'

# get all regions!

for reg_name in regions.keys():
    if os.path.exists(os.path.join(gcat_path, reg_name, 'metadata.txt')):
        print(f'Skipping region {reg_name}')
        continue
    print(f'Getting galaxies from region {reg_name}')
    sim = exp_simulation(origin, reg_name=reg_name)
    sim.gcat_from_snaps()