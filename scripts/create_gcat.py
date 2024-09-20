import sys
sys.path.append('/home/submit/aqc/frb_project')

import numpy as np
import os
from illustris_frb import exp_simulation
from illustris_frb.regions import regions

origin = 500 * np.array([50, 70, 23])

# get all regions!

for reg_name in regions.keys():
    sim = exp_simulation(origin, reg_name=reg_name)
    if os.path.exists(os.path.join(sim.gcat_path, 'metadata.txt')):
        print(f'Skipping region {reg_name}')
    else:
        print(f'Getting galaxies from region {reg_name}')
        sim.gcat_from_snaps()
    
    print(f'Processing magnitudes for region {reg_name}')
    sim.process_magnitudes()