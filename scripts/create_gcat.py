import sys
sys.path.append('/home/submit/aqc/frb_project')

import numpy as np
from illustris_frb import exp_simulation, region

## ---- INPUTS ----

origin = 500 * np.array([50, 70, 23])
reg = region((1.18, -np.pi/2), 1.11, 1.88)
region_name = 'A1'
snaps = 'all'
# reg = region((0, 0), 0, 0.1, size=0.18, res=0.001)
# region_name = 'confirm_test'

## -- END OF INPUTS --

sim = exp_simulation(origin, reg, region_name=region_name)
sim.gcat_from_snaps(snaps)