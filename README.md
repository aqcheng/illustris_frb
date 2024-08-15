# DIRECTORY ORGANIZATION 

THIS IS CURRENTLY VERY OUTDATED - I WILL UPDATE THIS LATER!

List of data runs:
- `test_flat`: Target region is a 0.18 rad x 0.18 rad square equatorial region, $\theta \in [\pi/2 - 0.09, \pi/2 + 0.09]$, $\phi \in [0.01, 0.19]$. This region was found using the notebook `notebooks/get_good_pixels.ipynb`, along with other potential target regions. The origin is set at `binsize * np.array([50, 70, 23])`, which is a void region within the simulation.
    - `test_flat_001` put FRBs at $z=0.4$ in a grid with pixel size 0.01 rad, with $180^2$ FRBs total. However, it was getting all galaxies within the cone.
    - `test_flat_001_shell` has the same FRB DMs, but only counts foreground galaxies in the shell corresponding to snapshot 84.

## data
`data` is a symbolic link, and its true path is `/data/submit/submit-illustris/april/data/`.
- `g_cats` is where the "galaxy catalogs" are stored. These are shells of galaxies, organized by snapshot, for a specified origin and path of sky. 
    - `metadata.txt` has information on the redshift range, etc. of each shell
- `g_maps` is where the galaxy _maps_ are stored, i.e. a big table of all the galaxies within the simulation box for each snapshot. The stored properties in each `.hdf5` file is `['x', 'y', 'z', 'Mass', 'SFR', 'Radius']`. These were created with `scripts/create_g_map.py`.
- `header_info` copies over header attributes from the IllustrisTNG files for easier access of metadata.
- `n_e_maps` are the electron number count maps for each snapshot, which is a 3D grid of the simulation box. Each `.hdf5` file is a pandas DataFrame that stores the bin index $i$ and $N_e$, the number of electrons ine each bin. These were created from `scripts/create_n_e_map.py`.
- `results` are where the outputs of data runs are stored; these are tables with the foreground galaxy count and FRB DM for each pixel.
- `tmp`: i literally don't know lmao

## illustris_frb
Where all the ray tracing code and all code that interface with Illustris are. Eventually, I'll expand this package to include helper scripts in `scripts`.

## notebooks
- `debugging.ipynb`: Trying to figure out why I'm getting no cross-correlation, and why my DM and electron count maps seem to be offset
- `DM_redshift.ipynb`: Sanity check of my DM code, plotting DM vs redshift.
- `get_good_pixels.ipynb`: Computes and plots all bad sky regions. (I need to document the math here, because I don't remember why or how it works anymore.)
- `xcorr_flatsky.ipynb`: Where my flat sky cross-correlation estimator code is, and where I'm plotting it. Will move some of this code into the `illustris_frb` package eventually.

## scripts
- `create_g_map.py` and `create_n_e_map.py` creates the simulation box's galaxy and electron density maps, respectively. 
- `get_g_catalog_flatsky.py` creates the galaxy catalog. Given a flat patch of sky and a distance, it gets all galaxies within that cone by filtering for galaxies in consecutive shells and snapshots.
- `parallel_get_flat_sky_patch_data.py` computes the DMs of the FRBs in a flat patch of sky. Ray tracing is parallelized using the `multiprocess` module, but also because it takes as input which partition (rows of the grid) are being computed, such that different rows are computed in different jobs. 
    - The outputs are written to a scratch directory in `/work/submit/aqc/`. 
    - The job submission command is in `batchsubmit_command.txt` and uses the job submission script `batchsubmit.sh`.
- `aggregate_flat_sky_patch_data.py` aggregates the outputs of the different jobs in the scratch directory. It also retrieves the foreground galaxy count from the galaxy catalog in `g_cats`. The output are the DMs and foreground galaxy counts of each pixel, saved as a table (`.hdf5` pandas DataFrame).

Currently depreciated:
- `get_flat_sky_patch_data.py` does not implement partitioning into different jobs
- `not_flat`: `get_flat_sky_data.py` and `get_g_catalog.py` use an array of good healpix pixels to specify its sky region, as well as `checkpixels` to figure out periodic box stacking (I'm not sure if this works?). Probably won't be used since using spherical harmonics for the power spectrum seems like overkill for such a small region.
