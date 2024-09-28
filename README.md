# illustris_frb: FRB cross-correlations in IllustrisTNG

This repository contains code for computing the dispersion measure (DM) of Fast Radio Bursts (FRBs) by ray tracing through electrons of the IllustrisTNG cosmological simulation. Tools include:
- Illustris-TNG data processing, including electron map and galaxy catalog creation
- Ray tracing through snapshots of Illustris-TNG
- A notebook `get_good_pixels.ipynb` for selecting sky areas without geometrical artifacts from periodic boundary conditions
- Scripts to generate a mock galaxy catalog from the light cone of a sky region
- A 2D power spectrum estimator for a field with a window function using the optimal quadratic estimator
- scripts for running all experiments in [ paper in prep ]

The data for this project can be found on [Zenodo](doi.org/10.5281/zenodo.13854755).

## Project Structure

- `illustris_frb/`: Main package for interfacing with Illustris-TNG, ray tracing, cross-correlations, and simulating selection effects

- `notebooks/`:
  - `DM_redshift.ipynb`: Sanity check for DM calculations
  - `get_good_pixels.ipynb`: Computes and plots allowed sky regions
  - `experiments.ipynb`: Produces all plots

- `scripts/`:
  - `create_g_map.py`: Creates a map of all galaxies within the simulation box. To be run on the IllustrisTNG JupyterHub
  - `create_n_e_map.py`: Creates a binned electron density map of the simulation box. To be run on the IllustrisTNG JupyterHub
  - `create_gcat.py`: Generates the galaxy catalogs for the given sky regions
  - `compute_DM_partition.py`: Computes FRB DMs with parallelization, partitioned in to subtasks
  - `getpickles.py`: Pre-interpolates all DM grids to make running experiments less expensive