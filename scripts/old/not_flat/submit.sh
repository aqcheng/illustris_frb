#!/bin/bash
#
#
#SBATCH --job-name=skypatch2            
#SBATCH --output=logs/flat_skypatch2.out         
#SBATCH --export=ALL                
#SBATCH --mem=75G                     
#SBATCH --mail-type=BEGIN,END,FAIL   
#SBATCH --mail-user=aqc@mit.edu 

# data path: /data/submit/submit-illustris/april/data

# activate anaconda environment
module purge                          # Start with a clean environment
source /home/submit/aqc/.bashrc
conda activate

python get_flat_sky_patch_data.py     # Run the script using the loaded Python module
