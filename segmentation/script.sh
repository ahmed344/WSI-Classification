#!/bin/bash
#SBATCH --job-name=crop_tissues_multislides
#SBATCH --output=logs/crop_tissues_multislides.out
#SBATCH --error=logs/crop_tissues_multislides.err
#SBATCH --time=24:00:00

echo "Cropping tissues from multislides"

# Initialize conda by sourcing the script directly
# This makes 'conda' commands available in this shell session
source /work/aabdelrahman/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate wsi-env

# Run the script
python segmentation/crop_tissues_multislides.py

# Deactivate the conda environment
conda deactivate

echo "Done"