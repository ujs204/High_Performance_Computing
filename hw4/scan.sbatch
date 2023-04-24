#!/bin/bash

#SBATCH --job-name=scan
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --mem=2GB 
#SBATCH --time=01:00:00 
#SBATCH --output=scan.txt
#SBATCH --verbose
#SBATCH --mail-type=END
#SBATCH --mail-user=ujs204@nyu.edu

module purge 
module load amber/openmpi/intel/20.06

mpiexec ./scan 10000000

