#!/bin/bash

# Configuration for 1 node, 4 cores and 5 minutes of execution time
#SBATCH --job-name=ex3
#SBATCH -p std
#SBATCH --output=out_argmax_%j.out
#SBATCH --error=out_argmax_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

make >> make.out || exit 1      # Exit if make fails

./argmax 100000000
