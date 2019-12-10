#!/bin/bash

#SBATCH --partition=batch
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=8
#SBATCH --time=0:05:00

declare -i n=$1
declare -i d=$2
declare -i k=$3

srun ./test_synchronous $n $d $k
