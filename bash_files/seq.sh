#!/bin/bash

#SBATCH --partition=batch
#SBATCH --time=0:05:00

declare -i n=$1
declare -i d=$2
declare -i k=$3

srun ./test_sequential $n $d $k
