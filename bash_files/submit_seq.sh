#!/bin/bash

module load gcc netlib-lapack

make test_sequential

declare -i min_n=4000
declare -i max_n=20000
declare -i inc_n=$min_n

declare -i min_d=20
declare -i max_d=100
declare -i inc_d=$min_d

declare -i min_k=10
declare -i max_k=40
declare -i inc_k=$min_k

declare -i n
declare -i d
declare -i k

for ((nc=$min_n; nc<=$max_n; nc += $inc_n))
do
    n=nc
    for ((dc=$min_d; dc<=$max_d; dc += $inc_d))
    do
        d=dc
        for ((kc=$min_k; kc<=$max_k; kc += $inc_k))
        do
            k=kc
            sbatch seq.sh $n $d $k
        done
    done
done
