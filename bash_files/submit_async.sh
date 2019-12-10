#!/bin/bash

module load gcc openmpi openblas netlib-lapack

make test_synchronous

declare -i p=32

declare -i min_n=8000
declare -i max_n=40000
declare -i inc_n=$min_n

declare -i min_d=50
declare -i max_d=100
declare -i inc_d=$min_d

declare -i min_k=20
declare -i max_k=20
declare -i inc_k=$min_k

declare -i n
declare -i d
declare -i k

for ((nc=$min_n; nc<=$max_n; nc += $inc_n))
do
    n=nc/p
    for ((dc=$min_d; dc<=$max_d; dc += $inc_d))
    do
        d=dc
        for ((kc=$min_k; kc<=$max_k; kc += $inc_k))
        do
            k=kc
            sbatch async.sh $n $d $k
        done
    done
done
