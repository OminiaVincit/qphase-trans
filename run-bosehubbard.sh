#!/usr/bin/bash
for L in 15 25 35 45 55 65 75 85 95
do
	python source/PH_parallel_BoseHubbardStatics.py --exp exp_20200318 --odir bosehubbard2 --size $L --postprocess 0
done
