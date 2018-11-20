#!/usr/bin/env bash
source scripts/parallelize.sh
njobs=20

obj_list="dog cat bicycle bus aeroplane tvmonitor train bird cow bottle diningtable chair car sheep boat pottedplant sofa motorbike horse person"


for obj in ${obj_list} ; do
    cmds="$cmds ; time python ./scripts/_run_one_loc.py ${obj}  > logs/run_${obj}"
done

echo "executing..."
f_ParallelExec $njobs "$cmds"
