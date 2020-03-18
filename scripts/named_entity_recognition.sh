#!/usr/bin/env bash

# Make sure to compile the cython files using 
# `./scripts/compile_cython.sh` 

source scripts/parallelize.sh
njobs=48

npasses=50
npasses_ns=100


SEEDS="0 1 2 3 4 5 6 7 8 9"
common=" --prefix ~/data  --train-file conll03_ner/eng.train --dev-file conll03_ner/eng.testa"
common="${common} --test-file conll03_ner/eng.testb"

for seed in ${SEEDS}
do

####################### MAIN EXPT #################################
algorithm=csvrg
k=5
L=32
warm_start=3
smoother=1
for l2reg in 1 0.1 0.01
do
	cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --L $L --K $k --smoother $smoother --l2reg $l2reg --seed $seed --warm_start ${warm_start} ${common} > logs/smooth/out_${algorithm}_${L}_${smoother}_${k}_${l2reg}_${seed} 2>&1"
done

algorithm=csvrg_lr
k=5
lr=2e-2
warm_start=2
decay_smoother=expo
smoother=2
for l2reg in 1 0.1 0.01
do
        cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --lr $lr --K $k --smoother $smoother --l2reg $l2reg --seed $seed --warm_start ${warm_start} --decay_smoother ${decay_smoother}  ${common} > logs/smooth/out_${algorithm}_${lr}_${smoother}_${k}_${l2reg}_${decay_smoother}_${seed} 2>&1"
done

algorithm=svrg
k=5
lr=2e-2
smoother=1
for l2reg in 1 0.1 0.01
do
	cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --lr $lr --K $k --smoother $smoother --l2reg $l2reg --seed $seed ${common} > logs/smooth/out_${algorithm}_${lr}_${smoother}_${k}_${l2reg}_${seed} 2>&1"
done

algorithm=bcfw
for l2reg in 1 0.1 0.01
do
	cmds="$cmds; time python main_ner.py  --num_passes ${npasses_ns} --algorithm ${algorithm} --l2reg $l2reg --seed $seed ${common} > logs/nonsmooth/out_${algorithm}_${l2reg}_${seed} 2>&1"
done

algorithm=pegasos
for l2reg in 1 0.1 0.01
do
	cmds="$cmds; time python main_ner.py  --num_passes ${npasses_ns} --algorithm ${algorithm} --l2reg $l2reg --seed $seed ${common} > logs/nonsmooth/out_${algorithm}_${l2reg}_${seed} 2>&1"
done

algorithm=sgd
declare -A sgd_lr=( \
	[1.0]=0.02 \
	[0.1]=0.04 \
	[0.01]=0.04 \
	)
t_sgd=150000.0
for l2reg in 1.0 0.1 0.01
do
	cmds="$cmds ; time python main_ner.py --num_passes ${npasses_ns} --algorithm ${algorithm} --l2reg $l2reg --seed $seed ${common} --lr ${sgd_lr[${l2reg}]} --lr-t ${t_sgd} > logs/nonsmooth/out_${algorithm}_${l2reg}_${lr}_${t_sgd}_${seed} "
done

####################### MAIN EXPT: NON-SMOOTH #################################
algorithm=csvrg
L=32
warm_start=3
for l2reg in 0.01
do
    cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --L $L --l2reg $l2reg --seed $seed --warm_start ${warm_start} ${common} > logs/nonsmooth/out_${algorithm}_${L}_${mu}_${seed} 2>&1"
done

algorithm=svrg
lr=2e-2
for l2reg in 0.01
do
    cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --lr $lr --l2reg $l2reg --seed $seed ${common} > logs/nonsmooth/out_${algorithm}_${lr}_${mu}_${seed} 2>&1"
done



############################ SMOOTHER EXPT PLOTS ###########################
algorithm=csvrg
k=5
L=32
warm_start=3
for smoother in 2 8 32
do
for l2reg in 1 0.01
do
	cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --L $L --K $k --smoother $smoother --l2reg $l2reg --seed $seed --warm_start ${warm_start} ${common} > logs/smooth/out_${algorithm}_${L}_${smoother}_${k}_${l2reg}_${seed} 2>&1"
done
done

algorithm=csvrg_lr
k=5
lr=2e-2
decay_smoother=expo
warm_start=2
for smoother in 1 8 32
do
for l2reg in 1 0.01
do
        cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --lr $lr --K $k --smoother $smoother --l2reg $l2reg --seed $seed --warm_start ${warm_start} --decay_smoother ${decay_smoother}  ${common} > logs/smooth/out_${algorithm}_${lr}_${smoother}_${k}_${l2reg}_${decay_smoother}_${seed} 2>&1"
done
done



############################ WARM START EXPT PLOTS ###########################
algorithm=csvrg
k=5
smoother=1
L=128
for warm_start in 1 2 3
do
for l2reg in 1 0.01
do
	cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --L $L --K $k --smoother $smoother --l2reg $l2reg --seed $seed --warm_start ${warm_start} ${common} > logs/smooth/out_${algorithm}_${L}_${smoother}_${k}_${l2reg}_${seed} 2>&1"
done
done

algorithm=csvrg_lr
k=5
decay_smoother=expo
smoother=2
lr=1e-2
for warm_start in 1 2 3
do
for l2reg in 1 0.01
do
        cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --lr $lr --K $k --smoother $smoother --l2reg $l2reg --seed $seed --warm_start ${warm_start} --decay_smoother ${decay_smoother}  ${common} > logs/smooth/out_${algorithm}_${lr}_${smoother}_${k}_${l2reg}_${decay_smoother}_${seed} 2>&1"
done
done


############################### K EXPTS #######################################

algorithm=csvrg
L=32
warm_start=3
smoother=1
l2reg=0.01
for k in 2 5 10
do
	cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --L $L --K $k --smoother $smoother --l2reg $l2reg --seed $seed --warm_start ${warm_start} ${common} > logs/smooth/out_${algorithm}_${L}_${smoother}_${k}_${l2reg}_${seed} 2>&1"
done

algorithm=csvrg_lr
lr=2e-2
warm_start=2
decay_smoother=expo
smoother=2
l2reg=0.01
for k in 2 5 10
do
        cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --lr $lr --K $k --smoother $smoother --l2reg $l2reg --seed $seed --warm_start ${warm_start} --decay_smoother ${decay_smoother}  ${common} > logs/smooth/out_${algorithm}_${lr}_${smoother}_${k}_${l2reg}_${decay_smoother}_${seed} 2>&1"
done


################################ KAPPA EXPTS ###########################################
algorithm=csvrg_lr
k=5
lr=2e-2
smoother=2
l2reg=1.0
for kappa in 0.01 0.1 10 100 1.001
do
for warm_start in 1 2 3
do
    cmds="$cmds; time python main_ner.py  --num_passes $npasses --algorithm ${algorithm} --lr $lr --K $k --smoother $smoother --l2reg $l2reg --kappa $kappa --seed $seed --warm_start ${warm_start} --decay_smoother ${decay_smoother}  ${common} > logs/smooth/out_${algorithm}_${lr}_${smoother}_${k}_${l2reg}_${decay_smoother}  2>&1"
done
done

done

############ DONE ###########

echo "executing..."
f_ParallelExec $njobs "$cmds"
