#!/bin/bash
trap "exit" INT
dataset=${1?Error: which dataset am I using, avenue, avenue_robust_on_rain}
model=${2?Error: which model am I using? single_branch, multi_branch_z}
ver=${3?Error:I forget to input the version, int}
datadir=/project_scratch/bo/anomaly_data/
expdir=/project/bo/exp_data/

# ./requirement.sh
python3 train_end2end_sum.py --datadir $datadir --expdir $expdir --num_bg 2 --model_type $model --version $ver --data_set $dataset --rain_type None --brightness 0

