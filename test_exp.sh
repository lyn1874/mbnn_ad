#!/bin/bash
trap "exit" INT
ve=${1?Error: experiment version}
opt=${2?Error: opt}
rain=${3?Error: None, heavy, torrential}
brightness=${4?Error: 0 1 2 3 4 5 6 7 8 9 10}
datadir=${5:-/project_scratch/bo/anomaly_data/}
expdir=${6:-/project/bo/exp_data/}
model_type=${7:-single_branch}
data=${8:-avenue}
ti=${9:-testing_video_21_}

if [ $opt = test ]; then
    python3 test_end2end_sum.py --datadir $datadir --expdir $expdir --num_bg 2 --model_type $model_type --version $ve --data_set $data --test_opt save_score_faster --rain_type $rain --brightness $brightness
elif [ $opt = fps ]; then
    python3 test_end2end_fps.py --datadir $datadir --expdir $expdir --num_bg 2 --model_type $model_type --version $ve --data_set $data --test_opt save_score_faster --rain_type $rain --brightness $brightness
else
    python3 test_end2end_sum.py --datadir $datadir --expdir $expdir --num_bg 2 --model_type $model_type --version $ve --data_set $data --test_opt $opt --rain_type $rain --brightness $brightness --test_index_use $ti
fi

