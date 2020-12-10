#!/bin/bash
trap "exit" INT
datapath=${1?Error: where do you want to save the data, default: frames/}
aug=${2:-false}
download=${3:-false}
train_or_test=${4:-testing}
raintype=${5:-heavy}
bright=${6:-8}
if [ -d "${datapath}" ]; then
    echo "next download and preprocess data"
else
    mkdir -p $datapath
fi

expdir=$(pwd)
ds=Avenue
if [ $ds = Avenue ]; then
    datapath=$datapath/$ds/
    if [ -d "${datapath}" ]; then
        echo "$ds dataset exists"
        trainfolder=${datapath}frames/testing/
        if [ -d "$trainfolder" ]; then
            echo "Frames already exist, YEAH!"
        else
            echo "Extract frames"
            python3 aug_data.py --datapath $datapath --option extract
        fi        
    else
        echo "Download the Avenue dataset...."
        mkdir $datapath
        cd $datapath
        wget http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip
        unzip Avenue_Dataset.zip
        mv 'Avenue Dataset'/* .
        echo "Successfully download the dataset, next extract frames from the Avenue dataset"
        cd $expdir
        python3 aug_data.py --datapath $datapath --option extract
    fi
fi

if [ $aug = true ]; then
    echo "Augmenting the avenue dataset"
    python3 aug_data.py --datapath $datapath --option augment --rain_type $raintype --train_or_test $train_or_test --bright $bright
fi

if [ $download = true ]; then
    echo "Download experiment ckpts"
    ckptdir=checkpoints
    if [ -d "${ckptdir}" ]; then
        wget https://cloud.ilabt.imec.be/index.php/s/ey8iNj3xBsRxiQF/download -O $ckptdir/multibranch.zip
    else
        mkdir $ckptdir
        wget https://cloud.ilabt.imec.be/index.php/s/ey8iNj3xBsRxiQF/download -O $ckptdir/multibranch.zip
    fi
    unzip -d checkpoints/ checkpoints/multibranch.zip
    mv checkpoints/multi_branch_exp/* checkpoints/
    rm checkpoints/multibranch.zip
    rm -rf checkpoints/multi_branch_exp
    echo "Finish downloading experiments"
fi
