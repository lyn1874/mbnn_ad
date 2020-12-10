#!/bin/bash
trap "exit" INT
pip install tensorflow-gpu==1.13.1
yes Y | conda install cudatoolkit=10.0
echo "Installed all the requirements"
