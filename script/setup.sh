#!/bin/bash

cd ..
echo "Creating conda environment"
# create a new conda environment
conda env create -f env.yaml
# activate the conda environment
conda activate wave
python setup.py
cd script