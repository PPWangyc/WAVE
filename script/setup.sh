#!/bin/bash

cd ..
echo "Creating conda environment"
# create a new conda environment
conda env create -f env.yaml
echo "finished creating conda environment"

# activate the conda environment
conda activate wave

echo "start setup.py"
python setup.py

cd script
source download.sh