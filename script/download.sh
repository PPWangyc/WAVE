#!/bin/bash

echo "Downloading the dataset and model checkpoints"
echo "warning: this script will require git lfs to download the dataset, install from https://git-lfs.com"

git lfs install

# download the dataset
echo "start downloading the dataset"
echo "The dataset is around 4GB, it may take a while to download"
cd ..
cd data
git clone https://huggingface.co/datasets/PPWangyc/WAVE-BOLD5000
echo "finished downloading the dataset"
echo "if download dataset is smaller than 4GB, please make sure git lfs is installed"

cd ..

cd script