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
echo "lfs pull to make sure all the files are downloaded"
cd WAVE-BOLD5000
git lfs pull
cd ..
echo "if download dataset is smaller than 4GB, please make sure git lfs is installed!"

cd ..

git lfs install
# download the model checkpoints
echo "start downloading the model checkpoints"
echo "The model checkpoints are around 15GB, it may take a while to download"
cd checkpoints
git clone https://huggingface.co/PPWangyc/WAVE-models
echo "finished downloading the model checkpoints"
echo "lfs pull to make sure all the files are downloaded"
cd WAVE-models
git lfs pull
cd ..
echo "if download model checkpoints is smaller than 15GB, please make sure git lfs is installed!"

cd ..

cd script