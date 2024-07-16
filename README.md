# WAVE: Whole Brain Analysis of Visual Experiences

<p align="center">
    <img src=assets/figure_1.png />
</p>

## Environment Setup

To get started, ensure you have git-lfs and anaconda installed. If git-lfs is not installed, please follow the installation instructions [here](https://git-lfs.github.com/).

Execute the following commands to set up the project environment:

```bash
cd script
# make sure you installed git-lfs and anaconda
source setup.sh
```

The `setup.sh` script will create a new conda environment named `wave` and install all the necessary packages. Moreover it will also automatically download the processed [BOLD5000 huggingface dataset](https://huggingface.co/datasets/PPWangyc/WAVE-BOLD5000) and the [pre-trained WAVE huggingface model](https://huggingface.co/PPWangyc/WAVE-models).

Important: Please make sure you have installed `git-lfs` before running the setup script. If you haven't installed `git-lfs`, you can install it by following the instructions [here](https://git-lfs.github.com/). We used `git-lfs` to download the datasets and model checkpoints.

## Part A: Contrastive Learning
We provide the code for training the contrastive learning model in the `src/train_bold5000_contrastive.py` python file. To train the model, run the following command:

```bash
cd script
source train_bold5000_contrastive.sh CSI1 False
# CSI1 is the subject name
# False means not using the visual network mask
```

Parameters:
- `CSI1` is the subject name, we also accept to use "all" as universal training for all subjects
- `False` means not using the visual network mask, we also accept to use "True" to use the visual network mask
- `training-steps` is the number of training steps, increase by 4 times if use "all", unversal training for all subjects
- `seed` is the random seed for training, default is 42
- `wandb` is whether to log the training process to wandb, default is True

The results are saved in the results/**/contrastive folder. 

## Part B: Diffusion Model
We provide the code for training the diffusion model in the `src/train_bold5000_decode.py` python file. We use versatile diffusion model to decode the fMRI data. You can also edit the model vd cache path in the script. To train the model, run the following command:

```bash
cd script
source train_bold5000_decode.sh CSI1 False
# CSI1 is the subject name
# False means not using the visual network mask
```

Same parameters as the contrastive learning model. The pre-trained contrastive learning model path is set in the script, you can also replace it with your own model path by changing the `CHECKPOINT_DIR` in the `train_decode.sh` script.

## Reconstruction and Evaluation
We have implemented the reconstruction and evaluation code in the `src/reconstruct.py` python file. We also include the necessary metrics, all the results can be found in local results folder or log to wandb. To reconstruct the fMRI data, run the following command:

```bash
cd script
source reconstruct.sh CSI1 region False 2 42
# CSI1 is the subject name
# region is the mode represent the whole brain region
# False means not using the visual network mask
# 2 is the recon sample number
# 42 is the random seed for reconstruction
```

Parameters:
- `CSI1` is the subject name, "all" to reconstruct all subjects test set data.
- `region` You can also replace it with "voxel" to reconstruct based on visual network voxel data.
- `False` means not using the visual network mask, we also accept to use "True" to use the visual network mask.
- `2` is the number of reconstruction samples, we also accept to use "1" to reconstruct only one sample. In the paper, we use 2 samples for reconstruction.
- `42` is the random seed for reconstruction, default is 42.

We provided pre-trained models for CSI1 and universal. If you trained your own model, you can replace the `CHECKPOINT_TRAINED_DIR` in the `reconstruct.sh` script with your own model path.

## Dataset and Model
To support the reproducibility of our work, we provide the processed BOLD5000 dataset and the pre-trained WAVE model. You can download the dataset and model from the following links: 
- [BOLD5000 huggingface dataset](https://huggingface.co/datasets/PPWangyc/WAVE-BOLD5000)
- [Pre-trained WAVE huggingface model](https://huggingface.co/PPWangyc/WAVE-models)

Our external fMRI imagination dataset is required IRB approval and DUA to access. If you are interested in the dataset, please contact the authors.

We use V100/A100 GPU for training and reconstruction.

## Acknowledgements

We extend our gratitude to several teams whose resources have significantly contributed to our project:

-   **BOLD5000 Team:** We thank them for making both their raw and pre-processed data publicly accessible. [Visit BOLD5000](https://bold5000-dataset.github.io/website/)
-   **Learning From Brains Project:** Our fMRI-Foundation model and data loader draw upon the methodologies developed by this team. [Learn more](https://github.com/athms/learning-from-brains)
-   **MindEye Team:** Our diffusion model builds upon the innovative work done by the MindEye team. [Explore their GitHub](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)
-   **Mind-Vis Team:** We appreciate their commitment to open science and reproducibility, which is demonstrated through their published code. [Check out their resources](https://github.com/zjc062/mind-vis)

We are immensely thankful to all the authors for their openness and willingness to share their codes and checkpoints with the community.