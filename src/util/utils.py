import torch
import numpy as np
import random
import os
import argparse
from typing import Dict
from datetime import datetime
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    # set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('seed set to {}'.format(seed))

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def patchfy1D(x, patch_size):
    """
    x: (N, C, L)
    return: (N, L//patch_size, D)
    """

    N, C, L = x.shape
    x = x.reshape(N, L//patch_size, patch_size * C)
    return x

def mask_ecg(x, mask):
    """
    x: (N, Path_num, D)
    mask: (N, L)
    """
    # make mask to (N, Path_num, D), mask be nan
    mask = mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
    # convert x to nan
    x_masked = x.where(mask == 0, torch.tensor(np.nan).to(x.device))
    return x_masked

def save_model(config, epoch, net, optimizer, loss, checkpoint_paths, name):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'config': config,
    }
    torch.save(to_save, os.path.join(checkpoint_paths, '{}.pth'.format(name)))

def pad_ecg(ecg, max_length):
    """
    ecg: (N,C, L)
    """
    N, C, L = ecg.shape
    ecg_padded = torch.zeros(N, C, max_length).to(ecg.device)
    ecg_padded[:, :, :L] = ecg
    return ecg_padded

def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""

    parser = argparse.ArgumentParser(
        description='run model training'
    )

    # Data pipeline settings:
    parser.add_argument(
        '--data',
        metavar='DIR',
        default='data/upstream',
        type=str,
        help='path to training data directory '
             '(default: data/upstream)'
    )
    parser.add_argument(
        '--frac-val-per-dataset',
        metavar='FLOAT',
        default=0.05,
        type=float,
        help='fraction of fMRI runs per dataset that '
             'are randomly selected as validation data '
             '(default: 0.05)'
    )
    parser.add_argument(
        '--n-val-subjects-per-dataset',
        metavar='INT',
        default=-1,
        type=int,
        help='number of subjects per dataset that are '
             'randomly selected as validation data. '
             '! overrides --frac-val-per-dataset and '
             'requires setting --n-train-subjects-per-dataset' 
    )
    parser.add_argument(
        '--n-test-subjects-per-dataset',
        metavar='INT',
        default=-1,
        type=int,
        help='number of subjects per dataset that are '
             'randomly selected as test data. '
             '! Test set is only created if this is set != -1'
    )
    parser.add_argument(
        '--n-train-subjects-per-dataset',
        metavar='INT',
        default=-1,
        type=int,
        help='number of subjects per dataset that are '
             'randomly selected as training data. '
             '! overrides --frac-val-per-dataset and '
             'requires setting --n-val-subjects-per-dataset' 
    )
    parser.add_argument(
        '--parcellation-dim',
        metavar='INT',
        default=1024,
        type=int,
        help='dimension of input data parcellation (default: 1024). '
             '! This is fixed for the current up-/downstream data.'
    )
    parser.add_argument(
        '--pretrained-model',
        metavar='DIR',
        type=str,
        default='none',
        help='checkpoint used to initialize model weights '
             '(default: none)'
    )


    # Embedder settings:    
    parser.add_argument(
        '--embedding-dim',
        metavar='INT',
        default=768,
        type=int,
        help='dimension of input embedding '
             '(default: 768)'
    )
    parser.add_argument(
        '--num-hidden-layers-embedding-model',
        metavar='INT',
        default=1,
        type=int,
        help='numer of layers of linear embedding model '
             '(default: 1)'
    )
    parser.add_argument(
        '--tr-max',
        metavar='INT',
        default=300,
        type=int,
        help='maximum number of TRs in TR-embeddings '
             '(in seconds; default: 300)'
    )
    parser.add_argument(
        '--tr-precision',
        metavar='FLOAT',
        default=0.2,
        type=float,
        help='precision (ie., frequency) of TR embeddings '
             '(in seconds; default: 0.2). '
             'When set to 0.2, embeddings are created for: '
             '0, 0.2, 0.4, ..., tr-max'
    )
    parser.add_argument(
        '--freeze-embedder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze embedder weights during training '
             '(default: False) '
    )


    # UnEmbedder settings:
    parser.add_argument(
        '--num-hidden-layers-unembedding-model',
        metavar='INT',
        default=1,
        type=int,
        help='numer of hidden layers for linear unembedding model '
             '(default: 1)'
    )
    parser.add_argument(
        '--freeze-unembedder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze unembedder weights during training '
             '(default: False) '
    )


    # Decoder settings:
    parser.add_argument(
        '--architecture',
        metavar='STR',
        default='GPT',
        choices=(
            'BERT',
            'GPT',
            'autoencoder',
            'NetBERT',
            'LinearBaseline'
        ),
        type=str,
        help='Model architecture used for sequence modeling / decoding. '
             'One of {BERT, NetBERT, GPT, autoencoder, LinearBaseline} '
             '(default: GPT) '
    )
    parser.add_argument(
        '--num-hidden-layers',
        metavar='INT',
        default=4,
        type=int,
        help='number of hidden model layers in --architecture '
             '(default: 4). '
             '! Does not apply to LinearBaseline; '
             '! Same number of hidden layers is used for decoder / encoder '
             'parts of autoencoder (ie., default creates encoder and decoder '
             'with 4 hidden layers each)'
    )
    parser.add_argument(
        '--num-attention-heads',
        metavar='INT',
        default=-1,
        type=int,
        help='number of attention heads per transformer layer '
             '(default: embedding-dim // 64). '
             '! Does not apply to non-transformer models'
    )
    parser.add_argument(
        '--intermediate-dim-factor',
        metavar='INT',
        default=4,
        type=int,
        help='scales feed-forward transformer layer dimension relative to '
             'embedding-dim: intermediate-dim-factor * embedding-dim '
             '(default: 4)'
    )
    parser.add_argument(
        '--hidden-activation',
        metavar='STR',
        default='gelu_new',
        choices=(
            'gelu',
            'gelu_new',
            'relu',
            'silu'
        ),
        type=str,
        help='type of hidden activation of transformer layers '
             '(default: gelu_new); '
             'one of {"gelu", "gelu_new", "relu", "silu"}. '
             '! Does not apply to non-transformer models'
    )
    parser.add_argument(
        '--n-positions',
        metavar='INT',
        default=512,
        type=int,
        help='maximum sequence length that transformer model might ever be used with '
             '(default: 512)'
    )
    parser.add_argument(
        '--freeze-decoder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze decoder model weights during training '
             'as specified by --architecture '
             '(default: False) '
    )
    parser.add_argument(
        '--freeze-decoder-without-pooler-heads',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze decoder model weights during training '
             'as specified by --architecture, without pooler layer and '
             ' is-next-pred / decoding heads '
             '(default: False) '
    )

    

    # Trainer settings:
    parser.add_argument(
        '--resume-from',
        metavar='DIR',
        type=str,
        default='none',
        help='continue training from specified checkpoint '
             '(default: none)'
    )
    parser.add_argument(
        '--training-style',
        metavar='STR',
        default='CSM',
        choices=(
            'CSM',
            'BERT',
            'NetBERT',
            'autoencoder',
            'decoding'
        ),
        type=str,
        help='training framework / style (default: CSM); '
             'one of {BERT, CSM, NetBERT, autoencoder, decoding}'
    )
    parser.add_argument(
        '--decoding-target',
        metavar='STR',
        default='none',
        type=str,
        help='key for decoding target variable in .tar-files in --data'
             '(default: none). '
             '! Must be specified when setting --training-style to "decoding"'
    )
    parser.add_argument(
        '--num-decoding-classes',
        metavar='INT',
        default=0,
        type=int,
        help='number of decoding classes (ie., mental states) in --data '
             '(default: 0). '
             '! Must be specified when setting --training-style to "decoding"'
    )
    parser.add_argument(
        '--training-steps',
        metavar='INT',
        default=400000,
        type=int,
        help='number of training steps to perform '
             '(default: 400000)'
    )
    parser.add_argument(
        '--validation-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='number of validation steps to perform at evaluation time '
             '(default: 1000)'
    )
    parser.add_argument(
        '--test-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='number of test steps to perform at test time'
             '(default: 2000). '
             '! Test evaluation only performed if test set created by '
             'setting --n-test-subjects-per-dataset != -1'
    )
    parser.add_argument(
        '--per-device-training-batch-size',
        metavar='INT',
        default=64,
        type=int,
        help='batch size during training per training device '
             '(default: 64)'
    )
    parser.add_argument(
        '--per-device-validation-batch-size',
        metavar='INT',
        default=64,
        type=int,
        help='batch size during evaluation per training device '
             '(default: 64)'
    )
    parser.add_argument(
        '--optim',
        metavar='STR',
        default='adamw_hf',
        type=str,
        help='optimizer to use for training '
             '(default: adamw_hf) -> adamw from HuggingFrace transformer library. '
             'For other options see Huggingface TrainerArgs.'
    )
    parser.add_argument(
        '--learning-rate',
        metavar='FLOAT',
        default=1e-4,
        type=float,
        help='maximum learning rate during training '
             '(default: 1e-4)'
    )
    parser.add_argument(
        '--warmup-ratio',
        metavar='FLOAT',
        default=0.01,
        type=float,
        help='warm-up steps for linear learning rate scheduler '
             'specified as fraction of --training-steps '
             '(default: 0.01)'
    )
    parser.add_argument(
        '--weight-decay',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='weight decay strength (indicating l2-regularisation strength) '
             '(default: 0.1)'
    )
    parser.add_argument(
        '--adam-beta-1',
        metavar='FLOAT',
        default=0.9,
        type=float,
        help='adam beta 1 (default: 0.9)'
    )
    parser.add_argument(
        '--adam-beta-2',
        metavar='FLOAT',
        default=0.999,
        type=float,
        help='adam beta 2 (default: 0.999)'
    )
    parser.add_argument(
        '--adam-epsilon',
        metavar='FLOAT',
        default=1e-8,
        type=float,
        help='adam beta 2 (default: 1e-8)'
    )
    parser.add_argument(
        '--max-grad-norm',
        metavar='FLOAT',
        default=1.0,
        type=float,
        help='maximum gradient clipping norm (default: 1.0)'
    )
    parser.add_argument(
        '--lr-scheduler',
        metavar='STR',
        default='linear',
        choices=(
            'linear',
            'constant_with_warmup',
            'none'
        ),
        type=str,
        help='learning rate scheduler; '
             'one of {linear, constant_with_warmup, none} '
             '(default: linear)'
    )
    parser.add_argument(
        '--sample-random-seq',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        help='whether or not to randomly sample input sequences '
             'from BOLD --data during training '
             '(default: True). '
             'Range for randomly sampled sequence lengths specified by '
             '--seq-min and --seq-max'
    )
    parser.add_argument(
        '--seq-min',
        metavar='INT',
        default=10,
        type=int,
        help='minimum length of randomly sampled BOLD input sequences '
             '(in number of TRs; default: 10)'
    )
    parser.add_argument(
        '--seq-max',
        metavar='INT',
        default=50,
        type=int,
        help='maximum length of randomly sampled BOLD input sequences '
             '(in number of TRs; default: 50)'
    )
    parser.add_argument(
        '--bert-seq-gap-min',
        metavar='INT',
        default=1,
        type=int,
        help='minimum TR gap between two input sequences for BERT-style training '
             '(default: 1). '
             'Gap is randomly sampled between --bert-seq-gap-min and --bert-seq-gap-max'
    )
    parser.add_argument(
        '--bert-seq-gap-max',
        metavar='INT',
        default=5,
        type=int,
        help='maximum TR gap between two input sequences for BERT-style training '
             '(default: 5). '
             'Gap is randomly sampled between --bert-seq-gap-min and --bert-seq-gap-max'
    )
    parser.add_argument(
        '--masking-rate',
        metavar='FLOAT',
        default=0.2,
        type=float,
        help='masking rate for BERT-style training '
             '(default: 0.15)'
    )
    parser.add_argument(
        '--dropout',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='dropout ratio for hidden layers of embedder and decoder model parts '
             '(default: 0.1)'
    )
    parser.add_argument(
        '--autoen-teacher-forcing-ratio',
        metavar='FLAOT',
        default=0.5,
        type=float,
        help='teacher forcing ratio for autoencoder training '
             '(default: 0.5)'
    )

    
    # Logging settings:
    parser.add_argument(
        '--log-dir',
        metavar='DIR',
        type=str,
        default='results/models/upstream',
        help='path where training is logged '
             '(default: results/models/upstream)'
    )
    parser.add_argument(
        '--log-every-n-steps',
        metavar='INT',
        default=10000,
        type=int,
        help='frequence of logging in training steps '
             '(default: 10000)'
    )
    parser.add_argument(
        '--run-name',
        metavar='STR',
        type=str,
        default='none',
        help='descriptor of the training run used for logging and wandb; '
             '! if set to "none", a unique identifier is automatically created'
    )
    parser.add_argument(
        '--wandb-mode',
        metavar='STR',
        choices=(
            'online',
            'offline',
            'disabled'
        ),
        default='disabled',
        help='track training w/ wandb online or offline or not at all '
             '(default: disabled) '
             '! requires setting up weights-and-bias for this machine; '
             'see: https://docs.wandb.ai/'
    )
    parser.add_argument(
        '--wandb-project-name',
        metavar='STR',
        type=str,
        default='learning-from-brains',
        help='name of wandb project where data is logged '
             '(default: learning-from-brains)'
    )
    

    # Other settings:
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=1234,
        type=int,
        help='random seed (default: 1234)'
    )
    parser.add_argument(
        '--set-seed',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        type=str,
        help='whether or not to set random seed (default: True)'
    )
    parser.add_argument(
        '--fp16',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        help='whether or not to use 16-bit precision GPU training '
             '(default: True)'
    )
    parser.add_argument(
        '--deepspeed',
        metavar='DIR',
        default="none",
        type=str,
        help='location of deepspeed configuration file; '
             'automatically adds deepspeed functionality to training if specified '
             '(default: none)'
    )
    parser.add_argument(
        '--local_rank',
        metavar='INT',
        default=-1,
        type=int,
        help='Rank of the process during distributed training '
             '(default: -1)'
    )
    parser.add_argument(
        '--num-workers',
        metavar='INT',
        default=0,
        type=int,
        help='number of data loading workers '
             '(default: 0 -> load in main process)'
    )
    parser.add_argument(
        '--plot-model-graph',
        metavar='BOOL',
        default="False",
        type=str,
        choices=('True', 'False'),
        help='whether or not to save an image of the model graph to log-dir '
             '(default: False)'
    )
    parser.add_argument(
        '--smoke-test',
        metavar='BOOL',
        default="False",
        type=str,
        choices=("True", "False"),
        help='whetehr or not to run training in smoke test-mode '
             '(default: False)'
             'If set to "True", training is restricted by setting: '
             '--per-device-training_batch_size 2 '
             '--per-device-validation_batch_size 2 '
             '--training-steps 2 '
             '--validation-steps 2 '
             '--test-steps 2 '
             '--log-every-n-steps 1'
    )
    parser.add_argument(
        '--bold-dummy-mode',
        metavar='BOOL',
        default='False',
        type=str,
        choices=('True', 'False'),
        help='whether or not to replace BOLD with dummy during training; '
             'for internal testing purposes only! '
             '(default: False)'
    )
    parser.add_argument(
        '--do-train',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether or not to run training '
             '(default: True). '
             'If "False", train() still returns trainer'
    )

    parser.add_argument(
        '--is-prompt',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether or not to use prompt learner for training '
             '(default: True). '
                'If "False", train() still returns trainer'
    )

    parser.add_argument(
        '--wandb',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether or not to use wandb for logging '
             '(default: True). '
                'If "False", train() still returns trainer'
    )

    parser.add_argument(
        '--l1-lambda',
        metavar='FLOAT',
        default=0.001,
        type=float,
        help='lambda for l1 regularization '
             '(default: 0.001)'
    )

    parser.add_argument(
        '--scheduler',
        metavar='STR',
        default='step',
        type=str,
        help='scheduler for training '
             '(default: step)'
    )

    parser.add_argument(
        '--scheduler-gamma',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='gamma for scheduler '
             '(default: 0.1)'
    )

    parser.add_argument(
        '--scheduler-step-size',
        metavar='INT',
        default=1000,
        type=int,
        help='step size for scheduler '
             '(default: 1000)'
    )

    parser.add_argument(
        '--sparse-ratio',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='ratio of sparse '
             '(default: 0.1)'
    )

    parser.add_argument(
        '--eval-every-n-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='evaluate every n steps '
                '(default: 1000)'
    )

    parser.add_argument(
        '--subject-id',
        metavar='STR',
        default='none',
        type=str,
        help='subject id for training '
                '(default: none)'
    )

    parser.add_argument(
        '--train-perc',
        metavar='FLOAT',
        default=0.8,
        type=float,
        help='train percentage for training '
                '(default: 0.8)'
    )

    parser.add_argument(
        '--eval-perc',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='eval percentage for training '
                '(default: 0.1)'
    )

    parser.add_argument(
        '--test-perc',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='test percentage for training '
                '(default: 0.1)'
    )

    parser.add_argument(
        '--only-visual',
        metavar='BOOL',
        default='False',
        type=str,
        choices=('True', 'False'),
        help='whether or not to only use visual data for training '
             '(default: False)'
    )

    parser.add_argument(
        '--prior-checkpoint-dir',
        metavar='STR',
        default='none',
        type=str,
        help='prior checkpoint dir for training '
                '(default: none)'
    )

    parser.add_argument(
        '--sd-ckpt-dir',
        metavar='STR',
        default='none',
        type=str,
        help='sd checkpoint dir for reconstruction '
    )

    parser.add_argument(
        '--vd-ckpt-dir',
        metavar='STR',
        default='none',
        type=str,
        help='vd checkpoint dir for reconstruction '
    )

    parser.add_argument(
        '--n-samples-save',
        metavar='INT',
        default=1,
        type=int,
        help='number of samples to save '
    )

    parser.add_argument(
        '--log-reconstruction-every-n-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='log reconstruction every n steps '
    )

    parser.add_argument(
        '--tarfile-paths-split',
        metavar='STR',
        default=None,
        type=str,
        help='path to tarfile paths split '
    )

    parser.add_argument(
        "--norm-embs",
        metavar="BOOL",
        default="False",
        type=str,
        choices=("True", "False"),
        help="whether or not to normalize embeddings",
    )

    parser.add_argument(
        "--trained-checkpoint-dir",
        metavar="STR",
        default="none",
        type=str,
    )

    parser.add_argument(
        "--top-regions-num",
        metavar="INT",
        default=20,
        type=int,
    )

    parser.add_argument(
        "--csv-path",
        metavar="STR",
        default="none",
        type=str,
    )

    parser.add_argument(
        "--train-mode",
        metavar="STR",
        default="none",
        type=str,
    )

    parser.add_argument(
        "--ckpt-trained-classifier-dir",
        metavar="STR",
        default="none",
        type=str,
    )

    parser.add_argument(
        "--classification-id",
        metavar="INT",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--feature-ablation",
        metavar="BOOL",
        default="False",
        type=str,
        choices=("True", "False"),
        help="whether or not to ablate features",
    )

    parser.add_argument(
        "--save-recon-embds",
        metavar="BOOL",
        default="False",
        type=str,
        choices=("True", "False"),
        help="whether or not to save recon embds",
    )

    parser.add_argument(
        "--vis-mask-json-path",
        metavar="STR",
        default="none",
        type=str,
    )

    parser.add_argument(
        "--avg-data-dir",
        metavar="STR",
        default="none",
        type=str,
    )

    parser.add_argument(
        "--use-train-set",
        metavar="BOOL",
        default="False",
        type=str,
    )

    parser.add_argument(
        "--voxel-encoder-path",
        metavar="STR",
        default="none",
        type=str,
    )

    parser.add_argument(
        "--hidden",
        metavar="BOOL",
        default="False",
        type=str,
    )

    parser.add_argument(
        "--autoenc-ckpt-dir",
        metavar="STR",
        default="none",
        type=str,
    )

    parser.add_argument(
        "--recons-per-sample",
        metavar="INT",
        default=1,
        type=int,
    )

    return parser

def get_config(args: argparse.Namespace=None) -> Dict:
    """
    Make config from command line arguments (as created by get_args()).
    Performs additional formating of args required for calling train().
    """

    if args is None:
        args = get_args().parse_args()

    if args.smoke_test == "True":
        args.per_device_training_batch_size =  2
        args.per_device_validation_batch_size = 2
        args.training_steps = 2
        args.validation_steps = 2
        args.test_steps = 2
        args.log_every_n_steps = 1

    if args.num_attention_heads == -1:
        assert (
            args.embedding_dim%64
         ) == 0, f'embedding-dim needs be be multiple of 64 (currently: {args.embedding_dim})' 
        args.num_attention_heads = args.embedding_dim//64

    if args.run_name == 'none':
        args.run_name = f'{args.architecture}'

        if args.architecture != 'LinearBaseline':
            
            if 'Pretrained' not in args.architecture:
                args.run_name += f'_lrs-{args.num_hidden_layers}'

                if args.architecture != 'autoencoder':
                    args.run_name += f'_hds-{args.num_attention_heads}'

            args.run_name += f'_embd-{args.embedding_dim}'
            args.run_name += f'_train-{args.training_style}'
            args.run_name += f'_lr-{str(args.learning_rate).replace(".", "")[1:]}'
            args.run_name += f'_bs-{args.per_device_training_batch_size}'
            args.run_name += f'_drp-{str(args.dropout).replace(".", "")}'

            if args.training_style not in {'decoding', 'autoencoder', 'CSM'}:
                args.run_name += f'_msk-{str(args.masking_rate).replace(".", "")}'

        else:
            args.run_name += f'_train-{args.training_style}'

        args.run_name += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if args.smoke_test == "True":
        args.run_name = f'smoke-test_{args.run_name}'

    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.wandb_mode = args.wandb_mode if args.wandb_mode in {'online', 'offline'} and args.local_rank in {-1, 0} else "disabled"
    
    config = vars(args)

    for arg in config:
        
        if config[arg] in {'True', 'False'}:
            config[arg] = config[arg] == 'True'
        
        elif config[arg] == 'none':
            config[arg] = None

        elif 'subjects_per_dataset' in arg:
            config[arg] = None if config[arg] == -1 else config[arg]

    return config

def move_batch_to_device(batch, device):

        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        return batch

def get_vis_mask(vis_mask_json_path, batch_size=1, device='cuda'):
    print('setting up visual mask to fmri')
    vis_mask_ids = json.load(open(vis_mask_json_path, 'r'))
    vis_mask_ids = torch.tensor(vis_mask_ids)
    # create mask
    vis_mask = torch.zeros(batch_size, 50, 1024, dtype=torch.bool)
    vis_mask[:, :, vis_mask_ids] = True
    vis_mask = vis_mask.to(device)
    return vis_mask

def get_avg_tarfile_paths(data_dir, subject_id):
    print('setting up avg dataset')
    import glob
    train_tarfile_paths = glob.glob(os.path.join(data_dir, subject_id, 'train', '*.tar'))
    test_tarfile_paths = glob.glob(os.path.join(data_dir, subject_id, 'test', '*.tar'))
    return train_tarfile_paths, test_tarfile_paths


def get_train_eval_test(tarfile_paths: list, subject_id, train_perc=0.8, eval_perc=0.1, test_perc=0.1):
    """
    Get train, eval and test tarfile paths from list of tarfile paths.
    """
    subj_tarf_paths = [tarfile_path for tarfile_path in tarfile_paths if subject_id in tarfile_path]
    random.shuffle(subj_tarf_paths)
    n_subj_tarf_paths = len(subj_tarf_paths)
    n_train = int(n_subj_tarf_paths * train_perc)
    n_eval = int(n_subj_tarf_paths * eval_perc)
    n_test = int(n_subj_tarf_paths * test_perc)
   
    train_tarf_paths = subj_tarf_paths[:n_train]
    eval_tarf_paths = subj_tarf_paths[n_train:n_train+n_eval]
    test_tarf_paths = subj_tarf_paths[n_train+n_eval:n_train+n_eval+n_test]

    # output as dict
    train_eval_test = {
        'train': train_tarf_paths,
        'validation': eval_tarf_paths,
        'test': test_tarf_paths
    }

    return train_eval_test

def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def normalize(x, mean=None, std=None):
    mean = x.mean() if mean is None else mean
    std = x.std() if std is None else std
    return (x - mean) / std

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def get_voxel_num(subj):
    origin_voxel_num = {'CSI1':1685, 'CSI2': 1129, 'CSI3': 1466, 'CSI4': 2787}
    patched_voxel_num = {'CSI1': 1696, 'CSI2': 1136, 'CSI3': 1472,'CSI4': 2800}
    return origin_voxel_num[subj], patched_voxel_num[subj]

def get_all_subjects_tarfile_paths(data_dir):
    print('setting up all subjects')
    import glob
    train_tarfile_paths = glob.glob(os.path.join(data_dir, '*', 'train', '*.tar'))
    test_tarfile_paths = glob.glob(os.path.join(data_dir, '*', 'test', '*.tar'))
    return train_tarfile_paths, test_tarfile_paths

@torch.no_grad()
def reconstruction(
    image, batch,
    clip_extractor=None,
    unet=None, 
    vae=None, 
    noise_scheduler=None,
    voxel2clip_cls=None,
    diffusion_priors=None,
    text_token = None,
    img_lowlevel = None,
    num_inference_steps = 50,
    recons_per_sample = 1,
    guidance_scale = 7.5,
    img2img_strength = .85,
    timesteps_prior = 100,
    seed = 0,
    retrieve=False,
    plotting=True,
    verbose=False,
    img_variations=False,
    n_samples_save=1,
    num_retrieved=16,
    use_image_features=False,
    return_recons_emb=False,
    train_mode='region',
):
        
    brain_recons = None

    for key in batch:
        if key in ["inputs", "attention_mask", "t_rs"]:
            batch[key] = batch[key][:1].to(device)
    image=image[:1]

    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    if diffusion_priors is not None:
        if not isinstance(diffusion_priors, list):
            diffusion_priors = [diffusion_priors]
        brain_clip_embeddings_sum = None
        for diffusion_prior in diffusion_priors:
            if train_mode=='mindeye':
                brain_clip_embeddings0, proj_embeddings = diffusion_prior.voxel2clip(batch['voxels'].float())
                if not img_variations:
                    brain_clip_embeddings0 = brain_clip_embeddings0.view(len(batch['voxels']),-1,768) # 768 is the clip size
            elif 'region' in train_mode and 'mindeye' in train_mode:
                brain_clip_embeddings0, proj_embeddings = diffusion_prior.voxel2clip(batch['inputs'].squeeze(1)[:,:5].flatten(1).float())
                if not img_variations:
                    brain_clip_embeddings0 = brain_clip_embeddings0.view(len(batch['voxels']),-1,768)
            elif 'voxel' in train_mode and 'mindeye' in train_mode:
                brain_clip_embeddings0, proj_embeddings = diffusion_prior.voxel2clip(batch['voxels'].float())
                if not img_variations:
                    brain_clip_embeddings0 = brain_clip_embeddings0.view(len(batch['voxels']),-1,768)
            else:
                outputs = diffusion_prior.voxel2clip(batch)
                if 'region' in train_mode:
                    if hasattr(diffusion_prior, 'dim_mapper'):
                        brain_clip_embeddings0, proj_embeddings = outputs['unproj_fmri_features'], outputs['fmri_features'].clone()
                        brain_clip_embeddings0 = diffusion_prior.dim_mapper(brain_clip_embeddings0)
                        brain_clip_embeddings0 = brain_clip_embeddings0.view(len(batch['inputs']),-1,768)
                    else:
                        brain_clip_embeddings0, proj_embeddings = outputs['fmri_features'], outputs['fmri_features'].clone()
                elif 'voxel' in train_mode:
                    brain_clip_embeddings0, proj_embeddings = outputs['voxel_features'], None
            if retrieve:
                continue
            # brain_clip_embeddings0 = brain_clip_embeddings0.view(len(batch[0]),-1,768) if isinstance(clip_extractor,Clipper) else brain_clip_embeddings0.view(len(batch[0]),-1,1024)
            # brain_clip_embeddings0 = brain_clip_embeddings0.unsqueeze(1)
            if recons_per_sample>0:
                if not img_variations:
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1, 1)
                    print("brain_clip_embeddings0",brain_clip_embeddings0.shape)
                    try:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior,
                                                generator=generator) 
                    except:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior)
                else:
                    brain_clip_embeddings0 = brain_clip_embeddings0.view(-1,768)
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1)
                    brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = 1000, #1000 timesteps used from nousr pretraining
                                                generator=generator)
                if brain_clip_embeddings_sum is None:
                    brain_clip_embeddings_sum = brain_clip_embeddings
                else:
                    brain_clip_embeddings_sum += brain_clip_embeddings

        # average embeddings for all diffusion priors
        if recons_per_sample>0:
            brain_clip_embeddings = brain_clip_embeddings_sum / len(diffusion_priors)
    
    # if voxel2clip_cls is not None:
    #     _, cls_embeddings = voxel2clip_cls(voxel.to(device).float())
    # else:
    #     cls_embeddings = proj_embeddings
    # if verbose: print("cls_embeddings.",cls_embeddings.shape)
    
    # if retrieve:
    #     image_retrieved = query_laion(emb=cls_embeddings.flatten(),groundtruth=None,num=num_retrieved,
    #                                clip_extractor=clip_extractor,device=device,verbose=verbose)          
    if return_recons_emb:
        assert brain_clip_embeddings.shape[0]==1, "return_recons_emb only works for one sample at a time"
        return brain_clip_embeddings.flatten()
    if retrieve and recons_per_sample==0:
        brain_recons = torch.Tensor(image_retrieved)
        brain_recons.to(device)
    elif recons_per_sample > 0:
        if not img_variations:
            if use_image_features:
                print(brain_clip_embeddings.shape)
                brain_clip_embeddings = diffusion_priors[0].voxel2clip(batch)['image_features'][0].unsqueeze(0)
                print("brain_clip_embeddings",brain_clip_embeddings.shape)
            for samp in range(len(brain_clip_embeddings)):
                brain_clip_embeddings[samp] = brain_clip_embeddings[samp]/(brain_clip_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        else:
            if use_image_features:
                brain_clip_embeddings = diffusion_priors[0].voxel2clip(batch)['image_features']
                print("brain_clip_embeddings",brain_clip_embeddings.shape)
                # brain_clip_embeddings = use_image_features[:n_samples_save].unsqueeze(1) * 27.712812921102035
            else:
                brain_clip_embeddings = brain_clip_embeddings.unsqueeze(1)
        
        input_embedding = brain_clip_embeddings#.repeat(recons_per_sample, 1, 1)
        if verbose: print("input_embedding",input_embedding.shape)

        if text_token is not None:
            prompt_embeds = text_token.repeat(recons_per_sample, 1, 1)
        else:
            prompt_embeds = torch.zeros(len(input_embedding),77,768)
        if verbose: print("prompt!",prompt_embeds.shape)

        if do_classifier_free_guidance:
            input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)

        # dual_prompt_embeddings
        if not img_variations:
            input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

        # 5b. Prepare latent variables
        batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
        shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        if img_lowlevel is not None: # use img_lowlevel for img2img initialization
            init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            latent_timestep = timesteps[:1].repeat(batch_size)
            
            if verbose: print("img_lowlevel", img_lowlevel.shape)
            # if not img_variations:
            #     img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            # else: 
            #     img_lowlevel_embeddings = img_lowlevel
            img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
            init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

            noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                                generator=generator, dtype=input_embedding.dtype)
            init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents
        else:
            timesteps = noise_scheduler.timesteps
            latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                  generator=generator, dtype=input_embedding.dtype)
            latents = latents * noise_scheduler.init_noise_sigma

        # 7. Denoising loop
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            if verbose: print("latent_model_input", latent_model_input.shape)
            if verbose: print("input_embedding", input_embedding.shape)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # TODO:
                # noise_pred = dynamic_cfg(noise_pred_uncond, noise_pred_text, guidance_scale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        recons = decode_latents(latents,vae).detach().cpu()

        brain_recons = recons.unsqueeze(0)

    if verbose: print("brain_recons",brain_recons.shape)              
    # pick best reconstruction out of several
    best_picks = np.zeros(n_samples_save).astype(np.int16)
    
    if retrieve==False:
        v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings),-1),dim=-1)
        sims=[]
        for im in range(recons_per_sample): 
            currecon = clip_extractor.embed_image(brain_recons[0,[im]].float()).to(proj_embeddings.device).to(proj_embeddings.dtype)
            currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
            cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
            sims.append(cursim.item())
        if verbose: print(sims)
        best_picks[0] = int(np.nanargmax(sims))   
        if verbose: print(best_picks)
    # else: 
    #     v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings),-1),dim=-1)
    #     retrieved_clips = clip_extractor.embed_image(torch.Tensor(image_retrieved).to(device)).float()
    #     sims=[]
    #     for ii,im in enumerate(retrieved_clips):
    #         currecon = nn.functional.normalize(im.flatten()[None],dim=-1)
    #         if verbose: print(v2c_reference_out.shape, currecon.shape)
    #         cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
    #         sims.append(cursim.item())
    #     if verbose: print(sims)
    #     best_picks[0] = int(np.nanargmax(sims)) 
    #     if verbose: print(best_picks)
    #     recon_img = image_retrieved[best_picks[0]]
    
    if recons_per_sample==0 and retrieve:
        recon_is_laion = True
        recons_per_sample = 1 # brain reconstruction will simply be the LAION nearest neighbor
    else:
        recon_is_laion = False
                    
    img2img_samples = 0 if img_lowlevel is None else 1
    laion_samples = 1 if retrieve else 0
    num_xaxis_subplots = 1+img2img_samples+laion_samples+recons_per_sample
    if plotting:
        fig, ax = plt.subplots(n_samples_save, num_xaxis_subplots, 
                           figsize=(num_xaxis_subplots*5,6*n_samples_save),facecolor=(1, 1, 1))
    else:
        fig = None
        recon_img = None
    
    im = 0
    if plotting:
        ax[0].set_title(f"Original Image")
        ax[0].imshow(torch_to_Image(image[im]))
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img_lowlevel[im].clamp(0,1)))
    for ii,i in enumerate(range(num_xaxis_subplots-laion_samples-recons_per_sample,num_xaxis_subplots-laion_samples)):
        recon = brain_recons[im][ii]
        if recon_is_laion:
            recon = brain_recons[best_picks[0]]
        if plotting:
            if ii == best_picks[im]:
                ax[i].set_title(f"Reconstruction",fontweight='bold')
                recon_img = recon
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
    if plotting:
        if retrieve and not recon_is_laion:
            ax[-1].set_title(f"LAION5b top neighbor")
            ax[-1].imshow(torch_to_Image(image_retrieved0))
        for i in range(num_xaxis_subplots):
            ax[i].axis('off')
    
    # print("brain_recons",brain_recons.shape)
    # print("recon_img",recon_img.shape)
    # print(best_picks)
    return fig, brain_recons, best_picks, recon_img
