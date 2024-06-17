import torch
from model.fvl import FVL
import os
import argparse
from typing import Dict
import json
from batcher import make_batcher
from trainer.prior_trainer import PriorTrainer
from model.fvl import FVL
import wandb
import torch
from util.utils import get_config, set_seed, get_vis_mask, get_avg_tarfile_paths, get_all_subjects_tarfile_paths
from model.diffusion import BrainDiffusionPriorOld
from model import make_model

if __name__ == '__main__':
     
    """Model training according to config.
        -> see get_args() below for all command 
        line arguments.
    """
    
    config = get_config()

    if config['wandb']:
        wandb.init(project="WAVE", name="decode-bold2img", config=config)

    if config['set_seed']:
        set_seed(config['seed'])

    if config['do_train']:
        os.makedirs(
            config["log_dir"],
            exist_ok=True
        )

        config_filepath = os.path.join(
            config["log_dir"],
            'train_config.json'
        )
        
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)

    assert config["training_style"] in {
        'CSM',
    }, f'{config["training_style"]} is not supported.'
    
    assert config["architecture"] in {
        'GPT',
    }, f'{config["architecture"]} is not supported.'

    if 'all' in config['subject_id']:
        train_tarfile_paths, validation_tarfile_paths = get_all_subjects_tarfile_paths(config['avg_data_dir'])
        pad_voxel = True
    else:
        train_tarfile_paths, validation_tarfile_paths = get_avg_tarfile_paths(config['avg_data_dir'], config['subject_id'])
        pad_voxel = False
    
    batcher = make_batcher(
        training_style=config["training_style"],
        sample_random_seq=config["sample_random_seq"],
        seq_min=config["seq_min"],
        seq_max=config["seq_max"],
        bert_seq_gap_min=config["bert_seq_gap_min"],
        bert_seq_gap_max=config["bert_seq_gap_max"],
        decoding_target=config["decoding_target"],
        bold_dummy_mode=config["bold_dummy_mode"],
        pad_voxel=pad_voxel,
    )
    train_dataset = batcher.dataset(
        tarfiles=train_tarfile_paths,
        length=config["training_steps"]*config["per_device_training_batch_size"]
    )
    validation_dataset = batcher.dataset(
        tarfiles=validation_tarfile_paths,
        length=len(validation_tarfile_paths)
    )

    def model_init(params: Dict=None):
        model_config = dict(config)

        if params is not None:
            model_config |= params

        return make_model(model_config)

    clip_size = 768
    # Create fmri encoder
    model = model_init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config["vis_mask_json_path"] != None and config['vis_mask_json_path'] != "None":
        vis_mask = get_vis_mask(config["vis_mask_json_path"], batch_size=config["per_device_training_batch_size"],
                                device=device)
    else:
        vis_mask = None

    if config['hidden']:
        hidden = True
        norm_embs = True
    else:
        hidden = False
        norm_embs = False
    model = FVL(model, 
                is_prompt=config['is_prompt'], 
                sparse_ratio=config['sparse_ratio'],  
                train_mode=config['train_mode'],
                norm_embs=False,
                device=device, 
                mask_vis=vis_mask, 
                hidden_state=False)
    model.from_pretrained(config['pretrained_model'])

    if hidden:
        guidance_scale = 3.5
        timesteps = 100
        out_dim = clip_size
        depth = 6
        dim_head = 64
        heads = clip_size // dim_head
        # clean cache
        torch.cuda.empty_cache()
        from model.models import VersatileDiffusionPriorNetwork,BrainNetwork
        prior_network = VersatileDiffusionPriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = 257,
            learned_query_mode="pos_emb"
            ).to(device)
        print("prior_network loaded")

        from model.clip import Clipper
        model.CLIP=None
        clip_model = Clipper(clip_variant="ViT-L/14", hidden_state=True, norm_embs=True,device=device)
        model.CLIP = clip_model

        dim_mapper = torch.nn.Linear(clip_size, out_dim * 257).to(device)

        from model.diffusion import BrainDiffusionPrior
        # custom version that can fix seeds
        diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
            voxel2clip=model,
            dim_mapper = dim_mapper,
        ).to(device)
    else:
        model.to(device)
        guidance_scale = 7.5
        timesteps = 1000
        diffusion_prior = BrainDiffusionPriorOld.from_pretrained(
            # kwargs for DiffusionPriorNetwork
            dict(),
            # kwargs for DiffusionNetwork
            dict(
                condition_on_text_encodings=False,
                timesteps=timesteps,
                voxel2clip=model,
            ),
            voxel2clip_path=None,
            ckpt_dir=config['prior_checkpoint_dir'],
        )
    
    # model = FV(fmri_encoder, device=device, sparse_ratio=config['sparse_ratio'])
    print("Using {} device".format(device))
    
    from model.models import Clipper
    clip_extractor = Clipper("ViT-L/14", hidden_state=False, norm_embs=True, device=device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if config['train_mode'] == 'region':
        print("Region-Mode: Add parameters")
        opt_grouped_parameters = [
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in diffusion_prior.voxel2clip.fmri_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.voxel2clip.fmri_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if hidden:
            opt_grouped_parameters = [
            {'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in diffusion_prior.voxel2clip.fmri_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in diffusion_prior.voxel2clip.fmri_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in diffusion_prior.dim_mapper.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in diffusion_prior.dim_mapper.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    elif config['train_mode'] == 'voxel':
        print("Voxel-Mode: Add parameters")
        opt_grouped_parameters = [
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in diffusion_prior.voxel2clip.voxel_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.voxel2clip.voxel_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=config["learning_rate"])
    # convert dict to argparse.Namespace
    train_config = {
        "wandb": config["wandb"],
        "train_batch_size": config["per_device_training_batch_size"],
        "eval_batch_size": config["per_device_validation_batch_size"],
        "device": device,
        "learning_rate": config["learning_rate"],
        "optimizer": optimizer,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "num_workers": 4,
        "weight_decay": config["weight_decay"],
        "l1_lambda": config["l1_lambda"],
        "seed": 42,
        "log_steps": config["log_every_n_steps"],
        "train_style": "step",
        "eval_steps": config["eval_every_n_steps"],
        "num_steps": len(train_dataset) // config["per_device_training_batch_size"],
        "log_dir": config["log_dir"],
        "save_steps": 100000,
        "scheduler": config["scheduler"],
        "scheduler_warmup_ratio": config["warmup_ratio"],
        "scheduler_warmup_steps": 0,
        "scheduler_last_epoch": -1,
        "scheduler_step_size": config["scheduler_step_size"],
        "scheduler_gamma": config['scheduler_gamma'],
        "only_visual": config["only_visual"],
        "sd_ckpt_dir": config["sd_ckpt_dir"],
        "guidance_scale": guidance_scale,
        "timesteps": timesteps,
        "log_reconstruction_every_n_steps": config["log_reconstruction_every_n_steps"],
        "n_samples_save": config["n_samples_save"],
        "train_mode": config["train_mode"],
        "hidden": hidden,
        "vd_ckpt_dir": config["vd_ckpt_dir"],
    }
    trainer_args = argparse.Namespace(**train_config)
    trainer = PriorTrainer(
        model=diffusion_prior,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_loss=None,
        clip_extractor=clip_extractor,
    )
    trainer.train()