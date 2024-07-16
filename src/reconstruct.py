import torch
from model.fvl import FVL
import PIL
import os
import scipy as sp
from typing import Dict
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from batcher import make_batcher
from model.fvl import FVL
import wandb
import torch
from util.utils import (get_config, set_seed, 
                        reconstruction, 
                        torch_to_Image, 
                        get_avg_tarfile_paths, 
                        get_vis_mask,
                        get_voxel_num,
                        get_all_subjects_tarfile_paths)
from model.diffusion import BrainDiffusionPriorOld
from model import make_model
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler


if __name__ == '__main__':
     
    """Model training according to config.
        -> see get_args() below for all command 
        line arguments.
    """
    
    config = get_config()

    if config['wandb']:
        wandb.init(project="WAVE", name="reconstruct-{}-{}".format(config['subject_id'], config['train_mode']), config=config)

    if config['set_seed']:
        set_seed(config['seed'])

    if config['do_train']:
        os.makedirs(
            config["log_dir"],
            exist_ok=True
        )

        resume_path = str(config["resume_from"]) if config["resume_from"] is not None else None
        
        config_filepath = os.path.join(
            config["log_dir"],
            'train_config.json'
        )
            
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)

        config["resume_from"] = None

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
        train_tarfile_paths, validation_tarfile_paths = get_avg_tarfile_paths(config['avg_data_dir'], 
                                                                              config['subject_id'])
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
        seed=config["seed"] if config['set_seed'] else np.random.choice(range(1, 100000)),
        pad_voxel=pad_voxel
    )

    train_dataset = batcher.dataset(
        tarfiles=train_tarfile_paths,
        length=len(train_tarfile_paths)
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

    # Create fmri encoder
    model = model_init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config['vis_mask_json_path'] != None and config['vis_mask_json_path'] != "None":
        vis_mask = get_vis_mask(config["vis_mask_json_path"], batch_size=config["per_device_training_batch_size"],
                                device=device)
    else:
        vis_mask = None

    hidden = config['hidden']
    if hidden:
        norm_embs = False
    else:
        norm_embs = True
    
    img2img_strength = 1
    train_mode = config['train_mode']
    if 'region' in config['trained_checkpoint_dir'] or 'region' in train_mode:
        num_voxels = 1024 * 5
    else:
        if "all" in config['subject_id']:
            num_voxels = 2787
        else:
            num_voxels,_ = get_voxel_num(config['subject_id'])

    autoenc_ckpt_path = config['autoenc_ckpt_dir']
    if os.path.exists(autoenc_ckpt_path) and 'full' in train_mode:
        from model.models import Voxel2StableDiffusionModel
        checkpoint = torch.load(autoenc_ckpt_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        in_dim = num_voxels
        if "low-v" in train_mode:
            if "all" in config['subject_id']:
                in_dim = 2787
            else:
                in_dim, _ = get_voxel_num(config['subject_id'])
        voxel2sd = Voxel2StableDiffusionModel(in_dim=in_dim)

        voxel2sd.load_state_dict(state_dict,strict=False)
        voxel2sd.eval()
        voxel2sd.to(device)
        print("Loaded low-level model!")
        img2img_strength = 0.85
    else:
        print("No valid path for low-level model specified; not using img2img!") 
    if "mindeye" in config['trained_checkpoint_dir']:
        from model.models import BrainNetwork
        out_dim = clip_size = 768
        out_dim = clip_size * 257 if hidden else out_dim
        voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim,clip_size=clip_size,use_projector=True)
        model = BrainNetwork(**voxel2clip_kwargs)
        
    else:
        model = FVL(model, is_prompt=config['is_prompt'], sparse_ratio=config['sparse_ratio'], 
                norm_embs=norm_embs, mask_vis=vis_mask, hidden_state=hidden,
                device=device)
    
    # model.from_pretrained(config['pretrained_model'])
    # fmri_encoder, _ = model.get_fmri_encoder()
    # model = FV(fmri_encoder, device=device, sparse_ratio=config['sparse_ratio'])
    print("Using {} device".format(device))
    model.to(device)
    # clip_extractor = None
    from model.models import Clipper
    if "mindeye" in config['trained_checkpoint_dir']:
        clip_extractor = Clipper("ViT-L/14", hidden_state=hidden, norm_embs=True, device=device)
    else:
        clip_extractor = Clipper("ViT-L/14", hidden_state=False, norm_embs=True, device=device)
    if hidden:
        guidance_scale = 3.5
        timesteps = 100
        clip_size = 768
        out_dim = clip_size
        depth = 6
        dim_head = 64
        heads = clip_size // dim_head
        # clean cache
        torch.cuda.empty_cache()
        from model.models import VersatileDiffusionPriorNetwork
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
        if 'mindeye' in config['trained_checkpoint_dir']:
            from model.models import BrainDiffusionPrior
            diffusion_prior = BrainDiffusionPrior(
                net=prior_network,
                image_embed_dim=out_dim,
                condition_on_text_encodings=False,
                timesteps=timesteps,
                cond_drop_prob=0.2,
                image_embed_scale=None,
                voxel2clip=model,
            ).to(device)
        else:
            from model.diffusion import BrainDiffusionPrior
            dim_mapper = torch.nn.Linear(768, out_dim * 257)
            # custom version that can fix seeds
            diffusion_prior = BrainDiffusionPrior(
                net=prior_network,
                image_embed_dim=out_dim,
                condition_on_text_encodings=False,
                timesteps=timesteps,
                cond_drop_prob=0.2,
                image_embed_scale=None,
                voxel2clip=model,
                dim_mapper=dim_mapper,
            ).to(device)
        # clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
    else:
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

    checkpoint = torch.load(config['trained_checkpoint_dir'])
    print('Loading pretrained prior + fmri encoder checkpoint from {}'.format(config['trained_checkpoint_dir']))
    if "mindeye" in config['trained_checkpoint_dir']:
        checkpoint = checkpoint['model_state_dict']
        # train_mode = 'mindeye'
    diffusion_prior.load_state_dict(checkpoint)
    diffusion_prior.to(device)
    if hidden:
        print('Creating versatile diffusion reconstruction pipeline...')
        from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
        from diffusers.models import DualTransformer2DModel
        vd_cache_dir = config['vd_ckpt_dir']
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            cache_dir=vd_cache_dir
            ).to('cpu')
        vd_pipe.image_unet.eval()
        vd_pipe.vae.eval()
        vd_pipe.image_unet.requires_grad_(False)
        vd_pipe.vae.requires_grad_(False)
        vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(
            "shi-labs/versatile-diffusion", 
            subfolder="scheduler",
            cache_dir=vd_cache_dir)

        # Set weighting of Dual-Guidance 
        text_image_ratio = .0 # .5 means equally weight text and image, 0 means use only image
        for name, module in vd_pipe.image_unet.named_modules():
            if isinstance(module, DualTransformer2DModel):
                module.mix_ratio = text_image_ratio
                for i, type in enumerate(("text", "image")):
                    if type == "text":
                        module.condition_lengths[i] = 77
                        module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                    else:
                        module.condition_lengths[i] = 257
                        module.transformer_index_for_condition[i] = 0  # use the first (image) transformer
                        
        unet = vd_pipe.image_unet.to(device)
        vae = vd_pipe.vae.to(device)
        noise_scheduler = vd_pipe.scheduler
    else:
        print('Creating SD image variations reconstruction pipeline...')
        print(config['sd_ckpt_dir'])
        sd_cache_dir = config['sd_ckpt_dir']
        unet = UNet2DConditionModel.from_pretrained(sd_cache_dir,subfolder="unet").to(device)

        unet.eval() # dont want to train model
        unet.requires_grad_(False) # dont need to calculate gradients

        vae = AutoencoderKL.from_pretrained(sd_cache_dir,subfolder="vae").to(device)
        vae.eval()
        vae.requires_grad_(False)

        noise_scheduler = UniPCMultistepScheduler.from_pretrained(sd_cache_dir, subfolder="scheduler")
    num_inference_steps = 20
    if config['save_recon_embds']:
        # cancat validation dataset and train dataset
        from torch.utils.data import ConcatDataset
        validation_dataset = ConcatDataset([train_dataset, validation_dataset])
    print('Creating dataloader..., the length of validation dataset is {}'.format(len(validation_dataset)))
    eval_data_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
    )
    diffusion_prior.eval()

    only_lowlevel = False
    if img2img_strength == 1:
        img2img = False
    elif img2img_strength == 0:
        img2img = True
        only_lowlevel = True
    else:
        img2img = True

    best_recon_imgs = []
    origin_imgs = []
    img_names = []
    img_labels = []
    grids = []
    # reconstruction one-at-a-time starts here
    with torch.no_grad(): 
        for i, batch in enumerate(eval_data_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            img_names.append(batch['img_name'][0][0])
            img_labels.append(batch['img_label'][0][0])
            if config['save_recon_embds']:
                recon_embds = reconstruction(
                    batch['img'], batch,
                    None, unet, vae, noise_scheduler, diffusion_priors =diffusion_prior,
                    num_inference_steps=num_inference_steps, n_samples_save= config['n_samples_save'], 
                    guidance_scale=guidance_scale, timesteps_prior=timesteps, seed = config['seed'],
                    retrieve=False, plotting=True, img_variations=not hidden,
                    verbose=False,
                    use_image_features=False,
                    return_recons_emb=True
                )
                # save embeds to .npy file
                # make dir if not exist
                os.makedirs(os.path.join(config['log_dir'],'recon_embds'), exist_ok=True)
                np.save(os.path.join(config['log_dir'],'recon_embds', '{}.npy'.format(batch['img_name'][0][0])), recon_embds.cpu().numpy())
                continue
            else:
                if img2img:
                    if "low-r" in train_mode:
                        ae_preds = voxel2sd(batch['inputs'].squeeze(1)[:,:5].flatten(1).float())
                        print(ae_preds.shape)
                    elif "low-v" in train_mode:
                        ae_preds = voxel2sd(batch['voxels'].squeeze(1).float())
                    blurry_recons = vae.decode(ae_preds.to(device)/0.18215).sample / 2 + 0.5

                else:
                    blurry_recons = None
                if only_lowlevel:
                    recon_embds = blurry_recons
                else:
                    grid, brain_recons, best_picks, recon_img = reconstruction(
                        batch['img'], batch,
                        clip_extractor, unet, vae, noise_scheduler, diffusion_priors =diffusion_prior,
                        num_inference_steps=num_inference_steps, n_samples_save= config['n_samples_save'], 
                        guidance_scale=guidance_scale, timesteps_prior=timesteps, seed = config['seed'],
                        recons_per_sample=config['recons_per_sample'],
                        retrieve=False, 
                        plotting=True, 
                        img_variations=not hidden,
                        verbose=False,
                        use_image_features=False, 
                        img2img_strength=img2img_strength, 
                        img_lowlevel=blurry_recons,
                        train_mode=train_mode
                    )
            print('The best pick is {}'.format(best_picks[0]))
            # convert tensor to pil image
            best_recon_img = brain_recons[0][best_picks[0]]
            origin_img = batch['img'].squeeze(1)[0]
            best_recon_imgs.append(best_recon_img)
            origin_imgs.append(origin_img)
            grids.append(grid)
    
    # sort the images by their img_names
    img_names = np.array(img_names)
    img_labels = np.array(img_labels)

    sort_idx = np.argsort(img_names)
    img_names = img_names[sort_idx]
    img_labels = img_labels[sort_idx]
    
    best_recon_imgs = torch.stack(best_recon_imgs).to(device)
    origin_imgs = torch.stack(origin_imgs).to(device)
    best_recon_imgs = best_recon_imgs[sort_idx]
    origin_imgs = origin_imgs[sort_idx]

    grids = np.array(grids)
    grids = grids[sort_idx]

    for i in range(len(best_recon_imgs)):
        _best_recon_img = torch_to_Image(best_recon_imgs[i])
        _origin_img = torch_to_Image(origin_imgs[i])
        _grid = grids[i]
        _grid.savefig(os.path.join(config['log_dir'], 'recon_img_{}.png'.format(i)))
        _grid = PIL.Image.open(os.path.join(config['log_dir'], 'recon_img_{}.png'.format(i)))
        if config['wandb']:
            wandb.log({"grid": [wandb.Image(_grid, caption=f"{img_names[i]}: {img_labels[i]}")],
                        "recon_img": [wandb.Image(_best_recon_img, caption=f"{img_names[i]}: {img_labels[i]}")],
                        "origin_img": [wandb.Image(_origin_img, caption=f"{img_names[i]}: {img_labels[i]}")],
                    })
    # start evaluation
    # calculate FID
    from model.fid_wrapper import fid_wrapper, calculate_snr
    fid_wrapper = fid_wrapper()
    fid = fid_wrapper(best_recon_imgs, origin_imgs)
    print("FID: ", fid)
    # calculate SNR
    snr = calculate_snr(best_recon_imgs, origin_imgs)
    print("SNR: ", snr)

    # calculate accuracy
    from util.metrics import n_way_top_k_acc
    from torchvision.models import ViT_H_14_Weights, vit_h_14
    weights = ViT_H_14_Weights.DEFAULT
    vit_model = vit_h_14(weights=weights)
    vit_model.to(device)
    vit_model.eval()
    preprocess = weights.transforms()
    acc_list = []
    std_list = []
    for i in range (len(best_recon_imgs)):
        image, recon_image = preprocess(origin_imgs[i].unsqueeze(0)).to(device), preprocess(best_recon_imgs[i].unsqueeze(0)).to(device)
        recon_image_out = vit_model(recon_image).squeeze(0).softmax(0).detach()
        gt_class_id = vit_model(image).squeeze(0).softmax(0).argmax().item()
        acc, std = n_way_top_k_acc(recon_image_out, gt_class_id, 50, 1000, 1)
        acc_list.append(acc)
        std_list.append(std)
    print("mean acc: {}, std acc: {}".format(np.mean(acc_list), np.std(acc_list)))
    if config['wandb']:
        wandb.log({"mean_fid": fid, "mean_snr": snr, "mean_acc": np.mean(acc_list), "std_acc": np.std(acc_list)})

    from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

    # 2-way identification
    @torch.no_grad()
    def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
        preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
        reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
        if feature_layer is None:
            preds = preds.float().flatten(1).cpu().numpy()
            reals = reals.float().flatten(1).cpu().numpy()
        else:
            preds = preds[feature_layer].float().flatten(1).cpu().numpy()
            reals = reals[feature_layer].float().flatten(1).cpu().numpy()

        r = np.corrcoef(reals, preds)
        r = r[:len(all_images), len(all_images):]
        congruents = np.diag(r)

        success = r < congruents
        success_cnt = np.sum(success, 0)

        if return_avg:
            perf = np.mean(success_cnt) / (len(all_images)-1)
            return perf, success_cnt / (len(all_images)-1)
        else:
            return success_cnt, len(all_images)-1
    # PixCorr 
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    all_images = origin_imgs
    all_brain_recons = best_recon_imgs
    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = preprocess(all_brain_recons).view(len(all_brain_recons), -1).cpu()

    print(all_images_flattened.shape)
    print(all_brain_recons_flattened.shape)

    results_df = pd.DataFrame()
    # column: img_name, img_label, pixcorr, ssim, alexnet2, alexnet5, inception, clip, effnet, swav, acc

    results_df['img_name'] = img_names
    results_df['img_label'] = img_labels
    results_df['50-way-top-1_acc'] = np.array(acc_list)
    
    corrsum = 0
    pixcorr_list = []
    for i in range(len(all_images_flattened)):
        corr = np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
        corrsum += corr
        pixcorr_list.append(corr)

    corrmean = corrsum / len(all_images_flattened)
    results_df['pixcorr'] = np.array(pixcorr_list)
    pixcorr = corrmean
    print(pixcorr)

    # see https://github.com/zijin-gu/meshconv-decoding/issues/3
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim

    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])

    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())
    print("converted, now calculating ssim...")

    ssim_score=[]
    for im,rec in zip(img_gray,recon_gray):
        ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

    ssim = np.mean(ssim_score)
    print(ssim)
    results_df['ssim'] = np.array(ssim_score)
        
    from torchvision.models import alexnet, AlexNet_Weights
    alex_weights = AlexNet_Weights.IMAGENET1K_V1

    alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
    alex_model.eval().requires_grad_(False)

    # see alex_weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    layer = 'early, AlexNet(2)'
    print(f"\n---{layer}---")
    all_per_correct, indiv_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess, 'features.4')
    results_df['alexnet2'] = np.array(indiv_per_correct)
    alexnet2 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet2:.4f}")

    layer = 'mid, AlexNet(5)'
    print(f"\n---{layer}---")
    all_per_correct, indiv_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess, 'features.11')
    results_df['alexnet5'] = np.array(indiv_per_correct)
    alexnet5 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet5:.4f}")

    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                            return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    all_per_correct, indiv_per_correct = two_way_identification(all_brain_recons, all_images,
                                            inception_model, preprocess, 'avgpool')
    results_df['inception'] = np.array(indiv_per_correct)
    inception = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {inception:.4f}")

    import clip
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    all_per_correct, indiv_per_correct = two_way_identification(all_brain_recons, all_images,
                                            clip_model.encode_image, preprocess, None) # final layer
    clip_ = np.mean(all_per_correct)
    results_df['clip'] = np.array(indiv_per_correct)
    print(f"2-way Percent Correct: {clip_:.4f}")

    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                        return_nodes=['avgpool']).to(device)
    eff_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = eff_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = eff_model(preprocess(all_brain_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    print("Distance:",effnet)
    results_df['effnet'] = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))])

    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_model, 
                                        return_nodes=['avgpool']).to(device)
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = swav_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = swav_model(preprocess(all_brain_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    print("Distance:",swav)
    results_df['swav'] = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))])

    # Create a dictionary to store variable names and their corresponding values
data = {
    "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV"],
    "Value": [pixcorr, ssim, alexnet2, alexnet5, inception, clip_, effnet, swav],
}
print("-----------------------")
df = pd.DataFrame(data)
print(df.to_string(index=False))

print(results_df)
if config['wandb']:
    wandb.log({"PixCorr": pixcorr, 
               "SSIM": ssim, 
               "AlexNet(2)": alexnet2, 
               "AlexNet(5)": alexnet5, 
               "InceptionV3": inception, 
               "CLIP": clip_, 
               "EffNet-B": effnet, 
               "SwAV": swav})
    # log the results_df
    wandb.log({"results_df": wandb.Table(dataframe=results_df)})
else:
    print("save results_df to csv")
    results_df.to_csv(os.path.join(config['log_dir'], 'results_df.csv'), index=False)