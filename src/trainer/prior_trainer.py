'''
This is the file to build a standard trainer for the model
1. Load the model
2. Load the dataset
3. Load the trainer
4. Train the model
6. Evaluate the model
7. Log the results
'''
import torch
import os
import wandb
from accelerate import Accelerator
import util.utils as utils

class PriorTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        compute_loss,
        args,
        clip_extractor = None,
        ) -> None:
        self.accelerator = Accelerator()
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_loss = compute_loss
        self.clip_extractor = clip_extractor
        self.train_dataloader = self._get_train_dataloader()
        self.eval_dataloader = self._get_eval_dataloader()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self._prepare_accelerator()
        self._make_reconstruct_modules()
        self.best_eval_loss = float("inf")

    def train(self):
        self.step = 0
        print("Start training")
        print(f"training args: {self.args}")
        print(f"training dataset length: {len(self.train_dataset)}")
        print(f"eval dataset length: {len(self.eval_dataset)}")
        if self.args.train_style == "epoch":
            print(f"training epochs: {self.args.num_epochs}, eval steps: {self.args.eval_steps}")
            for epoch in range(self.args.num_epochs):
                epoch += 1
                self._train_epoch(epoch)
                self._eval_epoch(epoch)
        elif self.args.train_style == "step":
                print(f"training steps: {self.args.num_steps}, eval steps: {self.args.eval_steps}")
                self._train_epoch(epoch=1)

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_voxel0 = train_image0 = None
        # running_fmri_image_loss = 0.0
        # running_fmri_text_loss = 0.0
        # running_b2v_acc = 0.0
        # running_b2l_acc = 0.0
        for batch in self.train_dataloader:
            self.step += 1
            self.optimizer.zero_grad()
            losses = self._compute_loss(batch)
            self.accelerator.backward(losses['loss'])
            self.optimizer.step()
            if train_image0 is None:
                    train_image0 = batch['img'].detach().clone()
                    train_voxel0 = batch.copy()
            running_loss += losses['loss'].item()
            # running_fmri_image_loss += losses['loss_fmri_image']
            # running_fmri_text_loss += losses['loss_fmri_text']
            # running_b2v_acc += losses['b2v_acc']
            # running_b2l_acc += losses['b2l_acc']
            self.scheduler.step()
            if self.step % self.args.log_steps == 0:
                # print(f"Epoch {epoch}, step {self.step}: loss {running_loss/self.args.log_steps} fmri_image_loss {running_fmri_image_loss/self.args.log_steps} fmri_text_loss {running_fmri_text_loss/self.args.log_steps} b2v_acc {running_b2v_acc/self.args.log_steps} b2l_acc {running_b2l_acc/self.args.log_steps} lr {self.optimizer.param_groups[0]['lr']}")
                print(f"Epoch {epoch}, step {self.step}: loss {running_loss/self.args.log_steps}")
                if self.args.wandb:
                    # wandb.log({"epoch": epoch,"train_loss": running_loss/self.args.log_steps,
                    #            "train_fmri_image_loss": running_fmri_image_loss/self.args.log_steps,
                    #            "train_fmri_text_loss": running_fmri_text_loss/self.args.log_steps,
                    #              "train_b2v_acc": running_b2v_acc/self.args.log_steps,
                    #                 "train_b2l_acc": running_b2l_acc/self.args.log_steps,
                    #            "Step": self.step, "lr": self.optimizer.param_groups[0]['lr']})
                    wandb.log({"epoch": epoch,"train_loss": running_loss/self.args.log_steps,
                               "Step": self.step, "lr": self.optimizer.param_groups[0]['lr'],
                               })
                running_loss = 0.0
            if self.step % self.args.log_reconstruction_every_n_steps == 0:
                grid, _, _, _ = self._reconstruct(train_image0, train_voxel0)
                if self.args.wandb:
                    wandb.log({"epoch": epoch, "step": self.step, "train_reconstruction": [wandb.Image(grid, caption=f"Image Label {list(train_voxel0['img_label'][0])[:self.args.n_samples_save]}")]})
                else:
                    grid.savefig(os.path.join(self.args.log_dir, f"train_reconstrction_{self.step}.png"))
                train_image0 = train_voxel0 = None
                # running_fmri_image_loss = 0.0
                # running_fmri_text_loss = 0.0
                # running_b2v_acc = 0.0
                # running_b2l_acc = 0.0

            if self.step % self.args.save_steps == 0:
                self._save_results()
    
            if self.step % self.args.eval_steps == 0:
                self._eval_epoch(epoch)

    def _eval_epoch(self, epoch):
        print("Start evaluating")
        self.model.eval()
        # self.model.voxel2clip.eval()
        val_voxel0 = val_image0 = None
        eval_loss = 0.0
        # eval_fmri_image_loss = 0.0
        # eval_fmri_text_loss = 0.0
        # eval_b2v_acc = 0.0
        # eval_b2l_acc = 0.0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                losses = self._compute_loss(batch)
                if val_image0 is None:
                    val_image0 = batch['img'].detach().clone()
                    val_voxel0 = batch.copy()
                eval_loss += losses['loss'].item() * batch['inputs'].shape[0]
                # eval_fmri_image_loss += losses['loss_fmri_image'] * self.args.eval_batch_size
                # eval_fmri_text_loss += losses['loss_fmri_text'] * self.args.eval_batch_size
                # eval_b2v_acc += losses['b2v_acc'] * self.args.eval_batch_size
                # eval_b2l_acc += losses['b2l_acc'] * self.args.eval_batch_size
        # print(f"Epoch {epoch}, Step {self.step}, eval loss: {eval_loss/len(self.eval_dataset)} eval_fmri_image_loss: {eval_fmri_image_loss/len(self.eval_dataset)} eval_fmri_text_loss: {eval_fmri_text_loss/len(self.eval_dataset)} eval_b2v_acc: {eval_b2v_acc/len(self.eval_dataset)} eval_b2l_acc: {eval_b2l_acc/len(self.eval_dataset)}")
        print(f"Epoch {epoch}, Step {self.step}, eval loss: {eval_loss/len(self.eval_dataset)}")
        grid,_,_,_ = self._reconstruct(val_image0, val_voxel0)
        if self.args.wandb:
            # wandb.log({"epoch": epoch, "step": self.step, "eval_loss": eval_loss/len(self.eval_dataset),
            #            "eval_fmri_image_loss": eval_fmri_image_loss/len(self.eval_dataset),
            #            "eval_fmri_text_loss": eval_fmri_text_loss/len(self.eval_dataset),
            #            "eval_b2v_acc": eval_b2v_acc/len(self.eval_dataset),
            #            "eval_b2l_acc": eval_b2l_acc/len(self.eval_dataset)})
            
            wandb.log({"epoch": epoch, "step": self.step, "eval_loss": eval_loss/len(self.eval_dataset), "eval_reconstruction": [wandb.Image(grid, caption=f"Image Label {list(val_voxel0['img_label'][0])[:self.args.n_samples_save]}")]})

        else:
            grid.savefig(os.path.join(self.args.log_dir, f"eval_reconstrction_{self.step}.png"))
        if self.best_eval_loss > eval_loss/len(self.eval_dataset):
            self.best_eval_loss = eval_loss/len(self.eval_dataset)
            self._save_results(name='best')
        self.model.train()

    def _reconstruct(self, val_image0, val_voxel0, n_samples_save=1):
        grid, brain_recons, best_picks, recon_img = utils.reconstruction(
            val_image0, val_voxel0,
            self.clip_extractor, self.unet, self.vae, self.noise_scheduler,
            diffusion_priors = self.model.module if self.distributed else self.model,
            num_inference_steps = self.num_inference_steps,
            n_samples_save = n_samples_save,
            guidance_scale = self.args.guidance_scale,
            timesteps_prior = self.args.timesteps,
            seed = self.args.seed,
            retrieve = False,
            plotting = True,
            use_image_features=False,
            img_variations = not self.args.hidden,
            verbose=False,
            train_mode=self.args.train_mode,
            recons_per_sample=2,          
        )
        return grid, brain_recons, best_picks, recon_img

    def _compute_loss(self, batch):
        batch = self._move_batch_to_device(batch)
        loss, pred = self.model(batch = batch, train_mode = self.args.train_mode)
        return {'loss': loss * self.prior_mult, 'pred': pred}
    
    def _move_batch_to_device(self, batch):

        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        return batch

    def _get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
    
    def _get_eval_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )

    def _get_optimizer(self):
        if type(self.args.optimizer) == str:
            if self.args.optimizer == "Adam":
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon
                )
            elif self.args.optimizer == "SGD":
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay
                )
            elif self.args.optimizer == "AdamW":
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon
                )
        else:
            assert isinstance(self.args.optimizer, torch.optim.Optimizer)
            optimizer = self.args.optimizer
        return optimizer
    
    def _save_results(self, name = ''):
        print('Saving results')
        self._save_model(name)
    
    def _save_model(self, name = ''):
        os.makedirs(self.args.log_dir, exist_ok=True)
        if len(name) > 0:
            torch.save(self.model.state_dict(), os.path.join(self.args.log_dir, f"model_{name}.bin"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.args.log_dir, "model_{}.bin".format(self.step)))
        print(f"Model saved to {self.args.log_dir}, step {self.step}")
    
    def _get_scheduler(self):
        if self.args.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1 - epoch / self.args.num_epochs,
                last_epoch=-1
            )
        elif self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.num_epochs,
                eta_min=self.args.min_lr
            )
        elif self.args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.scheduler_step_size,
                gamma=self.args.scheduler_gamma
            )
        return scheduler
    
    def _warmup_lr_schedule(self, step):
        if step < self.args.warmup_steps:
            # Linear warmup
            warmup_factor = float(step) / float(max(1, self.args.warmup_steps))
        else:
            # Post warmup: here we keep it constant, but you could implement decay
            warmup_factor = 1
        
        return warmup_factor
    
    def _prepare_accelerator(self):
        self.model, self.train_dataloader, self.eval_dataloader, self.optimizer = self.accelerator.prepare(
            self.model, self.train_dataloader, self.eval_dataloader, self.optimizer
        )
        self.device = self.accelerator.device
        # if multi-gpu
        if self.accelerator.num_processes > 1:
            # change optimizer learning rate
            self.optimizer.param_groups[:]['lr'] = self.optimizer.param_groups[:]['lr'] * self.accelerator.num_processes
            self.distributed = True
        else:
            self.distributed = False

    def _make_reconstruct_modules(self):
        if self.args.hidden:
            print('Creating versatile diffusion reconstruction pipeline...')
            from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
            from diffusers.models import DualTransformer2DModel
            vd_cache_dir = self.args.vd_ckpt_dir
            vd_pipe =  vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
                "shi-labs/versatile-diffusion",
                cache_dir=vd_cache_dir,
                ).to('cpu')
            vd_pipe.image_unet.eval()
            vd_pipe.vae.eval()
            vd_pipe.image_unet.requires_grad_(False)
            vd_pipe.vae.requires_grad_(False)

            vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(
                "shi-labs/versatile-diffusion", 
                subfolder="scheduler",
                cache_dir=vd_cache_dir)
            self.num_inference_steps = 20

            self.prior_mult = 30

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

            self.unet, self.vae, self.noise_scheduler = self.accelerator.prepare(
                vd_pipe.image_unet, vd_pipe.vae, vd_pipe.scheduler
            )
        else:
            print('Creating SD image variations reconstruction pipeline...')
            from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler

            sd_cache_dir = self.args.sd_ckpt_dir
            unet = UNet2DConditionModel.from_pretrained(sd_cache_dir,subfolder="unet").to(self.device)

            unet.eval() # dont want to train model
            unet.requires_grad_(False) # dont need to calculate gradients

            vae = AutoencoderKL.from_pretrained(sd_cache_dir,subfolder="vae").to(self.device)
            vae.eval()
            vae.requires_grad_(False)

            noise_scheduler = UniPCMultistepScheduler.from_pretrained(sd_cache_dir, subfolder="scheduler")
            self.num_inference_steps = 20

            self.prior_mult = .03

            self.unet, self.vae, self.noise_scheduler = self.accelerator.prepare(
                unet, vae, noise_scheduler
            )

    