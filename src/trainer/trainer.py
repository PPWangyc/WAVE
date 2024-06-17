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

class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        compute_loss,
        args
        ) -> None:
        self.model = model.to(args.device)
        model.set_device(args.device)
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_loss = compute_loss
        self.train_dataloader = self._get_train_dataloader()
        self.eval_dataloader = self._get_eval_dataloader()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.best_eval_br2v_acc = 0.0
        self.best_eval_bv2v_acc = 0.0

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
        running_region_image_loss = 0.0
        running_region_text_loss = 0.0
        running_voxel_image_loss = 0.0
        running_voxel_text_loss = 0.0
        running_region_voxel_loss = 0.0
        running_br2v_acc = 0.0
        running_br2l_acc = 0.0
        running_bv2v_acc = 0.0
        running_bv2l_acc = 0.0
        running_br2bv_acc = 0.0
        for batch in self.train_dataloader:
            self.step += 1
            self.optimizer.zero_grad()
            losses = self._compute_loss(batch)
            losses['loss'].backward()
            self.optimizer.step()
            # print(losses['loss'])
            running_loss += losses['loss'].item()
            if 'region' == self.args.train_mode:
                running_region_image_loss += losses['loss_region_image']
                running_region_text_loss += losses['loss_region_text']
                running_br2v_acc += losses['br2v_acc']
                running_br2l_acc += losses['br2l_acc']
            elif 'voxel' == self.args.train_mode:
                running_voxel_image_loss += losses['loss_voxel_image']
                running_voxel_text_loss += losses['loss_voxel_text']
                running_bv2v_acc += losses['bv2v_acc']
                running_bv2l_acc += losses['bv2l_acc']
            elif 'region-voxel' == self.args.train_mode:
                running_region_image_loss += losses['loss_region_image']
                running_region_text_loss += losses['loss_region_text']
                running_voxel_image_loss += losses['loss_voxel_image']
                running_voxel_text_loss += losses['loss_voxel_text']
                running_region_voxel_loss += losses['loss_region_voxel']
                running_br2bv_acc += losses['br2bv_acc']
                running_br2v_acc += losses['br2v_acc']
                running_br2l_acc += losses['br2l_acc']
                running_bv2v_acc += losses['bv2v_acc']
                running_bv2l_acc += losses['bv2l_acc']
            else:
                raise ValueError("Wrong train mode")
            self.scheduler.step()
            if self.step % self.args.log_steps == 0:
                if 'region' == self.args.train_mode:
                    print(f"Epoch {epoch}, step {self.step}: loss {running_loss/self.args.log_steps} region_image_loss {running_region_image_loss/self.args.log_steps} region_text_loss {running_region_text_loss/self.args.log_steps} br2v_acc {running_br2v_acc/self.args.log_steps} br2l_acc {running_br2l_acc/self.args.log_steps} lr {self.optimizer.param_groups[0]['lr']}")
                elif 'voxel' == self.args.train_mode:
                    print(f"Epoch {epoch}, step {self.step}: loss {running_loss/self.args.log_steps} voxel_image_loss {running_voxel_image_loss/self.args.log_steps} voxel_text_loss {running_voxel_text_loss/self.args.log_steps} bv2v_acc {running_bv2v_acc/self.args.log_steps} bv2l_acc {running_bv2l_acc/self.args.log_steps} lr {self.optimizer.param_groups[0]['lr']}")
                elif 'region-voxel' == self.args.train_mode:
                    print(f"Epoch {epoch}, step {self.step}: loss {running_loss/self.args.log_steps} region_image_loss {running_region_image_loss/self.args.log_steps} region_text_loss {running_region_text_loss/self.args.log_steps} voxel_image_loss {running_voxel_image_loss/self.args.log_steps} voxel_text_loss {running_voxel_text_loss/self.args.log_steps} region_voxel_loss {running_region_voxel_loss/self.args.log_steps} br2bv_acc {running_br2bv_acc/self.args.log_steps} br2v_acc {running_br2v_acc/self.args.log_steps} br2l_acc {running_br2l_acc/self.args.log_steps} bv2v_acc {running_bv2v_acc/self.args.log_steps} bv2l_acc {running_bv2l_acc/self.args.log_steps} lr {self.optimizer.param_groups[0]['lr']}")
                if self.args.wandb:
                    wandb.log({"epoch": epoch,"train_loss": running_loss/self.args.log_steps, 
                               "train_region_image_loss": running_region_image_loss/self.args.log_steps, 
                               "train_region_text_loss": running_region_text_loss/self.args.log_steps,
                               "train_voxel_image_loss": running_voxel_image_loss/self.args.log_steps,
                                "train_voxel_text_loss": running_voxel_text_loss/self.args.log_steps,
                                "train_region_voxel_loss": running_region_voxel_loss/self.args.log_steps,
                                 "train_br2v_acc": running_br2v_acc/self.args.log_steps,
                                    "train_br2l_acc": running_br2l_acc/self.args.log_steps,
                                    "train_bv2v_acc": running_bv2v_acc/self.args.log_steps,
                                    "train_bv2l_acc": running_bv2l_acc/self.args.log_steps,
                                    "train_br2bv_acc": running_br2bv_acc/self.args.log_steps,
                               "Step": self.step, "lr": self.optimizer.param_groups[0]['lr']})
                running_loss = 0.0
                running_region_image_loss = 0.0
                running_region_text_loss = 0.0
                running_voxel_image_loss = 0.0
                running_voxel_text_loss = 0.0
                running_region_voxel_loss = 0.0
                running_br2bv_acc = 0.0
                running_bv2v_acc = 0.0
                running_bv2l_acc = 0.0
                running_br2v_acc = 0.0
                running_br2l_acc = 0.0

            if self.step % self.args.save_steps == 0:
                self._save_results()
    
            if self.step % self.args.eval_steps == 0:
                self._eval_epoch(epoch)

    def _eval_epoch(self, epoch):
        print("Start evaluating")
        self.model.eval()
        eval_loss = 0.0
        eval_region_image_loss = 0.0
        eval_region_text_loss = 0.0
        eval_voxel_image_loss = 0.0
        eval_voxel_text_loss = 0.0
        eval_region_voxel_loss = 0.0
        eval_br2v_acc = 0.0
        eval_br2l_acc = 0.0
        eval_bv2v_acc = 0.0
        eval_bv2l_acc = 0.0
        eval_br2bv_acc = 0.0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                losses = self._compute_loss(batch)
                eval_loss += losses['loss'].item() * batch['inputs'].shape[0]
                if 'region' == self.args.train_mode:
                    eval_region_image_loss += losses['loss_region_image'] * batch['inputs'].shape[0]
                    eval_region_text_loss += losses['loss_region_text'] * batch['inputs'].shape[0]
                    eval_br2v_acc += losses['br2v_acc'] * batch['inputs'].shape[0]
                    eval_br2l_acc += losses['br2l_acc'] * batch['inputs'].shape[0]

                elif 'voxel' == self.args.train_mode:
                    eval_voxel_image_loss += losses['loss_voxel_image'] * batch['inputs'].shape[0]
                    eval_voxel_text_loss += losses['loss_voxel_text'] * batch['inputs'].shape[0]
                    eval_bv2v_acc += losses['bv2v_acc'] * batch['inputs'].shape[0]
                    eval_bv2l_acc += losses['bv2l_acc'] * batch['inputs'].shape[0]
                
                elif 'region-voxel' == self.args.train_mode:
                    eval_region_image_loss += losses['loss_region_image'] * batch['inputs'].shape[0]
                    eval_region_text_loss += losses['loss_region_text'] * batch['inputs'].shape[0]
                    eval_voxel_image_loss += losses['loss_voxel_image'] * batch['inputs'].shape[0]
                    eval_voxel_text_loss += losses['loss_voxel_text'] * batch['inputs'].shape[0]
                    eval_region_voxel_loss += losses['loss_region_voxel'] * batch['inputs'].shape[0]
                    eval_br2bv_acc += losses['br2bv_acc'] * batch['inputs'].shape[0]
                    eval_br2v_acc += losses['br2v_acc'] * batch['inputs'].shape[0]
                    eval_br2l_acc += losses['br2l_acc'] * batch['inputs'].shape[0]
                    eval_bv2v_acc += losses['bv2v_acc'] * batch['inputs'].shape[0]
                    eval_bv2l_acc += losses['bv2l_acc'] * batch['inputs'].shape[0]

                else:
                    raise ValueError("Wrong train mode")
        if 'region' == self.args.train_mode:
            print(f"Epoch {epoch}, Step {self.step}, eval loss: {eval_loss/len(self.eval_dataset)} eval_region_image_loss: {eval_region_image_loss/len(self.eval_dataset)} eval_region_text_loss: {eval_region_text_loss/len(self.eval_dataset)} eval_br2v_acc: {eval_br2v_acc/len(self.eval_dataset)} eval_br2l_acc: {eval_br2l_acc/len(self.eval_dataset)}")
        elif 'voxel' == self.args.train_mode:
            print(f"Epoch {epoch}, Step {self.step}, eval loss: {eval_loss/len(self.eval_dataset)} eval_voxel_image_loss: {eval_voxel_image_loss/len(self.eval_dataset)} eval_voxel_text_loss: {eval_voxel_text_loss/len(self.eval_dataset)} eval_bv2v_acc: {eval_bv2v_acc/len(self.eval_dataset)} eval_bv2l_acc: {eval_bv2l_acc/len(self.eval_dataset)}")
        elif 'region-voxel' == self.args.train_mode:
            print(f"Epoch {epoch}, Step {self.step}, eval loss: {eval_loss/len(self.eval_dataset)} eval_region_image_loss: {eval_region_image_loss/len(self.eval_dataset)} eval_region_text_loss: {eval_region_text_loss/len(self.eval_dataset)} eval_voxel_image_loss: {eval_voxel_image_loss/len(self.eval_dataset)} eval_voxel_text_loss: {eval_voxel_text_loss/len(self.eval_dataset)} eval_region_voxel_loss: {eval_region_voxel_loss/len(self.eval_dataset)} eval_br2bv_acc: {eval_br2bv_acc/len(self.eval_dataset)} eval_br2v_acc: {eval_br2v_acc/len(self.eval_dataset)} eval_br2l_acc: {eval_br2l_acc/len(self.eval_dataset)} eval_bv2v_acc: {eval_bv2v_acc/len(self.eval_dataset)} eval_bv2l_acc: {eval_bv2l_acc/len(self.eval_dataset)}")
        if self.best_eval_br2v_acc < eval_br2v_acc/len(self.eval_dataset) and 'region' in self.args.train_mode:
            self.best_eval_br2v_acc = eval_br2v_acc/len(self.eval_dataset)
            self._save_results(name='best')
        elif self.best_eval_bv2v_acc < eval_bv2v_acc/len(self.eval_dataset):
            self.best_eval_bv2v_acc = eval_bv2v_acc/len(self.eval_dataset)
            self._save_results(name='best')
        if self.args.wandb:
            wandb.log({"epoch": epoch, "step": self.step, "eval_loss": eval_loss/len(self.eval_dataset),
                       "eval_region_image_loss": eval_region_image_loss/len(self.eval_dataset),
                       "eval_region_text_loss": eval_region_text_loss/len(self.eval_dataset),
                          "eval_voxel_image_loss": eval_voxel_image_loss/len(self.eval_dataset),
                            "eval_voxel_text_loss": eval_voxel_text_loss/len(self.eval_dataset),
                            "eval_region_voxel_loss": eval_region_voxel_loss/len(self.eval_dataset),
                            "eval_br2bv_acc": eval_br2bv_acc/len(self.eval_dataset),
                          "eval_bv2v_acc": eval_bv2v_acc/len(self.eval_dataset),
                            "eval_bv2l_acc": eval_bv2l_acc/len(self.eval_dataset),
                       "eval_br2v_acc": eval_br2v_acc/len(self.eval_dataset),
                       "eval_br2l_acc": eval_br2l_acc/len(self.eval_dataset)})
        self.model.train()

    def _compute_loss(self, batch):
        batch = self._move_batch_to_device(batch)
        outputs = self.model(batch)
        loss = self.compute_loss(outputs, self.args.only_visual, self.args.train_mode)
        return loss
    
    def _move_batch_to_device(self, batch):

        batch = {
            key: value.to(self.args.device) if isinstance(value, torch.Tensor) else value
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
        return optimizer
    
    def _save_results(self, name = ''):
        print('Saving results')
        self._save_model(name=name)
    
    def _save_model(self,name=''):
        os.makedirs(self.args.log_dir, exist_ok=True)
        if len(name) > 0:
            torch.save(self.model.state_dict(), os.path.join(self.args.log_dir, "model_{}.bin".format(name)))
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

    