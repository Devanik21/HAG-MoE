import torch
import torch.nn as nn
from tqdm import tqdm
from .losses import HAGMoELoss
from .schedulers import HAGMoEScheduler

class HAGMoETrainer:
    def __init__(self, model, optimizer, train_dataloader, val_dataloader=None,
                 lr_scheduler=None, num_groups=4, experts_per_group=4,
                 device='cuda' if torch.cuda.is_available() else 'cpu', max_grad_norm=1.0, log_fn=print):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_fn = log_fn

        self.criterion = HAGMoELoss(num_groups, experts_per_group)
        self.lambda_scheduler = HAGMoEScheduler()
        self.global_step = 0

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch['input_ids'].to(self.device)
        if 'labels' in batch:
            targets = batch['labels'].to(self.device)
        else:
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(self.device)
            if 'labels' not in batch and mask.size(-1) > input_ids.size(-1):
                mask = mask[:, :, :-1, :-1]

        logits, all_aux_data = self.model(input_ids, mask)
        lambdas = self.lambda_scheduler.get_lambdas()

        seq_len = min(logits.size(1), targets.size(1))
        logits = logits[:, :seq_len, :]
        targets = targets[:, :seq_len]

        loss, loss_dict = self.criterion(logits, targets, all_aux_data, **lambdas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.lambda_scheduler.step()
        self.global_step += 1

        return loss_dict

    def train_epoch(self):
        epoch_losses = {}
        num_batches = 0
        pbar = tqdm(self.train_dataloader, desc="Training")
        for batch in pbar:
            loss_dict = self.train_step(batch)
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            num_batches += 1
            if num_batches % 10 == 0:
                pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}", 'lm': f"{loss_dict['lm']:.4f}"})

        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        return epoch_losses

    def evaluate(self):
        if self.val_dataloader is None:
            return {}
        self.model.eval()
        val_losses = {}
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                if 'labels' in batch:
                    targets = batch['labels'].to(self.device)
                else:
                    targets = input_ids[:, 1:].contiguous()
                    input_ids = input_ids[:, :-1].contiguous()
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)
                    if 'labels' not in batch and mask.size(-1) > input_ids.size(-1):
                        mask = mask[:, :, :-1, :-1]

                logits, all_aux_data = self.model(input_ids, mask)
                seq_len = min(logits.size(1), targets.size(1))
                logits = logits[:, :seq_len, :]
                targets = targets[:, :seq_len]
                lambdas = self.lambda_scheduler.get_lambdas()

                _, loss_dict = self.criterion(logits, targets, all_aux_data, **lambdas)
                for k, v in loss_dict.items():
                    val_losses[k] = val_losses.get(k, 0.0) + v
                num_batches += 1

        for k in val_losses:
            val_losses[k] /= max(1, num_batches)
        return val_losses

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.log_fn(f"Epoch {epoch+1}/{num_epochs}")
            train_losses = self.train_epoch()
            self.log_fn(f"Train Losses: {train_losses}")
            if self.val_dataloader is not None:
                val_losses = self.evaluate()
                self.log_fn(f"Val Losses: {val_losses}")
