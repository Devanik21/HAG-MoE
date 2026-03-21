#!/usr/bin/env python3
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from hagmoe.core.model import HAGMoETransformer
from hagmoe.training.trainer import HAGMoETrainer
from hagmoe.training.schedulers import get_lr_scheduler

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def dummy_dataloader(batch_size, seq_len, vocab_size, num_batches=100):
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return num_batches * batch_size
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, vocab_size, (seq_len,)),
                'labels': torch.randint(0, vocab_size, (seq_len,))
            }
    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)

def main():
    parser = argparse.ArgumentParser(description="Train HAG-MoE Model")
    parser.add_argument('--config', type=str, default='configs/small.yaml')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    model = HAGMoETransformer(
        vocab_size=config['vocab_size'], d_model=config['d_model'], num_layers=config['num_layers'],
        num_heads=config['num_heads'], num_groups=config['num_groups'], experts_per_group=config['experts_per_group'],
        hidden_dim=config['hidden_dim'], k_min=config['k_min'], k_max=config['k_max'],
        alpha=config.get('alpha', 10.0), d_r=config.get('d_r', None), dropout=config.get('dropout', 0.1),
        max_seq_len=config['max_seq_len']
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=config['warmup_steps'], max_steps=config['max_steps'])

    train_dl = dummy_dataloader(config['batch_size'], config['max_seq_len'] // 8, config['vocab_size'], num_batches=50)
    val_dl = dummy_dataloader(config['batch_size'], config['max_seq_len'] // 8, config['vocab_size'], num_batches=10)

    trainer = HAGMoETrainer(
        model=model, optimizer=optimizer, train_dataloader=train_dl, val_dataloader=val_dl,
        lr_scheduler=lr_scheduler, num_groups=config['num_groups'], experts_per_group=config['experts_per_group'],
        device=args.device
    )
    trainer.train(num_epochs=args.epochs)
    print("Training complete!")

if __name__ == "__main__":
    main()
