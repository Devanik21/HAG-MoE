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
    with open(config_path, 'r') as f: return yaml.safe_load(f)

def modify_config_for_ablation(config, variant):
    if variant == "HAG-MoE (full)": pass
    elif variant == "HAG-MoE-noFB":
        config['lambda_gamma'] = 0.0
        config['disable_feedback_learning'] = True
    elif variant == "HAG-MoE-fixK":
        config['k_min'] = 2
        config['k_max'] = 2
    elif variant == "HAG-MoE-fixK-noFB":
        config['k_min'] = 2
        config['k_max'] = 2
        config['lambda_gamma'] = 0.0
        config['disable_feedback_learning'] = True
    else: print(f"Warning: Variant '{variant}' not fully supported via config alone.")
    return config

def main():
    parser = argparse.ArgumentParser(description="Run HAG-MoE Ablations")
    parser.add_argument('--config', type=str, default='configs/small.yaml')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    variants = ["HAG-MoE (full)", "HAG-MoE-noFB", "HAG-MoE-fixK", "HAG-MoE-fixK-noFB"]
    base_config = load_config(args.config)
    print(f"Running automated ablations on device: {args.device}")
    print(f"Base config: {args.config}\n" + "-" * 50)

    def get_dl():
        class DummyDS(torch.utils.data.Dataset):
            def __len__(self): return 100 * base_config['batch_size']
            def __getitem__(self, idx):
                seq_len = base_config['max_seq_len'] // 8
                return {'input_ids': torch.randint(0, base_config['vocab_size'], (seq_len,)), 'labels': torch.randint(0, base_config['vocab_size'], (seq_len,))}
        return DataLoader(DummyDS(), batch_size=base_config['batch_size'])

    for variant in variants:
        print(f"\nTraining variant: {variant}\n" + "=" * 40)
        config = modify_config_for_ablation(base_config.copy(), variant)

        model = HAGMoETransformer(
            vocab_size=config['vocab_size'], d_model=config['d_model'], num_layers=config['num_layers'],
            num_heads=config['num_heads'], num_groups=config['num_groups'], experts_per_group=config['experts_per_group'],
            hidden_dim=config['hidden_dim'], k_min=config['k_min'], k_max=config['k_max'],
            alpha=config.get('alpha', 10.0), d_r=config.get('d_r', None), dropout=config.get('dropout', 0.1),
            max_seq_len=config['max_seq_len']
        )

        if config.get('disable_feedback_learning', False):
            print("  Disabling feedback learning (gamma frozen at 0)")
            for layer in model.layers:
                layer.feedback.gamma.requires_grad = False
                layer.feedback.w_r.weight.requires_grad = False
                layer.feedback.w_e.requires_grad = False

        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'], weight_decay=config['weight_decay'])
        lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=config['warmup_steps'], max_steps=config['max_steps'])

        trainer = HAGMoETrainer(
            model=model, optimizer=optimizer, train_dataloader=get_dl(), val_dataloader=get_dl(),
            lr_scheduler=lr_scheduler, num_groups=config['num_groups'], experts_per_group=config['experts_per_group'],
            device=args.device
        )
        if config.get('disable_feedback_learning', False):
            trainer.lambda_scheduler.gamma_scheduler.target_value = 0.0

        trainer.train(num_epochs=args.epochs)
        print(f"Completed {variant}")

if __name__ == "__main__":
    main()
