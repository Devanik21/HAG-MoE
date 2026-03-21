#!/usr/bin/env python3
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from hagmoe.core.model import HAGMoETransformer
from hagmoe.analysis.cardinality_stats import compute_cardinality_stats

def load_config(config_path):
    with open(config_path, 'r') as f: return yaml.safe_load(f)

def dummy_dataloader(batch_size, seq_len, vocab_size, num_batches=100):
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return num_batches * batch_size
        def __getitem__(self, idx):
            return {'input_ids': torch.randint(0, vocab_size, (seq_len,)), 'labels': torch.randint(0, vocab_size, (seq_len,))}
    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=False)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_k_i = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids).to(device)
            logits, all_aux_data = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
            for aux_data in all_aux_data:
                all_k_i.append(aux_data['k_i'].cpu().view(-1))

    if total_tokens == 0:
        return {'loss': 0.0, 'perplexity': float('inf'), 'k_i_stats': {}}

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    if not all_k_i:
        return {'loss': avg_loss, 'perplexity': perplexity, 'k_i_stats': {}}

    k_i_tensor = torch.cat(all_k_i)
    k_i_stats = compute_cardinality_stats(k_i_tensor)
    return {'loss': avg_loss, 'perplexity': perplexity, 'k_i_stats': k_i_stats}

def main():
    parser = argparse.ArgumentParser(description="Evaluate HAG-MoE Model")
    parser.add_argument('--config', type=str, default='configs/small.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
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
    ).to(args.device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("Warning: No checkpoint provided, evaluating randomly initialized model.")

    eval_dl = dummy_dataloader(config['batch_size'], config['max_seq_len'] // 8, config['vocab_size'], num_batches=20)
    results = evaluate(model, eval_dl, args.device)

    print("\n--- Evaluation Results ---")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.4f}")
    print("\nCardinality Stats (K_i):")
    stats = results['k_i_stats']
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']}")
    print(f"  Max: {stats['max']}")
    print("  Distribution:")
    for k, p in sorted(stats['percentages'].items()): print(f"    K={k}: {p:.1f}%")

if __name__ == "__main__":
    main()
