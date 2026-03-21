import torch
import torch.nn as nn

def compute_theoretical_k(attn_weights: torch.Tensor, threshold: float = 0.9):
    batch_size, seq_len, _ = attn_weights.shape
    sorted_weights, _ = torch.sort(attn_weights, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_weights, dim=-1)
    k_theory = torch.argmax((cumsum >= threshold).int(), dim=-1) + 1
    return k_theory

def analyze_entropy_k_correlation(model, dataloader, device='cuda'):
    model.eval()
    all_entropies = []
    all_k_i = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            _, aux_data_list = model(input_ids, mask)
            for aux_data in aux_data_list:
                all_entropies.append(aux_data['norm_entropy'].cpu().view(-1))
                all_k_i.append(aux_data['k_i'].cpu().view(-1))

    entropies = torch.cat(all_entropies)
    k_i = torch.cat(all_k_i).float()
    correlation = torch.corrcoef(torch.stack([entropies, k_i]))[0, 1].item()
    return correlation, entropies, k_i
