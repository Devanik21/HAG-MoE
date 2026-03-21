import torch
import matplotlib.pyplot as plt

def compute_cardinality_stats(k_i_values: torch.Tensor):
    if isinstance(k_i_values, torch.Tensor):
        k_i_values = k_i_values.float()
    stats = {
        'mean': k_i_values.mean().item(), 'std': k_i_values.std().item(),
        'min': k_i_values.min().item(), 'max': k_i_values.max().item(), 'median': k_i_values.median().item(),
    }
    unique_k, counts = torch.unique(k_i_values, return_counts=True)
    total_tokens = k_i_values.numel()
    percentages = {}
    for k, c in zip(unique_k.tolist(), counts.tolist()):
        percentages[int(k)] = (c / total_tokens) * 100
    stats['percentages'] = percentages
    return stats

def plot_cardinality_distribution(k_i_values: torch.Tensor, save_path: str = None):
    stats = compute_cardinality_stats(k_i_values)
    percentages = stats['percentages']
    k_vals = sorted(list(percentages.keys()))
    p_vals = [percentages[k] for k in k_vals]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(k_vals, p_vals, color='skyblue', edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom')
    plt.title('Dynamic Cardinality ($K_i$) Distribution')
    plt.xlabel('Number of Active Experts ($K_i$)')
    plt.ylabel('Percentage of Tokens (%)')
    plt.xticks(k_vals)
    if p_vals:
        plt.ylim(0, max(p_vals) * 1.15)
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    else: plt.show()
    plt.close()
