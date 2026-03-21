import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import List

def plot_expert_routing_heatmap(routing_counts: torch.Tensor, layer_idx: int = None, save_path: str = None):
    if isinstance(routing_counts, torch.Tensor):
        routing_counts = routing_counts.detach().cpu().numpy()

    plt.figure(figsize=(12, 8))
    sns.heatmap(routing_counts, annot=True, fmt='g', cmap='viridis')
    title = f'Expert Routing Heatmap' + (f' (Layer {layer_idx})' if layer_idx is not None else '')
    plt.title(title)
    plt.xlabel('Expert Index within Group')
    plt.ylabel('Group Index')
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_layerwise_group_routing(all_layers_g_counts: List[torch.Tensor], save_path: str = None):
    num_layers = len(all_layers_g_counts)
    num_groups = all_layers_g_counts[0].shape[0]
    data = torch.stack(all_layers_g_counts).detach().cpu().numpy()
    data_norm = data / data.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    bottom = np.zeros(num_layers)
    x = np.arange(num_layers)
    for g in range(num_groups):
        plt.bar(x, data_norm[:, g], bottom=bottom, label=f'Group {g}')
        bottom += data_norm[:, g]

    plt.title('Group Routing Distribution Across Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('Fraction of Tokens')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    else: plt.show()
    plt.close()
