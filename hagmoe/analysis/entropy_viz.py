import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_token_entropy_distribution(entropy_values: torch.Tensor, save_path: str = None):
    if isinstance(entropy_values, torch.Tensor):
        entropy_values = entropy_values.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    sns.histplot(entropy_values, bins=50, kde=True)
    plt.title('Distribution of Per-Token Attention Entropy')
    plt.xlabel('Normalized Entropy ($\\tilde{\\mathcal{H}}_i$)')
    plt.ylabel('Density')
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_entropy_vs_cardinality(entropy_values: torch.Tensor, k_values: torch.Tensor, save_path: str = None):
    if isinstance(entropy_values, torch.Tensor):
        entropy_values = entropy_values.detach().cpu().numpy()
    if isinstance(k_values, torch.Tensor):
        k_values = k_values.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=k_values, y=entropy_values)
    plt.title('Attention Entropy vs Dynamic Cardinality ($K_i$)')
    plt.xlabel('Number of Active Experts ($K_i$)')
    plt.ylabel('Normalized Entropy ($\\tilde{\\mathcal{H}}_i$)')
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    else: plt.show()
    plt.close()
