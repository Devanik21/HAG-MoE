import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(hidden)

class ExpertGroup(nn.Module):
    def __init__(self, num_experts: int, d_model: int, hidden_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, hidden_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, k_max = expert_idx.shape
        d_model = x.shape[-1]

        out = torch.zeros_like(x)

        # Flatten all tokens across batch and sequence length
        x_flat = x.view(-1, d_model)  # (B*S, d_model)
        idx_flat = expert_idx.view(-1, k_max)  # (B*S, K_max)
        scores_flat = scores.view(-1, k_max)  # (B*S, K_max)
        out_flat = out.view(-1, d_model)  # (B*S, d_model)

        for e in range(self.num_experts):
            # Find which tokens (and at which k slot) go to this expert
            mask = (idx_flat == e)  # (B*S, K_max)

            # Reduce along k_max dimension to see if a token goes to this expert at all
            # Since a token shouldn't be assigned to the same expert twice in top-k
            token_mask = mask.any(dim=-1)  # (B*S)

            if not token_mask.any():
                continue

            # Select tokens for this expert
            x_e = x_flat[token_mask]  # (num_tokens_for_e, d_model)

            # Compute expert output
            out_e = self.experts[e](x_e)  # (num_tokens_for_e, d_model)

            # Extract the corresponding scores.
            # We can use mask to extract the specific score for this expert for each token
            # mask[token_mask] gives shape (num_tokens_for_e, K_max) with exactly one True per row
            expert_scores = scores_flat[token_mask][mask[token_mask]].unsqueeze(-1)  # (num_tokens_for_e, 1)

            # Add weighted output back to flattened output tensor
            out_flat[token_mask] += out_e * expert_scores

        return out_flat.view(batch_size, seq_len, d_model)
