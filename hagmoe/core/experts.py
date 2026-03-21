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
        out = torch.zeros_like(x)

        for k in range(k_max):
            indices_k = expert_idx[:, :, k]
            scores_k = scores[:, :, k]
            valid_mask = indices_k >= 0

            if not valid_mask.any():
                continue

            for e in range(self.num_experts):
                e_mask = valid_mask & (indices_k == e)
                if not e_mask.any():
                    continue

                x_e = x[e_mask]
                out_e = self.experts[e](x_e)
                s_e = scores_k[e_mask].unsqueeze(-1)
                out[e_mask] += out_e * s_e

        return out
