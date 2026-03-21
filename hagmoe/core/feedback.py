import torch
import torch.nn as nn

class BidirectionalFeedback(nn.Module):
    def __init__(self, d_model: int, total_experts: int, d_r: int = None):
        super().__init__()
        self.d_model = d_model
        self.total_experts = total_experts
        self.d_r = d_model // 8 if d_r is None else d_r

        self.w_e = nn.Parameter(torch.Tensor(total_experts, self.d_r))
        nn.init.normal_(self.w_e, mean=0.0, std=1.0)

        self.w_r = nn.Linear(self.d_r, d_model, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, o_i: torch.Tensor, e_idx: torch.Tensor, s_val: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, k_max = e_idx.shape

        # Flatten the tensors for vectorization
        e_idx_flat = e_idx.view(-1)
        s_val_flat = s_val.view(-1, 1)

        valid_mask = e_idx_flat >= 0
        valid_indices = e_idx_flat[valid_mask]
        valid_scores = s_val_flat[valid_mask]

        r_accum_flat = torch.zeros(batch_size * seq_len * k_max, self.d_r, device=o_i.device)

        if valid_mask.any():
            selected_w_e = self.w_e[valid_indices]
            weighted_w_e = selected_w_e * valid_scores
            r_accum_flat[valid_mask] = weighted_w_e

        r_accum_flat = r_accum_flat.view(batch_size * seq_len, k_max, self.d_r)
        r_accum = r_accum_flat.sum(dim=1).view(batch_size, seq_len, self.d_r)

        r_i = self.w_r(r_accum)
        modulation = 1.0 + self.gamma * torch.tanh(r_i)
        o_i_mod = o_i * modulation

        return o_i_mod, self.gamma
