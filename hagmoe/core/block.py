import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .routing import EntropyGate, CoarseGate, FineGate
from .experts import ExpertGroup
from .feedback import BidirectionalFeedback

class HAGMoEBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_groups: int, experts_per_group: int,
                 hidden_dim: int, k_min: int, k_max: int, alpha: float = 10.0,
                 d_r: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.total_experts = num_groups * experts_per_group

        self.num_coarse_heads = num_heads // 2
        self.num_fine_heads = num_heads - self.num_coarse_heads

        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.moe_ln = nn.LayerNorm(d_model)

        self.entropy_gate = EntropyGate(k_min, k_max, alpha)
        self.coarse_gate = CoarseGate(d_model, num_groups)
        self.fine_gate = FineGate(d_model, num_groups, experts_per_group)

        self.expert_groups = nn.ModuleList([
            ExpertGroup(experts_per_group, d_model, hidden_dim)
            for _ in range(num_groups)
        ])

        self.feedback = BidirectionalFeedback(d_model, self.total_experts, d_r)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        residual = x
        x_norm = self.attn_ln(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, mask)
        x_prime = residual + self.dropout(attn_out)

        residual = x_prime
        x_prime_norm = self.moe_ln(x_prime)

        attn_weights = attn_weights.detach()
        attn_c_heads = attn_weights[:, :self.num_coarse_heads, :, :]
        attn_f_heads = attn_weights[:, self.num_coarse_heads:, :, :]

        a_i_c = attn_c_heads.mean(dim=1)
        a_i_f = attn_f_heads.mean(dim=1)

        c_i_c = torch.bmm(a_i_c, x_prime_norm)
        c_i_f = torch.bmm(a_i_f, x_prime_norm)

        k_i, norm_entropy = self.entropy_gate(a_i_c, self.training)
        g_i_star, p_g = self.coarse_gate(c_i_c)
        e_idx, s_val, p_e = self.fine_gate(c_i_f, g_i_star, k_i)

        o_i = torch.zeros_like(x_prime_norm)
        for g in range(self.num_groups):
            g_mask = (g_i_star == g)
            if not g_mask.any():
                continue

            # Select tokens and corresponding data for the current group
            group_x = x_prime_norm[g_mask]
            group_e_idx = e_idx[g_mask]
            group_s_val = s_val[g_mask]

            if group_x.numel() == 0:
                continue

            # Convert global expert indices to local indices for the group
            local_e_idx = group_e_idx - g * self.experts_per_group
            local_e_idx[group_e_idx < 0] = -1  # Preserve padding indices

            # The expert group expects a batch dimension, so we add one
            # and then remove it from the output.
            group_out = self.expert_groups[g](
                group_x.unsqueeze(0),
                local_e_idx.unsqueeze(0),
                group_s_val.unsqueeze(0)
            )

            # Place the output back into the correct positions
            o_i[g_mask] = group_out.squeeze(0)

        o_i_mod, gamma = self.feedback(o_i, e_idx, s_val)
        out = residual + self.dropout(o_i_mod)

        aux_loss_data = {
            'p_g': p_g, 'g_i_star': g_i_star, 'p_e': p_e,
            'a_i_c': a_i_c, 'a_i_f': a_i_f,
            'gamma': gamma, 'k_i': k_i, 'norm_entropy': norm_entropy
        }
        return out, aux_loss_data
