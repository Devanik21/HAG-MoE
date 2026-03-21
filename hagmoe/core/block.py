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
            group_out = self.expert_groups[g](x_prime_norm, e_idx, s_val)
            o_i += group_out * g_mask.unsqueeze(-1)

        o_i_mod, gamma = self.feedback(o_i, e_idx, s_val)
        out = residual + self.dropout(o_i_mod)

        aux_loss_data = {
            'p_g': p_g, 'g_i_star': g_i_star, 'p_e': p_e,
            'a_i_c': a_i_c, 'a_i_f': a_i_f,
            'gamma': gamma, 'k_i': k_i, 'norm_entropy': norm_entropy
        }
        return out, aux_loss_data
