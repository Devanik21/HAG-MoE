import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EntropyGate(nn.Module):
    def __init__(self, k_min: int, k_max: int, alpha: float = 10.0):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.alpha = alpha
        self.register_buffer('mu_H', torch.tensor(0.5))
        self.momentum = 0.99

    def forward(self, attn_c: torch.Tensor, training: bool = True):
        seq_len = attn_c.size(-1)
        entropy = -(attn_c * torch.log(attn_c.clamp(min=1e-8))).sum(dim=-1)
        norm_entropy = entropy / math.log(seq_len)

        if training:
            batch_mean = norm_entropy.detach().mean()
            self.mu_H = self.momentum * self.mu_H + (1 - self.momentum) * batch_mean

        prob = torch.sigmoid(self.alpha * (norm_entropy - self.mu_H))
        k_diff = (self.k_max - self.k_min) * prob
        k_i = self.k_min + torch.floor(k_diff).int()
        k_i = torch.clamp(k_i, self.k_min, self.k_max)

        return k_i, norm_entropy

class CoarseGate(nn.Module):
    def __init__(self, d_model: int, num_groups: int):
        super().__init__()
        self.d_model = d_model
        self.num_groups = num_groups
        self.w_g = nn.Linear(d_model, num_groups, bias=False)

    def forward(self, c_c: torch.Tensor):
        logits = self.w_g(c_c) / math.sqrt(self.d_model)
        p_g = F.softmax(logits, dim=-1)
        g_i_star = torch.argmax(p_g, dim=-1)
        return g_i_star, p_g

class FineGate(nn.Module):
    def __init__(self, d_model: int, num_groups: int, experts_per_group: int):
        super().__init__()
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.w_e = nn.Parameter(torch.Tensor(num_groups, d_model, experts_per_group))
        nn.init.normal_(self.w_e, mean=0.0, std=math.sqrt(1.0 / d_model))

    def forward(self, c_f: torch.Tensor, g_i_star: torch.Tensor, k_i: torch.Tensor):
        batch_size, seq_len, d_model = c_f.shape
        flat_c_f = c_f.view(-1, d_model)
        flat_g = g_i_star.view(-1)

        w_e_selected = self.w_e[flat_g]
        logits = torch.bmm(flat_c_f.unsqueeze(1), w_e_selected).squeeze(1)

        p_e = F.softmax(logits, dim=-1)
        k_max = k_i.max().item()
        topk_scores, topk_indices = torch.topk(p_e, k_max, dim=-1)

        flat_k_i = k_i.view(-1).unsqueeze(-1)
        k_mask = torch.arange(k_max, device=k_i.device).unsqueeze(0) < flat_k_i

        global_indices = flat_g.unsqueeze(-1) * self.experts_per_group + topk_indices
        e_idx = global_indices.masked_fill(~k_mask, -1)
        s_val = topk_scores.masked_fill(~k_mask, 0.0)

        e_idx = e_idx.view(batch_size, seq_len, k_max)
        s_val = s_val.view(batch_size, seq_len, k_max)
        p_e = p_e.view(batch_size, seq_len, self.experts_per_group)

        return e_idx, s_val, p_e
