import torch
import torch.nn as nn

def compute_group_load_balancing_loss(p_g: torch.Tensor, g_i_star: torch.Tensor, num_groups: int) -> torch.Tensor:
    batch_size, seq_len, _ = p_g.shape
    num_tokens = batch_size * seq_len

    p_g_flat = p_g.view(-1, num_groups)
    g_i_star_flat = g_i_star.view(-1)

    f_g = torch.zeros(num_groups, device=p_g.device)
    for g in range(num_groups):
        f_g[g] = (g_i_star_flat == g).float().sum() / num_tokens

    P_g = p_g_flat.mean(dim=0)
    loss = num_groups * torch.sum(f_g * P_g)
    return loss

def compute_expert_load_balancing_loss(p_e: torch.Tensor, g_i_star: torch.Tensor,
                                       num_groups: int, experts_per_group: int) -> torch.Tensor:
    batch_size, seq_len, _ = p_e.shape
    num_tokens = batch_size * seq_len

    p_e_flat = p_e.view(-1, experts_per_group)
    g_i_star_flat = g_i_star.view(-1)

    total_loss = torch.tensor(0.0, device=p_e.device)
    active_groups = 0

    for g in range(num_groups):
        g_mask = (g_i_star_flat == g)
        if not g_mask.any():
            continue

        active_groups += 1
        group_p_e = p_e_flat[g_mask]
        num_group_tokens = group_p_e.size(0)

        e_i_star = torch.argmax(group_p_e, dim=-1)
        f_e = torch.zeros(experts_per_group, device=p_e.device)
        for e in range(experts_per_group):
            f_e[e] = (e_i_star == e).float().sum() / num_group_tokens

        P_e = group_p_e.mean(dim=0)
        group_loss = experts_per_group * torch.sum(f_e * P_e)
        total_loss += group_loss

    if active_groups > 0:
        return total_loss / active_groups
    return total_loss

def compute_divergence_regularizer(a_i_c: torch.Tensor, a_i_f: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    a_i_c_safe = a_i_c.clamp(min=eps)
    a_i_f_safe = a_i_f.clamp(min=eps)

    kl_div = torch.sum(a_i_c_safe * (torch.log(a_i_c_safe) - torch.log(a_i_f_safe)), dim=-1)
    return -kl_div.mean()

class HAGMoELoss(nn.Module):
    def __init__(self, num_groups: int, experts_per_group: int):
        super().__init__()
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, all_aux_data: list,
                lambda_lb_g: float = 0.1, lambda_lb_e: float = 0.1,
                lambda_div: float = 0.01, lambda_gamma: float = 0.01):
        lm_loss = self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        total_lb_g_loss = 0.0
        total_lb_e_loss = 0.0
        total_div_loss = 0.0
        total_gamma_loss = 0.0
        num_layers = len(all_aux_data)

        for aux_data in all_aux_data:
            total_lb_g_loss += compute_group_load_balancing_loss(aux_data['p_g'], aux_data['g_i_star'], self.num_groups)
            total_lb_e_loss += compute_expert_load_balancing_loss(aux_data['p_e'], aux_data['g_i_star'], self.num_groups, self.experts_per_group)
            total_div_loss += compute_divergence_regularizer(aux_data['a_i_c'], aux_data['a_i_f'])
            total_gamma_loss += (aux_data['gamma'] ** 2).squeeze()

        total_lb_g_loss /= num_layers
        total_lb_e_loss /= num_layers
        total_div_loss /= num_layers
        total_gamma_loss /= num_layers

        total_loss = lm_loss + lambda_lb_g * total_lb_g_loss + lambda_lb_e * total_lb_e_loss + lambda_div * total_div_loss + lambda_gamma * total_gamma_loss

        loss_dict = {
            'total': total_loss.item(), 'lm': lm_loss.item(),
            'lb_g': total_lb_g_loss.item() if isinstance(total_lb_g_loss, torch.Tensor) else total_lb_g_loss,
            'lb_e': total_lb_e_loss.item() if isinstance(total_lb_e_loss, torch.Tensor) else total_lb_e_loss,
            'div': total_div_loss.item() if isinstance(total_div_loss, torch.Tensor) else total_div_loss,
            'gamma': total_gamma_loss.item() if isinstance(total_gamma_loss, torch.Tensor) else total_gamma_loss
        }
        return total_loss, loss_dict
