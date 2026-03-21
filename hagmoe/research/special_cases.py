import torch
import torch.nn as nn

def check_standard_moe_reduction(hag_moe_block, x, mask=None):
    with torch.no_grad():
        original_gamma = hag_moe_block.feedback.gamma.clone()
        hag_moe_block.feedback.gamma.fill_(0.0)

    out, aux_data = hag_moe_block(x, mask)

    with torch.no_grad():
        hag_moe_block.feedback.gamma.copy_(original_gamma)

    return out, aux_data

def check_switch_transformer_reduction(hag_moe_block, x, mask=None):
    with torch.no_grad():
        orig_k_min = hag_moe_block.entropy_gate.k_min
        orig_k_max = hag_moe_block.entropy_gate.k_max
        orig_gamma = hag_moe_block.feedback.gamma.clone()

        hag_moe_block.entropy_gate.k_min = 1
        hag_moe_block.entropy_gate.k_max = 1
        hag_moe_block.feedback.gamma.fill_(0.0)

    out, aux_data = hag_moe_block(x, mask)
    k_i = aux_data['k_i']
    is_switch = torch.all(k_i == 1).item()

    with torch.no_grad():
        hag_moe_block.entropy_gate.k_min = orig_k_min
        hag_moe_block.entropy_gate.k_max = orig_k_max
        hag_moe_block.feedback.gamma.copy_(orig_gamma)

    return is_switch, out
