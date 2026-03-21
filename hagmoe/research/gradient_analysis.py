import torch
import torch.nn as nn

def analyze_feedback_gradients(model, input_ids, targets, loss_fn):
    model.train()
    for block in model.layers:
        block.fine_gate.w_e.requires_grad_(True)
        block.coarse_gate.w_g.requires_grad_(True)
        block.feedback.gamma.requires_grad_(True)

    logits, _ = model(input_ids)
    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

    model.zero_grad()
    loss.backward()

    grad_norms = {}
    for layer_idx, block in enumerate(model.layers):
        layer_norms = {}
        if block.fine_gate.w_e.grad is not None:
            layer_norms['fine_gate_w_e_grad_norm'] = block.fine_gate.w_e.grad.norm().item()
        if block.coarse_gate.w_g.weight.grad is not None:
            layer_norms['coarse_gate_w_g_grad_norm'] = block.coarse_gate.w_g.weight.grad.norm().item()
        if block.feedback.gamma.grad is not None:
            layer_norms['gamma_grad'] = block.feedback.gamma.grad.item()
        if block.feedback.w_r.weight.grad is not None:
            layer_norms['w_r_grad_norm'] = block.feedback.w_r.weight.grad.norm().item()
        grad_norms[f'layer_{layer_idx}'] = layer_norms

    return grad_norms
