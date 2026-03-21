import torch
from hagmoe.core.routing import EntropyGate

def test_entropy_gate_bounds():
    k_min = 2
    k_max = 8
    alpha = 10.0
    gate = EntropyGate(k_min, k_max, alpha)
    batch_size = 4
    seq_len = 16
    attn_c = torch.softmax(torch.randn(batch_size, seq_len, seq_len), dim=-1)
    k_i, norm_entropy = gate(attn_c)
    assert k_i.shape == (batch_size, seq_len)
    assert norm_entropy.shape == (batch_size, seq_len)
    assert torch.all(k_i >= k_min)
    assert torch.all(k_i <= k_max)
    assert torch.all(norm_entropy >= 0.0)
    assert torch.all(norm_entropy <= 1.0)

def test_entropy_gate_monotonicity():
    k_min = 1
    k_max = 4
    alpha = 100.0
    gate = EntropyGate(k_min, k_max, alpha)
    seq_len = 8
    focused_attn = torch.zeros(1, 1, seq_len)
    focused_attn[0, 0, 0] = 1.0
    diffuse_attn = torch.ones(1, 1, seq_len) / seq_len
    k_focused, H_focused = gate(focused_attn, training=False)
    k_diffuse, H_diffuse = gate(diffuse_attn, training=False)
    assert H_diffuse.item() > H_focused.item()
    assert k_diffuse.item() >= k_focused.item()
    assert k_focused.item() == k_min
