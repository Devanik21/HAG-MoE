import torch
from hagmoe.core.routing import CoarseGate, FineGate

def test_coarse_gate():
    d_model = 128
    num_groups = 4
    batch_size = 2
    seq_len = 10
    gate = CoarseGate(d_model, num_groups)
    c_c = torch.randn(batch_size, seq_len, d_model)
    g_i_star, p_g = gate(c_c)
    assert g_i_star.shape == (batch_size, seq_len)
    assert p_g.shape == (batch_size, seq_len, num_groups)
    torch.testing.assert_close(p_g.sum(dim=-1), torch.ones(batch_size, seq_len))
    assert torch.all(g_i_star == torch.argmax(p_g, dim=-1))

def test_fine_gate():
    d_model = 128
    num_groups = 4
    experts_per_group = 8
    batch_size = 2
    seq_len = 10
    k_min = 2
    k_max = 4
    gate = FineGate(d_model, num_groups, experts_per_group)
    c_f = torch.randn(batch_size, seq_len, d_model)
    g_i_star = torch.randint(0, num_groups, (batch_size, seq_len))
    k_i = torch.randint(k_min, k_max + 1, (batch_size, seq_len))
    e_idx, s_val, p_e = gate(c_f, g_i_star, k_i)
    assert e_idx.shape == (batch_size, seq_len, k_max)
    assert s_val.shape == (batch_size, seq_len, k_max)
    assert p_e.shape == (batch_size, seq_len, experts_per_group)
    torch.testing.assert_close(p_e.sum(dim=-1), torch.ones(batch_size, seq_len))
    for b in range(batch_size):
        for s in range(seq_len):
            k = k_i[b, s].item()
            if k < k_max:
                assert torch.all(e_idx[b, s, k:] == -1)
                assert torch.all(s_val[b, s, k:] == 0.0)
    valid_mask = e_idx >= 0
    valid_indices = e_idx[valid_mask]
    assert torch.all(valid_indices >= 0)
    assert torch.all(valid_indices < num_groups * experts_per_group)
