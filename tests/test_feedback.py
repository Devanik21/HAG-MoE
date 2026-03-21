import torch
from hagmoe.core.feedback import BidirectionalFeedback

def test_feedback_initialization_reduces_to_moe():
    d_model = 128
    total_experts = 16
    d_r = 32
    feedback = BidirectionalFeedback(d_model, total_experts, d_r)
    assert feedback.gamma.item() == 0.0
    batch_size = 2
    seq_len = 10
    k_max = 4
    o_i = torch.randn(batch_size, seq_len, d_model)
    e_idx = torch.randint(0, total_experts, (batch_size, seq_len, k_max))
    s_val = torch.rand(batch_size, seq_len, k_max)
    o_i_mod, gamma = feedback(o_i, e_idx, s_val)
    torch.testing.assert_close(o_i, o_i_mod)
    assert gamma.item() == 0.0

def test_feedback_modulation_shape():
    d_model = 128
    total_experts = 16
    d_r = 32
    feedback = BidirectionalFeedback(d_model, total_experts, d_r)
    with torch.no_grad():
        feedback.gamma.fill_(0.5)
    batch_size = 2
    seq_len = 10
    k_max = 4
    o_i = torch.randn(batch_size, seq_len, d_model)
    e_idx = torch.randint(0, total_experts, (batch_size, seq_len, k_max))
    s_val = torch.rand(batch_size, seq_len, k_max)
    o_i_mod, gamma = feedback(o_i, e_idx, s_val)
    assert o_i_mod.shape == (batch_size, seq_len, d_model)
    assert not torch.allclose(o_i, o_i_mod)
