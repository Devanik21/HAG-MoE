import torch
from hagmoe.core.block import HAGMoEBlock
from hagmoe.research.special_cases import check_standard_moe_reduction, check_switch_transformer_reduction

def test_switch_reduction():
    block = HAGMoEBlock(d_model=128, num_heads=4, num_groups=2, experts_per_group=4, hidden_dim=256, k_min=2, k_max=4)
    x = torch.randn(2, 10, 128)
    with torch.no_grad(): block.feedback.gamma.fill_(1.0)
    is_switch, out = check_switch_transformer_reduction(block, x)
    assert is_switch
    assert block.feedback.gamma.item() == 1.0
    assert block.entropy_gate.k_min == 2
    assert block.entropy_gate.k_max == 4

def test_standard_moe_reduction():
    block = HAGMoEBlock(d_model=128, num_heads=4, num_groups=2, experts_per_group=4, hidden_dim=256, k_min=2, k_max=4)
    x = torch.randn(2, 10, 128)
    with torch.no_grad(): block.feedback.gamma.fill_(1.0)
    out, aux = check_standard_moe_reduction(block, x)
    assert out.shape == (2, 10, 128)
    assert block.feedback.gamma.item() == 1.0
