from .special_cases import check_standard_moe_reduction, check_switch_transformer_reduction
from .entropy_theory import compute_theoretical_k, analyze_entropy_k_correlation
from .gradient_analysis import analyze_feedback_gradients

__all__ = [
    'check_standard_moe_reduction', 'check_switch_transformer_reduction',
    'compute_theoretical_k', 'analyze_entropy_k_correlation', 'analyze_feedback_gradients'
]
