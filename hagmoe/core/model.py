import torch
import torch.nn as nn

from .block import HAGMoEBlock

class HAGMoETransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int,
                 num_groups: int, experts_per_group: int, hidden_dim: int,
                 k_min: int, k_max: int, alpha: float = 10.0, d_r: int = None,
                 dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            HAGMoEBlock(
                d_model=d_model, num_heads=num_heads, num_groups=num_groups,
                experts_per_group=experts_per_group, hidden_dim=hidden_dim,
                k_min=k_min, k_max=k_max, alpha=alpha, d_r=d_r, dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)

        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        all_aux_data = []
        for layer in self.layers:
            x, aux_data = layer(x, mask)
            all_aux_data.append(aux_data)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, all_aux_data
