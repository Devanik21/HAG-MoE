import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AttentionProbe:
    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        self.num_heads = num_heads
        self.head_features = []
        self.labels = []

    def extract_features(self, attn_weights: torch.Tensor) -> np.ndarray:
        num_heads, seq_len, _ = attn_weights.shape
        features = np.zeros((num_heads, 4))
        for h in range(num_heads):
            w = attn_weights[h].detach().cpu().numpy()
            if seq_len > 1:
                local_attn = np.mean([w[i, max(0, i-1)] for i in range(1, seq_len)])
            else:
                local_attn = 0.0
            features[h, 0] = local_attn
            features[h, 1] = np.mean(w[:, 0])
            w_safe = np.maximum(w, 1e-8)
            features[h, 2] = np.mean(-np.sum(w_safe * np.log(w_safe), axis=-1))
            spreads = []
            for i in range(seq_len):
                distances = np.abs(np.arange(seq_len) - i)
                spreads.append(np.sum(w[i] * distances))
            features[h, 3] = np.mean(spreads)
        return features

    def add_data(self, features: np.ndarray, is_syntactic: bool):
        self.head_features.append(features)
        self.labels.extend([1 if is_syntactic else 0] * len(features))

    def train_probe(self):
        X = np.vstack(self.head_features)
        y = np.array(self.labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        probe = LogisticRegression(random_state=42)
        probe.fit(X_train, y_train)
        acc = accuracy_score(y_test, probe.predict(X_test))
        return probe, acc
