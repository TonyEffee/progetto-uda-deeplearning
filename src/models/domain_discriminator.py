"""
domain_discriminator.py

Piccola rete (MLP) che prova a distinguere se una feature proviene dal
source domain o dal target domain. E' l'avversario nel training
adversariale di DANN: piu' si allena bene, piu' spinge (tramite la
Gradient Reversal Layer) il feature extractor a produrre feature
indistinguibili tra i due domini.
"""

import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2),  # 2 classi: source (0) / target (1)
        )

    def forward(self, x):
        return self.net(x)
