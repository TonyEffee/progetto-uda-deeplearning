"""
coral.py

Implementazione della loss CORAL (CORrelation ALignment) per il feature
alignment non supervisionato tra source e target domain.

Riferimento: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep
Domain Adaptation" (2016).

L'idea: allineare le statistiche del secondo ordine (matrici di covarianza)
delle feature estratte dal backbone su source e target, cosi' che il
classificatore addestrato sul source si trovi a lavorare su feature che
"assomigliano" a quelle del target, anche senza mai vedere le label del
target.
"""

import torch


def compute_covariance(features):
    """Calcola la matrice di covarianza (D x D) di un batch di feature (N x D)."""
    n = features.size(0)
    if n <= 1:
        n = 2  # evita divisioni per zero con batch di dimensione 1 (es. ultimo batch)
    features = features - features.mean(dim=0, keepdim=True)
    covariance = features.t() @ features / (n - 1)
    return covariance


def coral_loss(source_features, target_features):
    """
    CORAL loss: distanza (norma di Frobenius al quadrato) tra le matrici di
    covarianza di source e target, normalizzata per la dimensionalita' delle
    feature cosi' che la loss resti in una scala ragionevole indipendentemente
    da D.
    """
    d = source_features.size(1)
    source_cov = compute_covariance(source_features)
    target_cov = compute_covariance(target_features)
    loss = torch.sum((source_cov - target_cov) ** 2)
    loss = loss / (4 * d * d)
    return loss
