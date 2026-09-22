"""
feature_utils.py

Funzioni di supporto per estrarre le feature intermedie (pre-fully-connected)
da una ResNet-18, necessarie per le tecniche di UDA basate su feature
alignment (CORAL, DANN). La ResNet-18 standard di torchvision non espone
direttamente queste feature nel forward(), quindi replichiamo qui i passaggi
interni fino al layer di pooling.
"""

import torch


def forward_with_features(model, x):
    """
    Esegue il forward pass di una ResNet-18 (torchvision) restituendo sia i
    logit finali sia le feature 512-dimensionali usate come input al layer
    fully-connected (utili per CORAL/DANN).

    Funziona sull'architettura creata da src.models.baseline.get_baseline_model,
    che e' una models.resnet18 standard con il solo layer 'fc' sostituito.
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    features = torch.flatten(x, 1)  # (batch, 512) per ResNet-18

    logits = model.fc(features)
    return logits, features
