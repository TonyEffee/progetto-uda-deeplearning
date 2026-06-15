import torch
import torch.nn as nn
from torchvision import models

def get_baseline_model(num_classes):
    """
    Carica una ResNet-18 pre-addestrata e adatta l'ultimo layer
    al nostro specifico numero di classi.
    """
    # 1. Carichiamo la rete con i pesi pre-addestrati su ImageNet (Transfer Learning)
    # Questo ci fa risparmiare ore di calcolo e migliora l'accuratezza iniziale
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Otteniamo il numero di feature in ingresso all'ultimo livello (Fully Connected layer)
    num_ftrs = model.fc.in_features
    
    # 3. Sostituiamo l'ultimo livello per avere esattamente 'num_classes' in uscita
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Piccolo blocco di test
if __name__ == "__main__":
    # Testiamo il modello simulando di avere 4 classi
    test_model = get_baseline_model(num_classes=4)
    print(test_model)
    print("\n[SUCCESS] Modello inizializzato correttamente con 4 uscite!")