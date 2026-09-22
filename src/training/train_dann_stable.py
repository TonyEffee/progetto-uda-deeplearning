"""
train_dann_stable.py

Variante stabilizzata di train_dann.py (Domain-Adversarial Neural Network).

Rispetto alla versione originale, espone tre iperparametri aggiuntivi per
controllare la pressione adversariale della Gradient Reversal Layer,
introdotti dopo aver osservato instabilita' (val_acc oscillante,
domain_loss collassata a livello di chance gia' dalle prime epoche, e
target accuracy peggiore del baseline puro) con la schedula originale:

    - gamma: velocita' di crescita di lambda (piu' basso = piu' lento)
    - max_lambda: pressione adversariale massima raggiungibile
    - domain_loss_weight: peso della domain_loss nella loss totale

Con i default (gamma=10, max_lambda=1.0, domain_loss_weight=1.0) il
comportamento e' identico alla versione originale.

Riferimento: Ganin & Lempitsky, "Unsupervised Domain Adaptation by
Backpropagation" (2015).

Ad ogni batch di source (con label) viene affiancato un batch di target
(senza label), esattamente come per CORAL. La differenza e' nel
meccanismo di allineamento: invece di allineare direttamente le
statistiche delle feature (CORAL), qui un piccolo discriminatore prova a
indovinare da quale dominio proviene ciascuna feature, mentre il feature
extractor - tramite la Gradient Reversal Layer - viene allenato a
INGANNARLO. Il risultato, se l'allenamento converge bene, e' che le
feature diventano invarianti al dominio.

Come per CORAL, le etichette del target non vengono MAI usate nella
loss di training: compaiono solo nella valutazione di monitoraggio
stampata ad ogni epoca.

USO:
    python -m src.training.train_dann_stable --config experiments/configs/dann_stable.yaml
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam, SGD
from tqdm import tqdm

from src.datasets.datasets import get_source_dataloaders, get_target_dataloader
from src.models.baseline import get_baseline_model
from src.models.feature_utils import forward_with_features
from src.models.domain_discriminator import DomainDiscriminator
from src.models.grl import grad_reverse


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class InfiniteIterator:
    """Cicla indefinitamente su un DataLoader (il target e' molto piu' piccolo del source)."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iterator = iter(dataloader)

    def next(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


def evaluate_classification(model, dataloader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Valutazione", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, _ = forward_with_features(model, images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def get_lambda_schedule(progress, gamma=10.0, max_lambda=1.0):
    """
    Schedula del paper DANN originale: lambda cresce da 0 a max_lambda in
    modo sigmoidale al progredire del training (progress va da 0 a 1).
    Partire da lambda=0 stabilizza le prime iterazioni, quando le feature
    non sono ancora abbastanza buone da guidare bene il discriminatore.

    gamma controlla QUANTO VELOCE cresce (valori piu' bassi = crescita
    piu' lenta e graduale). max_lambda limita la pressione adversariale
    massima raggiungibile, utile se si osserva instabilita' con lambda=1.
    """
    return max_lambda * (2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0)


def train_one_epoch(model, domain_discriminator, source_loader, target_iterator,
                     device, cls_criterion, domain_criterion, optimizer,
                     epoch, total_epochs, batches_per_epoch,
                     gamma=10.0, max_lambda=1.0, domain_loss_weight=1.0):
    model.train()
    domain_discriminator.train()

    total_cls_loss, total_domain_loss, correct, total = 0.0, 0.0, 0, 0
    progress_bar = tqdm(source_loader, desc="  Training", leave=False)

    for batch_idx, (source_images, source_labels) in enumerate(progress_bar):
        target_images, _ = target_iterator.next()  # le label del target NON vengono usate

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)
        batch_size_source = source_images.size(0)
        batch_size_target = target_images.size(0)

        # Progresso complessivo del training (0 -> 1), per la schedula di lambda
        current_step = (epoch - 1) * batches_per_epoch + batch_idx
        total_steps = total_epochs * batches_per_epoch
        progress = current_step / max(total_steps, 1)
        lambda_grl = get_lambda_schedule(progress, gamma=gamma, max_lambda=max_lambda)

        optimizer.zero_grad()

        # --- Ramo di classificazione (solo source, ha le label) ---
        source_logits, source_features = forward_with_features(model, source_images)
        cls_loss = cls_criterion(source_logits, source_labels)

        # --- Ramo del discriminatore di dominio (source + target, senza label di classe) ---
        _, target_features = forward_with_features(model, target_images)

        source_domain_features = grad_reverse(source_features, lambda_grl)
        target_domain_features = grad_reverse(target_features, lambda_grl)

        source_domain_logits = domain_discriminator(source_domain_features)
        target_domain_logits = domain_discriminator(target_domain_features)

        source_domain_labels = torch.zeros(batch_size_source, dtype=torch.long, device=device)
        target_domain_labels = torch.ones(batch_size_target, dtype=torch.long, device=device)

        domain_loss = domain_loss_weight * (
            domain_criterion(source_domain_logits, source_domain_labels)
            + domain_criterion(target_domain_logits, target_domain_labels)
        )

        loss = cls_loss + domain_loss
        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item() * batch_size_source
        total_domain_loss += domain_loss.item() * batch_size_source
        preds = source_logits.argmax(dim=1)
        correct += (preds == source_labels).sum().item()
        total += batch_size_source

        progress_bar.set_postfix(cls=f"{cls_loss.item():.4f}",
                                  domain=f"{domain_loss.item():.4f}",
                                  lam=f"{lambda_grl:.3f}")

    return total_cls_loss / total, total_domain_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Training con DANN (adversarial domain adaptation)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config["training"].get("seed", 42))
    device = get_device()
    print(f"Device in uso: {device}")

    train_loader, val_loader, class_to_idx = get_source_dataloaders(
        data_dir=config["data"]["source_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"Classi (class_to_idx): {class_to_idx}")

    target_loader, _ = get_target_dataloader(
        data_dir=config["data"]["target_dir"],
        class_to_idx=class_to_idx,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    target_iterator = InfiniteIterator(target_loader)

    num_classes = config["model"]["num_classes"]
    model = get_baseline_model(num_classes=num_classes).to(device)

    # Warm start dal baseline "puro" (non da CORAL!), per confrontare le due
    # tecniche UDA a parita' di punto di partenza.
    warm_start_path = config["model"].get("warm_start_checkpoint")
    if warm_start_path and Path(warm_start_path).exists():
        checkpoint = torch.load(warm_start_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Pesi inizializzati da checkpoint baseline: {warm_start_path}")
    else:
        print("[INFO] Nessun warm start trovato, si parte da pesi ImageNet puri.")

    discriminator_hidden_dim = config["model"].get("discriminator_hidden_dim", 256)
    domain_discriminator = DomainDiscriminator(in_features=512, hidden_dim=discriminator_hidden_dim).to(device)

    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    # Un solo optimizer per feature extractor + classificatore + discriminatore:
    # la GRL si occupa da sola di invertire il segno del gradiente dove serve.
    all_params = list(model.parameters()) + list(domain_discriminator.parameters())
    if config["training"]["optimizer"].lower() == "adam":
        optimizer = Adam(all_params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(all_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / config["output"]["checkpoint_name"]
    log_path = log_dir / f"{config['experiment_name']}_log.csv"

    epochs = config["training"]["epochs"]
    batches_per_epoch = len(train_loader)
    best_val_acc = 0.0

    # Iperparametri di stabilizzazione della GRL (default = comportamento originale)
    gamma = config["training"].get("gamma", 10.0)
    max_lambda = config["training"].get("max_lambda", 1.0)
    domain_loss_weight = config["training"].get("domain_loss_weight", 1.0)
    print(f"GRL: gamma={gamma}, max_lambda={max_lambda}, domain_loss_weight={domain_loss_weight}")

    with open(log_path, "w", newline="") as f:
        log_writer = csv.writer(f)
        log_writer.writerow([
            "epoch", "cls_loss", "domain_loss", "train_acc",
            "val_loss", "val_acc", "target_acc_monitor", "time_sec",
        ])

        for epoch in range(1, epochs + 1):
            start = time.time()
            cls_loss, domain_loss, train_acc = train_one_epoch(
                model, domain_discriminator, train_loader, target_iterator,
                device, cls_criterion, domain_criterion, optimizer,
                epoch, epochs, batches_per_epoch,
                gamma=gamma, max_lambda=max_lambda, domain_loss_weight=domain_loss_weight,
            )
            val_loss, val_acc = evaluate_classification(model, val_loader, device, cls_criterion)
            # Come per CORAL: le label del target vengono usate SOLO per
            # monitorare il progresso, mai nella loss di training.
            _, target_acc_monitor = evaluate_classification(model, target_loader, device, cls_criterion)
            elapsed = time.time() - start

            print(f"[Epoch {epoch}/{epochs}] "
                  f"cls_loss={cls_loss:.4f} domain_loss={domain_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_acc={val_acc:.4f} | target_acc (monitor)={target_acc_monitor:.4f} | {elapsed:.1f}s")

            log_writer.writerow([epoch, cls_loss, domain_loss, train_acc, val_loss, val_acc,
                                  target_acc_monitor, elapsed])
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "domain_discriminator_state_dict": domain_discriminator.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "val_acc": val_acc,
                    "target_acc_monitor": target_acc_monitor,
                    "epoch": epoch,
                }, checkpoint_path)
                print(f"  -> Nuovo miglior checkpoint salvato (val_acc={val_acc:.4f}, "
                      f"target_acc_monitor={target_acc_monitor:.4f})")

    print(f"\n[SUCCESS] Training DANN completato. Miglior val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint salvato in: {checkpoint_path}")
    print(f"Log salvato in: {log_path}")


if __name__ == "__main__":
    main()
