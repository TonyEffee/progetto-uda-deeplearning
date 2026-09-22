"""
train_coral.py

Training del modello con CORAL come tecnica di Unsupervised Domain
Adaptation (Obiettivo Minimo 3 della consegna - prima delle due tecniche
richieste, insieme a DANN).

Ad ogni batch di source (con label) viene affiancato un batch di target
(SENZA usare le sue label) preso ciclicamente dal target dataset, dato che
il target ha molte meno immagini del source. La loss totale e':

    loss = cross_entropy(source_logits, source_labels)
           + coral_lambda * coral_loss(source_features, target_features)

Il secondo termine spinge le feature del target ad "assomigliare"
statisticamente a quelle del source, senza mai usare le etichette del
target durante il training. Le etichette del target vengono usate SOLO
per monitorare il progresso ad ogni epoca (stampate ma non usate per la
loss ne' per scegliere il checkpoint migliore, che si basa sulla source
val accuracy per restare metodologicamente corretti).

USO:
    python -m src.training.train_coral --config experiments/configs/coral.yaml
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam, SGD
from tqdm import tqdm

from src.datasets.datasets import get_source_dataloaders, get_target_dataloader
from src.models.baseline import get_baseline_model
from src.models.feature_utils import forward_with_features
from src.losses.coral import coral_loss


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


def train_one_epoch(model, source_loader, target_iterator, device, criterion, optimizer, coral_lambda):
    model.train()
    total_cls_loss, total_coral_loss, correct, total = 0.0, 0.0, 0, 0

    progress_bar = tqdm(source_loader, desc="  Training", leave=False)
    for source_images, source_labels in progress_bar:
        target_images, _ = target_iterator.next()  # le label del target NON vengono usate

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)

        optimizer.zero_grad()

        source_logits, source_features = forward_with_features(model, source_images)
        _, target_features = forward_with_features(model, target_images)

        cls_loss = criterion(source_logits, source_labels)
        align_loss = coral_loss(source_features, target_features)
        loss = cls_loss + coral_lambda * align_loss

        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item() * source_images.size(0)
        total_coral_loss += align_loss.item() * source_images.size(0)
        preds = source_logits.argmax(dim=1)
        correct += (preds == source_labels).sum().item()
        total += source_labels.size(0)

        progress_bar.set_postfix(cls=f"{cls_loss.item():.4f}", coral=f"{align_loss.item():.4f}")

    return total_cls_loss / total, total_coral_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Training con CORAL (feature alignment UDA)")
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

    # Warm start facoltativo da un checkpoint baseline gia' allenato:
    # partire da pesi gia' buoni sul source rende la convergenza piu'
    # rapida e stabile rispetto a ripartire da ImageNet puro.
    warm_start_path = config["model"].get("warm_start_checkpoint")
    if warm_start_path and Path(warm_start_path).exists():
        checkpoint = torch.load(warm_start_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Pesi inizializzati da checkpoint baseline: {warm_start_path}")
    else:
        print("[INFO] Nessun warm start trovato, si parte da pesi ImageNet puri.")

    criterion = nn.CrossEntropyLoss()
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    if config["training"]["optimizer"].lower() == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    coral_lambda = config["training"]["coral_lambda"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / config["output"]["checkpoint_name"]
    log_path = log_dir / f"{config['experiment_name']}_log.csv"

    epochs = config["training"]["epochs"]
    best_val_acc = 0.0

    with open(log_path, "w", newline="") as f:
        log_writer = csv.writer(f)
        log_writer.writerow([
            "epoch", "cls_loss", "coral_loss", "train_acc",
            "val_loss", "val_acc", "target_acc_monitor", "time_sec",
        ])

        for epoch in range(1, epochs + 1):
            start = time.time()
            cls_loss, align_loss, train_acc = train_one_epoch(
                model, train_loader, target_iterator, device, criterion, optimizer, coral_lambda
            )
            val_loss, val_acc = evaluate_classification(model, val_loader, device, criterion)
            # Le label del target vengono usate QUI solo per monitorare il
            # progresso dell'adattamento, mai per il training vero e proprio.
            _, target_acc_monitor = evaluate_classification(model, target_loader, device, criterion)
            elapsed = time.time() - start

            print(f"[Epoch {epoch}/{epochs}] "
                  f"cls_loss={cls_loss:.4f} coral_loss={align_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_acc={val_acc:.4f} | target_acc (monitor)={target_acc_monitor:.4f} | {elapsed:.1f}s")

            log_writer.writerow([epoch, cls_loss, align_loss, train_acc, val_loss, val_acc,
                                  target_acc_monitor, elapsed])
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "val_acc": val_acc,
                    "target_acc_monitor": target_acc_monitor,
                    "epoch": epoch,
                }, checkpoint_path)
                print(f"  -> Nuovo miglior checkpoint salvato (val_acc={val_acc:.4f}, "
                      f"target_acc_monitor={target_acc_monitor:.4f})")

    print(f"\n[SUCCESS] Training CORAL completato. Miglior val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint salvato in: {checkpoint_path}")
    print(f"Log salvato in: {log_path}")


if __name__ == "__main__":
    main()
