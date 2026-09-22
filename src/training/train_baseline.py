"""
train_baseline.py

Training del classificatore baseline (ResNet-18) sul source domain
(Intel Image Classification). Corrisponde all'Obiettivo Minimo 1 della
consegna: "Training and Evaluation on Source Domain".

USO:
    python -m src.training.train_baseline --config experiments/configs/baseline.yaml

Salva:
    - Il checkpoint con la migliore accuracy di validazione in
      experiments/checkpoints/<checkpoint_name>
    - Un log CSV (loss/accuracy per epoca) in experiments/logs/
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam, SGD

from src.datasets.datasets import get_source_dataloaders
from src.models.baseline import get_baseline_model


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, dataloader, device, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Training del baseline sul source domain")
    parser.add_argument("--config", type=str, required=True, help="Percorso al file YAML di config")
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config["training"].get("seed", 42))

    device = get_device()
    print(f"Device in uso: {device}")

    # --- Dati ---
    train_loader, val_loader, class_to_idx = get_source_dataloaders(
        data_dir=config["data"]["source_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    print(f"Classi (class_to_idx): {class_to_idx}")
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # --- Modello ---
    num_classes = config["model"]["num_classes"]
    assert num_classes == len(class_to_idx), (
        f"num_classes nel config ({num_classes}) non combacia con le classi "
        f"trovate nel dataset ({len(class_to_idx)}). Controlla experiments/configs/baseline.yaml"
    )
    model = get_baseline_model(num_classes=num_classes).to(device)

    # --- Loss e optimizer ---
    criterion = nn.CrossEntropyLoss()
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    if config["training"]["optimizer"].lower() == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # --- Cartelle di output ---
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / config["output"]["checkpoint_name"]
    log_path = log_dir / f"{config['experiment_name']}_log.csv"

    # --- Training loop ---
    epochs = config["training"]["epochs"]
    best_val_acc = 0.0

    with open(log_path, "w", newline="") as f:
        log_writer = csv.writer(f)
        log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_sec"])

        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            elapsed = time.time() - start

            print(f"[Epoch {epoch}/{epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

            log_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, elapsed])
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "val_acc": val_acc,
                    "epoch": epoch,
                }, checkpoint_path)
                print(f"  -> Nuovo miglior checkpoint salvato (val_acc={val_acc:.4f})")

    print(f"\n[SUCCESS] Training completato. Miglior val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint salvato in: {checkpoint_path}")
    print(f"Log salvato in: {log_path}")


if __name__ == "__main__":
    main()