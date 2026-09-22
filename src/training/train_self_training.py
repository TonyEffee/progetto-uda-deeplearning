"""
train_self_training.py

Self-training con pseudo-label (Extra Objective della consegna): un
"insegnante" gia' allenato (di default il checkpoint CORAL, la tecnica
migliore trovata finora) genera etichette sul target domain; le
predizioni piu' sicure vengono trattate come se fossero vere etichette e
usate per continuare il fine-tuning insieme al source, nella speranza
che rinforzino cio' che il modello gia' sa fare bene sul target.

Le vere etichette del target vengono usate SOLO per due cose che non
influenzano mai il training:
  1) calcolare, a puro scopo di analisi/report, quanto sono "pulite" le
     pseudo-label selezionate (stampato ma mai usato nella loss)
  2) il monitoraggio della target accuracy ad ogni epoca (come gia'
     fatto per CORAL e DANN)

USO:
    python -m src.training.train_self_training --config experiments/configs/self_training.yaml
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import Adam, SGD
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from src.datasets.datasets import get_eval_transforms, get_source_dataloaders, get_target_dataloader
from src.models.baseline import get_baseline_model
from src.models.feature_utils import forward_with_features


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PseudoLabeledDataset(Dataset):
    """Dataset costruito dalle immagini target selezionate, con la pseudo-label come target."""

    def __init__(self, samples, transform):
        # samples: lista di (path, pseudo_label)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


@torch.no_grad()
def generate_pseudo_labels(model, target_dataset, device, confidence_threshold, batch_size, idx_to_class):
    """
    Inferenza su tutto il target dataset con il modello "insegnante"; tiene
    solo le predizioni con confidenza (probabilita' softmax massima) sopra
    la soglia. Ritorna i sample selezionati (path, pseudo_label) e stampa
    statistiche di qualita' calcolate SOLO per l'analisi (mai usate per
    allenare nulla).
    """
    model.eval()
    loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_paths = [str(path) for path, _, _ in target_dataset.samples]
    all_true_labels = np.array([label for _, label, _ in target_dataset.samples])

    all_pred_labels, all_confidences = [], []
    for images, _ in tqdm(loader, desc="  Generazione pseudo-label", leave=False):
        images = images.to(device)
        logits, _ = forward_with_features(model, images)
        probs = F.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)
        all_pred_labels.extend(preds.cpu().numpy().tolist())
        all_confidences.extend(confidences.cpu().numpy().tolist())

    all_pred_labels = np.array(all_pred_labels)
    all_confidences = np.array(all_confidences)
    selected_mask = all_confidences >= confidence_threshold

    selected_samples = [
        (all_paths[i], int(all_pred_labels[i]))
        for i in range(len(all_paths)) if selected_mask[i]
    ]

    n_selected = int(selected_mask.sum())
    pseudo_label_accuracy = (
        (all_pred_labels[selected_mask] == all_true_labels[selected_mask]).mean()
        if n_selected > 0 else float("nan")
    )

    print(f"  Pseudo-label selezionate: {n_selected}/{len(all_paths)} "
          f"({n_selected / len(all_paths) * 100:.1f}%) con confidenza >= {confidence_threshold}")
    print(f"  [SOLO ANALISI, non usato nel training] Accuracy delle pseudo-label selezionate "
          f"rispetto alle vere etichette: {pseudo_label_accuracy * 100:.2f}%")

    print("  [SOLO ANALISI] Dettaglio per classe vera (quante selezionate / totale, e purezza):")
    for class_idx in sorted(set(all_true_labels.tolist())):
        class_mask_true = all_true_labels == class_idx
        class_mask_selected = class_mask_true & selected_mask
        n_class_total = class_mask_true.sum()
        n_class_selected = class_mask_selected.sum()
        if n_class_selected > 0:
            class_purity = (all_pred_labels[class_mask_selected] == class_idx).mean()
        else:
            class_purity = float("nan")
        print(f"    {idx_to_class[class_idx]:12s}: {n_class_selected}/{n_class_total} selezionate, "
              f"purezza {class_purity * 100:.1f}%")

    return selected_samples


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


def main():
    parser = argparse.ArgumentParser(description="Self-training con pseudo-label sul target")
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

    # Il target completo (le vere etichette qui dentro vengono usate SOLO
    # per l'analisi di qualita' delle pseudo-label e per il monitoraggio)
    _, target_dataset_full = get_target_dataloader(
        data_dir=config["data"]["target_dir"],
        class_to_idx=class_to_idx,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    target_monitor_loader = DataLoader(
        target_dataset_full, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
    )

    # --- Step 1: carica il modello "insegnante" e genera le pseudo-label ---
    num_classes = config["model"]["num_classes"]
    teacher_checkpoint_path = config["model"]["teacher_checkpoint"]
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)

    teacher_model = get_baseline_model(num_classes=num_classes).to(device)
    teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"])
    print(f"Modello insegnante caricato da: {teacher_checkpoint_path}")

    confidence_threshold = config["training"]["confidence_threshold"]
    selected_samples = generate_pseudo_labels(
        teacher_model, target_dataset_full, device, confidence_threshold,
        batch_size=config["data"]["batch_size"], idx_to_class=idx_to_class,
    )

    if len(selected_samples) == 0:
        raise RuntimeError(
            "Nessuna pseudo-label ha superato la soglia di confidenza: "
            "abbassa 'confidence_threshold' nel config e riprova."
        )

    pseudo_labeled_dataset = PseudoLabeledDataset(selected_samples, transform=get_eval_transforms())

    # --- Step 2: fine-tuning su source (etichettato) + target pseudo-labeled ---
    combined_dataset = ConcatDataset([train_loader.dataset, pseudo_labeled_dataset])
    combined_loader = DataLoader(
        combined_dataset, batch_size=config["data"]["batch_size"], shuffle=True,
        num_workers=config["data"]["num_workers"],
    )

    model = get_baseline_model(num_classes=num_classes).to(device)
    model.load_state_dict(teacher_checkpoint["model_state_dict"])  # si riparte dall'insegnante

    criterion = nn.CrossEntropyLoss()
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    if config["training"]["optimizer"].lower() == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

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
        log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                              "target_acc_monitor", "time_sec"])

        for epoch in range(1, epochs + 1):
            start = time.time()
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            progress_bar = tqdm(combined_loader, desc="  Training", leave=False)
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits, _ = forward_with_features(model, images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / total
            train_acc = correct / total

            val_loss, val_acc = evaluate_classification(model, val_loader, device, criterion)
            _, target_acc_monitor = evaluate_classification(model, target_monitor_loader, device, criterion)
            elapsed = time.time() - start

            print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_acc={val_acc:.4f} | target_acc (monitor)={target_acc_monitor:.4f} | {elapsed:.1f}s")

            log_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc,
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

    print(f"\n[SUCCESS] Self-training completato. Miglior val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint salvato in: {checkpoint_path}")
    print(f"Log salvato in: {log_path}")


if __name__ == "__main__":
    main()
