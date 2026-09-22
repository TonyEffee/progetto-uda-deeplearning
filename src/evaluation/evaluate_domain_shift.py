"""
evaluate_domain_shift.py

Valuta il modello baseline (allenato SOLO sul source) sia sul source val
set sia sul target domain (le tue foto), per quantificare il domain shift.
Corrisponde all'Obiettivo Minimo 2 della consegna.

Produce:
    - Accuracy globale su source e su target (il numero chiave da riportare
      nel report: "il modello passa da X% sul source a Y% sul target")
    - Accuracy per-classe su entrambi i domini (utile perche' il target e'
      sbilanciato tra le classi)
    - Una confusion matrix per il target, salvata in figures/
    - Una lista dei primi N esempi target classificati in modo errato
      (utile per i "qualitative failure cases" richiesti dalla consegna)

USO:
    python -m src.evaluation.evaluate_domain_shift --config experiments/configs/baseline.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, classification_report

from src.datasets.datasets import get_source_dataloaders, get_target_dataloader
from src.models.baseline import get_baseline_model


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def run_inference(model, dataloader, device):
    """Ritorna tutte le predizioni e le label vere per un intero dataloader."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_preds), np.array(all_labels)


def per_class_accuracy(preds, labels, idx_to_class, present_classes=None):
    """Calcola l'accuracy per ciascuna classe presente nelle label vere."""
    results = {}
    unique_labels = sorted(set(labels.tolist()))
    for label_idx in unique_labels:
        class_name = idx_to_class[label_idx]
        if present_classes is not None and class_name not in present_classes:
            continue
        mask = labels == label_idx
        acc = (preds[mask] == labels[mask]).mean()
        results[class_name] = (acc, mask.sum())
    return results


def plot_confusion_matrix(cm, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix salvata in: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Valuta il domain shift source->target")
    parser.add_argument("--config", type=str, required=True, help="Percorso al file YAML di config")
    parser.add_argument("--num-failure-examples", type=int, default=10,
                         help="Quanti esempi di errore sul target stampare")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    print(f"Device in uso: {device}")

    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / config["output"]["checkpoint_name"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint non trovato in {checkpoint_path}. "
            f"Devi prima lanciare: python -m src.training.train_baseline --config {args.config}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = checkpoint["idx_to_class"]
    num_classes = len(class_to_idx)

    model = get_baseline_model(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint caricato (val_acc al training: {checkpoint['val_acc']:.4f}, "
          f"epoca {checkpoint['epoch']})")

    # --- Source val set (per confermare l'accuracy di riferimento) ---
    _, source_val_loader, source_class_to_idx = get_source_dataloaders(
        data_dir=config["data"]["source_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    assert source_class_to_idx == class_to_idx, "class_to_idx del checkpoint non combacia col dataset source!"

    source_preds, source_labels = run_inference(model, source_val_loader, device)
    source_acc = (source_preds == source_labels).mean()

    # --- Target domain (le tue foto) ---
    target_loader, target_dataset = get_target_dataloader(
        data_dir=config["data"]["target_dir"],
        class_to_idx=class_to_idx,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    target_preds, target_labels = run_inference(model, target_loader, device)
    target_acc = (target_preds == target_labels).mean()

    # --- Riepilogo del domain shift (il numero chiave per il report) ---
    print("\n" + "=" * 60)
    print("RISULTATO: QUANTIFICAZIONE DEL DOMAIN SHIFT")
    print("=" * 60)
    print(f"Accuracy su SOURCE val set : {source_acc * 100:.2f}%")
    print(f"Accuracy su TARGET (foto tue): {target_acc * 100:.2f}%")
    print(f"Degrado di performance     : {(source_acc - target_acc) * 100:.2f} punti percentuali")
    print("=" * 60)

    # --- Accuracy per classe ---
    present_classes = set(target_dataset.get_class_names())

    print("\n--- Accuracy per classe (SOURCE val) ---")
    for class_name, (acc, n) in per_class_accuracy(source_preds, source_labels, idx_to_class).items():
        print(f"  {class_name:12s}: {acc * 100:5.2f}%  (n={n})")

    print("\n--- Accuracy per classe (TARGET) ---")
    for class_name, (acc, n) in per_class_accuracy(
            target_preds, target_labels, idx_to_class, present_classes=present_classes).items():
        print(f"  {class_name:12s}: {acc * 100:5.2f}%  (n={n})")

    missing_classes = set(class_to_idx.keys()) - present_classes
    if missing_classes:
        print(f"\n[NOTA] Classi assenti nel target (non riproducibili localmente): "
              f"{sorted(missing_classes)}")

    # --- Confusion matrix sul target ---
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_class_names_sorted = sorted(present_classes, key=lambda c: class_to_idx[c])
    target_label_indices = [class_to_idx[c] for c in target_class_names_sorted]

    cm = confusion_matrix(target_labels, target_preds, labels=target_label_indices)
    plot_confusion_matrix(
        cm, target_class_names_sorted,
        figures_dir / "confusion_matrix_target_baseline.png",
        title="Confusion Matrix - Target Domain (Baseline, no UDA)",
    )

    print("\n--- Classification report dettagliato (TARGET) ---")
    print(classification_report(
        target_labels, target_preds,
        labels=target_label_indices,
        target_names=target_class_names_sorted,
        zero_division=0,
    ))

    # --- Esempi di errore qualitativi (per la sezione "failure cases" del report) ---
    print(f"\n--- Primi {args.num_failure_examples} esempi di errore sul TARGET ---")
    wrong_mask = target_preds != target_labels
    wrong_indices = np.where(wrong_mask)[0]
    for i in wrong_indices[: args.num_failure_examples]:
        img_path, true_label, true_class_name = target_dataset.samples[i]
        pred_class_name = idx_to_class[int(target_preds[i])]
        print(f"  {img_path.name:30s} | vero: {true_class_name:10s} | predetto: {pred_class_name}")

    print(f"\nTotale errori sul target: {wrong_mask.sum()} / {len(target_labels)} "
          f"({wrong_mask.mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()