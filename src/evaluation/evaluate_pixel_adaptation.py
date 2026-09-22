"""
evaluate_pixel_adaptation.py

Valuta l'effetto dell'adattamento di dominio BASATO SUI PIXEL
(color/histogram matching) sul target domain, confrontando l'accuracy
prima e dopo l'adattamento, con uno o piu' checkpoint gia' allenati.

Nessun riaddestramento: l'adattamento avviene interamente in fase di
preprocessing delle immagini target, quindi qualsiasi modello gia'
allenato puo' beneficiarne senza modifiche.

Salva anche alcuni esempi visivi prima/dopo, utili per il report: il
metodo e' "pixel-based" proprio perche' il suo effetto si vede a occhio
sulle immagini, a differenza di CORAL/DANN che agiscono su feature
invisibili.

USO:
    python -m src.evaluation.evaluate_pixel_adaptation \
        --source-dir data/source \
        --target-dir data/target \
        --method reinhard
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.datasets.color_matching import apply_color_matching, compute_source_statistics
from src.datasets.datasets import get_eval_transforms, get_target_dataloader
from src.models.baseline import get_baseline_model
from src.models.feature_utils import forward_with_features

CHECKPOINTS_TO_TEST = [
    ("Baseline", "experiments/checkpoints/baseline_resnet18_best.pth"),
    ("CORAL", "experiments/checkpoints/coral_resnet18_best.pth"),
    ("Self-training", "experiments/checkpoints/self_training_resnet18_best.pth"),
]


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ColorAdaptedTargetDataset(Dataset):
    """Target dataset con color matching applicato al volo su ogni immagine."""

    def __init__(self, samples, source_stats, transform):
        self.samples = samples  # lista di (path, label, class_name)
        self.source_stats = source_stats
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = apply_color_matching(image, self.source_stats)
        return self.transform(image), label


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device, idx_to_class=None):
    """Ritorna accuracy globale e, se richiesto, accuracy per classe."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(dataloader, desc="  Valutazione", leave=False):
        images = images.to(device)
        logits, _ = forward_with_features(model, images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    per_class = {}
    if idx_to_class is not None:
        for label_idx in sorted(set(all_labels.tolist())):
            mask = all_labels == label_idx
            per_class[idx_to_class[label_idx]] = (all_preds[mask] == all_labels[mask]).mean()

    return accuracy, per_class


def save_before_after_examples(samples, source_stats, output_path, num_examples=4, seed=42):
    """Salva una figura con alcuni esempi di immagini target prima/dopo l'adattamento."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(samples), size=min(num_examples, len(samples)), replace=False)

    fig, axes = plt.subplots(2, len(indices), figsize=(4 * len(indices), 8))
    if len(indices) == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        img_path, _, class_name = samples[idx]
        original = Image.open(img_path).convert("RGB")
        adapted = apply_color_matching(original, source_stats)

        axes[0, col].imshow(original)
        axes[0, col].set_title(f"{class_name} - originale", fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(adapted)
        axes[1, col].set_title(f"{class_name} - adattata", fontsize=10)
        axes[1, col].axis("off")

    fig.suptitle("Adattamento pixel-based: immagini target prima e dopo il color matching", fontsize=13)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Esempi visivi salvati in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Valuta l'adattamento pixel-based sul target")
    parser.add_argument("--source-dir", type=str, default="data/source")
    parser.add_argument("--target-dir", type=str, default="data/target")
    parser.add_argument("--method", type=str, default="reinhard", choices=["reinhard", "histogram"])
    parser.add_argument("--num-source-samples", type=int, default=600,
                         help="Quante immagini source usare per calcolare le statistiche di riferimento")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--examples-output", type=str, default="figures/pixel_adaptation_examples.png")
    args = parser.parse_args()

    device = get_device()
    print(f"Device in uso: {device}")
    print(f"Metodo di adattamento: {args.method}\n")

    print("Calcolo delle statistiche di colore del source domain...")
    source_stats = compute_source_statistics(
        args.source_dir, num_samples=args.num_source_samples, method=args.method
    )

    # Serve un checkpoint qualsiasi per ricavare class_to_idx
    first_available = next((p for _, p in CHECKPOINTS_TO_TEST if Path(p).exists()), None)
    if first_available is None:
        raise FileNotFoundError("Nessun checkpoint trovato in experiments/checkpoints/")
    reference_checkpoint = torch.load(first_available, map_location=device)
    class_to_idx = reference_checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Target originale
    target_loader_original, target_dataset = get_target_dataloader(
        args.target_dir, class_to_idx, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Target con color matching applicato
    adapted_dataset = ColorAdaptedTargetDataset(
        target_dataset.samples, source_stats, transform=get_eval_transforms()
    )
    target_loader_adapted = DataLoader(
        adapted_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print("\nGenerazione degli esempi visivi prima/dopo...")
    save_before_after_examples(target_dataset.samples, source_stats, args.examples_output)

    print("\n" + "=" * 70)
    print("RISULTATI: ADATTAMENTO PIXEL-BASED (color matching)")
    print("=" * 70)

    results_summary = []

    for label, ckpt_path in CHECKPOINTS_TO_TEST:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"\n=== {label} === [SALTATO: checkpoint non trovato]")
            continue

        print(f"\n=== {label} ===")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = get_baseline_model(num_classes=len(class_to_idx)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        print("  Target ORIGINALE:")
        acc_original, per_class_original = evaluate_accuracy(
            model, target_loader_original, device, idx_to_class
        )
        print(f"    Accuracy: {acc_original * 100:.2f}%")

        print("  Target ADATTATO (color matching):")
        acc_adapted, per_class_adapted = evaluate_accuracy(
            model, target_loader_adapted, device, idx_to_class
        )
        print(f"    Accuracy: {acc_adapted * 100:.2f}%")

        delta = (acc_adapted - acc_original) * 100
        print(f"  Differenza: {delta:+.2f} punti percentuali")

        print("  Dettaglio per classe (originale -> adattato):")
        for class_name in sorted(per_class_original.keys()):
            original_acc = per_class_original[class_name] * 100
            adapted_acc = per_class_adapted.get(class_name, 0) * 100
            print(f"    {class_name:12s}: {original_acc:5.2f}% -> {adapted_acc:5.2f}% "
                  f"({adapted_acc - original_acc:+.2f}pp)")

        results_summary.append((label, acc_original, acc_adapted, delta))

    print("\n" + "=" * 70)
    print("RIEPILOGO")
    print("=" * 70)
    print(f"{'Modello':<20} {'Originale':>12} {'Adattato':>12} {'Delta':>10}")
    for label, acc_original, acc_adapted, delta in results_summary:
        print(f"{label:<20} {acc_original * 100:>11.2f}% {acc_adapted * 100:>11.2f}% {delta:>+9.2f}pp")


if __name__ == "__main__":
    main()
