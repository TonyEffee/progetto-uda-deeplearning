"""
visualize_tsne.py

Visualizza con t-SNE le feature (512-dimensionali, estratte prima del
layer fully-connected) di source e target domain, per ciascun checkpoint
allenato finora (baseline, CORAL, DANN, DANN stabilizzato). Serve a
"vedere" quello che i numeri di accuracy raccontano solo a parole: se le
nuvole di punti source/target si sovrappongono di piu' dopo l'adattamento,
l'allineamento sta funzionando davvero, non solo sulla carta.

Genera una griglia 2x2 di scatter plot, uno per checkpoint, colorati per
dominio (source in blu, target in arancione).

USO:
    python -m src.evaluation.visualize_tsne \
        --source-dir data/source \
        --target-dir data/target \
        --output figures/tsne_comparison.png

Se un checkpoint elencato in DEFAULT_CHECKPOINTS non esiste (es. non hai
allenato una delle varianti), il relativo pannello viene semplicemente
lasciato vuoto con una nota, senza bloccare gli altri.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from src.datasets.datasets import get_eval_transforms, get_target_dataloader
from src.models.baseline import get_baseline_model
from src.models.feature_utils import forward_with_features

DEFAULT_CHECKPOINTS = [
    ("Baseline", "experiments/checkpoints/baseline_resnet18_best.pth"),
    ("CORAL", "experiments/checkpoints/coral_resnet18_best.pth"),
    ("DANN", "experiments/checkpoints/dann_resnet18_best.pth"),
    ("DANN stabilizzato", "experiments/checkpoints/dann_stable_resnet18_best.pth"),
]


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_source_subset_loader(source_dir, num_per_class, batch_size, seed):
    """
    Campiona un numero fisso di immagini per classe dal source val set,
    cosi' il numero di punti source nel grafico e' paragonabile a quello
    del target (che ha solo ~1200 immagini in tutto) invece di sommergerlo.
    """
    val_dir = Path(source_dir) / "val" / "seg_test"
    dataset = datasets.ImageFolder(val_dir, transform=get_eval_transforms())

    rng = random.Random(seed)
    targets = np.array(dataset.targets)
    selected_indices = []
    for class_idx in sorted(set(targets.tolist())):
        class_indices = np.where(targets == class_idx)[0].tolist()
        rng.shuffle(class_indices)
        selected_indices.extend(class_indices[:num_per_class])

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, dataset.class_to_idx


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    for images, _ in tqdm(dataloader, desc="  Estrazione feature", leave=False):
        images = images.to(device)
        _, features = forward_with_features(model, images)
        all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Visualizza le feature con t-SNE")
    parser.add_argument("--source-dir", type=str, default="data/source")
    parser.add_argument("--target-dir", type=str, default="data/target")
    parser.add_argument("--output", type=str, default="figures/tsne_comparison.png")
    parser.add_argument("--num-source-per-class", type=int, default=150,
                         help="Quante immagini source campionare per classe")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Device in uso: {device}")

    source_loader, class_to_idx = build_source_subset_loader(
        args.source_dir, args.num_source_per_class, args.batch_size, args.seed
    )
    print(f"Campione source: {len(source_loader.dataset)} immagini")

    target_loader, target_dataset = get_target_dataloader(
        args.target_dir, class_to_idx, batch_size=args.batch_size, num_workers=0
    )
    print(f"Immagini target: {len(target_dataset)}")

    n_cols = 2
    n_rows = (len(DEFAULT_CHECKPOINTS) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (label, ckpt_path) in zip(axes, DEFAULT_CHECKPOINTS):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"\n=== {label} === [SALTATO: checkpoint non trovato in {ckpt_path}]")
            ax.set_title(f"{label} (checkpoint non trovato)")
            ax.axis("off")
            continue

        print(f"\n=== {label} ===")
        checkpoint = torch.load(ckpt_path, map_location=device)
        num_classes = len(checkpoint["class_to_idx"])
        model = get_baseline_model(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        source_features = extract_features(model, source_loader, device)
        target_features = extract_features(model, target_loader, device)

        all_features = np.concatenate([source_features, target_features], axis=0)
        domain_labels = np.array([0] * len(source_features) + [1] * len(target_features))

        print(f"  Eseguo t-SNE su {len(all_features)} punti...")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed, init="pca")
        embedding = tsne.fit_transform(all_features)

        source_emb = embedding[domain_labels == 0]
        target_emb = embedding[domain_labels == 1]

        ax.scatter(source_emb[:, 0], source_emb[:, 1], s=10, alpha=0.6, label="Source", c="#4C72B0")
        ax.scatter(target_emb[:, 0], target_emb[:, 1], s=10, alpha=0.6, label="Target", c="#DD8452")
        ax.set_title(label)
        ax.legend(loc="best", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(DEFAULT_CHECKPOINTS):]:
        ax.axis("off")

    fig.suptitle("t-SNE delle feature: Source vs Target, per tecnica", fontsize=14)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\n[SUCCESS] Figura salvata in: {output_path}")


if __name__ == "__main__":
    main()
