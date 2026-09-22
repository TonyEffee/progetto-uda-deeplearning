"""
visualize_tsne_by_class.py

Variante di visualize_tsne.py: invece di colorare i punti per dominio
(source/target), li colora per CLASSE, usando un marker diverso per
indicare il dominio (cerchio pieno = source, triangolo = target). Utile
per capire se, oltre all'allineamento tra domini, le classi restano ben
separate tra loro dopo l'adattamento — un allineamento "riuscito" che
pero' mescola anche le classi tra loro non sarebbe un buon segno.

Riusa l'estrazione delle feature e l'impostazione di visualize_tsne.py,
cambia solo la logica di plotting (colore per classe + marker per
dominio invece di colore per dominio).

USO:
    python -m src.evaluation.visualize_tsne_by_class \
        --source-dir data/source \
        --target-dir data/target \
        --output figures/tsne_by_class.png
"""

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.datasets.datasets import get_target_dataloader
from src.evaluation.visualize_tsne import (
    DEFAULT_CHECKPOINTS,
    build_source_subset_loader,
    extract_features,
    get_device,
)
from src.models.baseline import get_baseline_model

# Colori fissi per classe, coerenti con l'ordine alfabetico di
# ImageFolder (buildings=0, forest=1, glacier=2, mountain=3, sea=4, street=5)
CLASS_COLORS = {
    0: "#4C72B0",  # buildings
    1: "#55A868",  # forest
    2: "#8172B2",  # glacier (mai presente nel target)
    3: "#C44E52",  # mountain
    4: "#CCB974",  # sea
    5: "#64B5CD",  # street
}


def main():
    parser = argparse.ArgumentParser(description="Visualizza le feature con t-SNE, colorate per classe")
    parser.add_argument("--source-dir", type=str, default="data/source")
    parser.add_argument("--target-dir", type=str, default="data/target")
    parser.add_argument("--output", type=str, default="figures/tsne_by_class.png")
    parser.add_argument("--num-source-per-class", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Device in uso: {device}")

    source_loader, class_to_idx = build_source_subset_loader(
        args.source_dir, args.num_source_per_class, args.batch_size, args.seed
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Label di classe per il campione source, lette direttamente dai target
    # dell'ImageFolder sottostante (evita di ricaricare le immagini solo per
    # recuperare l'etichetta)
    subset = source_loader.dataset
    source_class_labels = np.array([subset.dataset.targets[i] for i in subset.indices])
    print(f"Campione source: {len(source_class_labels)} immagini")

    target_loader, target_dataset = get_target_dataloader(
        args.target_dir, class_to_idx, batch_size=args.batch_size, num_workers=0
    )
    target_class_labels = np.array([label for _, label, _ in target_dataset.samples])
    print(f"Immagini target: {len(target_class_labels)}")

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
        all_class_labels = np.concatenate([source_class_labels, target_class_labels])
        domain_labels = np.array([0] * len(source_features) + [1] * len(target_features))

        print(f"  Eseguo t-SNE su {len(all_features)} punti...")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed, init="pca")
        embedding = tsne.fit_transform(all_features)

        for class_idx, color in CLASS_COLORS.items():
            for domain_idx, marker in [(0, "o"), (1, "^")]:
                mask = (all_class_labels == class_idx) & (domain_labels == domain_idx)
                if mask.sum() == 0:
                    continue
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           s=14, alpha=0.65, color=color, marker=marker,
                           edgecolors="none")

        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(DEFAULT_CHECKPOINTS):]:
        ax.axis("off")

    # Legenda unica per l'intera figura: colore = classe, forma = dominio
    class_handles = [
        mlines.Line2D([], [], color=color, marker="o", linestyle="None",
                      markersize=8, label=idx_to_class[class_idx])
        for class_idx, color in CLASS_COLORS.items()
    ]
    domain_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=8, label="Source"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None",
                      markersize=8, label="Target"),
    ]

    fig.legend(handles=class_handles, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.04), fontsize=9, title="Classe (colore)")
    fig.legend(handles=domain_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.99), fontsize=9, title="Dominio (forma)")

    fig.suptitle("t-SNE delle feature, colorate per classe", fontsize=14, y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[SUCCESS] Figura salvata in: {output_path}")


if __name__ == "__main__":
    main()
