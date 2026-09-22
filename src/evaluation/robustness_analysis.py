"""
robustness_analysis.py

Extra Objective della consegna: valuta quanto il modello e' sensibile a
perturbazioni controllate (sfocatura, rumore, variazioni di colore/luce,
risoluzione), applicandole al SOURCE val set (dove il modello parte da
un'accuracy pulita e nota) per capire quale fattore, da solo, produce un
calo di accuracy paragonabile ai punti percentuali di degrado osservati
sul target reale.

Logica: se una perturbazione controllata (es. rumore forte) causa un calo
simile a quello misurato sul target vero, e' un indizio che quel fattore
specifico (es. rumore del sensore, differenze di illuminazione) e' un
contributo importante al domain shift reale osservato nel progetto.

Il modello NON viene mai riallenato qui: si usa un checkpoint gia'
allenato (di default il baseline, per studiare il fenomeno del domain
shift "puro", senza l'effetto delle tecniche UDA a confondere l'analisi).

USO:
    python -m src.evaluation.robustness_analysis \
        --source-dir data/source \
        --checkpoint experiments/checkpoints/baseline_resnet18_best.pth \
        --output figures/robustness_analysis.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as TF_module  # noqa: F401 (import esplicito per chiarezza, non usato direttamente)
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.baseline import get_baseline_model
from src.models.feature_utils import forward_with_features

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Parametri di ciascuna perturbazione, per livello di severita' (0 = nessuna perturbazione)
BLUR_SIGMAS = [0.0, 1.0, 2.0, 4.0]
NOISE_STDS = [0.0, 0.05, 0.10, 0.20]
COLOR_BRIGHTNESS = [1.0, 0.80, 0.60, 0.40]
COLOR_CONTRAST = [1.0, 0.85, 0.70, 0.55]
COLOR_SATURATION = [1.0, 0.80, 0.60, 0.40]
RESOLUTION_SCALES = [224, 112, 56, 28]

PERTURBATION_TYPES = ["blur", "noise", "color_shift", "resolution"]
SEVERITY_LEVELS = [0, 1, 2, 3]


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_transform(perturbation_type, level):
    """Costruisce la pipeline di trasformazione per una data perturbazione e severita'."""
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    if perturbation_type == "blur":
        sigma = BLUR_SIGMAS[level]
        if sigma > 0:
            ops.append(transforms.GaussianBlur(kernel_size=9, sigma=sigma))

    elif perturbation_type == "resolution":
        scale = RESOLUTION_SCALES[level]
        if scale < IMAGE_SIZE:
            # Riduce la risoluzione e poi la riporta a 224 (simula una foto acquisita a bassa risoluzione)
            ops.append(transforms.Resize((scale, scale)))
            ops.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))

    elif perturbation_type == "color_shift":
        brightness = COLOR_BRIGHTNESS[level]
        contrast = COLOR_CONTRAST[level]
        saturation = COLOR_SATURATION[level]
        if level > 0:
            ops.append(transforms.Lambda(lambda img: TF.adjust_brightness(img, brightness)))
            ops.append(transforms.Lambda(lambda img: TF.adjust_contrast(img, contrast)))
            ops.append(transforms.Lambda(lambda img: TF.adjust_saturation(img, saturation)))

    ops.append(transforms.ToTensor())

    if perturbation_type == "noise":
        noise_std = NOISE_STDS[level]
        if noise_std > 0:
            ops.append(transforms.Lambda(
                lambda t: torch.clamp(t + torch.randn_like(t) * noise_std, 0.0, 1.0)
            ))

    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(dataloader, desc="  Valutazione", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits, _ = forward_with_features(model, images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def get_severity_description(perturbation_type, level):
    """Descrizione leggibile del parametro usato a un dato livello, per il report."""
    if perturbation_type == "blur":
        return f"sigma={BLUR_SIGMAS[level]}"
    if perturbation_type == "noise":
        return f"std={NOISE_STDS[level]}"
    if perturbation_type == "color_shift":
        return (f"brightness={COLOR_BRIGHTNESS[level]}, "
                f"contrast={COLOR_CONTRAST[level]}, saturation={COLOR_SATURATION[level]}")
    if perturbation_type == "resolution":
        return f"{RESOLUTION_SCALES[level]}x{RESOLUTION_SCALES[level]} -> upscaled a 224"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Analisi di robustezza a perturbazioni controllate")
    parser.add_argument("--source-dir", type=str, default="data/source")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/baseline_resnet18_best.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-plot", type=str, default="figures/robustness_analysis.png")
    parser.add_argument("--output-csv", type=str, default="experiments/logs/robustness_analysis.csv")
    parser.add_argument("--target-accuracy-reference", type=float, default=0.7388,
                         help="Accuracy reale sul target (baseline, senza UDA), per confronto nel grafico")
    args = parser.parse_args()

    device = get_device()
    print(f"Device in uso: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = len(checkpoint["class_to_idx"])
    model = get_baseline_model(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint caricato: {args.checkpoint} (val_acc originale: {checkpoint['val_acc']:.4f})")

    val_dir = Path(args.source_dir) / "val" / "seg_test"
    results = {}  # {perturbation_type: [accuracy per livello]}

    for perturbation_type in PERTURBATION_TYPES:
        print(f"\n=== Perturbazione: {perturbation_type} ===")
        results[perturbation_type] = []
        for level in SEVERITY_LEVELS:
            transform = build_transform(perturbation_type, level)
            dataset = datasets.ImageFolder(val_dir, transform=transform)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)

            accuracy = evaluate_accuracy(model, dataloader, device)
            results[perturbation_type].append(accuracy)
            description = get_severity_description(perturbation_type, level)
            print(f"  Livello {level} ({description}): accuracy = {accuracy * 100:.2f}%")

    # --- Salva i risultati grezzi in CSV ---
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["perturbation_type", "severity_level", "description", "accuracy"])
        for perturbation_type in PERTURBATION_TYPES:
            for level in SEVERITY_LEVELS:
                description = get_severity_description(perturbation_type, level)
                writer.writerow([perturbation_type, level, description, results[perturbation_type][level]])
    print(f"\n[SUCCESS] Risultati grezzi salvati in: {csv_path}")

    # --- Riepilogo: qual e' il livello di ciascuna perturbazione che si avvicina
    #     di piu' al degrado osservato sul target reale? ---
    print("\n" + "=" * 60)
    print("RIEPILOGO: confronto con il degrado reale sul target")
    print("=" * 60)
    print(f"Accuracy reale sul target (riferimento): {args.target_accuracy_reference * 100:.2f}%")
    clean_accuracy = results[PERTURBATION_TYPES[0]][0]  # livello 0 e' identico per ogni categoria
    print(f"Accuracy pulita (nessuna perturbazione)  : {clean_accuracy * 100:.2f}%")
    real_degradation = clean_accuracy - args.target_accuracy_reference
    print(f"Degrado reale da spiegare                : {real_degradation * 100:.2f} punti percentuali\n")

    for perturbation_type in PERTURBATION_TYPES:
        accuracies = results[perturbation_type]
        degradations = [clean_accuracy - acc for acc in accuracies]
        # Trova il livello la cui degradazione si avvicina di piu' a quella reale
        closest_level = min(SEVERITY_LEVELS, key=lambda lv: abs(degradations[lv] - real_degradation))
        print(f"{perturbation_type:12s}: livello piu' vicino al degrado reale = {closest_level} "
              f"(degrado {degradations[closest_level] * 100:.2f}pp vs reale {real_degradation * 100:.2f}pp)")

    # --- Grafico ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for perturbation_type in PERTURBATION_TYPES:
        accuracies = [acc * 100 for acc in results[perturbation_type]]
        ax.plot(SEVERITY_LEVELS, accuracies, marker="o", label=perturbation_type)

    ax.axhline(y=args.target_accuracy_reference * 100, color="gray", linestyle="--",
               label="Accuracy reale sul target")
    ax.set_xlabel("Livello di severita'")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustezza a perturbazioni controllate (source val set)")
    ax.set_xticks(SEVERITY_LEVELS)
    ax.legend()
    ax.grid(alpha=0.3)

    output_path = Path(args.output_plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\n[SUCCESS] Grafico salvato in: {output_path}")


if __name__ == "__main__":
    main()
