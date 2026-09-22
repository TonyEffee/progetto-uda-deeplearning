"""
color_matching.py

Adattamento di dominio BASATO SUI PIXEL (pixel-based adaptation), la
seconda categoria di tecniche UDA indicata nella consegna accanto al
feature alignment.

A differenza di CORAL e DANN, che agiscono sullo spazio delle feature
interne alla rete, qui si interviene direttamente sulle IMMAGINI del
target, prima ancora che entrino nel classificatore: le statistiche di
colore del target vengono trasformate per assomigliare a quelle del
source, in modo che il modello - allenato sul source - si trovi davanti
immagini "stilisticamente" piu' familiari.

E' una versione molto piu' leggera di quello che farebbe un CycleGAN
(che impara una traduzione di stile con una rete generativa), ottenuta
con una trasformazione statistica chiusa invece che con un secondo
addestramento: nessuna rete aggiuntiva, nessun training, costo
computazionale trascurabile. In cambio, puo' correggere solo differenze
di colore/luminosita'/contrasto globali, non differenze strutturali
(prospettiva, contenuto, texture).

Sono implementati due metodi:

1) "reinhard" (Reinhard et al., 2001): trasferimento di colore nello
   spazio LAB. Per ogni canale, l'immagine target viene normalizzata
   (media 0, std 1) e poi riscalata con media e deviazione standard del
   source. Preserva bene l'aspetto naturale dell'immagine.

2) "histogram": histogram matching per canale RGB, che forza la
   distribuzione cumulativa di ciascun canale del target a coincidere
   con quella del source. Piu' aggressivo del metodo di Reinhard.

Le statistiche del source vengono calcolate UNA VOLTA su un campione del
source training set (vedi compute_source_statistics) e poi riusate per
tutte le immagini target: nessuna etichetta del target viene mai usata,
quindi il metodo resta pienamente non supervisionato.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color as skcolor
from skimage import exposure

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compute_source_statistics(source_dir, num_samples=600, seed=42, method="reinhard"):
    """
    Calcola le statistiche di colore di riferimento del source domain,
    campionando un sottoinsieme casuale di immagini dal training set.

    Per "reinhard": ritorna (media, std) per i tre canali LAB.
    Per "histogram": ritorna un'immagine di riferimento "media" costruita
    concatenando i pixel del campione, usata come riferimento per il
    matching degli istogrammi.
    """
    source_dir = Path(source_dir)
    train_dir = source_dir / "train" / "seg_train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Cartella source non trovata: {train_dir}")

    all_images = []
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            all_images.extend([p for p in class_dir.iterdir()
                                if p.suffix.lower() in IMAGE_EXTENSIONS])

    rng = random.Random(seed)
    rng.shuffle(all_images)
    sampled = all_images[:num_samples]
    print(f"  Statistiche source calcolate su {len(sampled)} immagini campionate")

    if method == "reinhard":
        # Accumula media e varianza nello spazio LAB su tutto il campione
        lab_pixels = []
        for img_path in sampled:
            img = np.asarray(Image.open(img_path).convert("RGB")) / 255.0
            lab = skcolor.rgb2lab(img)
            lab_pixels.append(lab.reshape(-1, 3))
        lab_pixels = np.concatenate(lab_pixels, axis=0)
        mean = lab_pixels.mean(axis=0)
        std = lab_pixels.std(axis=0)
        return {"method": "reinhard", "mean": mean, "std": std}

    elif method == "histogram":
        # Costruisce un riferimento concatenando i pixel del campione
        # (ridimensionati per contenere l'uso di memoria)
        reference_pixels = []
        for img_path in sampled:
            img = Image.open(img_path).convert("RGB").resize((64, 64))
            reference_pixels.append(np.asarray(img))
        reference = np.concatenate([p.reshape(-1, 1, 3) for p in reference_pixels], axis=0)
        return {"method": "histogram", "reference": reference}

    raise ValueError(f"Metodo non riconosciuto: {method}")


def apply_color_matching(pil_image, source_stats):
    """
    Applica l'adattamento di colore a una singola immagine PIL del target,
    usando le statistiche del source calcolate in precedenza.
    Ritorna una nuova immagine PIL.
    """
    method = source_stats["method"]
    img = np.asarray(pil_image.convert("RGB")) / 255.0

    if method == "reinhard":
        lab = skcolor.rgb2lab(img)
        target_mean = lab.reshape(-1, 3).mean(axis=0)
        target_std = lab.reshape(-1, 3).std(axis=0)

        # Evita divisioni per zero su immagini quasi uniformi
        safe_std = np.where(target_std < 1e-6, 1e-6, target_std)

        lab_adapted = (lab - target_mean) / safe_std * source_stats["std"] + source_stats["mean"]

        # Riporta i valori nei range validi dello spazio LAB
        lab_adapted[..., 0] = np.clip(lab_adapted[..., 0], 0, 100)
        lab_adapted[..., 1:] = np.clip(lab_adapted[..., 1:], -128, 127)

        rgb_adapted = skcolor.lab2rgb(lab_adapted)

    elif method == "histogram":
        img_uint8 = (img * 255).astype(np.uint8)
        matched = exposure.match_histograms(img_uint8, source_stats["reference"], channel_axis=-1)
        rgb_adapted = matched / 255.0

    else:
        raise ValueError(f"Metodo non riconosciuto: {method}")

    rgb_adapted = np.clip(rgb_adapted, 0.0, 1.0)
    return Image.fromarray((rgb_adapted * 255).astype(np.uint8))
