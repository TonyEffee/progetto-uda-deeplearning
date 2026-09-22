"""
resize_target_images.py

Ridimensiona in-place tutte le immagini in data/target/<classe>/ a una
dimensione fissa (default 224x224, quella richiesta in input dalla rete).

Perche' serve: le foto scattate con l'iPhone sono a risoluzione piena
(diversi megapixel), e durante il training vengono ridimensionate ad ogni
singola iterazione da PyTorch — con CORAL/DANN, che rileggono il target
molte volte per epoca, questo diventa un collo di bottiglia enorme (si
puo' arrivare a 10x piu' lenti). Ridimensionarle una volta sola qui evita
il problema del tutto.

Va lanciato UNA VOLTA in locale (o su Colab, come fatto la prima volta)
prima di rifare lo zip del progetto, cosi' il dataset che finisce nello
zip e' gia' leggero e veloce da usare in ogni sessione futura.

USO:
    python -m src.utils.resize_target_images --target-dir data/target
"""

import argparse
from pathlib import Path

from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(description="Ridimensiona in-place le immagini del target dataset")
    parser.add_argument("--target-dir", type=str, default="data/target",
                         help="Cartella con le sottocartelle per classe")
    parser.add_argument("--size", type=int, default=224,
                         help="Lato in pixel del ridimensionamento (quadrato)")
    parser.add_argument("--quality", type=int, default=90,
                         help="Qualita' JPEG di salvataggio (1-100)")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Cartella non trovata: {target_dir}")

    target_size = (args.size, args.size)
    count = 0

    for class_dir in sorted(target_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_count = 0
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in VALID_EXTENSIONS:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(target_size, Image.BILINEAR)
                img.save(img_path, "JPEG", quality=args.quality)
                count += 1
                class_count += 1
        if class_count:
            print(f"  {class_dir.name}: {class_count} immagini ridimensionate")

    print(f"\n[SUCCESS] Ridimensionate {count} immagini totali in {target_dir}")


if __name__ == "__main__":
    main()
