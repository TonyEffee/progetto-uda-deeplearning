"""
datasets.py

Dataset e DataLoader per il progetto di Domain Adaptation.

- get_source_dataloaders(...): usa torchvision.datasets.ImageFolder sul
  source domain (Intel Image Classification), che ha gia' la struttura
  data/source/train/seg_train/<classe> e data/source/val/seg_test/<classe>.

- TargetDataset: dataset per le immagini raccolte personalmente
  (data/target/<classe>/*.jpg). NON usa ImageFolder perche' il target
  contiene un sottoinsieme delle classi del source (manca 'glacier',
  non riproducibile a Ragusa) e vogliamo che gli indici di classe
  restino coerenti con quelli del source: altrimenti l'accuracy
  calcolata in fase di valutazione sarebbe semplicemente sbagliata
  (es. una foto di 'mountain' nel target potrebbe ricevere l'indice
  che nel source corrisponde a 'sea').
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Estensioni immagine valide nel target (i frame estratti dai video e le
# foto convertite sono sempre .jpg, ma teniamo anche png/jpeg per sicurezza)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def get_train_transforms():
    """Augmentation per il training sul source (aiuta la generalizzazione)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms():
    """Transform deterministica per validation/test (source e target)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_source_dataloaders(data_dir="data/source", batch_size=32, num_workers=4):
    """
    Ritorna train_loader, val_loader e class_to_idx per il source domain.

    class_to_idx e' il dizionario {nome_classe: indice} costruito da
    ImageFolder in ordine alfabetico (es. {'buildings': 0, 'forest': 1,
    'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}). Va sempre
    riutilizzato per costruire il TargetDataset, per garantire la
    coerenza delle etichette tra source e target.
    """
    train_dir = Path(data_dir) / "train" / "seg_train"
    val_dir = Path(data_dir) / "val" / "seg_test"

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(val_dir, transform=get_eval_transforms())

    assert train_dataset.class_to_idx == val_dataset.class_to_idx, (
        "Le classi di train e val non combaciano! Controlla le cartelle "
        f"{train_dir} e {val_dir}."
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.class_to_idx


class TargetDataset(Dataset):
    """
    Dataset per le immagini del target domain raccolte personalmente.

    Usa il class_to_idx del SOURCE domain (passato come argomento) per
    assegnare le label, cosi' da restare coerente anche se il target
    contiene solo un sottoinsieme delle classi del source.
    """

    def __init__(self, data_dir, class_to_idx, transform=None):
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform or get_eval_transforms()
        self.samples = []  # lista di (path, label, nome_classe)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Cartella target non trovata: {self.data_dir}")

        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                print(f"[WARN] Classe '{class_name}' presente nel target ma non nel "
                      f"source, verra' ignorata nella valutazione.")
                continue
            label = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((img_path, label, class_name))

        if not self.samples:
            raise ValueError(f"Nessuna immagine trovata in {self.data_dir}")

        # Segnala se qualche classe del source non ha alcun campione nel target
        # (nel nostro caso ci aspettiamo che accada per 'glacier')
        target_classes_present = {name for _, _, name in self.samples}
        missing = set(class_to_idx.keys()) - target_classes_present
        if missing:
            print(f"[INFO] Classi del source assenti nel target: {sorted(missing)} "
                  f"(atteso per classi non riproducibili localmente, es. glacier)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_names(self):
        """Utile per costruire la confusion matrix con i nomi delle classi presenti."""
        return [name for _, _, name in self.samples]


def get_target_dataloader(data_dir, class_to_idx, batch_size=32, num_workers=4):
    dataset = TargetDataset(data_dir, class_to_idx, transform=get_eval_transforms())
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, dataset