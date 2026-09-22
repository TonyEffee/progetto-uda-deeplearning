# Project 30: Unsupervised Domain Adaptation for Image Recognition under Domain Shift

[![Report](https://img.shields.io/badge/Report-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Informazioni sul gruppo e sul progetto

- **Group ID**: G31
- **Project ID**: 30
- **Autore**: Gianluca Diquattro

## Descrizione del progetto

Il progetto studia il domain shift nel riconoscimento di scene. Un classificatore ResNet-18 viene addestrato sul dataset Intel Image Classification (source) e valutato su un dataset di fotografie personali (target), misurando il calo di accuratezza. Vengono poi confrontate diverse tecniche di Unsupervised Domain Adaptation, che non usano mai le etichette del target durante l'addestramento: CORAL, DANN, self-training con pseudo-label e color matching a livello di pixel.

**Report completo**: metodologia, analisi dei risultati, limiti e dichiarazione sull'uso dell'IA sono in **[docs/REPORT.md](docs/REPORT.md)**.

### Risultati principali

| Modello                            | Source val |   Target   |
| :--------------------------------- | :--------: | :--------: |
| Baseline (ResNet-18)               |   92,73%   |   74,05%   |
| CORAL                              |   93,57%   |   75,37%   |
| DANN                               |   93,50%   |   69,92%   |
| DANN stabilizzato                  |   93,53%   |   67,02%   |
| Self-training (insegnante CORAL)   |   93,80%   |   75,95%   |
| **Self-training + color matching** |   93,80%   | **76,12%** |

## Riproducibilità tecnica

### 1. Ambiente

```bash
git clone [URL del repository]
cd progetto-uda-deeplearning
conda env create -f environment.yml
conda activate dl-project
```

Il file `environment.yml` è configurato per PyTorch con CUDA 11.8: modificare `pytorch-cuda` in base alla propria GPU, o rimuoverlo per un'installazione solo CPU.

Per la preparazione del dataset target serve anche **ffmpeg** installato a livello di sistema (non tramite conda), usato per estrarre i frame dai video.

**Hardware.** Tutti i training sono stati eseguiti su Google Colab con GPU NVIDIA T4 (circa 3 minuti per epoca con CORAL e DANN). Su CPU un'epoca della baseline richiede circa 40 minuti, quindi il training su CPU non è praticabile. Il notebook usato su Colab è in [`notebooks/UDA_Colab_Training.ipynb`](notebooks/UDA_Colab_Training.ipynb). Valutazioni e analisi (t-SNE, robustezza, color matching) sono invece eseguibili anche su CPU.

### 2. Dati

**Dominio source: Intel Image Classification.** Scaricabile da [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). Le cartelle vanno organizzate così:

```
data/source/
├── train/seg_train/<classe>/*.jpg    # 14.034 immagini
└── val/seg_test/<classe>/*.jpg       # 3.000 immagini
```

con le sei classi `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

**Dominio target: foto personali.** Il dataset target (1.210 immagini, 5 classi, senza `glacier`) è composto da fotografie personali e **non è incluso nel repository**. La struttura attesa dagli script è:

```
data/target/<classe>/*.jpg
```

Per costruirlo da foto e video grezzi organizzati in `data/target_raw/<classe>/`:

```bash
# Conversione HEIC -> JPG ed estrazione dei frame dai video
python -m src.utils.extract_target_frames --raw-dir data/target_raw --out-dir data/target --target-per-class 140

# Ridimensionamento offline a 224x224 (necessario per tempi di training accettabili)
python -m src.utils.resize_target_images --target-dir data/target
```

Il primo script genera anche `data/target/manifest.csv`, che traccia la provenienza di ogni immagine (foto o video di origine). Il secondo sovrascrive le immagini in `data/target/`, lasciando intatti gli originali in `data/target_raw/`.

### 3. Checkpoint

I checkpoint addestrati non sono inclusi nel repository per motivi di dimensione (circa 45 MB ciascuno). Sono scaricabili da: [\[link alla cartella Google Drive\]](https://drive.google.com/drive/folders/1Awa9e225pI9LkriYDXPQLB918Z4ewcaW?usp=sharing)

Vanno posizionati in `experiments/checkpoints/`:

| File                              | Modello           |
| :-------------------------------- | :---------------- |
| `baseline_resnet18_best.pth`      | Baseline          |
| `coral_resnet18_best.pth`         | CORAL             |
| `dann_resnet18_best.pth`          | DANN              |
| `dann_stable_resnet18_best.pth`   | DANN stabilizzato |
| `self_training_resnet18_best.pth` | Self-training     |

### 4. Training

Tutti i comandi vanno lanciati dalla radice del repository. Gli iperparametri di ogni esperimento sono nei file YAML in `experiments/configs/`.

```bash
# Baseline sul source
python -m src.training.train_baseline --config experiments/configs/baseline.yaml

# CORAL (warm start dalla baseline)
python -m src.training.train_coral --config experiments/configs/coral.yaml

# DANN (warm start dalla baseline)
python -m src.training.train_dann --config experiments/configs/dann.yaml

# DANN stabilizzato (gamma, lambda massimo e peso della loss di dominio ridotti)
python -m src.training.train_dann_stable --config experiments/configs/dann_stable.yaml

# Self-training con pseudo-label (richiede il checkpoint CORAL come insegnante)
python -m src.training.train_self_training --config experiments/configs/self_training.yaml
```

La baseline va addestrata per prima, perché tutte le altre tecniche partono dai suoi pesi. Ogni script salva il checkpoint con la migliore accuratezza sul validation set del source e un log CSV per epoca in `experiments/logs/`.

### 5. Valutazione

Per riprodurre le accuratezze su source e target della tabella dei risultati:

```bash
python -m src.evaluation.evaluate_domain_shift --config experiments/configs/baseline.yaml
python -m src.evaluation.evaluate_domain_shift --config experiments/configs/coral.yaml
python -m src.evaluation.evaluate_domain_shift --config experiments/configs/dann.yaml
python -m src.evaluation.evaluate_domain_shift --config experiments/configs/dann_stable.yaml
python -m src.evaluation.evaluate_domain_shift --config experiments/configs/self_training.yaml
```

Lo script stampa accuratezza globale e per classe, classification report ed esempi di errore, e salva la confusion matrix sul target in `figures/confusion_matrix_target_baseline.png`. **Attenzione**: il nome del file è sempre lo stesso, quindi ogni esecuzione sovrascrive la precedente. Rinominare il file dopo ogni valutazione se si vogliono conservare le matrici di tutti i modelli.

### 6. Analisi aggiuntive

```bash
# t-SNE delle feature colorate per dominio (richiede i checkpoint di baseline, CORAL, DANN e DANN stabilizzato)
python -m src.evaluation.visualize_tsne --source-dir data/source --target-dir data/target --output figures/tsne_comparison.png

# t-SNE delle feature colorate per classe
python -m src.evaluation.visualize_tsne_by_class --source-dir data/source --target-dir data/target --output figures/tsne_by_class.png

# Robustezza della baseline a sfocatura, rumore, colore e risoluzione
python -m src.evaluation.robustness_analysis --source-dir data/source --checkpoint experiments/checkpoints/baseline_resnet18_best.pth

# Color matching (adattamento pixel-based) applicato a baseline, CORAL e self-training
python -m src.evaluation.evaluate_pixel_adaptation --source-dir data/source --target-dir data/target --method reinhard
```

Le figure vengono salvate in `figures/`; i risultati numerici dell'analisi di robustezza anche in `experiments/logs/robustness_analysis.csv`.

## 📁 Struttura del repository

```
├── data/                     # Dataset (non inclusi, vedi sezione 2)
├── docs/
│   ├── REPORT.md             # Report del progetto
│   └── slides.pdf            # Slide della presentazione
├── experiments/
│   ├── configs/              # Iperparametri di ogni esperimento (YAML)
│   ├── checkpoints/          # Pesi dei modelli (non inclusi, vedi sezione 3)
│   └── logs/                 # Log CSV di training e analisi
├── figures/                  # Figure usate nel report
├── notebooks/                # Notebook Google Colab
└── src/
    ├── datasets/             # Dataset loader e color matching
    ├── evaluation/           # Valutazione, t-SNE, robustezza, color matching
    ├── losses/               # Loss CORAL
    ├── models/               # ResNet-18, GRL, discriminatore di dominio
    ├── training/             # Script di training
    └── utils/                # Preparazione del dataset target
```

---
