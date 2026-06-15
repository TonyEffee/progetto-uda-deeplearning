> **NOTE: This file is the official template for the technical README of your repository.**  
> Before starting, make sure you have carefully read the **[INSTRUCTIONS.md](INSTRUCTIONS.md)**.  
> This file must contain **exclusively the technical aspects** of the project (Setup, Run, baseline Results). The textual and theoretical report should be placed in the **[`docs/REPORT.md`](docs/REPORT.md)** file.
> _Delete this note block before submission._

# Project 30: Unsupervised Domain Adaptation for Image Recognition under Domain Shift[cite: 1, 3]

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)[cite: 3]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)[cite: 3]

## 🧑‍🤝‍🧑 Group and Project Information

- **Group ID**: G31
- **Project ID**: 30

## 📝 Project Description

This project addresses the phenomenon of domain shift in image recognition systems[cite: 1]. We establish a baseline by training a Convolutional Neural Network on a source domain (Intel Image Classification dataset) and quantify the performance degradation when evaluating the model on a structurally different target domain (custom personal photographs)[cite: 1]. Finally, we implement Unsupervised Domain Adaptation (UDA) strategies, such as Feature Alignment and Pixel-Based Adaptation, to mitigate the domain gap and improve generalization without relying on target labels[cite: 1].

> 📖 **Official Report**: For all theoretical details, performance analysis, the architecture used, and group contributions, please refer to our formal paper: **[REPORT.md](docs/REPORT.md)**.

## 💻 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisites:**
To reproduce the environment and run the code, clone the repository and build the Conda environment using the provided file.

````bash
git clone [https://github.com/yourusername/your-repo.git](https://github.com/yourusername/your-repo.git)
cd your-repo
conda env create -f environment.yml
conda activate uda_project

**Dataset:**
Explain in 2 lines where to download the data from and in which folder it needs to reside (e.g., `data/raw/`).

### 2. Network Training
Provide the **exact commands** to start the training.

**Baseline Training:**
```bash
python -m src.training.train --config experiments/configs/baseline.yaml
````

**Improved Model Training:**

```bash
python -m src.training.train --config experiments/configs/model_v1.yaml
```

### 3. Evaluation

Provide the commands to reproduce the numbers in your summary table.

```bash
python -m src.evaluation.evaluate --config experiments/configs/model_v1.yaml
```

---

_For the declaration of individual tasks and the use of AI, refer to `docs/REPORT.md`._
