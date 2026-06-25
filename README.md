# HERO: Hypergraph-enhanced Relation-aware Robust Reasoning for MMKGC

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-used-3776AB" />
  <img height="20" src="https://img.shields.io/badge/pytorch-%3E%3D2.0-EE4C2C" />
  <img height="20" src="https://img.shields.io/badge/numpy-%3E%3D1.24-013243" />
  <img height="20" src="https://img.shields.io/badge/tqdm-%3E%3D4.66-FFC107" />
  <img height="20" src="https://img.shields.io/badge/latex-manuscripts-008080" />
</p>

This repository contains the manuscript packages and reference implementation for
HERO, a hypergraph-enhanced relation-aware robust reasoning framework for multimodal
knowledge graph completion (MMKGC).

The source code is a reference implementation reconstructed from the manuscript. It
follows the equations and training protocol described in the paper, but it should not
be treated as the original experiment code.

## Repository Layout

```text
.
|-- README.md
|-- DATASETS.md
|-- requirements.txt
|-- train.py
|-- configs/
|   |-- db15k.json
|   |-- mkg_w.json
|   `-- mkg_y.json
|-- scripts/
|   |-- train_db15k.ps1
|   |-- train_mkg_w.ps1
|   |-- train_mkg_y.ps1
|   `-- train_all.ps1
|-- src/
|   `-- hero/
|       |-- data.py
|       |-- evaluate.py
|       |-- model.py
|       `-- utils.py
`-- papers/
    |-- cs/
    |-- springer/
    `-- submission_upload/
```

## Implemented Components

- Hypergraph-based multimodal KG pre-training.
- Relation-specific hyperedge construction from structural triples.
- Hypergraph convolution over the entity-relation incidence matrix.
- Triple discrimination with random head/tail corruption.
- Hyperedge-level contrastive learning.
- Perturbation-consistent robust scoring.
- Filtered link prediction with MRR and Hits@1/3/10.

## Installation

```bash
conda run -n pyg pip install -r requirements.txt
```

## Dataset Preparation

The paper uses three public MMKGC benchmarks: DB15K, MKG-W, and MKG-Y. This
repository does not download or redistribute the datasets. Place user-downloaded
datasets under:

```text
data/DB15K/
data/MKG-W/
data/MKG-Y/
```

Each dataset directory should contain:

```text
train.txt or train2id.txt
valid.txt or valid2id.txt
test.txt or test2id.txt
entity2id.txt
relation2id.txt
text_features.npy or text_features.pt
visual_features.npy or visual_features.pt
```

If mapping files are absent but triples use contiguous numeric IDs, the loader can
infer entity and relation counts. If multimodal features are absent, the code falls
back to trainable entity embeddings for structural debugging only.

See `DATASETS.md` for dataset source and redistribution notes.

## Training

Run a single dataset:

```bash
conda run -n pyg python train.py --config configs/db15k.json
conda run -n pyg python train.py --config configs/mkg_w.json
conda run -n pyg python train.py --config configs/mkg_y.json
```

Run with PowerShell scripts:

```powershell
.\scripts\train_db15k.ps1
.\scripts\train_mkg_w.ps1
.\scripts\train_mkg_y.ps1
.\scripts\train_all.ps1
```

## Verification

```bash
conda run -n pyg python -m py_compile train.py src/hero/__init__.py src/hero/data.py src/hero/model.py src/hero/evaluate.py src/hero/utils.py
```

## Paper Packages

- `papers/cs/`: CAS/CS journal manuscript package based on the Elsevier CAS template.
- `papers/springer/`: Springer Nature manuscript package.
- `papers/submission_upload/`: prepared submission package with PDF, source files,
  figure files, and upload notes.

## Badge Sources

The badges above are based on directly present repository files:

- `requirements.txt`: PyTorch, NumPy, and tqdm.
- `papers/**/*.tex`: LaTeX manuscript packages.

