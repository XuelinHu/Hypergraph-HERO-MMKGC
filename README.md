# HERO: Hypergraph-enhanced Relation-aware Robust Reasoning for MMKGC

This repository contains the manuscript materials and a reconstructed reference
implementation for HERO, a hypergraph-enhanced relation-aware robust reasoning
framework for multimodal knowledge graph completion.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── configs/                    # Experiment configuration files
├── src/                        # HERO reference implementation package
├── train.py                    # Training and evaluation entry point
├── DATASETS.md                 # Dataset source and license notes
└── papers/
    ├── cs/                     # CAS/CS journal manuscript package
    ├── springer/               # Springer Nature manuscript package
    └── submission_upload/      # Prepared upload package for submission
```

## Code

The implementation in `src/` is a reference implementation derived from the LaTeX
manuscript, not the original experimental code. It includes:

- hypergraph-based MKG pre-training;
- relation-specific hyperedge construction;
- triple discrimination and hyperedge contrastive losses;
- perturbation-consistent robust scoring;
- filtered link-prediction evaluation.

Run syntax check:

```bash
conda run -n pyg python -m py_compile train.py src/hero/__init__.py src/hero/data.py src/hero/model.py src/hero/evaluate.py src/hero/utils.py
```

Run training after preparing a dataset:

```bash
conda run -n pyg python train.py --config configs/db15k.json
```

## Datasets

The paper uses DB15K, MKG-W, and MKG-Y. These datasets are public MMKGC benchmarks
reused by several open research repositories, but this repository does not download
or redistribute them. Place user-downloaded datasets under:

```text
data/DB15K/
data/MKG-W/
data/MKG-Y/
```

See `DATASETS.md` for source and license notes.

## Papers

- `papers/cs/`: current CAS/CS manuscript version and related template files.
- `papers/springer/`: Springer Nature template version.
- `papers/submission_upload/`: submission-ready package, including PDF, LaTeX source,
  figures, and upload notes.
