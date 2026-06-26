# Run HERO training on all manuscript benchmark configurations.
conda run -n pyg python train.py --config configs/db15k.json
conda run -n pyg python train.py --config configs/mkg_w.json
conda run -n pyg python train.py --config configs/mkg_y.json
