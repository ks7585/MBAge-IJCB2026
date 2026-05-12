"""Build selective_channels.pkl for Sprint 23.

This is a one-time pre-processing step.  It ranks channels in each tongue
backbone by their R² decade-monotonicity score and writes the ranked index
arrays to data/selective_channels.pkl, which train_phase2_sprint23.py then
loads at startup.

Monotonicity criterion: for a channel to be "age-monotonic", the mean
activation across samples in each decade bin should increase (or decrease)
monotonically with age.  We quantify this as:

    R² of (decade_mean_activation ~ linear_age_midpoint)

High R² = the channel's average activation rises/falls smoothly with decade.
These are the channels least likely to add within-1Y noise.

Usage:
    python build_selective_channels.py --data_dir data
    # writes data/selective_channels.pkl

Verification output (printed to stdout):
    For each backbone: number of channels with R² > threshold and the actual
    count that will be selected (= min(n_monotonic, K)).
"""

import argparse
import pickle

import numpy as np

BACKBONE_INFO = {
    'ResNet18':       {'channels': 128, 'file': 'tongue_ResNet18_features_v2.0.pkl'},
    'MobileNetV2':    {'channels':  32, 'file': 'tongue_MobileNetV2_features_v2.0.pkl'},
    'EfficientNetB0': {'channels':  40, 'file': 'tongue_EfficientNetB0_features_v2.0.pkl'},
    'DenseNet121':    {'channels': 128, 'file': 'tongue_DenseNet121_features_v2.0.pkl'},
}

# How many channels to keep in the selective GAP per backbone.
# These match the R²>0.8 counts identified in the data analysis.
SELECTIVE_K = {
    'ResNet18':       50,
    'MobileNetV2':    24,
    'EfficientNetB0': 32,
    'DenseNet121':    46,
}

DECADE_STARTS   = [20, 30, 40, 50, 60]
DECADE_MIDPOINTS = [25., 35., 45., 55., 65.]


def decade_monotonicity_r2(feat_maps: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Return R² for linear fit of per-decade channel mean vs age midpoint.

    Args:
        feat_maps: (N, C, H, W)
        ages:      (N,) integer ages 20-69

    Returns:
        r2: (C,) array of R² values in [0, 1]
    """
    N, C, H, W = feat_maps.shape
    gap = feat_maps.mean(axis=(2, 3))     # (N, C)

    midpoints = np.array(DECADE_MIDPOINTS)   # (5,)
    r2 = np.zeros(C, dtype=np.float64)

    for c in range(C):
        decade_means = np.zeros(5)
        for di, ds in enumerate(DECADE_STARTS):
            mask = (ages >= ds) & (ages < ds + 10)
            if mask.sum() == 0:
                decade_means[di] = 0.0
            else:
                decade_means[di] = gap[mask, c].mean()

        # Linear fit: decade_means ~ a * midpoints + b
        # R² = 1 - SS_res / SS_tot
        coeffs = np.polyfit(midpoints, decade_means, deg=1)
        predicted = np.polyval(coeffs, midpoints)
        ss_res = np.sum((decade_means - predicted) ** 2)
        ss_tot = np.sum((decade_means - decade_means.mean()) ** 2)
        r2[c] = 1.0 - ss_res / (ss_tot + 1e-12)

    return r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--r2_threshold', type=float, default=0.8,
                   help='R² threshold for "monotonic" channel classification')
    args = p.parse_args()

    ages = None
    rankings = {}

    for name, info in BACKBONE_INFO.items():
        path = f"{args.data_dir}/{info['file']}"
        print(f'Loading {name} from {path} ...')
        with open(path, 'rb') as f:
            d = pickle.load(f)

        feat_maps = d['train']['features'].astype(np.float32)  # (N, C, H, W)
        if ages is None:
            ages = np.array(d['train']['age'], dtype=np.float32)

        print(f'  Shape: {feat_maps.shape}')

        r2 = decade_monotonicity_r2(feat_maps, ages)

        # Sort channels by descending R² (most monotonic first)
        sorted_idx = np.argsort(r2)[::-1].astype(np.int64)

        n_monotonic = int((r2 >= args.r2_threshold).sum())
        k = SELECTIVE_K[name]
        print(f'  R² ≥ {args.r2_threshold}: {n_monotonic}/{info["channels"]} channels')
        print(f'  Selecting top-{k} (R² range: '
              f'{r2[sorted_idx[0]]:.3f} … {r2[sorted_idx[k-1]]:.3f})')

        rankings[name] = sorted_idx  # full sorted list; train script takes top-K

    out_path = f'{args.data_dir}/selective_channels.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(rankings, f, protocol=4)

    print(f'\nSaved to {out_path}')
    print('Run train_phase2_sprint23.py — it will load this file automatically.')


if __name__ == '__main__':
    main()
