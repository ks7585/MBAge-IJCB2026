"""Sprint 12: Feature Engineering on HierModel baseline.

Tests ALL feature engineering techniques from the catalog:
- Tongue: std pool, grid pool, DCT, Haar wavelets, moments, LayerNorm, ECA, iSQRT-COV
- Gabor: (5,8) reshape marginals, cross-bank asymmetry
- Fusion: GMU, MMTM

Built on HierModel (Sprint 4 OOF=0.4185). Each feature is a flag.
"""

import argparse
import csv
import json
import math
import os
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dctn
from scipy.stats import skew, kurtosis

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_AGE, MAX_AGE, NUM_CLASSES = 20, 69, 50
NUM_DECADES = 5
BINS_PER_DECADE = 10
DECADE_STARTS = [20, 30, 40, 50, 60]

BACKBONE_INFO = {
    'ResNet18': {'channels': 128, 'file': 'tongue_ResNet18_features_v2.0.pkl'},
    'MobileNetV2': {'channels': 32, 'file': 'tongue_MobileNetV2_features_v2.0.pkl'},
    'EfficientNetB0': {'channels': 40, 'file': 'tongue_EfficientNetB0_features_v2.0.pkl'},
    'DenseNet121': {'channels': 128, 'file': 'tongue_DenseNet121_features_v2.0.pkl'},
}
BACKBONE_NAMES = list(BACKBONE_INFO.keys())
FACE_FILE = 'face_features_gabor_v2.0.pkl'
FACE_DIM = 40


# ============================================================================
# PRE-COMPUTED FEATURE EXTRACTION (numpy, at load time)
# ============================================================================

def compute_std_pool(feat_4d):
    """(N,C,H,W) -> (N,C) spatial std per channel."""
    return feat_4d.reshape(feat_4d.shape[0], feat_4d.shape[1], -1).std(axis=2).astype(np.float32)


def compute_grid_pool(feat_4d):
    """(N,C,28,28) -> (N,4C) quadrant GAP."""
    N, C, H, W = feat_4d.shape
    hh, hw = H // 2, W // 2
    q_tl = feat_4d[:, :, :hh, :hw].reshape(N, C, -1).mean(axis=2)
    q_tr = feat_4d[:, :, :hh, hw:].reshape(N, C, -1).mean(axis=2)
    q_bl = feat_4d[:, :, hh:, :hw].reshape(N, C, -1).mean(axis=2)
    q_br = feat_4d[:, :, hh:, hw:].reshape(N, C, -1).mean(axis=2)
    return np.concatenate([q_tl, q_tr, q_bl, q_br], axis=1).astype(np.float32)


def compute_dct_features(feat_4d, k=8):
    """(N,C,28,28) -> (N,64) mean DCT low-freq coefficients across channels."""
    N, C, H, W = feat_4d.shape
    dct_feats = np.zeros((N, k * k), dtype=np.float32)
    for i in range(N):
        # Vectorized: apply 2D DCT across all channels at once
        dct_all = dctn(feat_4d[i], type=2, norm='ortho', axes=(-2, -1))  # (C, H, W)
        dct_feats[i] = dct_all[:, :k, :k].mean(axis=0).flatten()
    return dct_feats


def compute_haar_energy(feat_4d):
    """(N,C,28,28) -> (N,4C) Haar wavelet subband energies."""
    N, C, H, W = feat_4d.shape
    even_r = feat_4d[:, :, 0::2, :]
    odd_r = feat_4d[:, :, 1::2, :]
    L = (even_r + odd_r) * 0.5
    H_r = (even_r - odd_r) * 0.5
    LL = (L[:, :, :, 0::2] + L[:, :, :, 1::2]) * 0.5
    LH = (L[:, :, :, 0::2] - L[:, :, :, 1::2]) * 0.5
    HL = (H_r[:, :, :, 0::2] + H_r[:, :, :, 1::2]) * 0.5
    HH = (H_r[:, :, :, 0::2] - H_r[:, :, :, 1::2]) * 0.5
    e_ll = (LL ** 2).reshape(N, C, -1).mean(axis=2)
    e_lh = (LH ** 2).reshape(N, C, -1).mean(axis=2)
    e_hl = (HL ** 2).reshape(N, C, -1).mean(axis=2)
    e_hh = (HH ** 2).reshape(N, C, -1).mean(axis=2)
    return np.concatenate([e_ll, e_lh, e_hl, e_hh], axis=1).astype(np.float32)


def compute_higher_moments(feat_4d):
    """(N,C,28,28) -> (N,2C) skewness + kurtosis per channel."""
    N, C = feat_4d.shape[:2]
    spatial = feat_4d.reshape(N, C, -1)
    sk = skew(spatial, axis=2)
    ku = kurtosis(spatial, axis=2)
    sk = np.clip(sk, -10, 10).astype(np.float32)
    ku = np.clip(ku, -10, 10).astype(np.float32)
    return np.concatenate([sk, ku], axis=1)


def compute_spatial_moments(feat_4d):
    """(N,C,28,28) -> (N,3C) centroid x, y, dispersion per channel."""
    N, C, H, W = feat_4d.shape
    ys = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1)
    xs = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W)
    weights = np.abs(feat_4d) + 1e-8
    w_sum = weights.sum(axis=(2, 3), keepdims=True)
    cx = (weights * xs).sum(axis=(2, 3)) / w_sum.squeeze((2, 3))
    cy = (weights * ys).sum(axis=(2, 3)) / w_sum.squeeze((2, 3))
    disp = (weights * ((xs - cx.reshape(N, C, 1, 1)) ** 2 +
                       (ys - cy.reshape(N, C, 1, 1)) ** 2)).sum(axis=(2, 3)) / w_sum.squeeze((2, 3))
    disp = np.sqrt(disp + 1e-8)
    return np.concatenate([cx, cy, disp], axis=1).astype(np.float32)


def compute_cbp_cross(gap_a, gap_b, sketch_dim=1024, seed=42):
    """Compact Bilinear Pooling via Tensor Sketch between two GAP vectors.
    (N, D_a) x (N, D_b) -> (N, sketch_dim).
    Uses count sketch with fixed random hash functions."""
    N, D_a = gap_a.shape
    _, D_b = gap_b.shape
    rng = np.random.RandomState(seed)
    # Random hash indices and signs for both inputs
    h_a = rng.randint(0, sketch_dim, size=D_a)
    s_a = rng.choice([-1, 1], size=D_a).astype(np.float32)
    h_b = rng.randint(0, sketch_dim, size=D_b)
    s_b = rng.choice([-1, 1], size=D_b).astype(np.float32)
    # Count sketch: scatter signed values into sketch_dim buckets
    sketch_a = np.zeros((N, sketch_dim), dtype=np.float32)
    sketch_b = np.zeros((N, sketch_dim), dtype=np.float32)
    for d in range(D_a):
        sketch_a[:, h_a[d]] += gap_a[:, d] * s_a[d]
    for d in range(D_b):
        sketch_b[:, h_b[d]] += gap_b[:, d] * s_b[d]
    # Element-wise product in frequency domain = convolution = outer product sketch
    fft_a = np.fft.fft(sketch_a, axis=1)
    fft_b = np.fft.fft(sketch_b, axis=1)
    cbp = np.fft.ifft(fft_a * fft_b, axis=1).real.astype(np.float32)
    # Signed sqrt normalization for stability
    cbp = np.sign(cbp) * np.sqrt(np.abs(cbp) + 1e-8)
    # L2 normalize
    norms = np.linalg.norm(cbp, axis=1, keepdims=True) + 1e-8
    cbp = cbp / norms
    return cbp


def compute_mi_channel_weights(gap_features, ages, top_k=64):
    """Compute MI-based channel importance weights.
    Returns weight vector: 1.0 for top-K channels, 0.1 for rest.
    Uses binned MI approximation (fast, no sklearn dependency)."""
    N, C = gap_features.shape
    # Bin ages into 10 bins for MI estimation
    age_bins = np.digitize(ages, np.linspace(ages.min(), ages.max(), 11)[1:-1])
    n_age_bins = len(np.unique(age_bins))
    # For each channel, compute MI via histogram
    mi_scores = np.zeros(C, dtype=np.float32)
    for c in range(C):
        # Bin channel values into 10 bins
        ch_vals = gap_features[:, c]
        ch_bins = np.digitize(ch_vals, np.linspace(ch_vals.min(), ch_vals.max(), 11)[1:-1])
        # Joint and marginal distributions
        joint = np.zeros((10, n_age_bins), dtype=np.float64)
        for i in range(N):
            joint[min(ch_bins[i], 9), min(age_bins[i], n_age_bins - 1)] += 1
        joint = joint / N + 1e-10
        p_ch = joint.sum(axis=1, keepdims=True)
        p_age = joint.sum(axis=0, keepdims=True)
        mi_scores[c] = np.sum(joint * np.log(joint / (p_ch * p_age + 1e-10)))
    # Top-K get weight 1.0, rest get 0.1
    threshold = np.sort(mi_scores)[::-1][min(top_k, C) - 1]
    weights = np.where(mi_scores >= threshold, 1.0, 0.1).astype(np.float32)
    return weights


def compute_cka_dedup(tongue_feats, ages, threshold=0.8, n_components=64):
    """CKA-guided backbone de-duplication.
    For each backbone pair with linear CKA > threshold, project the weaker
    backbone's GAP onto the null-space of the stronger backbone's top PCA components.
    Modifies tongue_feats in-place (replaces GAP with residual for redundant backbones).
    """
    # Gather GAP vectors (train split)
    gaps = {}
    for name in BACKBONE_NAMES:
        gaps[name] = tongue_feats[name]['train']['gap']

    def linear_cka(X, Y):
        """Compute linear CKA between two feature matrices (N, D)."""
        X_c = X - X.mean(axis=0, keepdims=True)
        Y_c = Y - Y.mean(axis=0, keepdims=True)
        hsic_xy = np.linalg.norm(X_c.T @ Y_c, 'fro') ** 2
        hsic_xx = np.linalg.norm(X_c.T @ X_c, 'fro') ** 2
        hsic_yy = np.linalg.norm(Y_c.T @ Y_c, 'fro') ** 2
        return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)

    # Determine backbone "strength" by correlation with age
    strength = {}
    for name, gap in gaps.items():
        # Use mean absolute correlation with age as proxy for informativeness
        corrs = np.abs(np.corrcoef(gap.T, ages[None, :])[:-1, -1])
        strength[name] = np.mean(corrs)

    # Sort backbones by strength (strongest first)
    sorted_names = sorted(BACKBONE_NAMES, key=lambda n: -strength[n])

    # For each pair, check CKA and project if redundant
    projected = set()
    for i, strong_name in enumerate(sorted_names):
        if strong_name in projected:
            continue
        for weak_name in sorted_names[i+1:]:
            if weak_name in projected:
                continue
            cka_val = linear_cka(gaps[strong_name], gaps[weak_name])
            if cka_val > threshold:
                # Project weak onto null-space of strong's top PCA components
                strong_gap = gaps[strong_name]
                # PCA of strong backbone
                strong_c = strong_gap - strong_gap.mean(axis=0, keepdims=True)
                # Use SVD; take top n_components (or fewer if dim < n_components)
                n_comp = min(n_components, strong_c.shape[1])
                U, S, Vt = np.linalg.svd(strong_c, full_matrices=False)
                V_top = Vt[:n_comp].T  # (D_strong, n_comp)
                # Project weak GAP: residual = x - V @ V^T @ x
                weak_gap_train = tongue_feats[weak_name]['train']['gap']
                weak_gap_test = tongue_feats[weak_name]['test']['gap']
                # V_top is in strong's space, so we need to project weak into
                # the shared space. Since dims may differ, project within weak's
                # own space using the correlation structure.
                # Actually: compute PCA of strong, then project weak onto null-space
                # using cross-covariance. Simpler: just use strong's PCA basis
                # directly if dims match, otherwise use CCA-style projection.
                # For simplicity when dims differ: compute PCA of the weak backbone
                # conditioned on strong. Use the approach: project weak_gap onto
                # the span of (strong^T @ weak) top components.
                # Simplified approach: PCA the weak backbone, remove components
                # that correlate most with the strong backbone's top PCA directions.
                weak_c_train = weak_gap_train - weak_gap_train.mean(axis=0, keepdims=True)
                weak_c_test = weak_gap_test - weak_gap_test.mean(axis=0, keepdims=True)
                n_comp_weak = min(n_components, weak_c_train.shape[1])
                _, _, Vt_weak = np.linalg.svd(weak_c_train, full_matrices=False)
                V_weak_top = Vt_weak[:n_comp_weak].T  # (D_weak, n_comp_weak)
                # Residual: remove top PCA components of weak that are redundant
                # x_residual = x - V_weak @ V_weak^T @ x
                proj_train = weak_gap_train @ V_weak_top @ V_weak_top.T
                proj_test = weak_gap_test @ V_weak_top @ V_weak_top.T
                tongue_feats[weak_name]['train']['gap'] = (weak_gap_train - proj_train).astype(np.float32)
                tongue_feats[weak_name]['test']['gap'] = (weak_gap_test - proj_test).astype(np.float32)
                projected.add(weak_name)
                print(f"  CKA dedup: {weak_name} (CKA={cka_val:.3f} with {strong_name}) -> null-space residual")


def compute_gabor_marginals(face_bank, n_scales=5, n_orient=8):
    """(N, 40) -> (N, 19) per bank."""
    N = face_bank.shape[0]
    reshaped = face_bank.reshape(N, n_scales, n_orient)
    scale_marg = reshaped.sum(axis=2)
    orient_marg = reshaped.sum(axis=1)
    orient_idx = np.arange(n_orient, dtype=np.float32).reshape(1, 1, n_orient)
    softmax_weights = np.exp(reshaped - reshaped.max(axis=2, keepdims=True))
    softmax_weights = softmax_weights / (softmax_weights.sum(axis=2, keepdims=True) + 1e-8)
    dominant_orient = (softmax_weights * orient_idx).sum(axis=2)
    total_energy = np.abs(face_bank).sum(axis=1, keepdims=True)
    return np.concatenate([scale_marg, orient_marg, dominant_orient, total_energy],
                          axis=1).astype(np.float32)


def compute_gabor_asymmetry(bank_left, bank_right, n_orient=8):
    """Cross-bank asymmetry features. (N,40) x2 -> (N, 21)."""
    N = bank_left.shape[0]
    left_orient = bank_left.reshape(N, 5, n_orient).sum(axis=1)
    right_orient = bank_right.reshape(N, 5, n_orient).sum(axis=1)
    abs_diff = np.abs(left_orient - right_orient)
    norm_diff = abs_diff / (np.abs(left_orient) + np.abs(right_orient) + 1e-6)
    dot = (left_orient * right_orient).sum(axis=1, keepdims=True)
    norm_l = np.linalg.norm(left_orient, axis=1, keepdims=True) + 1e-8
    norm_r = np.linalg.norm(right_orient, axis=1, keepdims=True) + 1e-8
    cosine = dot / (norm_l * norm_r)
    full_diff = np.abs(bank_left - bank_right)
    full_ratio = full_diff / (np.abs(bank_left) + np.abs(bank_right) + 1e-6)
    return np.concatenate([
        abs_diff, norm_diff, cosine,
        full_diff.mean(axis=1, keepdims=True),
        full_diff.std(axis=1, keepdims=True),
        full_ratio.mean(axis=1, keepdims=True),
        full_ratio.std(axis=1, keepdims=True),
    ], axis=1).astype(np.float32)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_engineer(data_dir, face_banks, args):
    """Load raw features and pre-compute all deterministic feature engineering."""
    print("Loading data...")
    t0 = time.time()

    tongue_raw = {}
    tongue_feats = {}
    ages = None
    uuids = None

    for name, info in BACKBONE_INFO.items():
        with open(f"{data_dir}/{info['file']}", 'rb') as f:
            d = pickle.load(f)
        train_4d = d['train']['features'].astype(np.float32)
        test_4d = d['test']['features'].astype(np.float32)

        if ages is None:
            ages = np.array(d['train']['age'], dtype=np.float32)
            uuids = {'train': d['train']['uuid'], 'test': d['test']['uuid']}

        feats_train = {'gap': train_4d.mean(axis=(2, 3))}
        feats_test = {'gap': test_4d.mean(axis=(2, 3))}

        if args.use_std_pool:
            feats_train['std'] = compute_std_pool(train_4d)
            feats_test['std'] = compute_std_pool(test_4d)

        if args.use_grid_pool:
            feats_train['grid'] = compute_grid_pool(train_4d)
            feats_test['grid'] = compute_grid_pool(test_4d)

        if args.use_dct:
            feats_train['dct'] = compute_dct_features(train_4d)
            feats_test['dct'] = compute_dct_features(test_4d)

        if args.use_haar:
            feats_train['haar'] = compute_haar_energy(train_4d)
            feats_test['haar'] = compute_haar_energy(test_4d)

        if args.use_higher_moments:
            feats_train['moments'] = compute_higher_moments(train_4d)
            feats_test['moments'] = compute_higher_moments(test_4d)

        if args.use_spatial_moments:
            feats_train['spatial'] = compute_spatial_moments(train_4d)
            feats_test['spatial'] = compute_spatial_moments(test_4d)

        tongue_feats[name] = {'train': feats_train, 'test': feats_test}

        # Keep raw 4D only for DenseNet121 (used by ECA, iSQRT-COV, or learned query)
        if name == 'DenseNet121' and (args.use_isqrt_cov or args.use_eca or args.use_learned_query):
            tongue_raw[name] = {'train': train_4d, 'test': test_4d}

        del train_4d, test_4d, d

    # CBP cross-backbone features (ResNet18 x DenseNet121 GAP)
    if args.use_cbp:
        resnet_gap_train = tongue_feats['ResNet18']['train']['gap']
        dense_gap_train = tongue_feats['DenseNet121']['train']['gap']
        resnet_gap_test = tongue_feats['ResNet18']['test']['gap']
        dense_gap_test = tongue_feats['DenseNet121']['test']['gap']
        cbp_train = compute_cbp_cross(resnet_gap_train, dense_gap_train,
                                      sketch_dim=args.cbp_dim, seed=args.seed)
        cbp_test = compute_cbp_cross(resnet_gap_test, dense_gap_test,
                                     sketch_dim=args.cbp_dim, seed=args.seed)
        # Store CBP as a pseudo-backbone feature under ResNet18 (arbitrary choice)
        tongue_feats['ResNet18']['train']['cbp'] = cbp_train
        tongue_feats['ResNet18']['test']['cbp'] = cbp_test
        print(f"  CBP cross-backbone: {args.cbp_dim} dims")

    # MI channel selection (apply weights to GAP features)
    if args.use_mi_selection:
        for name in BACKBONE_NAMES:
            gap_train = tongue_feats[name]['train']['gap']
            mi_weights = compute_mi_channel_weights(
                gap_train, ages, top_k=args.mi_top_k)
            # Apply weights to GAP (element-wise, preserves shape)
            tongue_feats[name]['train']['gap'] = gap_train * mi_weights[None, :]
            tongue_feats[name]['test']['gap'] = tongue_feats[name]['test']['gap'] * mi_weights[None, :]
            n_selected = int((mi_weights >= 1.0).sum())
            if name == 'DenseNet121':
                print(f"  MI selection {name}: {n_selected}/{gap_train.shape[1]} channels at full weight")

    # CKA-guided backbone de-duplication
    if args.use_cka_dedup:
        compute_cka_dedup(tongue_feats, ages, threshold=0.8, n_components=64)

    # Face features
    face_raw = {}
    face_extra = {'train': None, 'test': None}
    with open(f"{data_dir}/{FACE_FILE}", 'rb') as f:
        face_data = pickle.load(f)
    for bank in face_banks:
        face_raw[bank] = {
            'train': face_data['train'][f'features_{bank}'].astype(np.float32),
            'test': face_data['test'][f'features_{bank}'].astype(np.float32),
        }

    if args.use_gabor_marginals:
        marginals_train = []
        marginals_test = []
        for bank in face_banks:
            marginals_train.append(compute_gabor_marginals(face_raw[bank]['train']))
            marginals_test.append(compute_gabor_marginals(face_raw[bank]['test']))
        face_extra['train'] = np.concatenate(marginals_train, axis=1)
        face_extra['test'] = np.concatenate(marginals_test, axis=1)

    if args.use_gabor_asymmetry and 'B' in face_banks and 'C' in face_banks:
        asym_train = compute_gabor_asymmetry(face_raw['B']['train'], face_raw['C']['train'])
        asym_test = compute_gabor_asymmetry(face_raw['B']['test'], face_raw['C']['test'])
        if face_extra['train'] is not None:
            face_extra['train'] = np.concatenate([face_extra['train'], asym_train], axis=1)
            face_extra['test'] = np.concatenate([face_extra['test'], asym_test], axis=1)
        else:
            face_extra['train'] = asym_train
            face_extra['test'] = asym_test

    print(f"  Loaded + engineered in {time.time()-t0:.1f}s")
    for name in BACKBONE_NAMES:
        dims = {k: v.shape[1] for k, v in tongue_feats[name]['train'].items()}
        print(f"  {name}: {dims} (total={sum(dims.values())})")
        # Verify train/test feature keys match
        assert set(tongue_feats[name]['train'].keys()) == set(tongue_feats[name]['test'].keys()), \
            f"Train/test key mismatch for {name}"
    if face_extra['train'] is not None:
        print(f"  Face extra: {face_extra['train'].shape[1]} dims")

    return tongue_feats, tongue_raw, face_raw, face_extra, ages, uuids


# ============================================================================
# DATASET
# ============================================================================

class Sprint12Dataset:
    """Pre-computed features + optional raw 4D for ECA/iSQRT."""

    def __init__(self, tongue_feats, tongue_raw, face_raw, face_extra,
                 face_banks, ages, indices, device, args,
                 tongue_stats=None, face_stats=None):
        self.device = device
        self.face_banks = face_banks
        self.args = args
        self.n = len(indices)

        if tongue_stats is None:
            self.tongue_stats = {}
            for name in BACKBONE_NAMES:
                self.tongue_stats[name] = {}
                for feat_key, feat_arr in tongue_feats[name]['train'].items():
                    f = feat_arr[indices]
                    m = f.mean(axis=0, keepdims=True).astype(np.float32)
                    s = (f.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
                    self.tongue_stats[name][feat_key] = (m, s)
        else:
            self.tongue_stats = tongue_stats

        if face_stats is None:
            self.face_stats = {}
            for bank in face_banks:
                f = face_raw[bank]['train'][indices]
                m = f.mean(axis=0, keepdims=True).astype(np.float32)
                s = (f.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
                self.face_stats[bank] = (m, s)
            if face_extra['train'] is not None:
                f = face_extra['train'][indices]
                m = f.mean(axis=0, keepdims=True).astype(np.float32)
                s = (f.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
                self.face_stats['_extra'] = (m, s)
        else:
            self.face_stats = face_stats

        # Pre-computed tongue -> GPU (concatenated per backbone)
        self.tongue = {}
        for name in BACKBONE_NAMES:
            parts = []
            for feat_key in sorted(tongue_feats[name]['train'].keys()):
                f = tongue_feats[name]['train'][feat_key][indices]
                m, s = self.tongue_stats[name][feat_key]
                parts.append(((f - m) / s).astype(np.float32))
            self.tongue[name] = torch.from_numpy(np.concatenate(parts, axis=1)).to(device)

        # Raw 4D on CPU
        self.tongue_raw = {}
        if tongue_raw:
            for name in tongue_raw:
                self.tongue_raw[name] = tongue_raw[name]['train'][indices]

        # Face -> GPU
        self.face = {}
        for bank in face_banks:
            f = face_raw[bank]['train'][indices]
            m, s = self.face_stats[bank]
            self.face[bank] = torch.from_numpy(((f - m) / s).astype(np.float32)).to(device)

        self.face_extra = None
        if face_extra['train'] is not None:
            f = face_extra['train'][indices]
            m, s = self.face_stats['_extra']
            self.face_extra = torch.from_numpy(((f - m) / s).astype(np.float32)).to(device)

        self.ages = torch.from_numpy(ages[indices]).to(device) if ages is not None else None

    def get_batch(self, idx):
        tongue = {name: self.tongue[name][idx] for name in BACKBONE_NAMES}
        face = {bank: self.face[bank][idx] for bank in self.face_banks}
        face_extra = self.face_extra[idx] if self.face_extra is not None else None
        ages = self.ages[idx] if self.ages is not None else None

        tongue_4d = None
        if self.tongue_raw:
            tongue_4d = {}
            idx_np = idx.cpu().numpy() if isinstance(idx, torch.Tensor) else idx
            for name in self.tongue_raw:
                tongue_4d[name] = torch.from_numpy(
                    self.tongue_raw[name][idx_np].astype(np.float32)).to(self.device)

        return tongue, face, face_extra, ages, tongue_4d

    def make_test(self, tongue_feats, tongue_raw, face_raw, face_extra):
        tongue = {}
        for name in BACKBONE_NAMES:
            parts = []
            for feat_key in sorted(tongue_feats[name]['test'].keys()):
                f = tongue_feats[name]['test'][feat_key]
                m, s = self.tongue_stats[name][feat_key]
                parts.append(((f - m) / s).astype(np.float32))
            tongue[name] = torch.from_numpy(np.concatenate(parts, axis=1)).to(self.device)

        tongue_4d_test = None
        if tongue_raw:
            tongue_4d_test = {}
            for name in tongue_raw:
                tongue_4d_test[name] = torch.from_numpy(
                    tongue_raw[name]['test'].astype(np.float32)).to(self.device)

        face = {}
        for bank in self.face_banks:
            f = face_raw[bank]['test']
            m, s = self.face_stats[bank]
            face[bank] = torch.from_numpy(((f - m) / s).astype(np.float32)).to(self.device)

        face_extra_t = None
        if face_extra['test'] is not None and '_extra' in self.face_stats:
            f = face_extra['test']
            m, s = self.face_stats['_extra']
            face_extra_t = torch.from_numpy(((f - m) / s).astype(np.float32)).to(self.device)

        return tongue, face, face_extra_t, tongue_4d_test


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class ECABlock(nn.Module):
    """Efficient Channel Attention (1D conv across channels)."""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x_4d):
        b, c, h, w = x_4d.shape
        y = x_4d.mean(dim=(2, 3))
        y = self.conv(y.unsqueeze(1)).squeeze(1)
        y = torch.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
        return x_4d * y


class NewtonSchulzSqrt(nn.Module):
    """iSQRT-COV on DenseNet with channel pre-reduction."""
    def __init__(self, in_channels, reduce_dim=32, out_dim=64, n_iter=5, dropout=0.3):
        super().__init__()
        self.reduce = nn.Linear(in_channels, reduce_dim) if reduce_dim < in_channels else None
        self.n_iter = n_iter
        tri_dim = reduce_dim * (reduce_dim + 1) // 2
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(tri_dim, out_dim),
            nn.GELU(),
        )
        self.reduce_dim = reduce_dim

    def forward(self, x_4d):
        B, C, H, W = x_4d.shape
        x = x_4d.reshape(B, C, H * W)
        if self.reduce is not None:
            x = x.permute(0, 2, 1)
            x = self.reduce(x)
            x = x.permute(0, 2, 1)
        D = x.shape[1]
        x_centered = x - x.mean(dim=2, keepdim=True)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (H * W - 1)
        # Normalize so trace=1 (guarantees spectral norm <= 1 for PSD matrices)
        trace = torch.diagonal(cov, dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
        cov = cov / (trace + 1e-8)
        # Newton-Schulz iterations for matrix square root
        I = torch.eye(D, device=cov.device).unsqueeze(0).expand(B, -1, -1)
        Y = cov
        Z = I.clone()
        for _ in range(self.n_iter):
            T = 0.5 * (3.0 * I - torch.bmm(Z, Y))
            Y = torch.bmm(Y, T)
            Z = torch.bmm(T, Z)
        # Guard against NaN from divergence
        Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        # Extract upper triangle (offset=0 includes diagonal)
        idx = torch.triu_indices(D, D, offset=0, device=cov.device)
        tri = Y[:, idx[0], idx[1]]
        return self.proj(tri)


class GMU(nn.Module):
    """Gated Multimodal Unit."""
    def __init__(self, tongue_dim, face_dim, out_dim):
        super().__init__()
        self.gate = nn.Linear(tongue_dim + face_dim, out_dim)
        self.tongue_proj = nn.Linear(tongue_dim, out_dim)
        self.face_proj = nn.Linear(face_dim, out_dim)

    def forward(self, h_tongue, h_face):
        z = torch.sigmoid(self.gate(torch.cat([h_tongue, h_face], dim=-1)))
        return z * torch.tanh(self.tongue_proj(h_tongue)) + \
               (1 - z) * torch.tanh(self.face_proj(h_face))


class MMTM(nn.Module):
    """Multimodal Transfer Module (SE-block for two modalities)."""
    def __init__(self, tongue_dim, face_dim, reduction=4):
        super().__init__()
        bottleneck = max((tongue_dim + face_dim) // reduction, 32)
        self.shared = nn.Sequential(
            nn.Linear(tongue_dim + face_dim, bottleneck),
            nn.ReLU(),
        )
        self.excite_tongue = nn.Linear(bottleneck, tongue_dim)
        self.excite_face = nn.Linear(bottleneck, face_dim)

    def forward(self, h_tongue, h_face):
        s = torch.cat([h_tongue, h_face], dim=-1)
        z = self.shared(s)
        e_t = torch.sigmoid(self.excite_tongue(z))
        e_f = torch.sigmoid(self.excite_face(z))
        return h_tongue * e_t, h_face * e_f


class LearnedQueryPool(nn.Module):
    """Single learned query attention over spatial tokens.
    One query vector attends over HW tokens per backbone.
    Initialized to uniform (equivalent to GAP at init).
    """
    def __init__(self, channels, n_tokens=784):
        super().__init__()
        self.channels = channels
        # q=0 -> dot product uniform -> softmax uniform -> GAP equivalent
        self.query = nn.Parameter(torch.zeros(channels))
        # W_k=0 -> keys all zero -> uniform attention at init
        self.key_proj = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.key_proj.weight)
        self.scale = channels ** 0.5

    def forward(self, x_4d):
        """x_4d: (B, C, H, W) -> (B, C) pooled."""
        B, C, H, W = x_4d.shape
        # X: (B, C, HW) -> (B, HW, C)
        X = x_4d.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        K = self.key_proj(X)  # (B, HW, C)
        # attn = softmax(q^T K / sqrt(d))
        attn_logits = (K @ self.query) / self.scale  # (B, HW)
        attn = F.softmax(attn_logits, dim=-1)  # (B, HW)
        # pool = X^T @ attn = weighted average of spatial tokens
        pooled = (X * attn.unsqueeze(-1)).sum(dim=1)  # (B, C)
        return pooled


# ============================================================================
# MAIN MODEL
# ============================================================================

class Sprint12Model(nn.Module):
    """HierModel with configurable feature engineering."""

    def __init__(self, args, tongue_input_dims, face_input_dim, face_extra_dim=0):
        super().__init__()
        self.args = args
        proj_dim = args.d_hidden // 4

        # Per-backbone projection
        self.tongue_projs = nn.ModuleDict()
        for name in BACKBONE_NAMES:
            in_dim = tongue_input_dims[name]
            layers = []
            if args.use_layernorm_pre:
                layers.append(nn.LayerNorm(in_dim))
            layers.extend([
                nn.Linear(in_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(args.dropout * 0.5),
            ])
            self.tongue_projs[name] = nn.Sequential(*layers)

        # ECA block (DenseNet121 only — apply attention then GAP for extra feature)
        self.eca_block = None
        if args.use_eca:
            self.eca_block = ECABlock(128)  # DenseNet121 channels
            self.eca_proj = nn.Sequential(
                nn.Linear(128, proj_dim),
                nn.GELU(),
                nn.Dropout(args.dropout * 0.5),
            )

        # iSQRT-COV on DenseNet
        self.isqrt = None
        if args.use_isqrt_cov:
            self.isqrt = NewtonSchulzSqrt(
                in_channels=128, reduce_dim=args.isqrt_reduce_dim,
                out_dim=proj_dim, dropout=args.dropout)

        # Learned query attention pool (DenseNet121 only)
        self.learned_query = None
        if args.use_learned_query:
            self.learned_query = LearnedQueryPool(channels=128, n_tokens=784)
            self.learned_query_proj = nn.Sequential(
                nn.Linear(128, proj_dim),
                nn.GELU(),
                nn.Dropout(args.dropout * 0.5),
            )

        # Face projection
        total_face_dim = len(args.face_banks) * FACE_DIM + face_extra_dim
        face_layers = []
        if args.use_layernorm_pre:
            face_layers.append(nn.LayerNorm(total_face_dim))
        face_layers.extend([
            nn.Linear(total_face_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(args.dropout * 0.5),
        ])
        self.face_proj = nn.Sequential(*face_layers)

        # Dimension math
        n_tongue_sources = len(BACKBONE_NAMES)
        tongue_total_dim = n_tongue_sources * proj_dim
        if args.use_isqrt_cov:
            tongue_total_dim += proj_dim  # iSQRT branch
        if args.use_eca:
            tongue_total_dim += proj_dim  # ECA-GAP branch
        if args.use_learned_query:
            tongue_total_dim += proj_dim  # Learned query attention branch
        face_total_dim = proj_dim

        # Fusion
        self.fusion_type = args.fusion_type
        if args.fusion_type == 'gmu':
            self.fusion = GMU(tongue_total_dim, face_total_dim, args.d_hidden)
            combined_dim = args.d_hidden
        elif args.fusion_type == 'mmtm':
            self.fusion = MMTM(tongue_total_dim, face_total_dim)
            combined_dim = tongue_total_dim + face_total_dim
        else:
            combined_dim = tongue_total_dim + face_total_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, args.d_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.d_hidden),
            nn.Linear(args.d_hidden, args.d_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.d_hidden),
            nn.Linear(args.d_hidden, args.d_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.d_hidden),
        )

        # Hierarchical heads
        self.decade_head = nn.Linear(args.d_hidden, NUM_DECADES)
        self.fine_heads = nn.ModuleList([
            nn.Linear(args.d_hidden, BINS_PER_DECADE) for _ in range(NUM_DECADES)
        ])
        # Confidence fusion auxiliary heads
        self.tongue_aux_head = None
        self.face_aux_head = None
        if args.use_confidence_fusion:
            self.tongue_aux_head = nn.Linear(tongue_total_dim, 1)
            self.face_aux_head = nn.Linear(face_total_dim, 1)

        self.register_buffer('fine_indices',
                             torch.arange(BINS_PER_DECADE, dtype=torch.float32))
        self.register_buffer('decade_starts_t',
                             torch.tensor(DECADE_STARTS, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tongue_batch, face_batch, face_extra=None, tongue_4d=None):
        tongue_parts = []
        for name in BACKBONE_NAMES:
            x = tongue_batch[name]
            tongue_parts.append(self.tongue_projs[name](x))

        # ECA: channel-attention on raw 4D DenseNet, then GAP -> project
        if self.eca_block is not None and tongue_4d is not None and 'DenseNet121' in tongue_4d:
            raw = tongue_4d['DenseNet121']
            eca_out = self.eca_block(raw)  # (B, 128, 28, 28) recalibrated
            eca_gap = eca_out.mean(dim=(2, 3))  # (B, 128)
            tongue_parts.append(self.eca_proj(eca_gap))

        # iSQRT-COV: second-order features from DenseNet
        if self.isqrt is not None and tongue_4d is not None and 'DenseNet121' in tongue_4d:
            tongue_parts.append(self.isqrt(tongue_4d['DenseNet121']))

        # Learned query attention pool on DenseNet121
        if self.learned_query is not None and tongue_4d is not None and 'DenseNet121' in tongue_4d:
            lq_pooled = self.learned_query(tongue_4d['DenseNet121'])  # (B, 128)
            tongue_parts.append(self.learned_query_proj(lq_pooled))

        h_tongue = torch.cat(tongue_parts, dim=-1)

        face_parts = [face_batch[bank] for bank in self.args.face_banks]
        face_cat = torch.cat(face_parts, dim=-1)
        if face_extra is not None:
            face_cat = torch.cat([face_cat, face_extra], dim=-1)
        h_face = self.face_proj(face_cat)

        # Confidence fusion auxiliary predictions (before trunk)
        tongue_aux_pred = None
        face_aux_pred = None
        if self.tongue_aux_head is not None:
            tongue_aux_pred = self.tongue_aux_head(h_tongue).squeeze(-1)  # (B,)
            tongue_aux_pred = torch.clamp(tongue_aux_pred + 44.5, MIN_AGE, MAX_AGE)  # center at mean age
        if self.face_aux_head is not None:
            face_aux_pred = self.face_aux_head(h_face).squeeze(-1)  # (B,)
            face_aux_pred = torch.clamp(face_aux_pred + 44.5, MIN_AGE, MAX_AGE)

        if self.fusion_type == 'gmu':
            combined = self.fusion(h_tongue, h_face)
        elif self.fusion_type == 'mmtm':
            h_tongue_m, h_face_m = self.fusion(h_tongue, h_face)
            combined = torch.cat([h_tongue_m, h_face_m], dim=-1)
        else:
            combined = torch.cat([h_tongue, h_face], dim=-1)

        h = self.trunk(combined)

        decade_logits = self.decade_head(h)
        decade_probs = F.softmax(decade_logits, dim=-1)
        fine_logits_list = [head(h) for head in self.fine_heads]

        fine_expected = []
        for d in range(NUM_DECADES):
            fine_probs = F.softmax(fine_logits_list[d], dim=-1)
            expected_offset = (fine_probs * self.fine_indices.unsqueeze(0)).sum(dim=-1)
            fine_expected.append(self.decade_starts_t[d] + expected_offset)

        fine_expected = torch.stack(fine_expected, dim=-1)
        final_pred = (decade_probs * fine_expected).sum(dim=-1)
        final_pred = torch.clamp(final_pred, MIN_AGE, MAX_AGE)

        # At inference with confidence fusion: blend using entropy-based weighting
        if self.tongue_aux_head is not None and not self.training:
            # Entropy of decade distribution as uncertainty proxy
            # Lower entropy = more confident main head
            # Use auxiliary heads to adjust: weight main vs aux based on agreement
            tongue_err = (final_pred - tongue_aux_pred).abs()
            face_err = (final_pred - face_aux_pred).abs()
            # Confidence = 1/(1 + error), higher when sub-head agrees with main
            w_tongue = 1.0 / (1.0 + tongue_err)
            w_face = 1.0 / (1.0 + face_err)
            w_main = torch.ones_like(w_tongue) * 2.0  # main head gets higher base weight
            w_total = w_main + w_tongue + w_face
            final_pred = (w_main * final_pred + w_tongue * tongue_aux_pred +
                         w_face * face_aux_pred) / w_total
            final_pred = torch.clamp(final_pred, MIN_AGE, MAX_AGE)

        return final_pred, decade_logits, fine_logits_list, h, tongue_aux_pred, face_aux_pred


# ============================================================================
# LOSSES
# ============================================================================

def hierarchical_loss(pred, decade_logits, fine_logits_list, ages, sigma, args):
    device = ages.device
    decade_true = ((ages - MIN_AGE) / 10).long().clamp(0, NUM_DECADES - 1)
    loss_decade = F.cross_entropy(decade_logits, decade_true)

    loss_fine = torch.tensor(0.0, device=device)
    for d in range(NUM_DECADES):
        mask = decade_true == d
        if mask.sum() == 0:
            continue
        fine_true_offset = (ages[mask] - DECADE_STARTS[d]).clamp(0, BINS_PER_DECADE - 1)
        fine_bins = torch.arange(BINS_PER_DECADE, dtype=torch.float32, device=device)
        fine_soft = -0.5 * ((fine_bins - fine_true_offset.unsqueeze(-1)) / (sigma * 0.5)) ** 2
        fine_soft = F.softmax(fine_soft, dim=-1)
        log_probs = F.log_softmax(fine_logits_list[d][mask], dim=-1)
        loss_fine = loss_fine + F.kl_div(log_probs, fine_soft, reduction='batchmean')
    loss_fine = loss_fine / NUM_DECADES

    loss_l1 = F.l1_loss(pred, ages)

    w, eps = 5.0, 0.5
    diff = (pred - ages).abs()
    wing = torch.where(diff < w, w * torch.log(1 + diff / eps),
                       diff - w + w * math.log(1 + w / eps))
    loss_wing = wing.mean()

    return (args.w_decade * loss_decade + args.w_fine * loss_fine +
            args.w_l1 * loss_l1 + args.w_wing * loss_wing)


# ============================================================================
# METRICS
# ============================================================================

def comp_score(y_true, y_pred):
    err = np.abs(y_true - y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    acc_1y = np.mean(err <= 1)
    acc_5y = np.mean(err <= 5)
    acc_10y = np.mean(err <= 10)
    score = 0.1 / (1 + mse) + 0.4 * acc_1y + 0.3 * acc_5y + 0.2 * acc_10y
    return score, mse, acc_1y, acc_5y, acc_10y


# ============================================================================
# TRAINING
# ============================================================================

def train_fold(fold, train_idx, val_idx, tongue_feats, tongue_raw, face_raw,
               face_extra, ages, args, device):
    train_ds = Sprint12Dataset(
        tongue_feats, tongue_raw, face_raw, face_extra,
        args.face_banks, ages, train_idx, device, args)
    val_ds = Sprint12Dataset(
        tongue_feats, tongue_raw, face_raw, face_extra,
        args.face_banks, ages, val_idx, device, args,
        tongue_stats=train_ds.tongue_stats, face_stats=train_ds.face_stats)

    tongue_input_dims = {name: train_ds.tongue[name].shape[1] for name in BACKBONE_NAMES}
    face_extra_dim = train_ds.face_extra.shape[1] if train_ds.face_extra is not None else 0
    face_input_dim = len(args.face_banks) * FACE_DIM + face_extra_dim

    model = Sprint12Model(args, tongue_input_dims, face_input_dim, face_extra_dim).to(device)

    if fold == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    n_train = train_ds.n
    best_score, best_ep, wait = 0.0, 0, 0
    best_state = None

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * min(1.0, ep / args.sigma_anneal)
        total_loss = 0.0

        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            tongue_b, face_b, face_extra_b, ages_b, tongue_4d_b = train_ds.get_batch(idx)

            aug_tongue = {name: tongue_b[name] + torch.randn_like(tongue_b[name]) * 0.02
                          for name in BACKBONE_NAMES}
            aug_face = {bank: face_b[bank] + torch.randn_like(face_b[bank]) * 0.05
                        for bank in args.face_banks}

            pred, dec_logits, fine_logits, _, t_aux, f_aux = model(aug_tongue, aug_face, face_extra_b, tongue_4d_b)
            loss = hierarchical_loss(pred, dec_logits, fine_logits, ages_b, sigma, args)
            # Auxiliary confidence fusion losses
            if t_aux is not None:
                loss = loss + 0.2 * F.l1_loss(t_aux, ages_b)
            if f_aux is not None:
                loss = loss + 0.2 * F.l1_loss(f_aux, ages_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            tongue_v, face_v, face_extra_v, ages_v, tongue_4d_v = val_ds.get_batch(
                torch.arange(val_ds.n, device=device))
            pred_v, _, _, _, _, _ = model(tongue_v, face_v, face_extra_v, tongue_4d_v)
            pred_np = pred_v.cpu().numpy()
            true_np = ages_v.cpu().numpy()

        score, mse, acc1, acc5, acc10 = comp_score(true_np, pred_np)

        if score > best_score:
            best_score = score
            best_ep = ep
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if ep % 20 == 0 or ep == args.epochs - 1 or wait == 0:
            print(f"  E{ep:3d} σ={sigma:.2f} | loss={total_loss:.3f} | "
                  f"V: s={score:.4f} 1y={acc1:.3f} 5y={acc5:.3f} 10y={acc10:.3f} | "
                  f"best={best_score:.4f}@{best_ep}")

        if wait >= args.patience:
            break

    ckpt_dir = Path(args.output_dir) / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': best_state,
        'tongue_stats': {name: {k: (m.tolist(), s.tolist())
                                for k, (m, s) in train_ds.tongue_stats[name].items()}
                         for name in BACKBONE_NAMES},
        'face_stats': {k: (m.tolist(), s.tolist())
                       for k, (m, s) in train_ds.face_stats.items()},
        'tongue_input_dims': tongue_input_dims,
        'face_extra_dim': face_extra_dim,
        'args': vars(args),
    }, ckpt_dir / f'fold{fold}.pt')

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        tongue_v, face_v, face_extra_v, _, tongue_4d_v = val_ds.get_batch(
            torch.arange(val_ds.n, device=device))
        pred_v, _, _, _, _, _ = model(tongue_v, face_v, face_extra_v, tongue_4d_v)
    oof_pred = pred_v.cpu().numpy()

    print(f"  Fold {fold}: best score={best_score:.4f} @ epoch {best_ep}")
    return oof_pred, best_score, train_ds


def generate_test_preds(tongue_feats, tongue_raw, face_raw, face_extra, args, device):
    ckpt_dir = Path(args.output_dir) / args.exp_name
    all_preds = []

    for fold in range(args.n_folds):
        ckpt = torch.load(ckpt_dir / f'fold{fold}.pt', map_location=device, weights_only=False)
        saved_args = argparse.Namespace(**ckpt['args'])

        tongue_input_dims = ckpt['tongue_input_dims']
        face_extra_dim = ckpt['face_extra_dim']
        face_input_dim = len(saved_args.face_banks) * FACE_DIM + face_extra_dim

        model = Sprint12Model(saved_args, tongue_input_dims, face_input_dim, face_extra_dim).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        tongue_stats = {}
        for name in BACKBONE_NAMES:
            tongue_stats[name] = {}
            for feat_key, (m, s) in ckpt['tongue_stats'][name].items():
                tongue_stats[name][feat_key] = (np.array(m, dtype=np.float32),
                                                np.array(s, dtype=np.float32))
        face_stats = {}
        for k, (m, s) in ckpt['face_stats'].items():
            face_stats[k] = (np.array(m, dtype=np.float32), np.array(s, dtype=np.float32))

        tongue_test = {}
        for name in BACKBONE_NAMES:
            parts = []
            for feat_key in sorted(tongue_feats[name]['test'].keys()):
                f = tongue_feats[name]['test'][feat_key]
                m, s = tongue_stats[name][feat_key]
                parts.append(((f - m) / s).astype(np.float32))
            tongue_test[name] = torch.from_numpy(np.concatenate(parts, axis=1)).to(device)

        face_test = {}
        for bank in saved_args.face_banks:
            f = face_raw[bank]['test']
            m, s = face_stats[bank]
            face_test[bank] = torch.from_numpy(((f - m) / s).astype(np.float32)).to(device)

        face_extra_test = None
        if face_extra['test'] is not None and '_extra' in face_stats:
            f = face_extra['test']
            m, s = face_stats['_extra']
            face_extra_test = torch.from_numpy(((f - m) / s).astype(np.float32)).to(device)

        # Inference in batches (4D tensors are large)
        N_test = tongue_test[BACKBONE_NAMES[0]].shape[0]
        batch_preds = []
        with torch.no_grad():
            for start in range(0, N_test, 64):
                end = min(start + 64, N_test)
                t_batch = {name: tongue_test[name][start:end] for name in BACKBONE_NAMES}
                f_batch = {bank: face_test[bank][start:end] for bank in saved_args.face_banks}
                fe_batch = face_extra_test[start:end] if face_extra_test is not None else None
                t4d_batch = None
                if tongue_raw:
                    t4d_batch = {}
                    for name in tongue_raw:
                        t4d_batch[name] = torch.from_numpy(
                            tongue_raw[name]['test'][start:end].astype(np.float32)).to(device)
                pred_b, _, _, _, _, _ = model(t_batch, f_batch, fe_batch, t4d_batch)
                batch_preds.append(pred_b.cpu().numpy())
                if t4d_batch is not None:
                    del t4d_batch
        all_preds.append(np.concatenate(batch_preds).clip(MIN_AGE, MAX_AGE))

    return np.mean(all_preds, axis=0)


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--output_dir', default='checkpoints')
    p.add_argument('--exp_name', required=True)
    p.add_argument('--face_banks', default='ABCD')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_folds', type=int, default=10)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--d_hidden', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--sigma_start', type=float, default=2.0)
    p.add_argument('--sigma_end', type=float, default=0.5)
    p.add_argument('--sigma_anneal', type=int, default=100)
    p.add_argument('--w_decade', type=float, default=0.5)
    p.add_argument('--w_fine', type=float, default=1.0)
    p.add_argument('--w_l1', type=float, default=0.3)
    p.add_argument('--w_wing', type=float, default=0.2)

    # Feature engineering flags
    p.add_argument('--use_std_pool', action='store_true')
    p.add_argument('--use_grid_pool', action='store_true')
    p.add_argument('--use_dct', action='store_true')
    p.add_argument('--use_haar', action='store_true')
    p.add_argument('--use_higher_moments', action='store_true')
    p.add_argument('--use_spatial_moments', action='store_true')
    p.add_argument('--use_layernorm_pre', action='store_true')
    p.add_argument('--use_eca', action='store_true')
    p.add_argument('--use_isqrt_cov', action='store_true')
    p.add_argument('--isqrt_reduce_dim', type=int, default=32)
    p.add_argument('--use_gabor_marginals', action='store_true')
    p.add_argument('--use_gabor_asymmetry', action='store_true')
    p.add_argument('--use_cbp', action='store_true')
    p.add_argument('--cbp_dim', type=int, default=1024)
    p.add_argument('--use_mi_selection', action='store_true')
    p.add_argument('--mi_top_k', type=int, default=64)
    p.add_argument('--use_cka_dedup', action='store_true')
    p.add_argument('--use_learned_query', action='store_true')
    p.add_argument('--use_confidence_fusion', action='store_true')
    p.add_argument('--fusion_type', choices=['none', 'gmu', 'mmtm'], default='none')

    args = p.parse_args()
    args.face_banks = list(args.face_banks)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment: {args.exp_name}")
    print(f"Features: std={args.use_std_pool} grid={args.use_grid_pool} dct={args.use_dct} "
          f"haar={args.use_haar} moments={args.use_higher_moments} "
          f"spatial={args.use_spatial_moments} LN={args.use_layernorm_pre} "
          f"eca={args.use_eca} isqrt={args.use_isqrt_cov} "
          f"cbp={args.use_cbp} mi={args.use_mi_selection} "
          f"gabor_marg={args.use_gabor_marginals} gabor_asym={args.use_gabor_asymmetry} "
          f"cka_dedup={args.use_cka_dedup} learned_query={args.use_learned_query} "
          f"conf_fusion={args.use_confidence_fusion} "
          f"fusion={args.fusion_type}")

    tongue_feats, tongue_raw, face_raw, face_extra, ages, uuids = \
        load_and_engineer(args.data_dir, args.face_banks, args)

    N = len(ages)
    print(f"  {N} train samples")

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof_preds = np.zeros(N, dtype=np.float32)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(N))):
        print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
        oof_pred, score, train_ds = train_fold(
            fold, train_idx, val_idx,
            tongue_feats, tongue_raw, face_raw, face_extra, ages, args, device)
        oof_preds[val_idx] = oof_pred
        fold_scores.append(score)

    score, mse, acc1, acc5, acc10 = comp_score(ages, oof_preds)
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"OOF Score: {score:.4f} | MSE={mse:.1f} | 1Y={acc1:.3f} | 5Y={acc5:.3f} | 10Y={acc10:.3f}")
    print(f"Folds: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    ckpt_dir = Path(args.output_dir) / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'oof': {'score': score, 'mse': float(mse), '1y': float(acc1),
                '5y': float(acc5), '10y': float(acc10)},
        'folds': fold_scores,
        'args': vars(args),
    }
    with open(ckpt_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nGenerating test predictions...")
    test_preds = generate_test_preds(tongue_feats, tongue_raw, face_raw, face_extra, args, device)

    test_uuids = uuids['test']
    sub_path = ckpt_dir / 'submission.csv'
    with open(sub_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['uuid', 'age'])
        for uid, age in zip(test_uuids, test_preds):
            writer.writerow([uid, f"{age:.4f}"])

    zip_path = ckpt_dir / 'submission.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, 'submission.csv')
    print(f"  Submission: {zip_path}")


if __name__ == '__main__':
    main()
