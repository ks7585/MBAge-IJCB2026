"""Phase 2 Hierarchical: Decade + Within-Decade age prediction.

Purpose: Improve 1Y-ACC by using sharper within-decade distributions.
  - Coarse head: 5-class decade classifier (20s, 30s, 40s, 50s, 60s)
  - Fine heads: 5 separate 10-class DLDL within each decade
  - Soft routing: decade_probs * fine_expected_value = final prediction
  - 10-class DLDL is much sharper than 50-class => better 1Y-ACC
"""

import argparse
import csv
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

MIN_AGE, MAX_AGE, NUM_CLASSES = 20, 69, 50
NUM_DECADES = 5
BINS_PER_DECADE = 10

BACKBONE_INFO = {
    'ResNet18': {'channels': 128, 'file': 'tongue_ResNet18_features_v2.0.pkl'},
    'MobileNetV2': {'channels': 32, 'file': 'tongue_MobileNetV2_features_v2.0.pkl'},
    'EfficientNetB0': {'channels': 40, 'file': 'tongue_EfficientNetB0_features_v2.0.pkl'},
    'DenseNet121': {'channels': 128, 'file': 'tongue_DenseNet121_features_v2.0.pkl'},
}
BACKBONE_NAMES = list(BACKBONE_INFO.keys())

FACE_FILE = 'face_features_gabor_v2.0.pkl'
FACE_BANKS = ['features_B', 'features_C', 'features_D']
FACE_DIM = 40

DECADE_STARTS = [20, 30, 40, 50, 60]  # start age of each decade


def load_data(data_dir):
    tongue_features = {}
    ages = None
    uuids = None
    for name, info in BACKBONE_INFO.items():
        with open(f"{data_dir}/{info['file']}", 'rb') as f:
            d = pickle.load(f)
        tongue_features[name] = {
            'train': d['train']['features'].astype(np.float32),
            'test': d['test']['features'].astype(np.float32),
        }
        if ages is None:
            ages = np.array(d['train']['age'], dtype=np.float32)
            uuids = {'train': d['train']['uuid'], 'test': d['test']['uuid']}

    with open(f"{data_dir}/{FACE_FILE}", 'rb') as f:
        face_data = pickle.load(f)

    face_features = {}
    for bank in FACE_BANKS:
        face_features[bank] = {
            'train': face_data['train'][bank].astype(np.float32),
            'test': face_data['test'][bank].astype(np.float32),
        }
    return tongue_features, face_features, ages, uuids


class MultiModalDataset:
    def __init__(self, tongue_feats, face_feats, ages, indices, device,
                 tongue_stats=None, face_stats=None):
        self.device = device
        self.n = len(indices)

        if tongue_stats is None:
            self.tongue_stats = {}
            for name in BACKBONE_NAMES:
                f = tongue_feats[name]['train'][indices]
                m = f.mean(axis=(0, 2, 3), keepdims=True)
                s = f.std(axis=(0, 2, 3), keepdims=True) + 1e-6
                self.tongue_stats[name] = (m, s)
        else:
            self.tongue_stats = tongue_stats

        if face_stats is None:
            self.face_stats = {}
            for bank in FACE_BANKS:
                f = face_feats[bank]['train'][indices]
                m = f.mean(axis=0, keepdims=True)
                s = f.std(axis=0, keepdims=True) + 1e-6
                self.face_stats[bank] = (m, s)
        else:
            self.face_stats = face_stats

        self.tongue = {}
        for name in BACKBONE_NAMES:
            f = tongue_feats[name]['train'][indices]
            m, s = self.tongue_stats[name]
            self.tongue[name] = torch.from_numpy((f - m) / s).to(device)

        self.face = {}
        for bank in FACE_BANKS:
            f = face_feats[bank]['train'][indices]
            m, s = self.face_stats[bank]
            self.face[bank] = torch.from_numpy((f - m) / s).to(device)

        self.ages = torch.from_numpy(ages[indices]).to(device) if ages is not None else None

    def get_batch(self, idx):
        tongue_batch = {name: self.tongue[name][idx] for name in BACKBONE_NAMES}
        face_batch = {bank: self.face[bank][idx] for bank in FACE_BANKS}
        ages = self.ages[idx] if self.ages is not None else None
        return tongue_batch, face_batch, ages

    def make_test(self, tongue_feats, face_feats):
        tongue = {}
        for name in BACKBONE_NAMES:
            f = tongue_feats[name]['test']
            m, s = self.tongue_stats[name]
            tongue[name] = torch.from_numpy((f - m) / s).to(self.device)
        face = {}
        for bank in FACE_BANKS:
            f = face_feats[bank]['test']
            m, s = self.face_stats[bank]
            face[bank] = torch.from_numpy((f - m) / s).to(self.device)
        return tongue, face


# ============================================================================
# MODEL: Hierarchical Decade + Within-Decade DLDL
# ============================================================================

class HierarchicalAgeModel(nn.Module):
    """Hierarchical age predictor with decade classifier and within-decade DLDL heads.

    Architecture:
      - GAP per tongue backbone -> project -> concat
      - Face banks concat -> project
      - Shared trunk: 3-layer MLP with GELU, dropout, LayerNorm
      - Decade head: Linear -> 5 classes (softmax)
      - 5 fine heads: Linear -> 10 classes each (softmax DLDL)
      - Final: sum_d P(decade_d) * (decade_start_d + softmax(fine_d) @ [0..9])
    """

    def __init__(self, d_hidden: int = 256, dropout: float = 0.3):
        super().__init__()

        # Per-backbone projection (GAP only for simplicity)
        self.tongue_projs = nn.ModuleDict()
        total_tongue_dim = 0
        for name, info in BACKBONE_INFO.items():
            ch = info['channels']
            self.tongue_projs[name] = nn.Sequential(
                nn.Linear(ch, d_hidden // 4),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            total_tongue_dim += d_hidden // 4

        # Face projection
        total_face_dim = len(FACE_BANKS) * FACE_DIM
        self.face_proj = nn.Sequential(
            nn.Linear(total_face_dim, d_hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        combined_dim = total_tongue_dim + d_hidden // 4

        # Shared trunk: 3-layer MLP
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),

            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),

            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
        )

        # Decade classifier head: 5 classes
        self.decade_head = nn.Linear(d_hidden, NUM_DECADES)

        # 5 within-decade fine heads: each outputs 10 logits
        self.fine_heads = nn.ModuleList([
            nn.Linear(d_hidden, BINS_PER_DECADE) for _ in range(NUM_DECADES)
        ])

        # Precompute offset indices [0, 1, ..., 9] for expected value
        self.register_buffer(
            'fine_indices', torch.arange(BINS_PER_DECADE, dtype=torch.float32)
        )
        # Decade start ages
        self.register_buffer(
            'decade_starts_t', torch.tensor(DECADE_STARTS, dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tongue_batch, face_batch):
        # Tongue: GAP per backbone
        tongue_parts = []
        for name in BACKBONE_NAMES:
            x = tongue_batch[name]  # (B, C, H, W)
            gap = x.mean(dim=(2, 3))  # (B, C)
            tongue_parts.append(self.tongue_projs[name](gap))

        # Face: concat all banks
        face_parts = [face_batch[bank] for bank in FACE_BANKS]
        face_cat = torch.cat(face_parts, dim=-1)

        # Combine and pass through trunk
        combined = torch.cat(tongue_parts + [self.face_proj(face_cat)], dim=-1)
        h = self.trunk(combined)  # (B, d_hidden)

        # Decade logits and probabilities
        decade_logits = self.decade_head(h)  # (B, 5)
        decade_probs = F.softmax(decade_logits, dim=-1)  # (B, 5)

        # Fine head logits for each decade
        fine_logits_list = [head(h) for head in self.fine_heads]  # list of (B, 10)

        # Compute expected age per decade
        # E[age | decade_d] = decade_start_d + softmax(fine_logits_d) @ [0,1,...,9]
        fine_expected = []
        for d in range(NUM_DECADES):
            fine_probs = F.softmax(fine_logits_list[d], dim=-1)  # (B, 10)
            expected_offset = (fine_probs * self.fine_indices.unsqueeze(0)).sum(dim=-1)  # (B,)
            expected_age = self.decade_starts_t[d] + expected_offset  # (B,)
            fine_expected.append(expected_age)

        fine_expected = torch.stack(fine_expected, dim=-1)  # (B, 5)

        # Final prediction: weighted sum over decades
        final_pred = (decade_probs * fine_expected).sum(dim=-1)  # (B,)
        final_pred = torch.clamp(final_pred, MIN_AGE, MAX_AGE)

        return final_pred, decade_logits, fine_logits_list


# ============================================================================
# LABEL GENERATION
# ============================================================================

def make_decade_labels(ages: torch.Tensor) -> torch.Tensor:
    """Convert continuous ages to decade class indices (0-4)."""
    decade_idx = ((ages - MIN_AGE) / 10.0).long()
    return torch.clamp(decade_idx, 0, NUM_DECADES - 1)


def make_fine_gaussian_labels(ages: torch.Tensor, sigma: float,
                              device: torch.device) -> tuple:
    """Create Gaussian soft labels within the true decade.

    Returns:
        decade_indices: (B,) which decade each sample belongs to
        fine_labels: (B, 10) Gaussian distribution within that decade
    """
    decade_idx = make_decade_labels(ages)  # (B,)
    decade_start = torch.tensor(DECADE_STARTS, device=device, dtype=torch.float32)[decade_idx]  # (B,)

    # Offset within decade (can be fractional)
    offset = ages - decade_start  # (B,) in [0, 10)

    # Create Gaussian soft labels over 10 bins
    bins = torch.arange(BINS_PER_DECADE, device=device, dtype=torch.float32)  # (10,)
    # (B, 10) = gaussian centered at offset
    diff = bins.unsqueeze(0) - offset.unsqueeze(1)  # (B, 10)
    labels = torch.exp(-0.5 * (diff / sigma) ** 2)
    labels = labels / (labels.sum(dim=-1, keepdim=True) + 1e-8)

    return decade_idx, labels


# ============================================================================
# LOSSES
# ============================================================================

def wing_loss(pred: torch.Tensor, target: torch.Tensor,
              w: float = 2.0, eps: float = 0.5) -> torch.Tensor:
    diff = torch.abs(pred - target)
    c = w * (1.0 - math.log(1.0 + w / eps))
    loss = torch.where(
        diff < w,
        w * torch.log(1.0 + diff / eps),
        diff - c
    )
    return loss.mean()


def comp_score_np(true: np.ndarray, pred: np.ndarray) -> float:
    errors = np.abs(true - pred)
    mse = np.mean((true - pred) ** 2)
    return (0.1 / (1 + mse) + 0.4 * np.mean(errors <= 1)
            + 0.3 * np.mean(errors <= 5) + 0.2 * np.mean(errors <= 10))


def hierarchical_loss(final_pred: torch.Tensor,
                      decade_logits: torch.Tensor,
                      fine_logits_list: list,
                      ages: torch.Tensor,
                      sigma: float,
                      decade_ce_weight: float,
                      dldl_weight: float,
                      l1_weight: float,
                      wing_weight: float,
                      device: torch.device) -> torch.Tensor:
    """Compute combined hierarchical loss.

    Components:
      1. CrossEntropy on decade classification
      2. KL divergence on within-decade DLDL (true decade only)
      3. L1 on final prediction
      4. Wing loss on final prediction
    """
    # Decade classification loss
    decade_targets = make_decade_labels(ages)  # (B,)
    decade_ce = F.cross_entropy(decade_logits, decade_targets)

    # Within-decade DLDL loss (only for the true decade of each sample)
    decade_idx, fine_labels = make_fine_gaussian_labels(ages, sigma, device)

    # Gather fine logits for each sample's true decade
    batch_size = ages.shape[0]
    dldl_loss = torch.tensor(0.0, device=device)
    for d in range(NUM_DECADES):
        mask = (decade_idx == d)
        if mask.sum() == 0:
            continue
        fine_log_probs = F.log_softmax(fine_logits_list[d][mask], dim=-1)  # (n_d, 10)
        target_dist = fine_labels[mask]  # (n_d, 10)
        # KL(target || pred) = sum(target * (log(target) - log(pred)))
        kl = F.kl_div(fine_log_probs, target_dist, reduction='batchmean')
        dldl_loss = dldl_loss + kl * (mask.sum().float() / batch_size)

    # L1 regression loss
    l1 = F.l1_loss(final_pred, ages)

    # Wing loss
    wl = wing_loss(final_pred, ages)

    total = (decade_ce_weight * decade_ce
             + dldl_weight * dldl_loss
             + l1_weight * l1
             + wing_weight * wl)

    return total


# ============================================================================
# TRAINING
# ============================================================================

def train_fold(fold, train_idx, val_idx, tongue_feats, face_feats, ages,
               uuids, args, device):
    train_ds = MultiModalDataset(tongue_feats, face_feats, ages, train_idx, device)
    val_ds = MultiModalDataset(tongue_feats, face_feats, ages, val_idx, device,
                               tongue_stats=train_ds.tongue_stats,
                               face_stats=train_ds.face_stats)

    model = HierarchicalAgeModel(
        d_hidden=args.d_hidden,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_score, best_ep, wait = -1, 0, 0
    best_state = None
    n_train = train_ds.n
    snapshots = []  # for snapshot ensembling

    # Snapshot ensembling: use CosineAnnealingWarmRestarts
    if args.snapshots:
        T_0 = args.epochs // args.snap_cycles
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=1e-6)

    # Precompute age proximity matrix for C-Mixup
    if args.cmixup:
        all_train_ages = train_ds.ages  # (n_train,) on device

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)

        # Sigma annealing: linear from sigma_start to sigma_end
        progress = ep / max(args.epochs - 1, 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            tongue_b, face_b, ages_b = train_ds.get_batch(idx)

            # Augmentation: noise
            aug_tongue = {}
            for name in BACKBONE_NAMES:
                noise = torch.randn_like(tongue_b[name]) * args.noise_std
                aug_tongue[name] = tongue_b[name] + noise
            aug_face = {}
            for bank in FACE_BANKS:
                noise = torch.randn_like(face_b[bank]) * args.face_noise
                aug_face[bank] = face_b[bank] + noise

            # C-Mixup: label-proximity feature interpolation
            if args.cmixup and ages_b.shape[0] > 1:
                bs = ages_b.shape[0]
                # Sample mixing partners based on age proximity
                age_diffs = torch.abs(ages_b.unsqueeze(1) - all_train_ages.unsqueeze(0))
                # Gaussian kernel: closer ages get higher prob
                probs = torch.exp(-0.5 * (age_diffs / args.cmixup_tau) ** 2)
                # Zero out self-matches
                for bi in range(bs):
                    probs[bi, idx[bi]] = 0.0
                probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
                # Sample partner indices
                partner_idx = torch.multinomial(probs, 1).squeeze(1)
                tongue_p, face_p, ages_p = train_ds.get_batch(partner_idx)
                # Lambda from Beta distribution
                lam = torch.distributions.Beta(args.cmixup_alpha, args.cmixup_alpha).sample(
                    (bs,)).to(device)
                lam = torch.max(lam, 1.0 - lam)  # keep lam >= 0.5 so original dominates
                # Mix features
                for name in BACKBONE_NAMES:
                    lam_view = lam.view(-1, 1, 1, 1)
                    aug_tongue[name] = lam_view * aug_tongue[name] + (1 - lam_view) * tongue_p[name]
                for bank in FACE_BANKS:
                    lam_view = lam.view(-1, 1)
                    aug_face[bank] = lam_view * aug_face[bank] + (1 - lam_view) * face_p[bank]
                # Mix ages
                ages_b = lam * ages_b + (1 - lam) * ages_p

            final_pred, decade_logits, fine_logits_list = model(aug_tongue, aug_face)

            loss = hierarchical_loss(
                final_pred, decade_logits, fine_logits_list,
                ages_b, sigma,
                decade_ce_weight=args.decade_ce_weight,
                dldl_weight=args.dldl_weight,
                l1_weight=args.l1_weight,
                wing_weight=args.wing_weight,
                device=device,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Snapshot ensembling: save at cycle minima
        if args.snapshots:
            T_0 = args.epochs // args.snap_cycles
            if T_0 > 0 and (ep + 1) % T_0 == 0:
                snap_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                snapshots.append((ep, snap_state))
                print(f'    [SNAPSHOT] Saved snapshot at epoch {ep}')

        # Validation
        model.eval()
        with torch.no_grad():
            tongue_v, face_v, ages_v = val_ds.get_batch(
                torch.arange(val_ds.n, device=device))
            pred_v, _, _ = model(tongue_v, face_v)
            pred_np = pred_v.cpu().numpy()
            true_np = ages_v.cpu().numpy()

        score = comp_score_np(true_np, pred_np)
        errors = np.abs(true_np - pred_np)

        if score > best_score:
            best_score = score
            best_ep = ep
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if ep % 10 == 0 or ep < 5 or score > best_score - 0.001:
            print(f'    Fold {fold} ep{ep:3d}: loss={epoch_loss/n_batches:.4f} '
                  f'sigma={sigma:.2f} val={score:.4f} 1y={np.mean(errors<=1):.3f} '
                  f'5y={np.mean(errors<=5):.3f} best={best_score:.4f}@{best_ep}')

        if wait >= args.patience:
            print(f'    Early stop at epoch {ep}')
            break

    # Collect all model states to predict with
    # If snapshots enabled, include snapshot states + best state
    predict_states = []
    if args.snapshots and len(snapshots) > 0:
        for ep_snap, snap_state in snapshots:
            predict_states.append(("snap", ep_snap, snap_state))
        predict_states.append(("best", best_ep, best_state))
        print(f'    Using {len(predict_states)} models ({len(snapshots)} snapshots + best)')
    else:
        predict_states.append(("best", best_ep, best_state))

    test_tongue, test_face = train_ds.make_test(tongue_feats, face_feats)
    n_test = test_tongue[BACKBONE_NAMES[0]].shape[0]

    oof_preds_list = []
    test_preds_list = []

    for label, ep_s, state in predict_states:
        model.load_state_dict(state)
        model.eval()

        # OOF predictions
        with torch.no_grad():
            tongue_v, face_v, _ = val_ds.get_batch(
                torch.arange(val_ds.n, device=device))
            oof_p, _, _ = model(tongue_v, face_v)
            oof_preds_list.append(oof_p.cpu().numpy())

        # Test predictions (TTA: original + 3 noisy)
        with torch.no_grad():
            test_idx = torch.arange(n_test, device=device)
            tongue_t = {name: test_tongue[name][test_idx] for name in BACKBONE_NAMES}
            face_t = {bank: test_face[bank][test_idx] for bank in FACE_BANKS}

            p1, _, _ = model(tongue_t, face_t)
            preds = [p1.cpu().numpy()]
            for _ in range(3):
                noisy_tongue = {name: tongue_t[name] + torch.randn_like(tongue_t[name]) * 0.01
                                for name in BACKBONE_NAMES}
                noisy_face = {bank: face_t[bank] + torch.randn_like(face_t[bank]) * 0.01
                              for bank in FACE_BANKS}
                p_noisy, _, _ = model(noisy_tongue, noisy_face)
                preds.append(p_noisy.cpu().numpy())
            test_preds_list.append(np.mean(preds, axis=0))

    # Median ensemble if multiple states, else just use single
    if len(predict_states) > 1:
        oof_pred = np.median(oof_preds_list, axis=0)
        test_pred = np.median(test_preds_list, axis=0)
    else:
        oof_pred = oof_preds_list[0]
        test_pred = test_preds_list[0]

    # Save checkpoint
    save_dir = f'{args.output_dir}/hier_{args.exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state, f'{save_dir}/fold_{fold}.pt')
    if args.snapshots:
        for si, (ep_s, snap_state) in enumerate(snapshots):
            torch.save(snap_state, f'{save_dir}/fold_{fold}_snap{si}.pt')

    print(f'  Fold {fold}: score={best_score:.4f} ({best_ep}ep)')
    return oof_pred, test_pred, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--exp_name', default='hier_s42')
    parser.add_argument('--d_hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_std', type=float, default=0.02)
    parser.add_argument('--face_noise', type=float, default=0.05)
    # Loss weights
    parser.add_argument('--decade_ce_weight', type=float, default=1.0)
    parser.add_argument('--dldl_weight', type=float, default=0.5)
    parser.add_argument('--l1_weight', type=float, default=0.5)
    parser.add_argument('--wing_weight', type=float, default=0.5)
    # Sigma annealing for within-decade DLDL
    parser.add_argument('--sigma_start', type=float, default=2.0)
    parser.add_argument('--sigma_end', type=float, default=0.5)
    # C-Mixup: label-proximity feature interpolation
    parser.add_argument('--cmixup', action='store_true', help='Enable C-Mixup augmentation')
    parser.add_argument('--cmixup_alpha', type=float, default=0.4, help='Beta distribution alpha')
    parser.add_argument('--cmixup_tau', type=float, default=5.0, help='Age proximity kernel bandwidth')
    # Snapshot ensembling: save models at cosine restart minima
    parser.add_argument('--snapshots', action='store_true', help='Enable snapshot ensembling')
    parser.add_argument('--snap_cycles', type=int, default=4, help='Number of cosine restart cycles')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    print('Loading data...')
    t0 = time.time()
    tongue_feats, face_feats, ages, uuids = load_data(args.data_dir)
    print(f'  Loaded in {time.time()-t0:.1f}s')

    n_samples = len(ages)
    age_bins = np.clip((ages - MIN_AGE).astype(int), 0, NUM_CLASSES - 1)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    oof_preds = np.zeros(n_samples)
    test_preds_all = []
    fold_scores = []
    t_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), age_bins)):
        print(f'\n--- Fold {fold} ({len(train_idx)} train / {len(val_idx)} val) ---')

        if fold == 0:
            model_params = sum(p.numel() for p in HierarchicalAgeModel(
                d_hidden=args.d_hidden, dropout=args.dropout
            ).parameters())
            print(f'  Model params: {model_params:,}')

        oof_pred, test_pred, score = train_fold(
            fold, train_idx, val_idx, tongue_feats, face_feats, ages,
            uuids, args, device)

        oof_preds[val_idx] = oof_pred
        test_preds_all.append(test_pred)
        fold_scores.append(score)

    # Final OOF score
    elapsed = (time.time() - t_start) / 60
    oof_score = comp_score_np(ages, oof_preds)

    test_preds = np.mean(test_preds_all, axis=0)

    print(f'\n{"="*60}')
    print(f'Hierarchical DONE in {elapsed:.1f} min')
    print(f'OOF Score: {oof_score:.4f}')
    print(f'Fold scores: {[f"{s:.4f}" for s in fold_scores]}')
    errors = np.abs(ages - oof_preds)
    mse = np.mean((ages - oof_preds) ** 2)
    print(f'  MSE={mse:.1f}, 1Y={np.mean(errors<=1):.3f}, '
          f'5Y={np.mean(errors<=5):.3f}, 10Y={np.mean(errors<=10):.3f}')

    # Per-age breakdown
    print('\nPer-age-group OOF:')
    for lo in range(20, 65, 5):
        hi = lo + 5
        mask = (ages >= lo) & (ages < hi)
        if mask.sum() == 0:
            continue
        errs = np.abs(ages[mask] - oof_preds[mask])
        print(f'  [{lo}-{hi}): n={mask.sum()} MAE={errs.mean():.2f} '
              f'1Y={np.mean(errs<=1):.2f} 5Y={np.mean(errs<=5):.2f}')

    # Decade classification accuracy
    decade_true = np.clip(((ages - MIN_AGE) / 10).astype(int), 0, NUM_DECADES - 1)
    decade_pred = np.clip(((oof_preds - MIN_AGE) / 10).astype(int), 0, NUM_DECADES - 1)
    decade_acc = np.mean(decade_true == decade_pred)
    print(f'\nDecade classification accuracy (from predictions): {decade_acc:.3f}')

    # Save
    save_dir = f'{args.output_dir}/hier_{args.exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/oof_predictions.npy', oof_preds)
    np.save(f'{save_dir}/oof_ages.npy', ages)
    np.save(f'{save_dir}/test_predictions.npy', test_preds)

    # Also save to server_preds for blending
    os.makedirs('server_preds', exist_ok=True)
    np.save(f'server_preds/hier_{args.exp_name}_oof.npy', oof_preds)
    np.save(f'server_preds/hier_{args.exp_name}_test.npy', test_preds)

    # Save submission
    sub_path = f'{save_dir}/submission_hier_{args.exp_name}.csv'
    with open(sub_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['uuid', 'age'])
        for u, a in zip(uuids['test'], test_preds):
            w.writerow([u, f'{a:.4f}'])

    # Summary JSON
    with open(f'{save_dir}/summary.json', 'w') as f:
        json.dump({
            'oof_score': oof_score,
            'fold_scores': fold_scores,
            'elapsed_min': elapsed,
            'args': vars(args),
        }, f, indent=2)

    print(f'\nSubmission saved: {sub_path}')


if __name__ == '__main__':
    main()
