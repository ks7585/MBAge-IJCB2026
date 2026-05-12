"""Phase 2 v3: Full-recipe Multi-Modal Tongue + Face Fusion.

Restores ALL Phase 1 winning ingredients:
  - Phased training: DLDL-only -> add KD -> add MetricSurrogate
  - Knowledge Distillation (soft teacher targets)
  - Metric Surrogate Loss (differentiable competition score)
  - Wing Loss (amplifies small errors -> helps 1Y-ACC)
  - CORAL ordinal loss for triple head
  - Proper sigma annealing (warmup + cosine decay)
  - All augmentations (noise, cutout, backbone dropout, flip)

Face fusion modes:
  none   : tongue-only baseline (no face features)
  late   : concat face after transformer
  film   : FiLM conditioning (face modulates tongue features)
  gated  : learned per-sample modality gate

Usage:
  python train_phase2_v3.py --data_dir data --face_mode film --triple_head --seed 42
"""

import argparse
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

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_AGE, MAX_AGE, NUM_CLASSES = 20, 69, 50

BACKBONE_INFO = {
    'ResNet18': {'channels': 128, 'file': 'tongue_ResNet18_features_v2.0.pkl'},
    'MobileNetV2': {'channels': 32, 'file': 'tongue_MobileNetV2_features_v2.0.pkl'},
    'EfficientNetB0': {'channels': 40, 'file': 'tongue_EfficientNetB0_features_v2.0.pkl'},
    'DenseNet121': {'channels': 128, 'file': 'tongue_DenseNet121_features_v2.0.pkl'},
}
BACKBONE_NAMES = list(BACKBONE_INFO.keys())

FACE_FILE = 'face_features_gabor_v2.0.pkl'
FACE_BANKS = ['features_A', 'features_B', 'features_C', 'features_D']
FACE_DIM = 40  # per bank
FACE_TOTAL_DIM = FACE_DIM * len(FACE_BANKS)  # 160


# ============================================================================
# DATA
# ============================================================================

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
# MODEL COMPONENTS
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(),
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class BackboneEncoder(nn.Module):
    def __init__(self, in_ch, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d, 1), nn.BatchNorm2d(d), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, groups=d), nn.BatchNorm2d(d), nn.GELU(),
        )
        self.se = SEBlock(d)
    def forward(self, x):
        return self.se(self.net(x))


class DLDLHead(nn.Module):
    def __init__(self, in_f, hidden=256, drop=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_f, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(drop * 0.7),
            nn.Linear(hidden // 2, NUM_CLASSES))
        self.register_buffer('ages', torch.arange(MIN_AGE, MAX_AGE + 1, dtype=torch.float32))
    def forward(self, x):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        age = (probs * self.ages).sum(-1)
        return logits, probs, age


class RegressionHead(nn.Module):
    def __init__(self, in_f, hidden=128, drop=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_f, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, 1))
    def forward(self, x):
        return self.fc(x).squeeze(-1)


class OrdinalHead(nn.Module):
    def __init__(self, in_f, hidden=128, drop=0.3, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_f, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, num_classes - 1))
    def forward(self, x):
        logits = self.fc(x)
        probs = torch.sigmoid(logits)
        age = probs.sum(dim=-1) + MIN_AGE
        return logits, age


# ============================================================================
# MULTI-MODAL MODEL
# ============================================================================

class MultiModalModelV3(nn.Module):
    """Full-recipe multi-modal model with FiLM/gated/late/none fusion.

    Returns dict with all outputs for proper loss computation.
    """

    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.3,
                 face_mode='none', use_triple_head=False):
        super().__init__()
        self.face_mode = face_mode
        self.d_model = d_model

        channels = {n: info['channels'] for n, info in BACKBONE_INFO.items()}

        # Tongue encoders
        self.tongue_encoders = nn.ModuleDict({
            n: BackboneEncoder(c, d_model) for n, c in channels.items()
        })
        self.tongue_projs = nn.ModuleDict({
            n: nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(d_model, d_model), nn.GELU()
            ) for n in channels
        })

        # Face encoder (mode-dependent)
        if face_mode == 'film':
            # FiLM: face generates per-channel gamma/beta for tongue features
            self.film_net = nn.Sequential(
                nn.Linear(FACE_TOTAL_DIM, d_model * 2), nn.GELU(),
                nn.Linear(d_model * 2, d_model * 2),  # outputs [gamma, beta]
            )
        elif face_mode == 'gated':
            # Gated: learn per-sample fusion weight
            self.face_proj = nn.Sequential(
                nn.Linear(FACE_TOTAL_DIM, d_model), nn.GELU(),
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            )
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.GELU(),
                nn.Linear(d_model, 1), nn.Sigmoid(),
            )
        elif face_mode == 'late':
            self.face_proj = nn.Sequential(
                nn.Linear(FACE_TOTAL_DIM, d_model), nn.GELU(),
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            )

        # Transformer (always 4 tongue tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 2, dropout,
            activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Head dimensions
        if face_mode == 'late':
            feat_dim = d_model * 4 + d_model
        else:
            feat_dim = d_model * 4

        self.dldl_head = DLDLHead(feat_dim, hidden=256, drop=dropout)

        self.use_triple_head = use_triple_head
        if use_triple_head:
            self.reg_head = RegressionHead(feat_dim, hidden=128, drop=dropout)
            self.ord_head = OrdinalHead(feat_dim, hidden=128, drop=dropout)
            self.head_weights = nn.Parameter(torch.zeros(3))

    def _concat_face(self, face_feats):
        """Concatenate all face banks into a single vector."""
        return torch.cat([face_feats[b] for b in FACE_BANKS], dim=-1)  # (B, 160)

    def forward(self, tongue_feats, face_feats):
        # Encode tongue backbone features
        encoded = {}
        for name in BACKBONE_NAMES:
            encoded[name] = self.tongue_encoders[name](tongue_feats[name])

        # FiLM conditioning: modulate encoded tongue features before projection
        if self.face_mode == 'film':
            face_vec = self._concat_face(face_feats)
            film_params = self.film_net(face_vec)  # (B, d*2)
            gamma = film_params[:, :self.d_model].unsqueeze(-1).unsqueeze(-1)  # (B, d, 1, 1)
            beta = film_params[:, self.d_model:].unsqueeze(-1).unsqueeze(-1)
            for name in BACKBONE_NAMES:
                encoded[name] = gamma * encoded[name] + beta

        # Project to tokens
        tokens = []
        for name in BACKBONE_NAMES:
            tokens.append(self.tongue_projs[name](encoded[name]))
        tongue_tokens = torch.stack(tokens, dim=1)  # (B, 4, d)

        # Transformer
        x = self.transformer(tongue_tokens)
        tongue_flat = x.reshape(x.shape[0], -1)  # (B, 4*d)

        # Fusion
        if self.face_mode == 'late':
            face_vec = self._concat_face(face_feats)
            face_out = self.face_proj(face_vec)
            features = torch.cat([tongue_flat, face_out], dim=-1)
        elif self.face_mode == 'gated':
            face_vec = self._concat_face(face_feats)
            face_out = self.face_proj(face_vec)
            tongue_pool = tongue_flat[:, :self.d_model]  # use first token as summary
            gate_input = torch.cat([tongue_pool, face_out], dim=-1)
            g = self.gate(gate_input)  # (B, 1)
            # Gate modulates the tongue representation
            gated = tongue_flat * g + tongue_flat * (1 - g) * 0.5  # soft gate
            features = gated  # same dim as tongue-only
        else:
            features = tongue_flat

        # Heads
        logits, probs, age_dldl = self.dldl_head(features)

        result = {
            'logits': logits, 'probs': probs,
            'age': age_dldl, 'age_dldl': age_dldl,
            'features': features,
        }

        if self.use_triple_head:
            age_reg = self.reg_head(features)
            ord_logits, age_ord = self.ord_head(features)
            w = F.softmax(self.head_weights, dim=0)
            age_fused = w[0] * age_dldl + w[1] * age_reg + w[2] * age_ord
            result.update({
                'age': age_fused,
                'age_reg': age_reg, 'age_ord': age_ord,
                'ord_logits': ord_logits,
                'head_weights': w,
            })

        return result


# ============================================================================
# LOSSES (Full Phase 1 Recipe)
# ============================================================================

class MultiTaskDLDL(nn.Module):
    """KL + L1 + Variance loss with sigma annealing."""
    def __init__(self, sigma=3.0, dist='gaussian', w_kl=0.5, w_l1=0.3, w_var=0.2, lam_var=0.05):
        super().__init__()
        self.sigma = sigma
        self.dist = dist
        self.w_kl, self.w_l1, self.w_var, self.lam_var = w_kl, w_l1, w_var, lam_var
        self.register_buffer('ages_vec', torch.arange(MIN_AGE, MAX_AGE + 1, dtype=torch.float32))

    def set_sigma(self, s):
        self.sigma = s

    def forward(self, logits, true_ages):
        ages = self.ages_vec.unsqueeze(0)
        targets = true_ages.unsqueeze(1)
        if self.dist == 'gaussian':
            t = torch.exp(-0.5 * ((ages - targets) / self.sigma) ** 2)
        else:
            t = torch.exp(-torch.abs(ages - targets) / self.sigma)
        t = t + 1e-8
        t = t / t.sum(-1, keepdim=True)

        log_p = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(log_p, t, reduction='none').sum(-1)

        probs = F.softmax(logits, dim=-1)
        exp_age = (probs * ages).sum(-1)
        l1 = torch.abs(exp_age - true_ages)
        var = (probs * (ages - exp_age.unsqueeze(1)) ** 2).sum(-1)

        loss = self.w_kl * kl + self.w_l1 * l1 + self.w_var * self.lam_var * var
        return loss.mean()


class MetricSurrogateLoss(nn.Module):
    """Differentiable approximation of competition score."""
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.thresholds = [1.0, 5.0, 10.0]
        self.weights = [0.4, 0.3, 0.2]

    def forward(self, pred_ages, true_ages):
        errors = torch.abs(pred_ages - true_ages)
        mse = (errors ** 2).mean()
        mse_score = 0.1 / (1.0 + mse)
        acc_score = torch.tensor(0.0, device=pred_ages.device)
        for thresh, weight in zip(self.thresholds, self.weights):
            smooth_acc = torch.sigmoid((thresh - errors) / self.temperature)
            acc_score = acc_score + weight * smooth_acc.mean()
        return -(mse_score + acc_score)


class WingLoss(nn.Module):
    """Wing Loss -- amplifies small errors for 1Y-ACC optimization."""
    def __init__(self, w=2.0, epsilon=0.5):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        small = self.w * torch.log(1 + diff / self.epsilon)
        large = diff - self.C
        loss = torch.where(diff < self.w, small, large)
        return loss.mean()


class OrdinalBCELoss(nn.Module):
    """CORAL ordinal loss."""
    def __init__(self):
        super().__init__()
        self.register_buffer('thresholds',
                             torch.linspace(MIN_AGE + 0.5, MAX_AGE - 0.5, NUM_CLASSES - 1))
    def forward(self, logits, true_ages):
        labels = (true_ages.unsqueeze(1) > self.thresholds.unsqueeze(0)).float()
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')


class SigmaScheduler:
    """Cosine annealing for DLDL sigma: warmup -> cosine decay."""
    def __init__(self, start=3.0, end=0.5, warmup=20, anneal=100):
        self.start, self.end, self.warmup, self.anneal = start, end, warmup, anneal
    def __call__(self, epoch):
        if epoch < self.warmup:
            return self.start
        p = min(1.0, (epoch - self.warmup) / self.anneal)
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(p * math.pi))


# ============================================================================
# SCORING
# ============================================================================

def comp_score(true, pred):
    """Competition score: 0.1/(1+MSE) + 0.4*1Y + 0.3*5Y + 0.2*10Y."""
    errors = np.abs(true - pred)
    mse = np.mean((true - pred) ** 2)
    acc1 = np.mean(errors <= 1.0)
    acc5 = np.mean(errors <= 5.0)
    acc10 = np.mean(errors <= 10.0)
    return 0.1 / (1 + mse) + 0.4 * acc1 + 0.3 * acc5 + 0.2 * acc10


# ============================================================================
# AUGMENTATION
# ============================================================================

def augment_tongue(feats, noise_std=0.02, cutout=0.4, backbone_drop=0.1, flip_prob=0.5):
    """Augment tongue spatial features."""
    B = next(iter(feats.values())).shape[0]
    device = next(iter(feats.values())).device

    # Gaussian noise
    if noise_std > 0:
        feats = {k: v + torch.randn_like(v) * noise_std for k, v in feats.items()}

    # Spatial cutout
    if cutout > 0:
        mask = torch.rand(B, 1, 28, 28, device=device) > cutout
        feats = {k: v * mask for k, v in feats.items()}

    # Backbone dropout
    if backbone_drop > 0:
        for name in BACKBONE_NAMES:
            drop_mask = torch.rand(B, 1, 1, 1, device=device) > backbone_drop
            feats[name] = feats[name] * drop_mask

    # Horizontal flip
    if flip_prob > 0 and torch.rand(1).item() < flip_prob:
        feats = {k: v.flip(-1) for k, v in feats.items()}

    return feats


def augment_face(face_feats, noise_std=0.05, dropout=0.1):
    """Augment face Gabor features."""
    result = {}
    for bank in FACE_BANKS:
        f = face_feats[bank]
        if noise_std > 0:
            f = f + torch.randn_like(f) * noise_std
        if dropout > 0:
            mask = torch.rand_like(f) > dropout
            f = f * mask / (1 - dropout)
        result[bank] = f
    return result


# ============================================================================
# TRAINING
# ============================================================================

def train_fold(args, fold_idx, train_idx, val_idx,
               tongue_feats, face_feats, ages, distill_targets, device):
    """Train one fold with full Phase 1 recipe."""

    train_ds = MultiModalDataset(tongue_feats, face_feats, ages, train_idx, device)
    val_ds = MultiModalDataset(tongue_feats, face_feats, ages, val_idx, device,
                                tongue_stats=train_ds.tongue_stats,
                                face_stats=train_ds.face_stats)

    # Build model
    model = MultiModalModelV3(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, face_mode=args.face_mode,
        use_triple_head=args.triple_head,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if fold_idx == 0:
        print(f"  Model params: {n_params:,} (face_mode={args.face_mode}, "
              f"triple_head={args.triple_head})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Losses
    criterion = MultiTaskDLDL(sigma=args.sigma_start, dist=args.dist,
                               w_kl=args.w_kl, w_l1=args.w_l1, w_var=args.w_var).to(device)
    sigma_sched = SigmaScheduler(args.sigma_start, args.sigma_end,
                                  warmup=args.sigma_warmup, anneal=args.sigma_anneal)

    metric_loss_fn = MetricSurrogateLoss(temperature=2.0).to(device)
    wing_loss_fn = WingLoss(w=args.wing_w, epsilon=args.wing_eps).to(device) if args.wing_weight > 0 else None
    ordinal_loss_fn = OrdinalBCELoss().to(device) if args.triple_head else None

    # KD targets for this fold
    fold_distill = None
    if distill_targets is not None:
        fold_distill = torch.from_numpy(distill_targets[train_idx]).to(device)

    # Training state
    best_score = -1
    best_epoch = 0
    best_state = None
    patience_counter = 0
    warmup_epochs = 10
    bs = args.batch_size

    for epoch in range(args.epochs):
        model.train()
        sigma = sigma_sched(epoch)
        criterion.set_sigma(sigma)

        # Phased metric surrogate: ramp temperature from 2.0 -> 0.5
        p = min(1.0, epoch / max(1, args.epochs * 0.7))
        metric_loss_fn.temperature = 2.0 - 1.5 * p

        # Phased metric weight
        if epoch < args.metric_start:
            metric_w = 0.0
        elif epoch < args.metric_start + 20:
            ramp = (epoch - args.metric_start) / 20.0
            metric_w = args.metric_weight * ramp
        else:
            metric_w = args.metric_weight

        # Phased KD
        kd_active = (fold_distill is not None and epoch >= args.kd_start)

        perm = torch.randperm(train_ds.n, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, train_ds.n, bs):
            idx = perm[i:i + bs]
            tongue_batch, face_batch, age_batch = train_ds.get_batch(idx)

            # Augment tongue
            tongue_batch = augment_tongue(tongue_batch, noise_std=args.noise_std,
                                           cutout=args.cutout,
                                           backbone_drop=args.backbone_drop,
                                           flip_prob=args.flip_prob)
            # Augment face
            if args.face_mode != 'none':
                face_batch = augment_face(face_batch, noise_std=args.face_noise,
                                           dropout=args.face_dropout)

            # KD target blending
            if kd_active:
                kd_ages = fold_distill[idx]
                ages_target = (1 - args.kd_alpha) * age_batch + args.kd_alpha * kd_ages
            else:
                ages_target = age_batch

            # Forward
            out = model(tongue_batch, face_batch)

            # 1. DLDL loss (KL + L1 + Var)
            loss = criterion(out['logits'], ages_target)

            # 2. Wing Loss on predicted age
            if wing_loss_fn is not None:
                wl = wing_loss_fn(out['age'], age_batch)
                loss = loss + args.wing_weight * wl

            # 3. Ordinal loss for triple head
            if args.triple_head and ordinal_loss_fn is not None and 'ord_logits' in out:
                ol = ordinal_loss_fn(out['ord_logits'], age_batch)
                loss = loss + args.ordinal_weight * ol
                # Wing on sub-heads
                if wing_loss_fn is not None:
                    wl_sub = wing_loss_fn(out['age_reg'], age_batch) + \
                             wing_loss_fn(out['age_ord'], age_batch)
                    loss = loss + 0.2 * args.wing_weight * wl_sub

            # 4. Metric surrogate loss (phased)
            if metric_w > 0:
                ml = metric_loss_fn(out['age'], age_batch)
                loss = loss + metric_w * ml

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            tongue_val, face_val, ages_val = val_ds.get_batch(
                torch.arange(val_ds.n, device=device))
            out_val = model(tongue_val, face_val)
            val_preds = out_val['age'].cpu().numpy()
            val_ages = ages_val.cpu().numpy()
            val_preds = np.clip(val_preds, MIN_AGE, MAX_AGE)
            score = comp_score(val_ages, val_preds)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"    Fold {fold_idx} ep{epoch:3d}: loss={total_loss/n_batches:.4f} "
                  f"val={score:.4f} best={best_score:.4f}@{best_epoch} "
                  f"sigma={sigma:.2f} mw={metric_w:.2f}", flush=True)

        if patience_counter >= args.patience:
            print(f"    Early stop at epoch {epoch}", flush=True)
            break

    # Load best and generate predictions
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # OOF predictions
    with torch.no_grad():
        tongue_val, face_val, _ = val_ds.get_batch(torch.arange(val_ds.n, device=device))
        oof_preds = model(tongue_val, face_val)['age'].cpu().numpy()

    # Free validation data from GPU before test inference
    tongue_stats = train_ds.tongue_stats
    face_stats = train_ds.face_stats
    del val_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test predictions (with TTA: noise x3)
    test_tongue, test_face = train_ds.make_test(tongue_feats, face_feats)
    test_preds_list = []
    with torch.no_grad():
        # Clean pass
        out_test = model(test_tongue, test_face)
        test_preds_list.append(out_test['age'].cpu().numpy())

        # Noise TTA (3 passes) -- reuse tensors to save memory
        for _ in range(3):
            for k in test_tongue:
                test_tongue[k] = test_tongue[k] + torch.randn_like(test_tongue[k]) * 0.01
            if args.face_mode != 'none':
                for k in test_face:
                    test_face[k] = test_face[k] + torch.randn_like(test_face[k]) * 0.02
            out_tta = model(test_tongue, test_face)
            test_preds_list.append(out_tta['age'].cpu().numpy())

    test_preds = np.mean(test_preds_list, axis=0)
    del test_tongue, test_face
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Checkpoint
    ckpt = {
        'model_state': best_state,
        'score': best_score,
        'epoch': best_epoch,
        'fold': fold_idx,
        'args': vars(args),
    }

    return ckpt, oof_preds, val_idx, test_preds, best_score


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--data_dir', default='data')
    p.add_argument('--output_dir', default='checkpoints')
    p.add_argument('--exp_name', default='p2v3_default')
    # Model
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--face_mode', choices=['none', 'late', 'film', 'gated'], default='none')
    p.add_argument('--triple_head', action='store_true')
    # DLDL
    p.add_argument('--dist', default='gaussian')
    p.add_argument('--sigma_start', type=float, default=3.0)
    p.add_argument('--sigma_end', type=float, default=0.5)
    p.add_argument('--sigma_warmup', type=int, default=20)
    p.add_argument('--sigma_anneal', type=int, default=100)
    p.add_argument('--w_kl', type=float, default=0.3)
    p.add_argument('--w_l1', type=float, default=0.5)
    p.add_argument('--w_var', type=float, default=0.2)
    # Wing Loss
    p.add_argument('--wing_weight', type=float, default=0.5)
    p.add_argument('--wing_w', type=float, default=2.0)
    p.add_argument('--wing_eps', type=float, default=0.5)
    # Ordinal
    p.add_argument('--ordinal_weight', type=float, default=0.3)
    # Metric Surrogate
    p.add_argument('--metric_weight', type=float, default=1.0)
    p.add_argument('--metric_start', type=int, default=20)
    # Knowledge Distillation
    p.add_argument('--kd_targets', type=str, default=None, help='Path to .npy OOF predictions')
    p.add_argument('--kd_alpha', type=float, default=0.5)
    p.add_argument('--kd_start', type=int, default=15)
    # Training
    p.add_argument('--n_folds', type=int, default=10)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    # Augmentation
    p.add_argument('--noise_std', type=float, default=0.02)
    p.add_argument('--cutout', type=float, default=0.4)
    p.add_argument('--backbone_drop', type=float, default=0.1)
    p.add_argument('--flip_prob', type=float, default=0.5)
    p.add_argument('--face_noise', type=float, default=0.05)
    p.add_argument('--face_dropout', type=float, default=0.1)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: face_mode={args.face_mode}, triple_head={args.triple_head}, "
          f"seed={args.seed}, wing={args.wing_weight}, metric={args.metric_weight}", flush=True)

    # Load data
    print("Loading data...", flush=True)
    t0 = time.time()
    tongue_feats, face_feats, ages, uuids = load_data(args.data_dir)
    print(f"  Loaded in {time.time()-t0:.1f}s: {len(ages)} train, "
          f"{len(uuids['test'])} test", flush=True)

    # Load KD targets
    distill_targets = None
    if args.kd_targets and os.path.exists(args.kd_targets):
        distill_targets = np.load(args.kd_targets)
        print(f"  KD targets loaded: {distill_targets.shape}", flush=True)

    # Output dir
    out_dir = Path(args.output_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cross-validation
    bins = np.digitize(ages, np.arange(25, 70, 5))
    kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    all_oof = np.zeros(len(ages))
    all_test = []
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(ages, bins)):
        print(f"\n--- Fold {fold_idx} ({len(train_idx)} train / {len(val_idx)} val) ---",
              flush=True)

        ckpt, oof_preds, val_indices, test_preds, score = train_fold(
            args, fold_idx, train_idx, val_idx,
            tongue_feats, face_feats, ages, distill_targets, device)

        all_oof[val_idx] = oof_preds
        all_test.append(test_preds)
        fold_scores.append(score)

        torch.save(ckpt, out_dir / f"fold_{fold_idx}.pt")
        print(f"  Fold {fold_idx}: score={score:.4f} ({ckpt['epoch']}ep)", flush=True)

    # Summary
    oof_score = comp_score(ages, np.clip(all_oof, MIN_AGE, MAX_AGE))
    test_avg = np.mean(all_test, axis=0)
    test_avg = np.clip(test_avg, MIN_AGE, MAX_AGE)

    oof_errors = np.abs(ages - np.clip(all_oof, MIN_AGE, MAX_AGE))
    oof_mse = np.mean((ages - np.clip(all_oof, MIN_AGE, MAX_AGE)) ** 2)
    oof_1y = np.mean(oof_errors <= 1.0)
    oof_5y = np.mean(oof_errors <= 5.0)
    oof_10y = np.mean(oof_errors <= 10.0)

    print(f"\n{'='*60}")
    print(f"OOF Score: {oof_score:.4f}")
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Test: mean={test_avg.mean():.2f}, std={test_avg.std():.2f}")
    print(f"  MSE={oof_mse:.1f}, 1Y={oof_1y:.3f}, 5Y={oof_5y:.3f}, 10Y={oof_10y:.3f}")
    print(f"\nSaved to {out_dir}", flush=True)

    # Save OOF predictions
    np.save(out_dir / "oof_predictions.npy", all_oof)

    # Save test predictions
    np.save(out_dir / "test_predictions.npy", test_avg)

    # Save summary
    summary = {
        'oof_score': float(oof_score),
        'fold_scores': [float(s) for s in fold_scores],
        'args': vars(args),
        'n_params': sum(p.numel() for p in MultiModalModelV3(
            args.d_model, args.n_heads, args.n_layers, args.dropout,
            args.face_mode, args.triple_head).parameters() if p.requires_grad),
        'oof_mse': float(oof_mse),
        'oof_1y': float(oof_1y),
        'oof_5y': float(oof_5y),
        'oof_10y': float(oof_10y),
    }
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save submission CSV
    import csv
    csv_path = out_dir / "submission.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['uuid', 'age'])
        for uid, age in zip(uuids['test'], test_avg):
            writer.writerow([uid, f"{age:.4f}"])
    print(f"Submission: {csv_path}", flush=True)


if __name__ == '__main__':
    main()
