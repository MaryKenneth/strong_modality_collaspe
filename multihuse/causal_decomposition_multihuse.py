"""
CAUSAL DECOMPOSITION — MultiHuSE (2x2 grid: attention x classifier-freezing)
===============================================================================
Supports MODEL_TYPE = 'distilled' or 'baseline' (set below). 
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import copy
import json

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
SPLITS_DIR      = r'multihuse_dataset\splits_strict_v2'
FEATURES_DIR    = r'multihuse_dataset'
CHECKPOINTS_DIR = 'checkpoints'

MODEL_TYPE = 'baseline'   # ← CHANGE: 'distilled' or 'baseline'

OUTPUT_DIR = f'causal_decomposition_results_multihuse_{MODEL_TYPE}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIO_DIM    = 1280
NUM_CLASSES  = 5
LABEL_COLUMN = 'humor_style'
FUSION_DIM   = 512
BATCH_SIZE   = 64
SEED         = 999

CELL2_EPOCHS        = 100
CELL2_LR            = 5e-5
CELL2_WEIGHT_DECAY  = 1e-3
CELL2_PATIENCE      = 40
CELL2_MIXUP_ALPHA    = 0.05
CELL2_WARMUP_EPOCHS  = 15

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}  |  MODEL_TYPE={MODEL_TYPE}")

# ─── Per-MODEL_TYPE combination configs (checkpoint naming differs) ─────────

COMBINATIONS_DISTILLED = [
    {'name': 'bert_audio_videomae', 'text_features': 'bert_text_embeddings.csv',
     'text_checkpoint': 'multihuse_distill_bert_from_e5', 'text_type': 'bert_student', 'text_dim': 768,
     'audio_checkpoint': 'multihuse_distill_audio_from_e5',
     'video_features': 'videomae_visual_embeddings.csv', 'video_checkpoint': 'multihuse_distill_videomae_from_e5', 'video_dim': 768,
     'unimodal_ceiling': 70.2},
    {'name': 'bert_audio_pecore', 'text_features': 'bert_text_embeddings.csv',
     'text_checkpoint': 'multihuse_distill_bert_from_e5', 'text_type': 'bert_student', 'text_dim': 768,
     'audio_checkpoint': 'multihuse_distill_audio_from_e5',
     'video_features': 'pe_core_visual_embeddings.csv', 'video_checkpoint': 'multihuse_distill_video_pe_from_e5', 'video_dim': 1024,
     'unimodal_ceiling': 70.2},
    {'name': 'e5_audio_videomae', 'text_features': 'e5_text_embeddings.csv',
     'text_checkpoint': 'text_e5_text_embeddings', 'text_type': 'e5_baseline', 'text_dim': 1024,
     'audio_checkpoint': 'multihuse_distill_audio_from_e5',
     'video_features': 'videomae_visual_embeddings.csv', 'video_checkpoint': 'multihuse_distill_videomae_from_e5', 'video_dim': 768,
     'unimodal_ceiling': 74.9},
    {'name': 'e5_audio_pecore', 'text_features': 'e5_text_embeddings.csv',
     'text_checkpoint': 'text_e5_text_embeddings', 'text_type': 'e5_baseline', 'text_dim': 1024,
     'audio_checkpoint': 'multihuse_distill_audio_from_e5',
     'video_features': 'pe_core_visual_embeddings.csv', 'video_checkpoint': 'multihuse_distill_video_pe_from_e5', 'video_dim': 1024,
     'unimodal_ceiling': 74.9},
]

COMBINATIONS_BASELINE = [
    {'name': 'bert_audio_videomae', 'text_features': 'bert_text_embeddings.csv',
     'text_checkpoint': 'text_bert_text_embeddings', 'text_dim': 768,
     'audio_checkpoint': 'audio_dasheng_audio_embeddings',
     'video_features': 'videomae_visual_embeddings.csv', 'video_checkpoint': 'video_videomae_visual_embeddings', 'video_dim': 768,
     'unimodal_ceiling': None},
    {'name': 'bert_audio_pecore', 'text_features': 'bert_text_embeddings.csv',
     'text_checkpoint': 'text_bert_text_embeddings', 'text_dim': 768,
     'audio_checkpoint': 'audio_dasheng_audio_embeddings',
     'video_features': 'pe_core_visual_embeddings.csv', 'video_checkpoint': 'video_pe_core_visual_embeddings', 'video_dim': 1024,
     'unimodal_ceiling': None},
    {'name': 'e5_audio_videomae', 'text_features': 'e5_text_embeddings.csv',
     'text_checkpoint': 'text_e5_text_embeddings', 'text_dim': 1024,
     'audio_checkpoint': 'audio_dasheng_audio_embeddings',
     'video_features': 'videomae_visual_embeddings.csv', 'video_checkpoint': 'video_videomae_visual_embeddings', 'video_dim': 768,
     'unimodal_ceiling': None},
    {'name': 'e5_audio_pecore', 'text_features': 'e5_text_embeddings.csv',
     'text_checkpoint': 'text_e5_text_embeddings', 'text_dim': 1024,
     'audio_checkpoint': 'audio_dasheng_audio_embeddings',
     'video_features': 'pe_core_visual_embeddings.csv', 'video_checkpoint': 'video_pe_core_visual_embeddings', 'video_dim': 1024,
     'unimodal_ceiling': None},
]

COMBINATIONS = COMBINATIONS_DISTILLED if MODEL_TYPE == 'distilled' else COMBINATIONS_BASELINE


# ==========================================================================
# BACKBONE MODELS — distilled variants
class TextModel(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x)
    def get_features(self, x): return self.net[:4](x)
    def get_classifier(self): return self.net[4]


class BERTStudent(nn.Module):
    """Distilled-only text backbone."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x): return self.classifier(self.encoder(x))
    def get_features(self, x): return self.encoder(x)
    def get_classifier(self): return self.classifier


class AudioStudent(nn.Module):
    """Distilled-only audio backbone"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(AUDIO_DIM, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 768),       nn.BatchNorm1d(768),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, 512),        nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x): return self.classifier(self.encoder(x))
    def get_features(self, x): return self.encoder(x)


class VideoStudent(nn.Module):
    """Distilled-only video backbone"""
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 768),       nn.BatchNorm1d(768),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, 512),        nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x): return self.classifier(self.encoder(x))
    def get_features(self, x): return self.encoder(x)


# ==========================================================================
# BACKBONE MODELS — baseline variants 
class BaselineAudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUDIO_DIM, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),   # [0-3]
            nn.Linear(1024, 768),       nn.BatchNorm1d(768),  nn.ReLU(), nn.Dropout(0.3),   # [4-7]
            nn.Linear(768, 512),        nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.2),   # [8-11] ← 512-dim
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(0.2),                          # [12-14]
            nn.Linear(256, NUM_CLASSES)                                                       # [15]
        )
    def forward(self, x): return self.net(x)
    def get_features(self, x): return self.net[:12](x)


class BaselineVideoModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),   # [0-3]
            nn.Linear(1024, 512),       nn.ReLU(), nn.Dropout(0.3),                         # [4-6] ← 512-dim
            nn.Linear(512, NUM_CLASSES)                                                     # [7]
        )
    def forward(self, x): return self.net(x)
    def get_features(self, x): return self.net[:7](x)


def load_text_model(combo, fold):
    if MODEL_TYPE == 'distilled' and combo.get('text_type') == 'bert_student':
        m = BERTStudent().to(device)
    else:
        m = TextModel(combo['text_dim']).to(device)
    m.load_state_dict(torch.load(
        f"{CHECKPOINTS_DIR}/{combo['text_checkpoint']}_fold_{fold}.pth", map_location=device))
    return m

def load_audio_model(combo, fold):
    if MODEL_TYPE == 'distilled':
        m = AudioStudent().to(device)
    else:
        m = BaselineAudioModel().to(device)
    m.load_state_dict(torch.load(
        f"{CHECKPOINTS_DIR}/{combo['audio_checkpoint']}_fold_{fold}.pth", map_location=device))
    return m


def load_video_model(combo, fold):
    if MODEL_TYPE == 'distilled':
        m = VideoStudent(combo['video_dim']).to(device)
    else:
        m = BaselineVideoModel(combo['video_dim']).to(device)
    m.load_state_dict(torch.load(
        f"{CHECKPOINTS_DIR}/{combo['video_checkpoint']}_fold_{fold}.pth", map_location=device))
    return m


class CrossModalAnchor(nn.Module):
    def __init__(self, d_model=FUSION_DIM):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, anchor, context):
        a_seq = anchor.unsqueeze(1)
        out, _ = self.cross_attn(query=a_seq, key=context, value=context)
        return self.norm(anchor + self.dropout(out.squeeze(1)))


def get_text_classifier(text_model):
    if hasattr(text_model, 'get_classifier'):
        return text_model.get_classifier()
    return text_model.net[4]   # baseline TextModel 


# ═══════════════════════════════════════════════════════════════════════════
# CELL 1: original frozen-head IAF (dominant=text always, per these combos)
# ═══════════════════════════════════════════════════════════════════════════

class InvertedAsymmetricFusionFrozen(nn.Module):
    """Matches the ORIGINAL frozen-head IAF exactly (Cell 1) — works for both
    distilled and baseline backbones via get_text_classifier()."""
    def __init__(self, text_model, audio_model, video_model):
        super().__init__()
        self.text_backbone, self.audio_backbone, self.video_backbone = text_model, audio_model, video_model
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters(): p.requires_grad = False
        self.audio_anchor = CrossModalAnchor(FUSION_DIM)
        self.video_anchor = CrossModalAnchor(FUSION_DIM)
        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 3),
        )
        self.text_classifier = copy.deepcopy(get_text_classifier(text_model))
        for p in self.text_classifier.parameters(): p.requires_grad = False
        self.audio_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)
        self.video_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, t_feat, a_feat, v_feat):
        out_t = t_feat
        ctx_a = torch.stack([t_feat, v_feat], dim=1)
        out_a = self.audio_anchor(a_feat, ctx_a)
        ctx_v = torch.stack([t_feat, a_feat], dim=1)
        out_v = self.video_anchor(v_feat, ctx_v)
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        logits = (w_t * self.text_classifier(out_t) + w_a * self.audio_classifier(out_a)
                  + w_v * self.video_classifier(out_v))
        return logits, weights

    @torch.no_grad()
    def text_pathway_logits(self, t_feat):
        return self.text_classifier(t_feat)


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2: trainable-pure ablation.
# ═══════════════════════════════════════════════════════════════════════════

class InvertedAsymmetricFusionTrainable(nn.Module):
    def __init__(self, text_model, audio_model, video_model):
        super().__init__()
        self.text_backbone, self.audio_backbone, self.video_backbone = text_model, audio_model, video_model
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters(): p.requires_grad = False
        self.audio_anchor = CrossModalAnchor(FUSION_DIM)
        self.video_anchor = CrossModalAnchor(FUSION_DIM)
        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 3),
        )
        self.text_classifier = copy.deepcopy(get_text_classifier(text_model))
        for p in self.text_classifier.parameters(): p.requires_grad = True
        self.audio_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)
        self.video_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, t_feat, a_feat, v_feat):
        out_t = t_feat
        ctx_a = torch.stack([t_feat, v_feat], dim=1)
        out_a = self.audio_anchor(a_feat, ctx_a)
        ctx_v = torch.stack([t_feat, a_feat], dim=1)
        out_v = self.video_anchor(v_feat, ctx_v)
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        logits = (w_t * self.text_classifier(out_t) + w_a * self.audio_classifier(out_a)
                  + w_v * self.video_classifier(out_v))
        return logits, weights

    @torch.no_grad()
    def text_pathway_logits(self, t_feat):
        return self.text_classifier(t_feat)


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3 + 4
# ═══════════════════════════════════════════════════════════════════════════

class SymmetricTriModalFusion(nn.Module):
    def __init__(self, text_model, audio_model, video_model):
        super().__init__()
        self.text_backbone, self.audio_backbone, self.video_backbone = text_model, audio_model, video_model
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters(): p.requires_grad = False
        self.text_anchor  = CrossModalAnchor(FUSION_DIM)
        self.audio_anchor = CrossModalAnchor(FUSION_DIM)
        self.video_anchor = CrossModalAnchor(FUSION_DIM)
        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 3),
        )
        self.classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, t_feat, a_feat, v_feat):
        ctx_t = torch.stack([a_feat, v_feat], dim=1)
        ctx_a = torch.stack([t_feat, v_feat], dim=1)
        ctx_v = torch.stack([t_feat, a_feat], dim=1)
        out_t = self.text_anchor(t_feat, ctx_t)
        out_a = self.audio_anchor(a_feat, ctx_a)
        out_v = self.video_anchor(v_feat, ctx_v)
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        fused = w_t * out_t + w_a * out_a + w_v * out_v
        logits = self.classifier(fused)
        return logits, weights, out_t   # out_t exposed for Cell 3 / Cell 4 pathway eval

    def attended_text_feature(self, t_feat, a_feat, v_feat):
        ctx_t = torch.stack([a_feat, v_feat], dim=1)
        return self.text_anchor(t_feat, ctx_t)


# ==========================================================================
# DATA
def load_fold_split(fold_num):
    train_df = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_train.csv')
    test_df  = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_test.csv')
    return train_df['file_name'].values, test_df['file_name'].values


def load_features(feature_file, file_names, feat_prefix):
    df = pd.read_csv(f'{FEATURES_DIR}/{feature_file}')
    df = df[df['file_name'].isin(file_names)].copy()
    df = df.sort_values('file_name').reset_index(drop=True)
    feat_cols = sorted([c for c in df.columns if c.startswith(feat_prefix)])
    return (df[feat_cols].values.astype(np.float32), df[LABEL_COLUMN].values, df['file_name'].values)


class FusionDataset(Dataset):
    def __init__(self, t, a, v, labels, fnames):
        to_t = lambda x: x.detach().clone().float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        self.t, self.a, self.v = to_t(t), to_t(a), to_t(v)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.fnames = fnames
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.t[idx].clone(), self.a[idx].clone(), self.v[idx].clone(), self.labels[idx].clone(), self.fnames[idx]


@torch.no_grad()
def extract_backbone_features(txt, aud, vid, X_t, X_a, X_v, batch_size=64):
    txt.eval(); aud.eval(); vid.eval()
    n = len(X_t)
    t_out, a_out, v_out = [], [], []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        if (e - s) == 1 and e < n:
            e = min(s + 2, n)
        t_out.append(txt.get_features(torch.tensor(X_t[s:e], dtype=torch.float32).to(device)).cpu())
        a_out.append(aud.get_features(torch.tensor(X_a[s:e], dtype=torch.float32).to(device)).cpu())
        v_out.append(vid.get_features(torch.tensor(X_v[s:e], dtype=torch.float32).to(device)).cpu())
    return torch.cat(t_out), torch.cat(a_out), torch.cat(v_out)


@torch.no_grad()
def evaluate_text_pathway(model, loader):
    model.eval()
    preds, trues = [], []
    for t_b, a_b, v_b, y_b, _ in loader:
        t_b = t_b.to(device)
        logits = model.text_pathway_logits(t_b)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y_b.numpy())
    return accuracy_score(trues, preds) * 100


@torch.no_grad()
def evaluate_cell3_frozen_head(sym_model, frozen_head, loader):
    sym_model.eval(); frozen_head.eval()
    preds, trues = [], []
    for t_b, a_b, v_b, y_b, _ in loader:
        t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
        out_t = sym_model.attended_text_feature(t_b, a_b, v_b)
        logits = frozen_head(out_t)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y_b.numpy())
    return accuracy_score(trues, preds) * 100



# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 (BASELINE ONLY) — TRAINING UTILITIES. 
# ═══════════════════════════════════════════════════════════════════════════
def mixup_data(t, a, v, y, alpha=CELL2_MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(t.size(0)).to(t.device)
    return (lam * t + (1 - lam) * t[idx], lam * a + (1 - lam) * a[idx],
            lam * v + (1 - lam) * v[idx], y, y[idx], lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def apply_modality_dropout(t, a, v, epoch):
    if epoch < CELL2_WARMUP_EPOCHS:
        return t, a, v
    B = t.size(0)
    mask_t = torch.ones(B, 1, device=t.device)
    mask_a = torch.ones(B, 1, device=a.device)
    mask_v = torch.ones(B, 1, device=v.device)
    probs = np.random.rand(B)
    for i in range(B):
        p = probs[i]
        if   p < 0.40: pass
        elif p < 0.50: mask_v[i] = 0
        elif p < 0.60: mask_a[i] = 0
        elif p < 0.70: mask_t[i] = 0
        elif p < 0.80: mask_a[i] = mask_v[i] = 0
        elif p < 0.90: mask_t[i] = mask_v[i] = 0
        else:          mask_t[i] = mask_a[i] = 0
    return t * mask_t, a * mask_a, v * mask_v


def train_one_epoch_cell2(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for t_b, a_b, v_b, y_b, _ in loader:
        t_b, a_b, v_b, y_b = t_b.to(device), a_b.to(device), v_b.to(device), y_b.to(device)
        t_m, a_m, v_m, y_a, y_b_mix, lam = mixup_data(t_b, a_b, v_b, y_b)
        t_in, a_in, v_in = apply_modality_dropout(t_m, a_m, v_m, epoch)
        optimizer.zero_grad()
        logits, _ = model(t_in, a_in, v_in)
        loss = mixup_criterion(criterion, logits, y_a, y_b_mix, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == y_b).sum().item()
        total += y_b.size(0)
    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def evaluate_cell2(model, loader):
    model.eval()
    all_preds, all_trues = [], []
    for t_b, a_b, v_b, y_b, _ in loader:
        t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
        logits, _ = model(t_b, a_b, v_b)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_trues.extend(y_b.numpy())
    return accuracy_score(all_trues, all_preds) * 100


def train_cell2_baseline_ablation(txt_model, aud_model, vid_model, train_loader, test_loader, ckpt_path):
    model = InvertedAsymmetricFusionTrainable(txt_model, aud_model, vid_model).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=CELL2_LR, weight_decay=CELL2_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CELL2_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc, trigger = 0.0, 0
    for epoch in range(CELL2_EPOCHS):
        train_loss, train_acc = train_one_epoch_cell2(model, train_loader, optimizer, criterion, epoch)
        test_acc = evaluate_cell2(model, test_loader)
        scheduler.step()
        if test_acc > best_acc:
            best_acc, trigger = test_acc, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            trigger += 1
        if (epoch + 1) % 10 == 0:
            print(f"    [Cell2 ablation] Epoch {epoch+1}: Train={train_acc:.2f}%  "
                  f"Test={test_acc:.2f}% (Best={best_acc:.2f}%)")
        if trigger >= CELL2_PATIENCE:
            print(f"    [Cell2 ablation] Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, best_acc


@torch.no_grad()
def evaluate_cell4_shared_head(sym_model, loader):
    sym_model.eval()
    preds, trues = [], []
    for t_b, a_b, v_b, y_b, _ in loader:
        t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
        out_t = sym_model.attended_text_feature(t_b, a_b, v_b)
        logits = sym_model.classifier(out_t)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y_b.numpy())
    return accuracy_score(trues, preds) * 100


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT NAME HELPERS (differ by MODEL_TYPE)
# ═══════════════════════════════════════════════════════════════════════════

def cell1_ckpt_path(combo, fold):
    tag = 'distilled' if MODEL_TYPE == 'distilled' else 'baseline'
    return f"{CHECKPOINTS_DIR}/inverted_asymmetric_{tag}_{combo['name']}_fold_{fold}.pth"

def cell2_ckpt_path(combo, fold):
    if MODEL_TYPE == 'distilled':
        return f"{CHECKPOINTS_DIR}/iaf_trainable_pure_ablation_{combo['name']}_fold_{fold}.pth"
    return f"{CHECKPOINTS_DIR}/iaf_trainable_pure_ablation_baseline_{combo['name']}_fold_{fold}.pth"

def cell34_ckpt_path(combo, fold):
    tag = 'distilled' if MODEL_TYPE == 'distilled' else 'baseline'
    return f"{CHECKPOINTS_DIR}/symmetric_{tag}_{combo['name']}_fold_{fold}.pth"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN

all_grid = {}

for combo in COMBINATIONS:
    ceiling_str = f"{combo['unimodal_ceiling']:.1f}%" if combo['unimodal_ceiling'] is not None else "UNKNOWN — fill in"
    print(f"\n{'='*80}")
    print(f"COMBINATION: {combo['name'].upper()}  (dominant=text, ceiling={ceiling_str})")
    print(f"{'='*80}")

    fold_cells = {'cell1_no_attn_frozen_head': [], 'cell2_no_attn_trainable_head': [],
                  'cell3_attn_frozen_head': [], 'cell4_attn_trainable_head': []}

    for fold in range(1, 6):
        train_files, test_files = load_fold_split(fold)

        X_text_test,  y_test,  test_fn = load_features(combo['text_features'], test_files, 'text_feat_')
        X_audio_test, _, _ = load_features('dasheng_audio_embeddings.csv', test_files, 'audio_feat_')
        X_video_test, _, _ = load_features(combo['video_features'], test_files, 'visual_feat_')

        need_cell2_training = (MODEL_TYPE != 'distilled'
                                and not os.path.exists(cell2_ckpt_path(combo, fold)))

        if need_cell2_training:
            X_text_train,  y_train, train_fn = load_features(combo['text_features'], train_files, 'text_feat_')
            X_audio_train, _, _ = load_features('dasheng_audio_embeddings.csv', train_files, 'audio_feat_')
            X_video_train, _, _ = load_features(combo['video_features'], train_files, 'visual_feat_')
            le = LabelEncoder()
            le.fit(np.concatenate([y_train, y_test]))
            y_test_enc  = le.transform(y_test)
            y_train_enc = le.transform(y_train)
        else:
            le = LabelEncoder()
            y_test_enc = le.fit_transform(y_test)

        txt_model = load_text_model(combo, fold)
        aud_model = load_audio_model(combo, fold)
        vid_model = load_video_model(combo, fold)

        t_test_f, a_test_f, v_test_f = extract_backbone_features(txt_model, aud_model, vid_model, X_text_test, X_audio_test, X_video_test)
        test_loader = DataLoader(FusionDataset(t_test_f, a_test_f, v_test_f, y_test_enc, test_fn), batch_size=BATCH_SIZE, shuffle=False)

        if need_cell2_training:
            t_train_f, a_train_f, v_train_f = extract_backbone_features(
                txt_model, aud_model, vid_model, X_text_train, X_audio_train, X_video_train)
            train_loader = DataLoader(FusionDataset(t_train_f, a_train_f, v_train_f, y_train_enc, train_fn),
                                       batch_size=BATCH_SIZE, shuffle=True)

        # ── Cell 1 ──
        try:
            iaf_frozen = InvertedAsymmetricFusionFrozen(txt_model, aud_model, vid_model).to(device)
            iaf_frozen.load_state_dict(torch.load(cell1_ckpt_path(combo, fold), map_location=device))
            fold_cells['cell1_no_attn_frozen_head'].append(evaluate_text_pathway(iaf_frozen, test_loader))
        except FileNotFoundError:
            print(f"  [fold {fold}] Cell 1 checkpoint missing: {cell1_ckpt_path(combo, fold)}")
            fold_cells['cell1_no_attn_frozen_head'].append(np.nan)

        # ── Cell 2 ──
        if MODEL_TYPE == 'distilled':
            try:
                iaf_trainable = InvertedAsymmetricFusionTrainable(txt_model, aud_model, vid_model).to(device)
                iaf_trainable.load_state_dict(torch.load(cell2_ckpt_path(combo, fold), map_location=device))
                fold_cells['cell2_no_attn_trainable_head'].append(evaluate_text_pathway(iaf_trainable, test_loader))
            except FileNotFoundError:
                print(f"  [fold {fold}] Cell 2 checkpoint missing: {cell2_ckpt_path(combo, fold)}")
                fold_cells['cell2_no_attn_trainable_head'].append(np.nan)
        else:
            ckpt_path = cell2_ckpt_path(combo, fold)
            if need_cell2_training:
                print(f"  [fold {fold}] Cell 2: training baseline trainable-pure-head ablation "
                      f"(checkpoint not found: {ckpt_path})...")
                iaf_trainable, best_acc = train_cell2_baseline_ablation(
                    txt_model, aud_model, vid_model, train_loader, test_loader, ckpt_path)
                print(f"  [fold {fold}] Cell 2 ablation trained — best test acc during training: {best_acc:.2f}%")
            else:
                iaf_trainable = InvertedAsymmetricFusionTrainable(txt_model, aud_model, vid_model).to(device)
                iaf_trainable.load_state_dict(torch.load(ckpt_path, map_location=device))
            fold_cells['cell2_no_attn_trainable_head'].append(evaluate_text_pathway(iaf_trainable, test_loader))

        # ── Cell 3 & 4 ──
        try:
            sym_model = SymmetricTriModalFusion(txt_model, aud_model, vid_model).to(device)
            sym_model.load_state_dict(torch.load(cell34_ckpt_path(combo, fold), map_location=device))

            frozen_head = copy.deepcopy(get_text_classifier(txt_model)).to(device)
            for p in frozen_head.parameters(): p.requires_grad = False

            fold_cells['cell3_attn_frozen_head'].append(evaluate_cell3_frozen_head(sym_model, frozen_head, test_loader))
            fold_cells['cell4_attn_trainable_head'].append(evaluate_cell4_shared_head(sym_model, test_loader))
        except FileNotFoundError:
            print(f"  [fold {fold}] Cell 3/4 checkpoint missing: {cell34_ckpt_path(combo, fold)}")
            fold_cells['cell3_attn_frozen_head'].append(np.nan)
            fold_cells['cell4_attn_trainable_head'].append(np.nan)

        print(f"  Fold {fold}: Cell1={fold_cells['cell1_no_attn_frozen_head'][-1]:.2f}%  "
              f"Cell2={fold_cells['cell2_no_attn_trainable_head'][-1]:.2f}%  "
              f"Cell3={fold_cells['cell3_attn_frozen_head'][-1]:.2f}%  "
              f"Cell4={fold_cells['cell4_attn_trainable_head'][-1]:.2f}%")

    cell = {k: float(np.nanmean(v)) for k, v in fold_cells.items()}
    cell_std = {f"{k}_std": float(np.nanstd(v)) for k, v in fold_cells.items()}
    ceiling = combo['unimodal_ceiling']

    print(f"\n{'-'*80}")
    print(f"2x2 GRID (mean over 5 folds) — {combo['name']}  (ceiling={ceiling_str})")
    print(f"{'-'*80}")
    if ceiling is not None:
        print(f"                    Head frozen              Head trainable")
        print(f"No cross-attn       {cell['cell1_no_attn_frozen_head']:.2f}% ({ceiling-cell['cell1_no_attn_frozen_head']:+.2f}pp)      {cell['cell2_no_attn_trainable_head']:.2f}% ({ceiling-cell['cell2_no_attn_trainable_head']:+.2f}pp)")
        print(f"Cross-attn          {cell['cell3_attn_frozen_head']:.2f}% ({ceiling-cell['cell3_attn_frozen_head']:+.2f}pp)      {cell['cell4_attn_trainable_head']:.2f}% ({ceiling-cell['cell4_attn_trainable_head']:+.2f}pp)")
    else:
        print(f"                    Head frozen              Head trainable")
        print(f"No cross-attn       {cell['cell1_no_attn_frozen_head']:.2f}%                  {cell['cell2_no_attn_trainable_head']:.2f}%")
        print(f"Cross-attn          {cell['cell3_attn_frozen_head']:.2f}%                  {cell['cell4_attn_trainable_head']:.2f}%")
        print(f"  (ceiling unknown for baseline — pp-drop columns omitted; fill in unimodal_ceiling to get them)")

    cell.update(cell_std)
    cell['unimodal_ceiling'] = ceiling
    cell['dominant'] = 'text'
    all_grid[combo['name']] = cell

print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")
for name, c in all_grid.items():
    if np.isnan(c['cell2_no_attn_trainable_head']):
        print(f"\n{name}: Cell 2 unavailable, skipping interpretation for this combination.")
        continue
    attn_effect  = c['cell1_no_attn_frozen_head'] - c['cell3_attn_frozen_head']
    drift_effect = c['cell1_no_attn_frozen_head'] - c['cell2_no_attn_trainable_head']
    combined      = c['cell1_no_attn_frozen_head'] - c['cell4_attn_trainable_head']
    print(f"\n{name}:")
    print(f"  Attention-alone effect  (Cell1 - Cell3): {attn_effect:+.2f}pp")
    print(f"  Classifier-drift effect (Cell1 - Cell2): {drift_effect:+.2f}pp")
    print(f"  Combined effect         (Cell1 - Cell4): {combined:+.2f}pp")
    additive = attn_effect + drift_effect
    if abs(combined) > abs(additive) * 1.5:
        print(f"  -> Interaction effect: combined damage ({combined:+.2f}pp) far exceeds the sum "
              f"of individual effects ({additive:+.2f}pp)")
    elif attn_effect > drift_effect:
        print(f"  -> Cross-modal attention is the dominant collapse mechanism")
    else:
        print(f"  -> Classifier drift is the dominant collapse mechanism")

out_path = f'{OUTPUT_DIR}/causal_decomposition_multihuse_{MODEL_TYPE}.json'
with open(out_path, 'w') as f:
    json.dump(all_grid, f, indent=4, default=str)
print(f"\nSaved to {out_path}")
print(f"{'='*80}\n")
