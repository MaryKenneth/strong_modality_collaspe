"""
SYMMETRIC FUSION – MAKD (DISTILLED) STUDENTS (MUStARD)
========================================================
Uses distilled text models with symmetric fusion.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import json
import os
from pathlib import Path
import copy

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION – SET THIS FOR EACH RUN
# ══════════════════════════════════════════════════════════════════════════════
CASCADE_CONFIG = 'bert_videomae'   # ← CHANGE: 'bert_videomae', 'bert_pe_core',
                                   #           'e5_videomae', or 'e5_pe_core'

CONFIGS = {
    'bert_videomae': {
        'name': 'BERT (Audio+E5 Cascade) + Audio + VideoMAE',
        'text_type': 'bert_context_punchline',
        'text_checkpoint': 'mustard_cascade_bert_audio_e5',
        'text_dim': 1536,
        'video_type': 'videomae',
        'video_checkpoint': 'mustard_video_videomae',
        'video_dim': 768,
    },
    'bert_pe_core': {
        'name': 'BERT (Audio+E5 Cascade) + Audio + Video_PE',
        'text_type': 'bert_context_punchline',
        'text_checkpoint': 'mustard_cascade_bert_audio_e5',
        'text_dim': 1536,
        'video_type': 'pe_core',
        'video_checkpoint': 'mustard_video_pe_core',
        'video_dim': 1024,
    },
    'e5_videomae': {
        'name': 'E5 (Audio+Video Cascade) + Audio + VideoMAE',
        'text_type': 'e5_context_punchline',
        'text_checkpoint': 'mustard_cascade_e5_audio_video',
        'text_dim': 2048,
        'video_type': 'videomae',
        'video_checkpoint': 'mustard_video_videomae',
        'video_dim': 768,
    },
    'e5_pe_core': {
        'name': 'E5 (Audio+Video Cascade) + Audio + Video_PE',
        'text_type': 'e5_context_punchline',
        'text_checkpoint': 'mustard_cascade_e5_audio_video',
        'text_dim': 2048,
        'video_type': 'pe_core',
        'video_checkpoint': 'mustard_video_pe_core',
        'video_dim': 1024,
    },
}

if CASCADE_CONFIG not in CONFIGS:
    raise ValueError(f"Invalid CASCADE_CONFIG. Must be one of {list(CONFIGS.keys())}")

CONFIG = CONFIGS[CASCADE_CONFIG]
print(f"\n{'='*70}")
print(f"CASCADE SYMMETRIC FUSION: {CONFIG['name']}")
print(f"{'='*70}")
print(f"Text: {CONFIG['text_type']} (CASCADE distilled)")
print(f"Video: {CONFIG['video_type']}")
print(f"Audio: Dasheng (punchline)")
print(f"Architecture: ALL modalities attend (Text ✓  Audio ✓  Video ✓)")
print(f"Single shared classifier (no per-pathway hierarchy)")
print(f"{'='*70}\n")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
MUSTARD_DIR       = r'mustard_dataset'
FEATURES_DIR      = f'{MUSTARD_DIR}/mustard_features_pkl_FIXED'
SPLITS_FILE       = f'{MUSTARD_DIR}/mustard_dataset_master/data/split_indices.p'
SARCASM_DATA_PATH = f'{MUSTARD_DIR}/mustard_dataset_master/data/sarcasm_data.json'
CHECKPOINTS_DIR   = 'checkpoints'
OUTPUT_DIR        = f'predictions_output_fusion_symmetric_cascade_{CASCADE_CONFIG}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/context/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/context/not_sarcasm',
}
PUNCHLINE_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/utterances/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/utterances/not_sarcasm',
}

NUM_CLASSES   = 2
FUSION_DIM    = 512
AUDIO_DIM     = 1280

# Hyperparameters
BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 1e-3
WEIGHT_DECAY  = 1e-2
PATIENCE      = 35
MIXUP_ALPHA   = 0.05
WARMUP_EPOCHS = 15
SEED          = 999

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class AudioTeacher(nn.Module):
    """Dasheng audio teacher (78%)"""
    def __init__(self, input_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 11 and return_features:
                features = x.clone()
        return (x, features) if return_features else x


class VideoBaseline(nn.Module):
    """Video baseline (PE-Core or VideoMAE)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 6 and return_features:
                features = x.clone()
        return (x, features) if return_features else x


class SimplifiedTextStudent(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim > 1500:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(768, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits   = self.classifier(features)
        return (logits, features) if return_features else logits


# ─────────────────────────────────────────────────────────────────────────────
# SYMMETRIC FUSION ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAnchor(nn.Module):
    def __init__(self, d_model=FUSION_DIM):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, anchor, context):
        """anchor: (B,D)  context: (B,2,D)"""
        a_seq = anchor.unsqueeze(1)
        out, _ = self.cross_attn(query=a_seq, key=context, value=context)
        return self.norm(anchor + self.dropout(out.squeeze(1)))


class SymmetricFusion(nn.Module):
    def __init__(self, text_backbone, audio_backbone, video_backbone):
        super().__init__()

        # Frozen backbones
        self.text_backbone  = text_backbone
        self.audio_backbone = audio_backbone
        self.video_backbone = video_backbone
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters():
                p.requires_grad = False

        # Symmetric cross-modal anchors – one per modality
        self.text_anchor  = CrossModalAnchor(FUSION_DIM)   # Text  ← Audio+Video
        self.audio_anchor = CrossModalAnchor(FUSION_DIM)   # Audio ← Text+Video
        self.video_anchor = CrossModalAnchor(FUSION_DIM)   # Video ← Text+Audio

        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),
        )

        # Single shared classifier
        self.classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, text_feat, audio_feat, video_feat):
        """ All inputs are pre-extracted 512-dim features from frozen backbones."""
        # All three modalities attend to the other two
        out_text  = self.text_anchor(
            text_feat,  torch.stack([audio_feat, video_feat], dim=1))
        out_audio = self.audio_anchor(
            audio_feat, torch.stack([text_feat,  video_feat], dim=1))
        out_video = self.video_anchor(
            video_feat, torch.stack([text_feat,  audio_feat], dim=1))

        # Gate on raw (pre-attention) features
        gate_in = torch.cat([text_feat, audio_feat, video_feat], dim=1)
        weights  = torch.softmax(self.gate_net(gate_in), dim=1)
        w_text  = weights[:, 0].unsqueeze(1)
        w_audio = weights[:, 1].unsqueeze(1)
        w_video = weights[:, 2].unsqueeze(1)

        # Weighted sum → shared classifier
        fused  = w_text * out_text + w_audio * out_audio + w_video * out_video
        logits = self.classifier(fused)
        return logits, weights


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP + CURRICULUM MODALITY DROPOUT
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(t, a, v, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(t.size(0)).to(t.device)
    return (lam * t + (1 - lam) * t[idx],
            lam * a + (1 - lam) * a[idx],
            lam * v + (1 - lam) * v[idx],
            y, y[idx], lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def apply_modality_dropout(t, a, v, epoch):
    """
    Phase 1 (epoch < WARMUP_EPOCHS): all modalities present.
    Phase 2 (epoch >= WARMUP_EPOCHS): stochastic 7-case masking.
    """
    if epoch < WARMUP_EPOCHS:
        return t, a, v

    B = t.size(0)
    mask_t = torch.ones(B, 1, device=t.device)
    mask_a = torch.ones(B, 1, device=a.device)
    mask_v = torch.ones(B, 1, device=v.device)
    probs  = np.random.rand(B)

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


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except:
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_features_from_pkl(pkl_path, feature_type):
    feature_keys = {
        'bert': 'text_bert', 'e5': 'text_e5',
        'dasheng': 'audio_dasheng', 'pe_core': 'video_pe', 'videomae': 'video_mae',
    }
    fallback_dims = {
        'bert': 768, 'e5': 1024, 'dasheng': 1280, 'pe_core': 1024, 'videomae': 768,
    }
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        feat = data.get(feature_keys[feature_type],
                        np.zeros((1, fallback_dims[feature_type])))
        return feat[0] if len(feat.shape) > 1 else feat
    except:
        return np.zeros(fallback_dims[feature_type])


def prepare_features(fold_indices, text_type, video_type, sarcasm_data, identifiers):
    X_text, X_audio, X_video, y_list, file_names = [], [], [], [], []
    text_feature_type = 'bert' if 'bert' in text_type else 'e5'

    for split_idx in fold_indices:
        if split_idx >= len(identifiers):
            continue
        identifier = identifiers[split_idx]
        label      = 1 if sarcasm_data[identifier]['sarcasm'] else 0
        label_str  = 'sarcasm' if label == 1 else 'not_sarcasm'

        context_path   = Path(CONTEXT_PATHS[label_str])  / f"{identifier}.pkl"
        punchline_path = Path(PUNCHLINE_PATHS[label_str]) / f"{identifier}.pkl"

        text_feat  = np.concatenate([
            load_features_from_pkl(context_path,   text_feature_type),
            load_features_from_pkl(punchline_path, text_feature_type),
        ])
        audio_feat = load_features_from_pkl(punchline_path, 'dasheng')
        video_feat = load_features_from_pkl(punchline_path, video_type)

        X_text.append(text_feat)
        X_audio.append(audio_feat)
        X_video.append(video_feat)
        y_list.append(label)
        file_names.append(identifier)

    return (np.array(X_text,  dtype=np.float32),
            np.array(X_audio, dtype=np.float32),
            np.array(X_video, dtype=np.float32),
            np.array(y_list), file_names)


class FusionDataset(Dataset):
    def __init__(self, text_data, audio_data, video_data, labels, file_names):
        if isinstance(text_data, torch.Tensor):
            self.text, self.audio, self.video = text_data, audio_data, video_data
        else:
            self.text  = torch.tensor(text_data,  dtype=torch.float32)
            self.audio = torch.tensor(audio_data, dtype=torch.float32)
            self.video = torch.tensor(video_data, dtype=torch.float32)
        self.labels = (labels if isinstance(labels, torch.Tensor)
                       else torch.tensor(labels, dtype=torch.long))
        self.file_names = file_names

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (self.text[idx].clone(), self.audio[idx].clone(),
                self.video[idx].clone(), self.labels[idx].clone(),
                self.file_names[idx])


@torch.no_grad()
def extract_backbone_features(text_model, audio_model, video_model,
                               text_data, audio_data, video_data, batch_size=64):
    """Pre-extract 512-dim features from all frozen backbones once."""
    text_model.eval(); audio_model.eval(); video_model.eval()
    n = len(text_data)
    text_feats, audio_feats, video_feats = [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end - start == 1 and start + batch_size < n:
            end = min(start + 2, n)

        t_b = torch.tensor(text_data[start:end],  dtype=torch.float32).to(device)
        a_b = torch.tensor(audio_data[start:end], dtype=torch.float32).to(device)
        v_b = torch.tensor(video_data[start:end], dtype=torch.float32).to(device)

        _, t_feat = text_model(t_b,  return_features=True)
        _, a_feat = audio_model(a_b, return_features=True)
        _, v_feat = video_model(v_b, return_features=True)

        text_feats.append(t_feat.cpu())
        audio_feats.append(a_feat.cpu())
        video_feats.append(v_feat.cpu())

    return (torch.cat(text_feats,  dim=0),
            torch.cat(audio_feats, dim=0),
            torch.cat(video_feats, dim=0))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for t_b, a_b, v_b, y_b, _ in loader:
        t_b, a_b, v_b, y_b = (t_b.to(device), a_b.to(device),
                               v_b.to(device), y_b.to(device))
        t_m, a_m, v_m, y_a, y_b_mix, lam = mixup_data(t_b, a_b, v_b, y_b)
        t_in, a_in, v_in = apply_modality_dropout(t_m, a_m, v_m, epoch)

        optimizer.zero_grad()
        logits, _ = model(t_in, a_in, v_in)
        loss = mixup_criterion(criterion, logits, y_a, y_b_mix, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(dim=1) == y_b).sum().item()
        total      += y_b.size(0)

    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_weights, all_fnames = [], [], [], []

    for t_b, a_b, v_b, y_b, fnames in loader:
        t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
        logits, weights = model(t_b, a_b, v_b)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_b.numpy())
        all_weights.append(weights.cpu().numpy())
        all_fnames.extend(fnames)

    acc   = accuracy_score(all_labels, all_preds) * 100
    avg_w = np.concatenate(all_weights, axis=0).mean(axis=0)
    return acc, all_preds, all_labels, avg_w, all_fnames


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

splits_data = load_pickle(SPLITS_FILE)
with open(SARCASM_DATA_PATH, 'r') as f:
    sarcasm_data = json.load(f)
identifiers = list(sarcasm_data.keys())
print(f"Total samples: {len(identifiers)}\n")

fold_accs = []
fold_weights = []       # collect gate weights per fold for analysis
all_preds, all_trues = [], []
all_pred_rows = []      # per-sample predictions across all folds, for saving
fold_summary_rows = []  # per-fold accuracy + gate weights, for saving

for fold in range(5):
    print(f"{'─'*70}")
    print(f"FOLD {fold}")
    print(f"{'─'*70}")

    train_indices = splits_data[fold][0]
    test_indices  = splits_data[fold][1]

    X_text_train, X_audio_train, X_video_train, y_train, train_fnames = prepare_features(
        train_indices, CONFIG['text_type'], CONFIG['video_type'], sarcasm_data, identifiers)
    X_text_test, X_audio_test, X_video_test, y_test, test_fnames = prepare_features(
        test_indices, CONFIG['text_type'], CONFIG['video_type'], sarcasm_data, identifiers)

    # Load frozen backbones
    text_student = SimplifiedTextStudent(CONFIG['text_dim']).to(device)
    text_student.load_state_dict(torch.load(
        f"{CHECKPOINTS_DIR}/{CONFIG['text_checkpoint']}_fold_{fold}.pth",
        map_location=device))

    audio_baseline = AudioTeacher(AUDIO_DIM).to(device)
    audio_baseline.load_state_dict(torch.load(
        f"{CHECKPOINTS_DIR}/mustard_audio_dasheng_fold_{fold}.pth",
        map_location=device))

    video_baseline = VideoBaseline(CONFIG['video_dim']).to(device)
    video_baseline.load_state_dict(torch.load(
        f"{CHECKPOINTS_DIR}/{CONFIG['video_checkpoint']}_fold_{fold}.pth",
        map_location=device))

    # Pre-extract features once
    print("Pre-extracting features from frozen backbones...")
    text_train_feats, audio_train_feats, video_train_feats = extract_backbone_features(
        text_student, audio_baseline, video_baseline,
        X_text_train, X_audio_train, X_video_train)
    text_test_feats, audio_test_feats, video_test_feats = extract_backbone_features(
        text_student, audio_baseline, video_baseline,
        X_text_test, X_audio_test, X_video_test)

    train_ds = FusionDataset(text_train_feats, audio_train_feats, video_train_feats,
                              y_train, train_fnames)
    test_ds  = FusionDataset(text_test_feats,  audio_test_feats,  video_test_feats,
                              y_test,  test_fnames)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = SymmetricFusion(text_student, audio_baseline, video_baseline).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc, trigger_times = 0.0, 0
    ckpt_path = f"{CHECKPOINTS_DIR}/fusion_symmetric_cascade_{CASCADE_CONFIG}_fold_{fold}.pth"

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch)
        test_acc, _, _, test_w, _ = evaluate(model, test_loader)
        scheduler.step()

        if test_acc > best_acc:
            best_acc, trigger_times = test_acc, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            trigger_times += 1

        if (epoch + 1) % 20 == 0:
            phase = "warmup" if epoch < WARMUP_EPOCHS else "dropout"
            print(f"  Epoch {epoch+1} [{phase}]: Train={train_acc:.2f}% "
                  f"Test={test_acc:.2f}% (Best={best_acc:.2f}%) "
                  f"W[T={test_w[0]:.3f} A={test_w[1]:.3f} V={test_w[2]:.3f}]")

        if trigger_times >= PATIENCE:
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    acc, preds, trues, weights, fnames = evaluate(model, test_loader)

    fold_accs.append(acc)
    all_preds.extend(preds)
    all_trues.extend(trues)
    fold_weights.append(weights)
    all_pred_rows.extend([
        {'fold': fold, 'identifier': fn, 'true_label': t, 'pred_label': p}
        for fn, t, p in zip(fnames, trues, preds)
    ])
    fold_summary_rows.append({
        'fold': fold, 'accuracy': acc,
        'gate_weight_text': float(weights[0]),
        'gate_weight_audio': float(weights[1]),
        'gate_weight_video': float(weights[2]),
    })
    print(f"Fold {fold} Final: {acc:.2f}%\n")

print(f"\n{'='*70}")
print(f"FINAL RESULTS – {CONFIG['name']}")
print(f"{'='*70}")
print(f"Average Accuracy: {np.mean(fold_accs):.2f}% (±{np.std(fold_accs):.2f}%)")
print(f"Per-fold: {[f'{a:.2f}%' for a in fold_accs]}")
print(f"\nConfusion Matrix:\n{confusion_matrix(all_trues, all_preds)}")
print(f"\nClassification Report:")
print(classification_report(all_trues, all_preds,
                             target_names=['not_sarcasm', 'sarcasm']))

avg_w = np.mean(fold_weights, axis=0)
print(f"\nAvg gate weights across folds:")
print(f"  T={avg_w[0]:.3f}  A={avg_w[1]:.3f}  V={avg_w[2]:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
pd.DataFrame(all_pred_rows).to_csv(
    f"{OUTPUT_DIR}/predictions_{CASCADE_CONFIG}.csv", index=False)

pd.DataFrame(fold_summary_rows).to_csv(
    f"{OUTPUT_DIR}/fold_summary_{CASCADE_CONFIG}.csv", index=False)

report_dict = classification_report(
    all_trues, all_preds, target_names=['not_sarcasm', 'sarcasm'],
    output_dict=True, zero_division=0)
pd.DataFrame(report_dict).transpose().to_csv(
    f"{OUTPUT_DIR}/classification_report_{CASCADE_CONFIG}.csv")

summary = {
    'config': CASCADE_CONFIG,
    'config_name': CONFIG['name'],
    'architecture': 'symmetric',
    'variant': 'cascade',
    'fold_accuracies': fold_accs,
    'mean_accuracy': float(np.mean(fold_accs)),
    'std_accuracy': float(np.std(fold_accs)),
    'confusion_matrix': confusion_matrix(all_trues, all_preds).tolist(),
    'classification_report': report_dict,
    'avg_gate_weights': {
        'text': float(avg_w[0]), 'audio': float(avg_w[1]), 'video': float(avg_w[2]),
    },
    'per_fold_gate_weights': fold_summary_rows,
}
with open(f"{OUTPUT_DIR}/summary_{CASCADE_CONFIG}.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved predictions, fold summary, classification report, and JSON "
      f"summary to {OUTPUT_DIR}/")
