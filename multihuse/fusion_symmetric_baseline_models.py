"""
SYMMETRIC FUSION WITH BASELINE MODELS - 4 COMBINATIONS (MultiHUSE)
===================================================================
Symmetric variant: ALL modalities attend to each other, features are
fused via a weighted sum and passed through a SINGLE shared classifier.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
SPLITS_DIR      = r'multihuse_dataset\splits_strict_v2'
FEATURES_DIR    = r'multihuse_dataset'
CHECKPOINTS_DIR = 'checkpoints'
OUTPUT_DIR      = 'predictions_output_symmetric_baseline'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMBINATIONS = [
    {
        'name':             'bert_audio_videomae',
        'text_features':    'bert_text_embeddings.csv',
        'text_checkpoint':  'text_bert_text_embeddings',
        'text_dim':         768,
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_features':   'videomae_visual_embeddings.csv',
        'video_checkpoint': 'video_videomae_visual_embeddings',
        'video_dim':        768,
    },
    {
        'name':             'bert_audio_pecore',
        'text_features':    'bert_text_embeddings.csv',
        'text_checkpoint':  'text_bert_text_embeddings',
        'text_dim':         768,
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_features':   'pe_core_visual_embeddings.csv',
        'video_checkpoint': 'video_pe_core_visual_embeddings',
        'video_dim':        1024,
    },
    {
        'name':             'e5_audio_videomae',
        'text_features':    'e5_text_embeddings.csv',
        'text_checkpoint':  'text_e5_text_embeddings',
        'text_dim':         1024,
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_features':   'videomae_visual_embeddings.csv',
        'video_checkpoint': 'video_videomae_visual_embeddings',
        'video_dim':        768,
    },
    {
        'name':             'e5_audio_pecore',
        'text_features':    'e5_text_embeddings.csv',
        'text_checkpoint':  'text_e5_text_embeddings',
        'text_dim':         1024,
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_features':   'pe_core_visual_embeddings.csv',
        'video_checkpoint': 'video_pe_core_visual_embeddings',
        'video_dim':        1024,
    },
]

AUDIO_DIM    = 1280
NUM_CLASSES  = 5
LABEL_COLUMN = 'humor_style'
FUSION_DIM   = 512

# Hyperparameters 
BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 5e-5
WEIGHT_DECAY  = 1e-3
PATIENCE      = 35
MIXUP_ALPHA   = 0.05
WARMUP_EPOCHS = 15
SEED          = 999

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print(f"\n{'='*70}")
print(f"SYMMETRIC FUSION — BASELINE MODELS (MultiHUSE)")
print(f"{'='*70}")


# ==========================================================================
# BASELINE MODEL DEFINITIONS  
class TextModel(nn.Module):
    """Baseline text model (BERT 768 or E5 1024 → 512 → NUM_CLASSES)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),   # [0]
            nn.BatchNorm1d(512),         # [1]
            nn.ReLU(),                   # [2]
            nn.Dropout(0.3),             # [3]
            nn.Linear(512, NUM_CLASSES)  # [4]
        )

    def forward(self, x):
        return self.net(x)

    def get_features(self, x):
        return self.net[:4](x)   # 512-dim


class BaselineAudioModel(nn.Module):
    """Baseline Dasheng audio model (1280 → 1024 → 768 → 512 encoder)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUDIO_DIM, 1024),  # [0]
            nn.BatchNorm1d(1024),        # [1]
            nn.ReLU(),                   # [2]
            nn.Dropout(0.4),             # [3]
            nn.Linear(1024, 768),        # [4]
            nn.BatchNorm1d(768),         # [5]
            nn.ReLU(),                   # [6]
            nn.Dropout(0.3),             # [7]
            nn.Linear(768, 512),         # [8]
            nn.BatchNorm1d(512),         # [9]
            nn.ReLU(),                   # [10]
            nn.Dropout(0.2),             # [11]  ← 512-dim feature point
            nn.Linear(512, 256),         # [12]
            nn.ReLU(),                   # [13]
            nn.Dropout(0.2),             # [14]
            nn.Linear(256, NUM_CLASSES)  # [15]
        )

    def forward(self, x):
        return self.net(x)

    def get_features(self, x):
        return self.net[:12](x)   # 512-dim


class BaselineVideoModel(nn.Module):
    """Baseline video model (VideoMAE 768 or PE-Core 1024 → 1024 → 512 encoder)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),  # [0]
            nn.BatchNorm1d(1024),        # [1]
            nn.ReLU(),                   # [2]
            nn.Dropout(0.5),             # [3]
            nn.Linear(1024, 512),        # [4]
            nn.ReLU(),                   # [5]
            nn.Dropout(0.3),             # [6]  ← 512-dim feature point
            nn.Linear(512, NUM_CLASSES)  # [7]
        )

    def forward(self, x):
        return self.net(x)

    def get_features(self, x):
        return self.net[:7](x)   # 512-dim

# ==========================================================================
# SYMMETRIC FUSION ARCHITECTURE

class CrossModalAnchor(nn.Module):
    """Cross-attention block: anchor attends to two-modality context."""
    def __init__(self, d_model=FUSION_DIM):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, anchor, context):
        """anchor: (B, D)  context: (B, 2, D)"""
        a_seq = anchor.unsqueeze(1)
        out, _ = self.cross_attn(query=a_seq, key=context, value=context)
        return self.norm(anchor + self.dropout(out.squeeze(1)))


class SymmetricTriModalFusion(nn.Module):
    def __init__(self, text_model, audio_model, video_model):
        super().__init__()

        self.text_backbone  = text_model
        self.audio_backbone = audio_model
        self.video_backbone = video_model
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters():
                p.requires_grad = False

        # SYMMETRIC: all three modalities get a cross-modal anchor
        self.text_anchor  = CrossModalAnchor(FUSION_DIM)
        self.audio_anchor = CrossModalAnchor(FUSION_DIM)
        self.video_anchor = CrossModalAnchor(FUSION_DIM)

        # Gate network with BatchNorm
        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),
        )

        # SHARED single classifier
        self.classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, t_feat, a_feat, v_feat):
        """t_feat, a_feat, v_feat: pre-extracted 512-dim features (B, 512)."""

        # All modalities attend to each other
        ctx_t = torch.stack([a_feat, v_feat], dim=1)   # Text  context: Audio + Video
        ctx_a = torch.stack([t_feat, v_feat], dim=1)   # Audio context: Text  + Video
        ctx_v = torch.stack([t_feat, a_feat], dim=1)   # Video context: Text  + Audio

        out_t = self.text_anchor(t_feat,  ctx_t)
        out_a = self.audio_anchor(a_feat, ctx_a)
        out_v = self.video_anchor(v_feat, ctx_v)

        # Gating on raw pre-attention features
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights  = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t = weights[:, 0].unsqueeze(1)
        w_a = weights[:, 1].unsqueeze(1)
        w_v = weights[:, 2].unsqueeze(1)

        # Weighted fusion → shared classifier
        fused  = w_t * out_t + w_a * out_a + w_v * out_v
        logits = self.classifier(fused)

        return logits, weights


# ==========================================================================
# MIXUP + CURRICULUM MODALITY DROPOUT

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
    2-phase curriculum:
      Phase 1 (epoch < WARMUP_EPOCHS): all modalities present.
      Phase 2 (epoch >= WARMUP_EPOCHS): stochastic per-sample masking.
        40% — full trimodal
        10% each — one modality missing (3 cases)
        10% each — two modalities missing (3 cases)
    """
    if epoch < WARMUP_EPOCHS:
        return t, a, v

    B      = t.size(0)
    mask_t = torch.ones(B, 1, device=t.device)
    mask_a = torch.ones(B, 1, device=a.device)
    mask_v = torch.ones(B, 1, device=v.device)
    probs  = np.random.rand(B)

    for i in range(B):
        p = probs[i]
        if   p < 0.40: pass                         # 40% full trimodal
        elif p < 0.50: mask_v[i] = 0                # 10% Text + Audio
        elif p < 0.60: mask_a[i] = 0                # 10% Text + Video
        elif p < 0.70: mask_t[i] = 0                # 10% Audio + Video
        elif p < 0.80: mask_a[i] = mask_v[i] = 0    # 10% Text only
        elif p < 0.90: mask_t[i] = mask_v[i] = 0    # 10% Audio only
        else:          mask_t[i] = mask_a[i] = 0    # 10% Video only

    return t * mask_t, a * mask_a, v * mask_v


# ==========================================================================
# DATA LOADING

def load_fold_split(fold_num):
    train_df = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_train.csv')
    test_df  = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_test.csv')
    return train_df['file_name'].values, test_df['file_name'].values


def load_features(feature_file, file_names, feat_prefix):
    df = pd.read_csv(f'{FEATURES_DIR}/{feature_file}')
    df = df[df['file_name'].isin(file_names)].copy()
    df = df.sort_values('file_name').reset_index(drop=True)
    feat_cols = sorted([c for c in df.columns if c.startswith(feat_prefix)])
    return (df[feat_cols].values.astype(np.float32),
            df[LABEL_COLUMN].values,
            df['file_name'].values)


class FusionDataset(Dataset):
    def __init__(self, t, a, v, labels, fnames):
        to_t = lambda x: (x.detach().clone().float()
                          if isinstance(x, torch.Tensor)
                          else torch.tensor(x, dtype=torch.float32))
        self.t      = to_t(t)
        self.a      = to_t(a)
        self.v      = to_t(v)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.fnames = fnames

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (self.t[idx].clone(), self.a[idx].clone(),
                self.v[idx].clone(), self.labels[idx].clone(),
                self.fnames[idx])


@torch.no_grad()
def extract_backbone_features(text_model, audio_model, video_model,
                               X_text, X_audio, X_video, batch_size=64):
    """Pre-extract 512-dim features from all frozen backbones once."""
    text_model.eval(); audio_model.eval(); video_model.eval()
    n = len(X_text)
    t_feats, a_feats, v_feats = [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if (end - start) == 1 and end < n:
            end = min(start + 2, n)
        t_b = torch.tensor(X_text[start:end],  dtype=torch.float32).to(device)
        a_b = torch.tensor(X_audio[start:end], dtype=torch.float32).to(device)
        v_b = torch.tensor(X_video[start:end], dtype=torch.float32).to(device)
        t_feats.append(text_model.get_features(t_b).cpu())
        a_feats.append(audio_model.get_features(a_b).cpu())
        v_feats.append(video_model.get_features(v_b).cpu())

    return (torch.cat(t_feats, dim=0),
            torch.cat(a_feats, dim=0),
            torch.cat(v_feats, dim=0))


# ==========================================================================
# TRAINING & EVALUATION

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
        correct    += (logits.argmax(1) == y_b).sum().item()
        total      += y_b.size(0)

    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_weights = [], [], []

    for t_b, a_b, v_b, y_b, _ in loader:
        t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
        logits, weights = model(t_b, a_b, v_b)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y_b.numpy())
        all_weights.append(weights.cpu().numpy())

    acc   = accuracy_score(all_labels, all_preds) * 100
    avg_w = np.concatenate(all_weights, axis=0).mean(axis=0)
    return acc, all_preds, all_labels, avg_w


# ==========================================================================
# MAIN

all_results = []

for combo in COMBINATIONS:
    print(f"\n{'='*70}")
    print(f"COMBINATION: {combo['name'].upper()}")
    print(f"{'='*70}")
    print(f"Text:  {combo['text_features']} (dim: {combo['text_dim']})")
    print(f"Audio: dasheng_audio_embeddings.csv (dim: {AUDIO_DIM})")
    print(f"Video: {combo['video_features']} (dim: {combo['video_dim']})")
    print(f"{'='*70}\n")

    fold_accs = []
    all_preds, all_trues = [], []
    le = None

    for fold in range(1, 6):
        print(f"{'─'*70}")
        print(f"FOLD {fold}/5")
        print(f"{'─'*70}")

        train_files, test_files = load_fold_split(fold)

        X_text_train,  y_train, train_fnames = load_features(
            combo['text_features'], train_files, 'text_feat_')
        X_text_test,   y_test,  test_fnames  = load_features(
            combo['text_features'], test_files,  'text_feat_')

        X_audio_train, _, _ = load_features(
            'dasheng_audio_embeddings.csv', train_files, 'audio_feat_')
        X_audio_test,  _, _ = load_features(
            'dasheng_audio_embeddings.csv', test_files,  'audio_feat_')

        X_video_train, _, _ = load_features(
            combo['video_features'], train_files, 'visual_feat_')
        X_video_test,  _, _ = load_features(
            combo['video_features'], test_files,  'visual_feat_')

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)

        # Load frozen baseline backbones
        print("Loading baseline backbone models...")
        txt_model = TextModel(combo['text_dim']).to(device)
        txt_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{combo['text_checkpoint']}_fold_{fold}.pth",
            map_location=device))

        aud_model = BaselineAudioModel().to(device)
        aud_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{combo['audio_checkpoint']}_fold_{fold}.pth",
            map_location=device))

        vid_model = BaselineVideoModel(combo['video_dim']).to(device)
        vid_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{combo['video_checkpoint']}_fold_{fold}.pth",
            map_location=device))

        # Pre-extract 512-dim features once
        print("Pre-extracting features from frozen backbones...")
        t_train_f, a_train_f, v_train_f = extract_backbone_features(
            txt_model, aud_model, vid_model,
            X_text_train, X_audio_train, X_video_train)
        t_test_f, a_test_f, v_test_f = extract_backbone_features(
            txt_model, aud_model, vid_model,
            X_text_test, X_audio_test, X_video_test)

        train_ds = FusionDataset(t_train_f, a_train_f, v_train_f, y_train_enc, train_fnames)
        test_ds  = FusionDataset(t_test_f,  a_test_f,  v_test_f,  y_test_enc,  test_fnames)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

        model = SymmetricTriModalFusion(txt_model, aud_model, vid_model).to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p   = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total_p:,}")

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best_acc, trigger_times = 0.0, 0
        ckpt_path = f"{CHECKPOINTS_DIR}/symmetric_baseline_{combo['name']}_fold_{fold}.pth"

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, epoch)
            test_acc, _, _, test_w = evaluate(model, test_loader)
            scheduler.step()

            if test_acc > best_acc:
                best_acc, trigger_times = test_acc, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                trigger_times += 1

            if (epoch + 1) % 20 == 0:
                phase = "warmup" if epoch < WARMUP_EPOCHS else "dropout"
                print(f"  Epoch {epoch+1} [{phase}]: Train={train_acc:.2f}%  "
                      f"Test={test_acc:.2f}% (Best={best_acc:.2f}%)  "
                      f"W[T={test_w[0]:.3f} A={test_w[1]:.3f} V={test_w[2]:.3f}]")

            if trigger_times >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        acc, preds, trues, weights = evaluate(model, test_loader)

        fold_accs.append(acc)
        all_preds.extend(preds)
        all_trues.extend(trues)
        print(f"Fold {fold} Final: {acc:.2f}%  "
              f"W[T={weights[0]:.3f} A={weights[1]:.3f} V={weights[2]:.3f}]\n")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)

    print(f"\n{'='*70}")
    print(f"COMBINATION {combo['name'].upper()} — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Average Accuracy : {mean_acc:.2f}% (±{std_acc:.2f}%)")
    print(f"Per-fold         : {[f'{a:.2f}%' for a in fold_accs]}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(all_trues, all_preds)}")
    print(f"\nClassification Report:")
    print(classification_report(all_trues, all_preds, target_names=le.classes_))

    all_results.append({
        'combination': combo['name'],
        'mean_accuracy': mean_acc,
        'std_accuracy':  std_acc,
    })

print(f"\n{'='*70}")
print(f"SYMMETRIC BASELINE — ALL COMBINATIONS SUMMARY")
print(f"{'='*70}")
for r in all_results:
    print(f"  {r['combination']:30s}  {r['mean_accuracy']:.2f}% (±{r['std_accuracy']:.2f}%)")
print(f"{'='*70}")
