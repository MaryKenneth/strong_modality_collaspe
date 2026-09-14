"""
INVERTED ASYMMETRIC FUSION (IAf) WITH BASELINE MODELS - 4 COMBINATIONS (MultiHUSE)
=============================================================================
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
import copy

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
SPLITS_DIR      = r'multihuse_dataset\splits_strict_v2'
FEATURES_DIR    = r'multihuse_dataset'
CHECKPOINTS_DIR = 'checkpoints'
OUTPUT_DIR      = 'predictions_output_inverted_asymmetric_baseline'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMBINATIONS = [
    {
        'name': 'bert_audio_videomae',
        'text_features':    'bert_text_embeddings.csv',
        'text_checkpoint':  'text_bert_text_embeddings',
        'text_dim':         768,
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_features':   'videomae_visual_embeddings.csv',
        'video_checkpoint': 'video_videomae_visual_embeddings',
        'video_dim':        768,
    },
    {
        'name': 'bert_audio_pecore',
        'text_features':    'bert_text_embeddings.csv',
        'text_checkpoint':  'text_bert_text_embeddings',
        'text_dim':         768,
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_features':   'pe_core_visual_embeddings.csv',
        'video_checkpoint': 'video_pe_core_visual_embeddings',
        'video_dim':        1024,
    },
    {
        'name': 'e5_audio_videomae',
        'text_features':    'e5_text_embeddings.csv',
        'text_checkpoint':  'text_e5_text_embeddings',
        'text_dim':         1024,
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_features':   'videomae_visual_embeddings.csv',
        'video_checkpoint': 'video_videomae_visual_embeddings',
        'video_dim':        768,
    },
    {
        'name': 'e5_audio_pecore',
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"\n{'='*70}")
print(f"INVERTED ASYMMETRIC FUSION — BASELINE MODELS (MultiHUSE)")
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
    """
    Baseline Dasheng audio model.
    Encoder: 1280 → 1024 → 768 → 512 (layers 0–11)
    Classifier: 512 → 256 → NUM_CLASSES (layers 12–15)
    """
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
    """
    Baseline video model (VideoMAE 768 or PE-Core 1024).
    Encoder: input_dim → 1024 → 512 (layers 0–6)
    Classifier: 512 → NUM_CLASSES (layer 7)
    """
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
# INVERTED ASYMMETRIC FUSION ARCHITECTURE

class CrossModalAnchor(nn.Module):
    """Single cross-attention block — weak modality attends to two context modalities."""
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


class InvertedAsymmetricFusion(nn.Module):
    """
    MultiHUSE inverted asymmetric fusion:
      Text  (STRONG): NO attention — stays pure
      Audio (WEAK):   ATTENDS to Text+Video
      Video (WEAK):   ATTENDS to Text+Audio

    Frozen backbones; only fusion layers are trained.
    """
    def __init__(self, text_model, audio_model, video_model):
        super().__init__()

        self.text_backbone  = text_model
        self.audio_backbone = audio_model
        self.video_backbone = video_model

        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters():
                p.requires_grad = False

        # INVERTED: weak modalities have cross-modal anchors; text has none
        self.audio_anchor = CrossModalAnchor(FUSION_DIM)
        self.video_anchor = CrossModalAnchor(FUSION_DIM)

        # Gate network — with BN
        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),
        )

        # Text classifier: frozen copy from text backbone
        self.text_classifier = copy.deepcopy(text_model.net[4])
        for p in self.text_classifier.parameters():
            p.requires_grad = False

        # Audio & Video classifiers: new trainable heads
        self.audio_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)
        self.video_classifier  = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, t_feat, a_feat, v_feat):
        """Inputs are pre-extracted 512-dim features from frozen backbones."""
        # Text: PURE — no attention
        out_t = t_feat

        # Audio: attends to Text + Video
        ctx_a = torch.stack([t_feat, v_feat], dim=1)
        out_a = self.audio_anchor(a_feat, ctx_a)

        # Video: attends to Text + Audio
        ctx_v = torch.stack([t_feat, a_feat], dim=1)
        out_v = self.video_anchor(v_feat, ctx_v)

        # Dynamic gating on raw pre-attention features
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights = torch.softmax(self.gate_net(gate_in), dim=1)

        w_t = weights[:, 0].unsqueeze(1)
        w_a = weights[:, 1].unsqueeze(1)
        w_v = weights[:, 2].unsqueeze(1)

        logits_t = self.text_classifier(out_t)
        logits_a = self.audio_classifier(out_a)
        logits_v = self.video_classifier(out_v)

        final_logits = w_t * logits_t + w_a * logits_a + w_v * logits_v
        return final_logits, weights


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
    2-phase curriculum (identical schedule to MUStARD fusion script):
      Phase 1 (epoch < WARMUP_EPOCHS): all modalities present.
      Phase 2 (epoch >= WARMUP_EPOCHS): stochastic per-sample masking.
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
        if   p < 0.40: pass                          # 40% trimodal
        elif p < 0.50: mask_v[i] = 0                 # 10% Text + Audio
        elif p < 0.60: mask_a[i] = 0                 # 10% Text + Video
        elif p < 0.70: mask_t[i] = 0                 # 10% Audio + Video
        elif p < 0.80: mask_a[i] = mask_v[i] = 0     # 10% Text only
        elif p < 0.90: mask_t[i] = mask_v[i] = 0     # 10% Audio only
        else:          mask_t[i] = mask_a[i] = 0     # 10% Video only

    return t * mask_t, a * mask_a, v * mask_v


# ==========================================================================
# DATA LOADING

def load_fold_split(fold_num):
    train_df = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_train.csv')
    test_df  = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_test.csv')
    return train_df['file_name'].values, test_df['file_name'].values


def load_features(feature_file, file_names_subset, feat_prefix):
    df = pd.read_csv(f'{FEATURES_DIR}/{feature_file}')
    df = df[df['file_name'].isin(file_names_subset)].copy()
    df = df.sort_values('file_name').reset_index(drop=True)

    feat_cols  = sorted([c for c in df.columns if c.startswith(feat_prefix)])
    X          = df[feat_cols].values.astype(np.float32)
    labels     = df[LABEL_COLUMN].values
    file_names = df['file_name'].values

    return X, labels, file_names


class FusionDataset(Dataset):
    """Dataset operating on pre-extracted 512-dim backbone features."""
    def __init__(self, t, a, v, y, fnames):
        self.t      = t.detach().clone().float() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        self.a      = a.detach().clone().float() if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)
        self.v      = v.detach().clone().float() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.long)
        self.fnames = fnames

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (self.t[i].clone(), self.a[i].clone(),
                self.v[i].clone(), self.y[i].clone(), self.fnames[i])


@torch.no_grad()
def extract_backbone_features(text_model, audio_model, video_model,
                               X_text, X_audio, X_video, batch_size=64):
    """Pre-extract 512-dim features once — avoids redundant forward passes during training."""
    text_model.eval()
    audio_model.eval()
    video_model.eval()

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
# TRAINING

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
    all_preds, all_trues, all_weights, epoch_data = [], [], [], []

    for t_b, a_b, v_b, y_b, fnames in loader:
        t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
        logits, weights = model(t_b, a_b, v_b)

        preds   = logits.argmax(dim=1).cpu().numpy()
        w_np    = weights.cpu().numpy()
        y_np    = y_b.numpy()

        all_preds.extend(preds)
        all_trues.extend(y_np)
        all_weights.append(w_np)

        for i, fname in enumerate(fnames):
            epoch_data.append({
                'file_name':    fname,
                'true_label':   int(y_np[i]),
                'pred_label':   int(preds[i]),
                'weight_text':  float(w_np[i, 0]),
                'weight_audio': float(w_np[i, 1]),
                'weight_video': float(w_np[i, 2]),
            })

    acc   = accuracy_score(all_trues, all_preds) * 100
    avg_w = np.concatenate(all_weights, axis=0).mean(axis=0)
    return acc, all_preds, all_trues, avg_w, epoch_data


# ==========================================================================
# MAIN

all_results = []

for combination in COMBINATIONS:
    print(f"\n{'='*80}")
    print(f"COMBINATION: {combination['name'].upper()}  [BASELINE MODELS]")
    print(f"{'='*80}")

    fold_accuracies = []
    all_fold_preds  = []
    le = None

    for fold in range(1, 6):
        print(f"\n{'─'*70}")
        print(f"FOLD {fold}/5")
        print(f"{'─'*70}")

        train_files, test_files = load_fold_split(fold)

        X_text_train,  y_train, train_fnames = load_features(
            combination['text_features'], train_files, 'text_feat_')
        X_text_test,   y_test,  test_fnames  = load_features(
            combination['text_features'], test_files,  'text_feat_')

        X_audio_train, _, _ = load_features(
            'dasheng_audio_embeddings.csv', train_files, 'audio_feat_')
        X_audio_test,  _, _ = load_features(
            'dasheng_audio_embeddings.csv', test_files,  'audio_feat_')

        X_video_train, _, _ = load_features(
            combination['video_features'], train_files, 'visual_feat_')
        X_video_test,  _, _ = load_features(
            combination['video_features'], test_files,  'visual_feat_')

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)

        # Load frozen baseline models
        print("Loading baseline models...")
        txt_model = TextModel(combination['text_dim']).to(device)
        txt_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{combination['text_checkpoint']}_fold_{fold}.pth",
            map_location=device))

        aud_model = BaselineAudioModel().to(device)
        aud_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{combination['audio_checkpoint']}_fold_{fold}.pth",
            map_location=device))

        vid_model = BaselineVideoModel(combination['video_dim']).to(device)
        vid_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{combination['video_checkpoint']}_fold_{fold}.pth",
            map_location=device))

        # Pre-extract 512-dim features from frozen backbones
        print("Pre-extracting features from frozen backbones...")
        t_train_f, a_train_f, v_train_f = extract_backbone_features(
            txt_model, aud_model, vid_model,
            X_text_train, X_audio_train, X_video_train)
        t_test_f,  a_test_f,  v_test_f  = extract_backbone_features(
            txt_model, aud_model, vid_model,
            X_text_test,  X_audio_test,  X_video_test)

        # Build datasets from pre-extracted features
        train_ds = FusionDataset(t_train_f, a_train_f, v_train_f, y_train_enc, train_fnames)
        val_ds   = FusionDataset(t_test_f,  a_test_f,  v_test_f,  y_test_enc,  test_fnames)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        # Build fusion model
        model = InvertedAsymmetricFusion(txt_model, aud_model, vid_model).to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,}")

        # Optimizer + scheduler
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best_acc      = 0.0
        trigger_times = 0
        best_epoch_data = []

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, epoch)
            val_acc, _, _, val_w, epoch_data = evaluate(model, val_loader)
            scheduler.step()

            # Fold + combo metadata for saving
            for rec in epoch_data:
                rec['fold']        = fold
                rec['epoch']       = epoch + 1
                rec['combination'] = combination['name']

            if val_acc > best_acc:
                best_acc        = val_acc
                trigger_times   = 0
                best_epoch_data = epoch_data
                torch.save(model.state_dict(),
                           f"{CHECKPOINTS_DIR}/inverted_asymmetric_baseline_"
                           f"{combination['name']}_fold_{fold}.pth")
            else:
                trigger_times += 1

            if (epoch + 1) % 20 == 0:
                phase = "warmup" if epoch < WARMUP_EPOCHS else "dropout"
                print(f"  Epoch {epoch+1} [{phase}]: Train={train_acc:.2f}% "
                      f"Val={val_acc:.2f}% (Best={best_acc:.2f}%) "
                      f"W[T={val_w[0]:.3f} A={val_w[1]:.3f} V={val_w[2]:.3f}]")

            if trigger_times >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        print(f"Fold {fold} Best Acc: {best_acc:.2f}%")
        fold_accuracies.append(best_acc)
        all_fold_preds.extend(best_epoch_data)

    # ── Combination results ────────────────────────────────────────────────
    mean_acc = np.mean(fold_accuracies)
    std_acc  = np.std(fold_accuracies)

    print(f"\n{'='*80}")
    print(f"COMBINATION {combination['name'].upper()} [BASELINE] — FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Mean Accuracy: {mean_acc:.2f}% (±{std_acc:.2f}%)")

    combo_df = pd.DataFrame(all_fold_preds)
    combo_df.to_csv(
        f"{OUTPUT_DIR}/inverted_asymmetric_baseline_{combination['name']}_predictions.csv",
        index=False)

    trues = combo_df['true_label'].values
    preds = combo_df['pred_label'].values
    print("\nConfusion Matrix:")
    print(confusion_matrix(trues, preds))
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=le.classes_))
    print(f"\nAverage Gate Weights:")
    print(f"  Text:  {combo_df['weight_text'].mean():.4f}")
    print(f"  Audio: {combo_df['weight_audio'].mean():.4f}")
    print(f"  Video: {combo_df['weight_video'].mean():.4f}")

    all_results.append({
        'combination':      combination['name'],
        'model_variant':    'baseline',
        'mean_accuracy':    mean_acc,
        'std_accuracy':     std_acc,
        'avg_weight_text':  combo_df['weight_text'].mean(),
        'avg_weight_audio': combo_df['weight_audio'].mean(),
        'avg_weight_video': combo_df['weight_video'].mean(),
    })

# ── Final summary ──────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("INVERTED ASYMMETRIC FUSION (BASELINE MODELS) — ALL COMBINATIONS SUMMARY")
print(f"{'='*80}")
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUTPUT_DIR}/inverted_asymmetric_baseline_summary.csv", index=False)
print(f"\nSummary saved to {OUTPUT_DIR}/inverted_asymmetric_baseline_summary.csv")