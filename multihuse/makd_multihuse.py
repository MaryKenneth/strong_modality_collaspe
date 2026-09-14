"""
MULTIHUSE MAKD — E5 TEACHER → 4 STUDENT TYPES
==============================================================
Distills knowledge from the strongest MultiHUSE model (E5 text) into
weaker modality students: BERT, Audio (Dasheng), VideoMAE, Video_PE.

Architecture: All students bottleneck to 512-dim before classifier.
KD Losses: 3-term (hard CE + soft KD + feature alignment)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os

# ==========================================================================
# CONFIGURATION — SET EXPERIMENT
EXPERIMENT_CONFIG = 'video_pe_from_e5'   # ← CHANGE: 'bert_from_e5', 'audio_from_e5',
                                         #           'videomae_from_e5', 'video_pe_from_e5'

EXPERIMENTS = {
    'bert_from_e5': {
        'name': 'BERT Student ← E5 Teacher',
        'student_type': 'bert',
        'student_input_dim': 768,
        'student_features': 'bert_text_embeddings.csv',
        'student_feat_prefix': 'text_feat_',
        'student_lr': 1e-3,
    },
    'audio_from_e5': {
        'name': 'Audio Student (Dasheng) ← E5 Teacher',
        'student_type': 'audio',
        'student_input_dim': 1280,
        'student_features': 'dasheng_audio_embeddings.csv',
        'student_feat_prefix': 'audio_feat_',
        'student_lr': 1e-4,
    },
    'videomae_from_e5': {
        'name': 'VideoMAE Student ← E5 Teacher',
        'student_type': 'video',
        'student_input_dim': None,   # GET dynamically from CSV
        'student_features': 'videomae_visual_embeddings.csv',
        'student_feat_prefix': 'visual_feat_',
        'student_lr': 9e-5,  #9e-5,  # slightly lower LR for VideoMAE to stabilize training
    },
    'video_pe_from_e5': {
        'name': 'Video PE-Core Student ← E5 Teacher',
        'student_type': 'video',
        'student_input_dim': None,   # GET dynamically from CSV
        'student_features': 'pe_core_visual_embeddings.csv',
        'student_feat_prefix': 'visual_feat_',
        'student_lr': 9e-5,  #9e-5,  # slightly lower LR for PE-Core to stabilize training
    },
}

if EXPERIMENT_CONFIG not in EXPERIMENTS:
    raise ValueError(f"Invalid EXPERIMENT_CONFIG. Must be one of {list(EXPERIMENTS.keys())}")

CONFIG = EXPERIMENTS[EXPERIMENT_CONFIG]
print(f"\n{'='*70}")
print(f"MULTIHUSE DISTILLATION EXPERIMENT: {CONFIG['name']}")
print(f"{'='*70}")
print(f"Student Type    : {CONFIG['student_type']} ({CONFIG['student_features']})")
print(f"Teacher         : E5 text (strongest MultiHUSE baseline)")
print(f"Task            : 5-class humor style classification")
print(f"{'='*70}\n")

# PATHS
SPLITS_DIR   = r'multihuse_dataset\splits_strict_v2'
FEATURES_DIR = r'multihuse_dataset'
CHECKPOINTS_DIR = 'checkpoints'

LABEL_COLUMN = 'humor_style'
E5_FEATURES_FILE   = 'e5_text_embeddings.csv'
E5_FEAT_PREFIX     = 'text_feat_'
E5_TEACHER_DIM     = 1024   # raw E5 input dim
E5_TEACHER_CKPT    = 'text_e5_text_embeddings'  # checkpoint prefix

PREDICTIONS_DIR = f'predictions_output_distill_{EXPERIMENT_CONFIG}'
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# HYPERPARAMETERS
NUM_CLASSES  = 5
BATCH_SIZE   = 64
EPOCHS       = 85
WEIGHT_DECAY = 1e-3  #1e-2
SEED         = 999
GRAD_CLIP    = 1.0

# KD parameters 
ALPHA_HARD = 0.40   # hard CE loss weight 0.40 
BETA_SOFT  = 0.35   # soft KD (KL-div) weight 0.35
GAMMA_FEAT = 0.25   # feature alignment (MSE) weight 0.25
TEMPERATURE = 3.5

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ───────────────────────────────────────────────────────────────────────────
# TEACHER MODEL  (must match the saved E5 baseline architecture)
# ───────────────────────────────────────────────────────────────────────────
class E5Teacher(nn.Module):
    def __init__(self):
        super(E5Teacher, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(E5_TEACHER_DIM, 512),   # 0
            nn.BatchNorm1d(512),               # 1
            nn.ReLU(),                         # 2
            nn.Dropout(0.3),                   # 3  ← feature extraction point
            nn.Linear(512, NUM_CLASSES)        # 4
        )

    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 3 and return_features:
                features = x.clone()
        if return_features:
            return x, features
        return x


# ───────────────────────────────────────────────────────────────────────────
# STUDENT MODELS — all bottleneck to 512-dim before classifier
# ───────────────────────────────────────────────────────────────────────────
class BERTStudent(nn.Module):
    def __init__(self):
        super(BERTStudent, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class AudioStudent(nn.Module):
    def __init__(self):
        super(AudioStudent, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1280, 1024),
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
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class VideoStudent(nn.Module):
    def __init__(self, input_dim):
        super(VideoStudent, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def build_student(student_type, input_dim):
    if student_type == 'bert':
        return BERTStudent().to(device)
    elif student_type == 'audio':
        return AudioStudent().to(device)
    elif student_type == 'video':
        return VideoStudent(input_dim).to(device)
    else:
        raise ValueError(f"Unknown student_type: {student_type}")


# ───────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ───────────────────────────────────────────────────────────────────────────
class DistillationDataset(Dataset):
    def __init__(self, student_data, teacher_data, labels, file_names):
        self.student = student_data
        self.teacher = teacher_data
        self.labels  = labels
        self.file_names = file_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.student[idx], dtype=torch.float32),
            torch.tensor(self.teacher[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],  dtype=torch.long),
            self.file_names[idx]
        )

def load_fold_split(fold_num):
    train_df = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_train.csv')
    test_df  = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_test.csv')
    return train_df['file_name'].values, test_df['file_name'].values

def load_features(feature_file, file_names_subset, feat_prefix):
    """Load feature matrix and labels for a subset of file names."""
    df = pd.read_csv(f'{FEATURES_DIR}/{feature_file}')
    df = df[df['file_name'].isin(file_names_subset)].copy()
    df = df.sort_values('file_name').reset_index(drop=True)

    feat_cols  = sorted([c for c in df.columns if c.startswith(feat_prefix)])
    X          = df[feat_cols].values.astype(np.float32)
    labels     = df[LABEL_COLUMN].values
    file_names = df['file_name'].values

    return X, labels, file_names


def get_feature_dim(feature_file, feat_prefix):
    """Dynamically resolve feature dimension from CSV column count."""
    df = pd.read_csv(f'{FEATURES_DIR}/{feature_file}', nrows=1)
    return len([c for c in df.columns if c.startswith(feat_prefix)])


# ───────────────────────────────────────────────────────────────────────────
# DISTILLATION LOSS — 3-term
# ───────────────────────────────────────────────────────────────────────────

def distillation_loss(student_logits, student_feats, teacher_logits, teacher_feats, labels):
    # 1. Hard label loss
    hard_loss = F.cross_entropy(student_logits, labels)

    # 2. Soft label KD
    soft_student = F.log_softmax(student_logits / TEMPERATURE, dim=1)
    soft_teacher = F.softmax(teacher_logits  / TEMPERATURE, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (TEMPERATURE ** 2)

    # 3. Feature alignment (student 512 -- teacher 512)
    feat_loss = F.mse_loss(student_feats, teacher_feats.detach())

    total = ALPHA_HARD * hard_loss + BETA_SOFT * kd_loss + GAMMA_FEAT * feat_loss

    return total, {
        'hard': hard_loss.item(),
        'kd':   kd_loss.item(),
        'feat': feat_loss.item(),
        'total': total.item(),
    }


# ───────────────────────────────────────────────────────────────────────────
# TRAINING
# ───────────────────────────────────────────────────────────────────────────
def train_distillation(teacher_model, student_model, train_loader, val_loader,
                       fold_num, config):
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=config['student_lr'],
        weight_decay=WEIGHT_DECAY
    )
    teacher_model.eval()

    best_acc        = 0.0
    best_epoch_data = None

    print(f"  Training {config['student_type']} student with E5 teacher (fold {fold_num})...")

    for epoch in range(EPOCHS):
        student_model.train()
        epoch_losses = {'hard': 0, 'kd': 0, 'feat': 0, 'total': 0}

        for x_student, x_teacher, y_batch, _ in train_loader:
            x_student = x_student.to(device)
            x_teacher = x_teacher.to(device)
            y_batch   = y_batch.to(device)

            # Teacher forward (frozen)
            with torch.no_grad():
                teacher_logits, teacher_feats = teacher_model(x_teacher, return_features=True)

            # Student forward
            student_logits, student_feats = student_model(x_student, return_features=True)

            # Loss
            total_loss, loss_dict = distillation_loss(
                student_logits, student_feats,
                teacher_logits, teacher_feats,
                y_batch
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), GRAD_CLIP)
            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]

        # ── Validation ──────────────────────────────────────────────────
        student_model.eval()
        epoch_predictions = []

        with torch.no_grad():
            for x_student, _, y_batch, fnames in val_loader:
                logits = student_model(x_student.to(device))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = np.argmax(probs, axis=1)

                for i, fname in enumerate(fnames):
                    pred_entry = {
                        'file_name':  fname,
                        'true_label': y_batch[i].item(),
                        'pred_label': preds[i],
                        'fold':       fold_num,
                        'epoch':      epoch + 1,
                    }
                    for c in range(NUM_CLASSES):
                        pred_entry[f'conf_{c}'] = probs[i, c]
                    epoch_predictions.append(pred_entry)

        trues   = [p['true_label'] for p in epoch_predictions]
        preds_  = [p['pred_label'] for p in epoch_predictions]
        val_acc = accuracy_score(trues, preds_) * 100

        if val_acc > best_acc:
            best_acc        = val_acc
            best_epoch_data = epoch_predictions
            ckpt_name = f"multihuse_distill_{EXPERIMENT_CONFIG}_fold_{fold_num}.pth"
            torch.save(student_model.state_dict(), f"{CHECKPOINTS_DIR}/{ckpt_name}")

        if (epoch + 1) % 40 == 0:
            avg = {k: v / len(train_loader) for k, v in epoch_losses.items()}
            print(f"    Epoch {epoch+1}: Val Acc = {val_acc:.2f}% (Best: {best_acc:.2f}%)")
            print(f"      Losses — Hard: {avg['hard']:.4f}, "
                  f"KD: {avg['kd']:.4f}, Feat: {avg['feat']:.4f}")

    best_preds = [p['pred_label'] for p in best_epoch_data]
    best_trues = [p['true_label'] for p in best_epoch_data]

    return best_acc, best_trues, best_preds, best_epoch_data


# ───────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────
student_input_dim = CONFIG['student_input_dim']
if student_input_dim is None:
    student_input_dim = get_feature_dim(
        CONFIG['student_features'], CONFIG['student_feat_prefix']
    )
    print(f"Resolved student input dim from CSV: {student_input_dim}")

print(f"\n{'='*70}")
print(f"Running: {CONFIG['name']}")
print(f"{'='*70}\n")

fold_accs             = []
all_fold_predictions  = []
all_preds             = []
all_trues             = []

for fold in range(1, 6):   # MultiHUSE: folds 1–5
    print(f"\n{'─'*70}")
    print(f"FOLD {fold}")
    print(f"{'─'*70}")

    train_files, test_files = load_fold_split(fold)

    # ── Load student features ──────────────────────────────────────────
    X_student_train, y_train, train_fnames = load_features(
        CONFIG['student_features'], train_files, CONFIG['student_feat_prefix'])
    X_student_test,  y_test,  test_fnames  = load_features(
        CONFIG['student_features'], test_files,  CONFIG['student_feat_prefix'])

    # ── Load E5 teacher features ───────────────────────────────────────
    X_teacher_train, _, _ = load_features(E5_FEATURES_FILE, train_files, E5_FEAT_PREFIX)
    X_teacher_test,  _, _ = load_features(E5_FEATURES_FILE, test_files,  E5_FEAT_PREFIX)

    # ── Encode labels ──────────────────────────────────────────────────
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    # ── Load E5 teacher model ──────────────────────────────────────────
    teacher_model = E5Teacher().to(device)
    teacher_ckpt  = f"{CHECKPOINTS_DIR}/{E5_TEACHER_CKPT}_fold_{fold}.pth"
    teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher_model.eval()

    # ── Build student ──────────────────────────────────────────────────
    student_model = build_student(CONFIG['student_type'], student_input_dim)

    # ── Datasets & loaders ─────────────────────────────────────────────
    train_ds = DistillationDataset(X_student_train, X_teacher_train, y_train_enc, train_fnames)
    val_ds   = DistillationDataset(X_student_test,  X_teacher_test,  y_test_enc,  test_fnames)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Train ──────────────────────────────────────────────────────────
    acc, true_labels, pred_labels, fold_predictions = train_distillation(
        teacher_model, student_model, train_loader, val_loader, fold, CONFIG
    )

    fold_accs.append(acc)
    all_fold_predictions.extend(fold_predictions)
    all_preds.extend(pred_labels)
    all_trues.extend(true_labels)

    print(f"  Fold {fold} Best Acc: {acc:.2f}%")

# ── Save predictions ───────────────────────────────────────────────────────
predictions_df = pd.DataFrame(all_fold_predictions)
predictions_df.to_csv(
    f'{PREDICTIONS_DIR}/distill_{EXPERIMENT_CONFIG}_predictions.csv',
    index=False
)

# ── Final report ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"FINAL RESULTS — {CONFIG['name']}")
print(f"{'='*70}")
print(f"Average Accuracy: {np.mean(fold_accs):.2f}% (±{np.std(fold_accs):.2f}%)")
print(f"Per-fold        : {[f'{a:.2f}%' for a in fold_accs]}")
print(f"\nConfusion Matrix:\n{confusion_matrix(all_trues, all_preds)}")
print(f"\nClassification Report:")
print(classification_report(all_trues, all_preds, target_names=le.classes_))
print(f"\n{'─'*50}")
print(f"Experiment Context:")
print(f"  Teacher         : E5 text (strongest MultiHUSE baseline)")
print(f"  Student         : {CONFIG['student_type']} — {CONFIG['student_features']}")
print(f"  Student dim     : {student_input_dim}")
print(f"  KD weights      : Hard={ALPHA_HARD}, Soft={BETA_SOFT}, Feat={GAMMA_FEAT}")
print(f"  Temperature     : {TEMPERATURE}")
print(f"{'='*70}")
