"""
DISTILLATION FOR MUSTARD — 4 EXPERIMENTAL CONFIGURATIONS
=================================================================
Tests distillation strategies for static text features (E5 & BERT).

Usage:
  Set EXPERIMENT_CONFIG to one of: 'e5_single', 'bert_audio_e5', 
                                    'bert_audio_video', 'e5_audio_video'
  Run 4 times to collect all results.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
import json
import os
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — SET EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════
EXPERIMENT_CONFIG = 'e5_audio_video'   # ← CHANGE: 'e5_single', 'bert_audio_e5', 
                                   #           'bert_audio_video', 'e5_audio_video'

EXPERIMENTS = {
    'e5_single': {
        'name': 'E5 (Audio Only) — Baseline',
        'student_type': 'e5_context_punchline',
        'primary_teacher': 'audio',
        'secondary_teacher': None,
        'cascade_weight': None,
        'student_lr':  5e-4,
    },
    'bert_audio_e5': {
        'name': 'BERT (Audio + E5) — Cross+Same Modality',
        'student_type': 'bert_context_punchline',
        'primary_teacher': 'audio',
        'secondary_teacher': 'e5_baseline',
        'cascade_weight': 0.3,  # 70% audio, 30% E5
        'student_lr': 1e-4,
    },
    'bert_audio_video': {
        'name': 'BERT (Audio + Video_PE) — Dual Cross-Modal',
        'student_type': 'bert_context_punchline',
        'primary_teacher': 'audio',
        'secondary_teacher': 'video_pe',
        'cascade_weight': 0.2,  # 20% audio, 80% video
        'student_lr': 1e-4,
    },
    'e5_audio_video': {
        'name': 'E5 (Audio + Video_PE) — Fair Dual Supervision',
        'student_type': 'e5_context_punchline',
        'primary_teacher': 'audio',
        'secondary_teacher': 'video_pe',
        'cascade_weight': 0.8,  # 80% audio, 20% video
        'student_lr': 5e-3,
    },
}

if EXPERIMENT_CONFIG not in EXPERIMENTS:
    raise ValueError(f"Invalid EXPERIMENT_CONFIG. Must be one of {list(EXPERIMENTS.keys())}")

CONFIG = EXPERIMENTS[EXPERIMENT_CONFIG]
print(f"\n{'='*70}")
print(f"DISTILLATION EXPERIMENT: {CONFIG['name']}")
print(f"{'='*70}")
print(f"Student       : {CONFIG['student_type']}")
print(f"Primary Teacher : {CONFIG['primary_teacher']}")
print(f"Secondary Teacher: {CONFIG['secondary_teacher']}")
if CONFIG['cascade_weight']:
    print(f"Cascade Weight  : {CONFIG['cascade_weight']:.1f} primary + "
          f"{1-CONFIG['cascade_weight']:.1f} secondary")
print(f"{'='*70}\n")

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
MUSTARD_DIR = r'mustard_dataset'
FEATURES_DIR = f'{MUSTARD_DIR}/mustard_features_pkl_FIXED'
SPLITS_FILE = f'{MUSTARD_DIR}/mustard_dataset_master/data/split_indices.p'
SARCASM_DATA_PATH = f'{MUSTARD_DIR}/mustard_dataset_master/data/sarcasm_data.json'
CHECKPOINTS_DIR = 'checkpoints'

CONTEXT_PATHS = {
    'sarcasm': f'{FEATURES_DIR}/context/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/context/not_sarcasm'
}
PUNCHLINE_PATHS = {
    'sarcasm': f'{FEATURES_DIR}/utterances/sarcasm', 
    'not_sarcasm': f'{FEATURES_DIR}/utterances/not_sarcasm'
}

PREDICTIONS_DIR = f'predictions_output_cascade_{EXPERIMENT_CONFIG}'
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────
FEATURE_DIMS = {
    'bert_context_punchline': 1536,
    'e5_context_punchline': 2048,
    'dasheng': 1280,
    'pe_core': 1024,
}

NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 85
WEIGHT_DECAY = 1e-3
SEED = 999

# KD PARAMETERS
ALPHA_HARD = 0.4    # hard CE
BETA_SOFT  = 0.35   # soft KD
GAMMA_FEAT = 0.25   # feature alignment
TEMPERATURE = 3.5
GRAD_CLIP = 1.0

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────
# TEACHER MODELS
# ─────────────────────────────────────────────────────────

class AudioTeacher(nn.Module):
    """Dasheng audio teacher"""
    def __init__(self, input_dim):
        super(AudioTeacher, self).__init__()
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
            nn.Linear(256, NUM_CLASSES)
        )
    
    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 11 and return_features:  # After second dropout (512 dims)
                features = x.clone()
        
        if return_features:
            return x, features
        return x


class VideoTeacher(nn.Module):
    """PE-Core video teacher"""
    def __init__(self, input_dim):
        super(VideoTeacher, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
    
    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 6 and return_features:  # After second dropout (512 dims)
                features = x.clone()
        
        if return_features:
            return x, features
        return x


class TextTeacher(nn.Module):
    """E5 baseline text teacher (for experiments)"""
    def __init__(self, input_dim):
        super(TextTeacher, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
    
    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 3 and return_features:  # After dropout (512 dims)
                features = x.clone()
        
        if return_features:
            return x, features
        return x


# ─────────────────────────────────────────────────────────
# TEXT STUDENT 
# ─────────────────────────────────────────────────────────

class SimplifiedTextStudent(nn.Module):
    def __init__(self, input_dim):
        super(SimplifiedTextStudent, self).__init__()
        
        # Adaptive encoder
        if input_dim > 1500:  # context+punchline variants (E5=2048, BERT=1536)
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
                nn.Dropout(0.2)
            )
        else:  # fallback for smaller dims
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        
        # Main classifier only
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


# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────

class DistillationDataset(Dataset):
    def __init__(self, text_data, teacher_data, labels, file_names):
        """
        teacher_data can be:
          - Single modality: np.array
          - Dual teachers: dict {'primary': np.array, 'secondary': np.array}
        """
        self.text = text_data
        self.teacher = teacher_data
        self.labels = labels
        self.file_names = file_names
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.text[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if isinstance(self.teacher, dict):
            # Dual teachers
            teacher_data = {
                k: torch.tensor(v[idx], dtype=torch.float32) 
                for k, v in self.teacher.items()
            }
        else:
            # Single teacher
            teacher_data = torch.tensor(self.teacher[idx], dtype=torch.float32)
        
        return text_tensor, teacher_data, label_tensor, self.file_names[idx]


def custom_collate(batch):
    """Handle both single and dual teacher data"""
    text_batch = []
    teacher_batch = []
    labels = []
    fnames = []
    
    is_dict = isinstance(batch[0][1], dict)
    
    for text, teacher_data, label, fname in batch:
        text_batch.append(text)
        teacher_batch.append(teacher_data)
        labels.append(label)
        fnames.append(fname)
    
    text_tensor = torch.stack(text_batch)
    labels_tensor = torch.stack(labels)
    
    if is_dict:
        keys = teacher_batch[0].keys()
        teacher_tensor = {}
        for key in keys:
            teacher_tensor[key] = torch.stack([t[key] for t in teacher_batch])
    else:
        teacher_tensor = torch.stack(teacher_batch)
    
    return text_tensor, teacher_tensor, labels_tensor, fnames


def load_mustard_splits():
    """Load MUStARD 5-fold cross-validation splits"""
    try:
        with open(SPLITS_FILE, 'rb') as f:
            splits_data = pickle.load(f, encoding='latin1')
    except UnicodeDecodeError:
        with open(SPLITS_FILE, 'rb') as f:
            splits_data = pickle.load(f, encoding='bytes')
    return splits_data


def load_features_from_pkl(pkl_path, feature_type):
    """Load features from a single .pkl file"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if feature_type == 'bert':
            return data.get('text_bert', np.zeros((1, 768)))[0]
        elif feature_type == 'e5':
            return data.get('text_e5', np.zeros((1, 1024)))[0]
        elif feature_type == 'dasheng':
            return data.get('audio_dasheng', np.zeros((1, 1280)))[0]
        elif feature_type == 'pe_core':
            return data.get('video_pe', np.zeros((1, 1024)))[0]
        else:
            return None
    except:
        if feature_type == 'bert':
            return np.zeros(768)
        elif feature_type == 'e5':
            return np.zeros(1024)
        elif feature_type == 'dasheng':
            return np.zeros(1280)
        elif feature_type == 'pe_core':
            return np.zeros(1024)


def prepare_student_features(fold_indices, student_type, sarcasm_data, identifiers):
    """Prepare text features for student"""
    X_list = []
    y_list = []
    file_names = []
    
    for split_idx in fold_indices:
        if split_idx >= len(identifiers):
            continue
            
        identifier = identifiers[split_idx]
        sarcasm_label = sarcasm_data[identifier]['sarcasm']
        label = 1 if sarcasm_label else 0
        
        if label == 1:
            context_path = Path(CONTEXT_PATHS['sarcasm']) / f"{identifier}.pkl"
            punchline_path = Path(PUNCHLINE_PATHS['sarcasm']) / f"{identifier}.pkl"
        else:
            context_path = Path(CONTEXT_PATHS['not_sarcasm']) / f"{identifier}.pkl"
            punchline_path = Path(PUNCHLINE_PATHS['not_sarcasm']) / f"{identifier}.pkl"
        
        # Extract text features (context + punchline only)
        if student_type == 'bert_context_punchline':
            context_feat = load_features_from_pkl(context_path, 'bert')
            punchline_feat = load_features_from_pkl(punchline_path, 'bert')
            text_feat = np.concatenate([context_feat, punchline_feat])
        elif student_type == 'e5_context_punchline':
            context_feat = load_features_from_pkl(context_path, 'e5')
            punchline_feat = load_features_from_pkl(punchline_path, 'e5')
            text_feat = np.concatenate([context_feat, punchline_feat])
        
        X_list.append(text_feat)
        y_list.append(label)
        file_names.append(identifier)
    
    return np.array(X_list, dtype=np.float32), np.array(y_list), file_names


def prepare_teacher_features(fold_indices, teacher_name, sarcasm_data, identifiers):
    """Prepare teacher features for a single teacher"""
    X_list = []
    
    feature_map = {
        'audio': 'dasheng',
        'video_pe': 'pe_core',
        'e5_baseline': 'e5',
    }
    feature_type = feature_map[teacher_name]
    
    for split_idx in fold_indices:
        if split_idx >= len(identifiers):
            continue
            
        identifier = identifiers[split_idx]
        sarcasm_label = sarcasm_data[identifier]['sarcasm']
        label = 1 if sarcasm_label else 0
        
        if label == 1:
            context_path = Path(CONTEXT_PATHS['sarcasm']) / f"{identifier}.pkl"
            punchline_path = Path(PUNCHLINE_PATHS['sarcasm']) / f"{identifier}.pkl"
        else:
            context_path = Path(CONTEXT_PATHS['not_sarcasm']) / f"{identifier}.pkl"
            punchline_path = Path(PUNCHLINE_PATHS['not_sarcasm']) / f"{identifier}.pkl"
        
        # E5 baseline needs context+punchline (like the student and teacher)
        if teacher_name == 'e5_baseline':
            context_feat = load_features_from_pkl(context_path, feature_type)
            punchline_feat = load_features_from_pkl(punchline_path, feature_type)
            feat = np.concatenate([context_feat, punchline_feat])
        else:
            # Audio and Video_PE use punchline only
            feat = load_features_from_pkl(punchline_path, feature_type)
        
        X_list.append(feat)
    
    return np.array(X_list, dtype=np.float32)


# ─────────────────────────────────────────────────────────
# DISTILLATION LOSS (3-term: Hard + Soft + Feature)
# ─────────────────────────────────────────────────────────

def distillation_loss(student_logits, student_feats, teacher_logits, teacher_feats, labels):
    """
    3-term KD loss matching UR-FUNNY:
      ALPHA_HARD  × cross-entropy (hard labels)
      BETA_SOFT   × KL-div (soft labels, temperature-scaled)
      GAMMA_FEAT  × MSE (feature alignment)
    """
    # 1. Hard label loss
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 2. Soft label KD
    soft_student = F.log_softmax(student_logits / TEMPERATURE, dim=1)
    soft_teacher = F.softmax(teacher_logits / TEMPERATURE, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (TEMPERATURE ** 2)
    
    # 3. Feature alignment
    feat_loss = F.mse_loss(student_feats, teacher_feats.detach())
    
    total = ALPHA_HARD * hard_loss + BETA_SOFT * kd_loss + GAMMA_FEAT * feat_loss
    
    return total, {
        'hard': hard_loss.item(),
        'kd': kd_loss.item(),
        'feat': feat_loss.item(),
        'total': total.item(),
    }


def cascade_distillation_loss(student_logits, student_feats, 
                              primary_logits, primary_feats,
                              secondary_logits, secondary_feats,
                              labels, cascade_weight):
    """
    Dual-teacher cascade loss (Option A: Weighted Combined Loss)
    
    cascade_weight: weight for primary teacher (e.g., 0.7)
                   secondary gets (1 - cascade_weight)
    """
    # Loss from primary teacher
    loss_primary, dict_primary = distillation_loss(
        student_logits, student_feats, primary_logits, primary_feats, labels)
    
    # Loss from secondary teacher
    loss_secondary, dict_secondary = distillation_loss(
        student_logits, student_feats, secondary_logits, secondary_feats, labels)
    
    # Weighted combination
    total_loss = cascade_weight * loss_primary + (1 - cascade_weight) * loss_secondary
    
    return total_loss, {
        'hard': cascade_weight * dict_primary['hard'] + (1-cascade_weight) * dict_secondary['hard'],
        'kd': cascade_weight * dict_primary['kd'] + (1-cascade_weight) * dict_secondary['kd'],
        'feat': cascade_weight * dict_primary['feat'] + (1-cascade_weight) * dict_secondary['feat'],
        'total': total_loss.item(),
        'primary_weight': cascade_weight,
        'secondary_weight': 1 - cascade_weight,
    }


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────

def train_distillation(primary_teacher, secondary_teacher, student_model, 
                      train_loader, val_loader, fold_num, config):
    """
    Train student with single or dual teachers.
    
    Args:
        primary_teacher: Main teacher model
        secondary_teacher: Optional second teacher (None for single-teacher)
        student_model: Student to train
        config: Experiment configuration dict
    """
    optimizer = optim.AdamW(student_model.parameters(), 
                           lr=config['student_lr'], 
                           weight_decay=WEIGHT_DECAY)
    
    primary_teacher.eval()
    if secondary_teacher is not None:
        secondary_teacher.eval()
    
    best_acc = 0.0
    best_epoch_data = None
    
    is_dual = secondary_teacher is not None
    teacher_desc = f"{config['primary_teacher']}" + \
                  (f"+{config['secondary_teacher']}" if is_dual else "")
    
    print(f"Training {config['student_type']} with {teacher_desc} (fold {fold_num})...")
    
    for epoch in range(EPOCHS):
        student_model.train()
        epoch_losses = {'hard': 0, 'kd': 0, 'feat': 0, 'total': 0}
        
        for x_text, teacher_data, y_batch, _ in train_loader:
            x_text, y_batch = x_text.to(device), y_batch.to(device)
            
            # Get student predictions
            student_logits, student_feats = student_model(x_text, return_features=True)
            
            # Get teacher predictions (no grad for teachers)
            with torch.no_grad():
                if is_dual:
                    # Dual teachers
                    primary_data = teacher_data['primary'].to(device)
                    secondary_data = teacher_data['secondary'].to(device)
                    
                    primary_logits, primary_feats = primary_teacher(primary_data, return_features=True)
                    secondary_logits, secondary_feats = secondary_teacher(secondary_data, return_features=True)
                else:
                    # Single teacher
                    teacher_input = teacher_data.to(device)
                    teacher_logits, teacher_feats = primary_teacher(teacher_input, return_features=True)
            
            # Compute loss (OUTSIDE no_grad block so student gradients flow)
            if is_dual:
                total_loss, loss_dict = cascade_distillation_loss(
                    student_logits, student_feats,
                    primary_logits, primary_feats,
                    secondary_logits, secondary_feats,
                    y_batch, config['cascade_weight']
                )
            else:
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
        
        # Validation
        student_model.eval()
        epoch_predictions = []
        
        with torch.no_grad():
            for x_text, teacher_data, y_batch, fnames in val_loader:
                logits = student_model(x_text.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                for i, fname in enumerate(fnames):
                    epoch_predictions.append({
                        'file_name': fname,
                        'true_label': y_batch[i].item(),
                        'pred_label': preds[i],
                        'conf_0': probs[i, 0],
                        'conf_1': probs[i, 1],
                        'fold': fold_num,
                        'epoch': epoch + 1
                    })
        
        trues = [p['true_label'] for p in epoch_predictions]
        preds = [p['pred_label'] for p in epoch_predictions]
        val_acc = accuracy_score(trues, preds) * 100
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch_data = epoch_predictions
            ckpt_name = f"mustard_cascade_{EXPERIMENT_CONFIG}_fold_{fold_num}.pth"
            torch.save(student_model.state_dict(), 
                      f"{CHECKPOINTS_DIR}/{ckpt_name}")
        
        if (epoch + 1) % 40 == 0:
            avg_losses = {k: v/len(train_loader) for k, v in epoch_losses.items()}
            print(f"  Epoch {epoch+1}: Val Acc = {val_acc:.2f}% (Best: {best_acc:.2f}%)")
            print(f"    Losses - Hard: {avg_losses['hard']:.4f}, "
                  f"KD: {avg_losses['kd']:.4f}, Feat: {avg_losses['feat']:.4f}")
    
    best_preds = [p['pred_label'] for p in best_epoch_data]
    best_trues = [p['true_label'] for p in best_epoch_data]
    
    return best_acc, best_trues, best_preds, best_epoch_data


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

splits_data = load_mustard_splits()

# Load sarcasm_data.json
with open(SARCASM_DATA_PATH, 'r') as f:
    sarcasm_data = json.load(f)

identifiers = list(sarcasm_data.keys())
print(f"Total samples in JSON: {len(identifiers)}")

print(f"\n{'='*70}")
print(f"Running: {CONFIG['name']}")
print(f"{'='*70}\n")

fold_accs = []
all_fold_predictions = []
all_preds = []
all_trues = []

for fold in range(5):  # 0-4 for MUStARD
    print(f"\n{'─'*70}")
    print(f"FOLD {fold}")
    print(f"{'─'*70}")
    
    # Get train/test indices
    train_indices = splits_data[fold][0]
    test_indices = splits_data[fold][1]
    
    # Load student features
    X_text_train, y_train, train_fnames = prepare_student_features(
        train_indices, CONFIG['student_type'], sarcasm_data, identifiers)
    X_text_test, y_test, test_fnames = prepare_student_features(
        test_indices, CONFIG['student_type'], sarcasm_data, identifiers)
    
    # Load teacher models and features
    teacher_configs = {
        'audio': {'class': AudioTeacher, 'dim': 1280, 
                 'ckpt': f"{CHECKPOINTS_DIR}/mustard_audio_dasheng_fold_{fold}.pth"},
        'video_pe': {'class': VideoTeacher, 'dim': 1024,
                    'ckpt': f"{CHECKPOINTS_DIR}/mustard_video_pe_core_fold_{fold}.pth"},
        'e5_baseline': {'class': TextTeacher, 'dim': 2048,
                       'ckpt': f"{CHECKPOINTS_DIR}/mustard_text_e5_context_punchline_fold_{fold}.pth"},
    }
    
    # Load primary teacher
    primary_cfg = teacher_configs[CONFIG['primary_teacher']]
    primary_teacher = primary_cfg['class'](primary_cfg['dim']).to(device)
    primary_teacher.load_state_dict(torch.load(primary_cfg['ckpt'], map_location=device))
    
    # Load primary teacher features
    X_primary_train = prepare_teacher_features(
        train_indices, CONFIG['primary_teacher'], sarcasm_data, identifiers)
    X_primary_test = prepare_teacher_features(
        test_indices, CONFIG['primary_teacher'], sarcasm_data, identifiers)
    
    # Handle secondary teacher (if dual)
    if CONFIG['secondary_teacher'] is not None:
        secondary_cfg = teacher_configs[CONFIG['secondary_teacher']]
        secondary_teacher = secondary_cfg['class'](secondary_cfg['dim']).to(device)
        secondary_teacher.load_state_dict(torch.load(secondary_cfg['ckpt'], map_location=device))
        
        # Load secondary teacher features
        X_secondary_train = prepare_teacher_features(
            train_indices, CONFIG['secondary_teacher'], sarcasm_data, identifiers)
        X_secondary_test = prepare_teacher_features(
            test_indices, CONFIG['secondary_teacher'], sarcasm_data, identifiers)
        
        # Package as dict
        X_teacher_train = {'primary': X_primary_train, 'secondary': X_secondary_train}
        X_teacher_test = {'primary': X_primary_test, 'secondary': X_secondary_test}
    else:
        secondary_teacher = None
        X_teacher_train = X_primary_train
        X_teacher_test = X_primary_test
    
    # Create datasets
    train_ds = DistillationDataset(X_text_train, X_teacher_train, y_train, train_fnames)
    val_ds = DistillationDataset(X_text_test, X_teacher_test, y_test, test_fnames)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    
    # Create student model
    student_dim = FEATURE_DIMS[CONFIG['student_type']]
    student_model = SimplifiedTextStudent(student_dim).to(device)
    
    # Train
    acc, true_labels, pred_labels, fold_predictions = train_distillation(
        primary_teacher, secondary_teacher, student_model,
        train_loader, val_loader, fold, CONFIG
    )
    
    fold_accs.append(acc)
    all_fold_predictions.extend(fold_predictions)
    all_preds.extend(pred_labels)
    all_trues.extend(true_labels)
    
    print(f"Fold {fold} Best Acc: {acc:.2f}%")

# Save predictions
predictions_df = pd.DataFrame(all_fold_predictions)
predictions_df.to_csv(
    f'{PREDICTIONS_DIR}/cascade_{EXPERIMENT_CONFIG}_predictions.csv', 
    index=False
)

print(f"\n{'='*70}")
print(f"FINAL RESULTS — {CONFIG['name']}")
print(f"{'='*70}")
print(f"Average Accuracy: {np.mean(fold_accs):.2f}% (±{np.std(fold_accs):.2f}%)")
print(f"Per-fold: {[f'{a:.2f}%' for a in fold_accs]}")
print(f"\nConfusion Matrix:\n{confusion_matrix(all_trues, all_preds)}")
print(f"\nClassification Report:")
print(classification_report(all_trues, all_preds, 
                          target_names=['not_sarcasm', 'sarcasm']))
print(f"\n{'─'*50}")
print(f"Experiment Context:")
print(f"  Teacher Baselines: Audio=78%, Video_PE=77.10%, E5=70.29%")
print(f"  Student: {CONFIG['student_type']}")
print(f"  Primary Teacher: {CONFIG['primary_teacher']}")
print(f"  Secondary Teacher: {CONFIG['secondary_teacher'] or 'None (single teacher)'}")
if CONFIG['cascade_weight']:
    print(f"  Cascade Weight: {CONFIG['cascade_weight']:.1f} primary + "
          f"{1-CONFIG['cascade_weight']:.1f} secondary")
print(f"{'='*70}")