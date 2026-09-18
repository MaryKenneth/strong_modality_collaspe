"""
LATE FUSION – BASELINE UNIMODAL MODELS (MUStARD), TRIMODAL ONLY
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import json
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
MUSTARD_DIR       = r'mustard_dataset'
FEATURES_DIR      = f'{MUSTARD_DIR}/mustard_features_pkl_FIXED'
SPLITS_FILE       = f'{MUSTARD_DIR}/mustard_dataset_master/data/split_indices.p'
SARCASM_DATA_PATH = f'{MUSTARD_DIR}/mustard_dataset_master/data/sarcasm_data.json'
CHECKPOINTS_DIR   = 'checkpoints'
PREDICTIONS_DIR   = 'predictions_output_late_fusion'
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

CONTEXT_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/context/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/context/not_sarcasm',
}
PUNCHLINE_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/utterances/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/utterances/not_sarcasm',
}

NUM_CLASSES   = 2
BATCH_SIZE    = 64
EPOCHS        = 50
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15
SEED          = 999
FEAT_DIM      = 512   # penultimate feature dim for every modality

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')

# ─────────────────────────────────────────────────────────────────────────────
# FUSION CONFIGURATIONS — trimodal only
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DIMS = {
    'bert_context_punchline': 1536,
    'e5_context_punchline':   2048,
    'bert_punchline_only':     768,
    'e5_punchline_only':      1024,
    'dasheng':                1280,
    'videomae':                768,
    'pe_core':                1024,
}

FUSION_CONFIGS = [
    {'name': 'text_bert_context_audio_video_pe',       'text': 'bert_context_punchline', 'audio': 'dasheng', 'video': 'pe_core'},
    {'name': 'text_bert_context_audio_video_videomae', 'text': 'bert_context_punchline', 'audio': 'dasheng', 'video': 'videomae'},
    {'name': 'text_e5_context_audio_video_pe',         'text': 'e5_context_punchline',   'audio': 'dasheng', 'video': 'pe_core'},
    {'name': 'text_e5_context_audio_video_videomae',   'text': 'e5_context_punchline',   'audio': 'dasheng', 'video': 'videomae'},
    {'name': 'text_bert_punchline_audio_video_pe',       'text': 'bert_punchline_only', 'audio': 'dasheng', 'video': 'pe_core'},
    {'name': 'text_bert_punchline_audio_video_videomae', 'text': 'bert_punchline_only', 'audio': 'dasheng', 'video': 'videomae'},
    {'name': 'text_e5_punchline_audio_video_pe',         'text': 'e5_punchline_only',   'audio': 'dasheng', 'video': 'pe_core'},
    {'name': 'text_e5_punchline_audio_video_videomae',   'text': 'e5_punchline_only',   'audio': 'dasheng', 'video': 'videomae'},
]

# ─────────────────────────────────────────────────────────────────────────────
# UNIMODAL MODEL DEFINITIONS 
# ─────────────────────────────────────────────────────────────────────────────

class TextModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )
    def forward(self, x):
        return self.net(x)

    def extract_features(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 3:
                return x
        return x


class AudioModel(nn.Module):
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
    def forward(self, x):
        return self.net(x)

    def extract_features(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 11:
                return x
        return x


class VideoModel(nn.Module):
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
    def forward(self, x):
        return self.net(x)

    def extract_features(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 6:
                return x
        return x


class FusionModel(nn.Module):
    def __init__(self, text_dim=FEAT_DIM, audio_dim=FEAT_DIM, video_dim=FEAT_DIM):
        super().__init__()
        total_dim = text_dim + audio_dim + video_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, text_feat, audio_feat, video_feat):
        combined = torch.cat([text_feat, audio_feat, video_feat], dim=1)
        return self.fusion(combined)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING 
# ─────────────────────────────────────────────────────────────────────────────

def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_feature(pkl_path, feature_type):
    key_map = {
        'bert':     'text_bert',
        'e5':       'text_e5',
        'dasheng':  'audio_dasheng',
        'videomae': 'video_mae',
        'pe_core':  'video_pe',
    }
    fallback = {'bert': 768, 'e5': 1024, 'dasheng': 1280, 'videomae': 768, 'pe_core': 1024}
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        feat = data.get(key_map[feature_type], np.zeros((1, fallback[feature_type])))
        return feat[0] if feat.ndim > 1 else feat
    except Exception:
        return np.zeros(fallback[feature_type])


def load_modality_data(fold_indices, modality_type, sarcasm_data, identifiers):
    if modality_type is None:
        return None

    if modality_type == 'bert_context_punchline':
        base, use_context = 'bert', True
    elif modality_type == 'e5_context_punchline':
        base, use_context = 'e5', True
    elif modality_type == 'bert_punchline_only':
        base, use_context = 'bert', False
    elif modality_type == 'e5_punchline_only':
        base, use_context = 'e5', False
    else:
        base, use_context = modality_type, False   # audio / video

    feats = []
    for split_idx in fold_indices:
        if split_idx >= len(identifiers):
            continue
        identifier = identifiers[split_idx]
        label_str  = 'sarcasm' if sarcasm_data[identifier]['sarcasm'] else 'not_sarcasm'
        ctx_path   = Path(CONTEXT_PATHS[label_str])  / f'{identifier}.pkl'
        punch_path = Path(PUNCHLINE_PATHS[label_str]) / f'{identifier}.pkl'

        if use_context:
            feat = np.concatenate([load_feature(ctx_path, base),
                                   load_feature(punch_path, base)])
        else:
            feat = load_feature(punch_path, base)
        feats.append(feat)

    return np.array(feats, dtype=np.float32)


def load_labels(fold_indices, sarcasm_data, identifiers):
    labels, fnames = [], []
    for split_idx in fold_indices:
        if split_idx >= len(identifiers):
            continue
        identifier = identifiers[split_idx]
        labels.append(1 if sarcasm_data[identifier]['sarcasm'] else 0)
        fnames.append(identifier)
    return np.array(labels), fnames


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TrimodalDataset(Dataset):
    def __init__(self, text_data, audio_data, video_data, labels, file_names):
        self.text   = torch.tensor(text_data,  dtype=torch.float32)
        self.audio  = torch.tensor(audio_data, dtype=torch.float32)
        self.video  = torch.tensor(video_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.fnames = file_names

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.text[idx], self.audio[idx], self.video[idx], self.labels[idx], self.fnames[idx]


# ─────────────────────────────────────────────────────────────────────────────
# UNIMODAL MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_unimodal(model_cls, input_dim, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f'  WARNING: checkpoint not found: {ckpt_path}')
        return None
    model = model_cls(input_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL — feature-concat fusion head
# ─────────────────────────────────────────────────────────────────────────────

def train_fusion(fusion_model, text_model, audio_model, video_model,
                  train_loader, test_loader, fold, config_name):
    optimizer = optim.Adam(fusion_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    trigger_times = 0
    best_epoch_data = None
    ckpt = f'{CHECKPOINTS_DIR}/late_fusion_concat_{config_name}_fold_{fold}.pth'

    for epoch in range(EPOCHS):
        fusion_model.train()
        for t_b, a_b, v_b, y_b, _ in train_loader:
            y_b = y_b.to(device)
            with torch.no_grad():
                t_feat = text_model.extract_features(t_b.to(device))
                a_feat = audio_model.extract_features(a_b.to(device))
                v_feat = video_model.extract_features(v_b.to(device))

            optimizer.zero_grad()
            outputs = fusion_model(t_feat, a_feat, v_feat)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

        fusion_model.eval()
        epoch_predictions = []
        with torch.no_grad():
            for t_b, a_b, v_b, y_b, fnames in test_loader:
                t_feat = text_model.extract_features(t_b.to(device))
                a_feat = audio_model.extract_features(a_b.to(device))
                v_feat = video_model.extract_features(v_b.to(device))
                outputs = fusion_model(t_feat, a_feat, v_feat)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                for i, fname in enumerate(fnames):
                    epoch_predictions.append({
                        'file_name': fname,
                        'true_label': y_b[i].item(),
                        'pred_label': preds[i],
                        'fold': fold,
                        'epoch': epoch + 1,
                    })

        trues = [p['true_label'] for p in epoch_predictions]
        preds = [p['pred_label'] for p in epoch_predictions]
        acc = accuracy_score(trues, preds)

        if acc > best_acc:
            best_acc = acc
            trigger_times = 0
            best_epoch_data = epoch_predictions
            torch.save(fusion_model.state_dict(), ckpt)
        else:
            trigger_times += 1

        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}: Acc={acc*100:.2f}% (Best={best_acc*100:.2f}%)')

        if trigger_times >= PATIENCE:
            print(f'    Early stopping at epoch {epoch+1}')
            break

    return best_acc, best_epoch_data


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

splits_data = load_pickle(SPLITS_FILE)
with open(SARCASM_DATA_PATH, 'r') as f:
    sarcasm_data = json.load(f)
identifiers = list(sarcasm_data.keys())
print(f'Total samples: {len(identifiers)}\n')

all_results = []

for config in FUSION_CONFIGS:
    config_name = config['name']
    text_type, audio_type, video_type = config['text'], config['audio'], config['video']

    print(f"\n{'='*65}")
    print(f"LATE FUSION (feature-concat): {config_name}")
    print(f"{'='*65}")

    fold_accs = []
    all_preds, all_trues = [], []

    for fold in range(5):
        print(f"\n  ── Fold {fold} ──")

        train_idx = splits_data[fold][0]
        test_idx  = splits_data[fold][1]

        X_text_tr  = load_modality_data(train_idx, text_type,  sarcasm_data, identifiers)
        X_audio_tr = load_modality_data(train_idx, audio_type, sarcasm_data, identifiers)
        X_video_tr = load_modality_data(train_idx, video_type, sarcasm_data, identifiers)
        y_train, train_fnames = load_labels(train_idx, sarcasm_data, identifiers)

        X_text_te  = load_modality_data(test_idx, text_type,  sarcasm_data, identifiers)
        X_audio_te = load_modality_data(test_idx, audio_type, sarcasm_data, identifiers)
        X_video_te = load_modality_data(test_idx, video_type, sarcasm_data, identifiers)
        y_test, test_fnames = load_labels(test_idx, sarcasm_data, identifiers)

        train_ds = TrimodalDataset(X_text_tr, X_audio_tr, X_video_tr, y_train, train_fnames)
        test_ds  = TrimodalDataset(X_text_te, X_audio_te, X_video_te, y_test,  test_fnames)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

        text_dim  = FEATURE_DIMS[text_type]
        audio_dim = FEATURE_DIMS[audio_type]
        video_dim = FEATURE_DIMS[video_type]

        text_model  = load_unimodal(TextModel,  text_dim,  f'{CHECKPOINTS_DIR}/mustard_text_{text_type}_fold_{fold}.pth')
        audio_model = load_unimodal(AudioModel, audio_dim, f'{CHECKPOINTS_DIR}/mustard_audio_{audio_type}_fold_{fold}.pth')
        video_model = load_unimodal(VideoModel, video_dim, f'{CHECKPOINTS_DIR}/mustard_video_{video_type}_fold_{fold}.pth')

        fusion_model = FusionModel().to(device)

        acc, best_epoch_data = train_fusion(fusion_model, text_model, audio_model, video_model,
                                             train_loader, test_loader, fold, config_name)

        print(f'  Fold {fold} Best Acc: {acc*100:.2f}%')
        fold_accs.append(acc)

        best_trues = [p['true_label'] for p in best_epoch_data]
        best_preds = [p['pred_label'] for p in best_epoch_data]
        all_preds.extend(best_preds)
        all_trues.extend(best_trues)

    predictions_df = pd.DataFrame([
        {'true_label': t, 'pred_label': p} for t, p in zip(all_trues, all_preds)
    ])
    predictions_df.to_csv(f'{PREDICTIONS_DIR}/late_fusion_mustard_{config_name}_predictions.csv', index=False)

    report_dict = classification_report(all_trues, all_preds, target_names=['not_sarcasm', 'sarcasm'],
                                         output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(
        f'{PREDICTIONS_DIR}/late_fusion_mustard_{config_name}_classification_report.csv')
    macro = report_dict['macro avg']
    weighted = report_dict['weighted avg']

    print(f"\n{'='*65}")
    print(f"Results: {config_name}")
    print(f"Mean Acc: {np.mean(fold_accs)*100:.2f}% (+/- {np.std(fold_accs)*100:.2f}%)")
    print(f"Macro F1: {macro['f1-score']:.4f}  Weighted F1: {weighted['f1-score']:.4f}")
    print(f"{'='*65}")
    print("Confusion Matrix:\n", confusion_matrix(all_trues, all_preds))

    all_results.append({
        'combination': config_name,
        'mean_acc': np.mean(fold_accs) * 100,
        'std_acc': np.std(fold_accs) * 100,
        'fold_accs': [round(a * 100, 2) for a in fold_accs],
        'macro_precision': macro['precision'],
        'macro_recall': macro['recall'],
        'macro_f1': macro['f1-score'],
        'weighted_precision': weighted['precision'],
        'weighted_recall': weighted['recall'],
        'weighted_f1': weighted['f1-score'],
    })

print(f"\n{'='*65}")
print("LATE FUSION (MUStARD, TRIMODAL, FEATURE-CONCAT) — SUMMARY")
print(f"{'='*65}")
for r in all_results:
    print(f"  {r['combination']:40s}  Acc={r['mean_acc']:.2f}% (+/-{r['std_acc']:.2f})  "
          f"MacroF1={r['macro_f1']:.4f}  WeightedF1={r['weighted_f1']:.4f}")
pd.DataFrame(all_results).to_csv(f'{PREDICTIONS_DIR}/late_fusion_mustard_summary.csv', index=False)
