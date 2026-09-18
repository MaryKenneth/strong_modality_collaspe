"""
CAUSAL DECOMPOSITION — MUStARD (2x2 grid x attention x classifier-freezing)
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import json
import os
import copy
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

MUSTARD_DIR       = r'mustard_dataset'
FEATURES_DIR      = f'{MUSTARD_DIR}/mustard_features_pkl_FIXED'
SPLITS_FILE       = f'{MUSTARD_DIR}/mustard_dataset_master/data/split_indices.p'
SARCASM_DATA_PATH = f'{MUSTARD_DIR}/mustard_dataset_master/data/sarcasm_data.json'
CHECKPOINTS_DIR   = 'checkpoints'
OUTPUT_DIR        = 'causal_decomposition_results_mustard'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/context/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/context/not_sarcasm',
}
PUNCHLINE_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/utterances/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/utterances/not_sarcasm',
}

CONFIGS = {
    'bert_videomae': {
        'name': 'BERT + Audio + VideoMAE', 'text_type': 'bert_context_punchline', 'text_dim': 1536,
        'baseline_text_checkpoint': 'mustard_text_bert_context_punchline',
        'cascade_text_checkpoint':  'mustard_cascade_bert_audio_e5',
        'video_type': 'videomae', 'video_checkpoint': 'mustard_video_videomae', 'video_dim': 768,
    },
    'bert_pe_core': {
        'name': 'BERT + Audio + Video_PE', 'text_type': 'bert_context_punchline', 'text_dim': 1536,
        'baseline_text_checkpoint': 'mustard_text_bert_context_punchline',
        'cascade_text_checkpoint':  'mustard_cascade_bert_audio_e5',
        'video_type': 'pe_core', 'video_checkpoint': 'mustard_video_pe_core', 'video_dim': 1024,
    },
    'e5_videomae': {
        'name': 'E5 + Audio + VideoMAE', 'text_type': 'e5_context_punchline', 'text_dim': 2048,
        'baseline_text_checkpoint': 'mustard_text_e5_context_punchline',
        'cascade_text_checkpoint':  'mustard_cascade_e5_audio_video',
        'video_type': 'videomae', 'video_checkpoint': 'mustard_video_videomae', 'video_dim': 768,
    },
    'e5_pe_core': {
        'name': 'E5 + Audio + Video_PE', 'text_type': 'e5_context_punchline', 'text_dim': 2048,
        'baseline_text_checkpoint': 'mustard_text_e5_context_punchline',
        'cascade_text_checkpoint':  'mustard_cascade_e5_audio_video',
        'video_type': 'pe_core', 'video_checkpoint': 'mustard_video_pe_core', 'video_dim': 1024,
    },
}

VARIANTS          = ['baseline', 'cascade']
DOMINANT_PATHWAYS = ['audio', 'video']   # which dominant modality's grid to compute

NUM_CLASSES  = 2
FUSION_DIM   = 512
AUDIO_DIM    = 1280

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


# ═══════════════════════════════════════════════════════════════════════════
# BACKBONE / FUSION MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
class AudioTeacher(nn.Module):
    def __init__(self, input_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES))

    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 11 and return_features:
                features = x.clone()
        return (x, features) if return_features else x

    def get_features(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 11:
                return x


class VideoBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES))

    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 6 and return_features:
                features = x.clone()
        return (x, features) if return_features else x

    def get_features(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 6:
                return x


class TextBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES))

    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 3 and return_features:
                features = x.clone()
        return (x, features) if return_features else x

    def get_features(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 3:
                return x


class SimplifiedTextStudent(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim > 1500:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2))
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES))

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits

    def get_features(self, x):
        return self.encoder(x)


class CrossModalAnchor(nn.Module):
    def __init__(self, d_model=FUSION_DIM):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, anchor, context):
        a_seq = anchor.unsqueeze(1)
        out, _ = self.cross_attn(query=a_seq, key=context, value=context)
        return self.norm(anchor + self.dropout(out.squeeze(1)))


class SymmetricFusion(nn.Module):
    def __init__(self, text_backbone, audio_backbone, video_backbone):
        super().__init__()
        self.text_backbone, self.audio_backbone, self.video_backbone = \
            text_backbone, audio_backbone, video_backbone
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters(): p.requires_grad = False

        self.text_anchor  = CrossModalAnchor(FUSION_DIM)
        self.audio_anchor = CrossModalAnchor(FUSION_DIM)
        self.video_anchor = CrossModalAnchor(FUSION_DIM)
        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 3))
        self.classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, t_feat, a_feat, v_feat):
        out_t = self.text_anchor(t_feat,  torch.stack([a_feat, v_feat], dim=1))
        out_a = self.audio_anchor(a_feat, torch.stack([t_feat, v_feat], dim=1))
        out_v = self.video_anchor(v_feat, torch.stack([t_feat, a_feat], dim=1))
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        fused = w_t * out_t + w_a * out_a + w_v * out_v
        return self.classifier(fused), weights

    def attended_feature(self, t_feat, a_feat, v_feat, pathway):
        if pathway == 'audio':
            return self.audio_anchor(a_feat, torch.stack([t_feat, v_feat], dim=1))
        elif pathway == 'video':
            return self.video_anchor(v_feat, torch.stack([t_feat, a_feat], dim=1))
        raise ValueError(f"Unknown dominant pathway: {pathway}")

    @torch.no_grad()
    def individual_pathway_logits(self, t_feat, a_feat, v_feat, pathway):
        """Cell 4: dominant modality routed through symmetric fusion's SHARED
        trainable classifier, other modalities still present as attention context."""
        feat = self.attended_feature(t_feat, a_feat, v_feat, pathway)
        return self.classifier(feat)


class InvertedAsymmetricFusion(nn.Module):
    """for (Cell 1)."""
    def __init__(self, text_backbone, audio_backbone, video_backbone):
        super().__init__()
        self.text_backbone, self.audio_backbone, self.video_backbone = \
            text_backbone, audio_backbone, video_backbone
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters(): p.requires_grad = False

        self.text_anchor = CrossModalAnchor(FUSION_DIM)
        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 3))

        self.audio_classifier = copy.deepcopy(audio_backbone.net[12:])
        self.video_classifier = copy.deepcopy(video_backbone.net[7:])
        for p in self.audio_classifier.parameters(): p.requires_grad = False
        for p in self.video_classifier.parameters(): p.requires_grad = False
        self.text_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, t_feat, a_feat, v_feat):
        out_text = self.text_anchor(t_feat, torch.stack([a_feat, v_feat], dim=1))
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        logits_text  = self.text_classifier(out_text)
        logits_audio = self.audio_classifier(a_feat)
        logits_video = self.video_classifier(v_feat)
        final = w_t * logits_text + w_a * logits_audio + w_v * logits_video
        return final, weights

    @torch.no_grad()
    def pure_pathway_logits(self, a_feat, v_feat, pathway):
        """Cell 1: the tested dominant modality's frozen PURE classifier
        applied directly to its own raw feature."""
        if pathway == 'audio':
            return self.audio_classifier(a_feat)
        elif pathway == 'video':
            return self.video_classifier(v_feat)
        raise ValueError(f"Unknown dominant pathway: {pathway}")


class TrainablePureHeadFusionMustard(nn.Module):
    """Cell 2 ablation: identical to InvertedAsymmetricFusion except the
    TESTED dominant modality's classifier is warm-started (same init as
    Cell 1) then left TRAINABLE. The OTHER dominant modality stays frozen,
    exactly as in standard IAF — only one pathway's drift is isolated at a
    time. Text still attends with a fresh trainable head, same as IAF."""
    def __init__(self, text_backbone, audio_backbone, video_backbone, ablate_dominant):
        super().__init__()
        assert ablate_dominant in ('audio', 'video')
        self.ablate_dominant = ablate_dominant
        self.text_backbone, self.audio_backbone, self.video_backbone = \
            text_backbone, audio_backbone, video_backbone
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters(): p.requires_grad = False

        self.text_anchor = CrossModalAnchor(FUSION_DIM)
        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 3))

        # Warm-started from the same trained unimodal classifier as Cell 1,
        # then the ablated one is explicitly unfrozen.
        self.audio_classifier = copy.deepcopy(audio_backbone.net[12:])
        self.video_classifier = copy.deepcopy(video_backbone.net[7:])
        for p in self.audio_classifier.parameters():
            p.requires_grad = (ablate_dominant == 'audio')
        for p in self.video_classifier.parameters():
            p.requires_grad = (ablate_dominant == 'video')
        self.text_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, t_feat, a_feat, v_feat):
        out_text = self.text_anchor(t_feat, torch.stack([a_feat, v_feat], dim=1))
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        logits_text  = self.text_classifier(out_text)
        logits_audio = self.audio_classifier(a_feat)
        logits_video = self.video_classifier(v_feat)
        final = w_t * logits_text + w_a * logits_audio + w_v * logits_video
        return final, weights

    @torch.no_grad()
    def pure_pathway_logits(self, a_feat, v_feat, pathway):
        if pathway == 'audio':
            return self.audio_classifier(a_feat)
        elif pathway == 'video':
            return self.video_classifier(v_feat)
        raise ValueError(f"Unknown dominant pathway: {pathway}")


# ═══════════════════════════════════════════════════════════════════════════
# MIXUP + MODALITY DROPOUT
# ═══════════════════════════════════════════════════════════════════════════
def mixup_data(t, a, v, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(t.size(0)).to(t.device)
    return (lam * t + (1 - lam) * t[idx], lam * a + (1 - lam) * a[idx],
            lam * v + (1 - lam) * v[idx], y, y[idx], lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def apply_modality_dropout(t, a, v, epoch):
    if epoch < WARMUP_EPOCHS:
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


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_features_from_pkl(pkl_path, feature_type):
    feature_keys = {
        'bert': 'text_bert', 'e5': 'text_e5', 'dasheng': 'audio_dasheng',
        'pe_core': 'video_pe', 'videomae': 'video_mae',
    }
    fallback_dims = {'bert': 768, 'e5': 1024, 'dasheng': 1280, 'pe_core': 1024, 'videomae': 768}
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        feat = data.get(feature_keys[feature_type], np.zeros((1, fallback_dims[feature_type])))
        return feat[0] if len(feat.shape) > 1 else feat
    except Exception:
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
        text_feat = np.concatenate([
            load_features_from_pkl(context_path,   text_feature_type),
            load_features_from_pkl(punchline_path, text_feature_type)])
        audio_feat = load_features_from_pkl(punchline_path, 'dasheng')
        video_feat = load_features_from_pkl(punchline_path, video_type)
        X_text.append(text_feat); X_audio.append(audio_feat); X_video.append(video_feat)
        y_list.append(label); file_names.append(identifier)
    return (np.array(X_text, dtype=np.float32), np.array(X_audio, dtype=np.float32),
            np.array(X_video, dtype=np.float32), np.array(y_list), file_names)


class FusionDataset(Dataset):
    def __init__(self, text_data, audio_data, video_data, labels):
        to_t = lambda x: x.detach().clone().float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        self.text, self.audio, self.video = to_t(text_data), to_t(audio_data), to_t(video_data)
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.text[idx].clone(), self.audio[idx].clone(), self.video[idx].clone(), self.labels[idx].clone()


@torch.no_grad()
def extract_backbone_features(text_model, audio_model, video_model,
                               text_data, audio_data, video_data, batch_size=64):
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
        text_feats.append(text_model.get_features(t_b).cpu())
        audio_feats.append(audio_model.get_features(a_b).cpu())
        video_feats.append(video_model.get_features(v_b).cpu())
    return (torch.cat(text_feats, dim=0), torch.cat(audio_feats, dim=0), torch.cat(video_feats, dim=0))


def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for t_b, a_b, v_b, y_b in loader:
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
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for t_b, a_b, v_b, y_b in loader:
        t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
        logits, _ = model(t_b, a_b, v_b)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y_b.numpy())
    return accuracy_score(all_labels, all_preds) * 100


@torch.no_grad()
def evaluate_pure_pathway(model, a_feat, v_feat, labels, pathway):
    model.eval()
    logits = model.pure_pathway_logits(a_feat, v_feat, pathway)
    preds = logits.argmax(1).cpu().numpy()
    return accuracy_score(labels, preds) * 100


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
splits_data = load_pickle(SPLITS_FILE)
with open(SARCASM_DATA_PATH, 'r') as f:
    sarcasm_data = json.load(f)
identifiers = list(sarcasm_data.keys())

all_grid_rows = []

for config_name, cfg in CONFIGS.items():
    for fold in range(5):
        train_indices = splits_data[fold][0]
        test_indices  = splits_data[fold][1]

        X_text_train, X_audio_train, X_video_train, y_train, _ = prepare_features(
            train_indices, cfg['text_type'], cfg['video_type'], sarcasm_data, identifiers)
        X_text_test, X_audio_test, X_video_test, y_test, _ = prepare_features(
            test_indices, cfg['text_type'], cfg['video_type'], sarcasm_data, identifiers)

        audio_model = AudioTeacher(AUDIO_DIM).to(device)
        audio_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/mustard_audio_dasheng_fold_{fold}.pth", map_location=device))
        video_model = VideoBaseline(cfg['video_dim']).to(device)
        video_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{cfg['video_checkpoint']}_fold_{fold}.pth", map_location=device))

        for variant in VARIANTS:
            fusion_tag = 'baseline' if variant == 'baseline' else 'cascade'
            text_ckpt = (f"{CHECKPOINTS_DIR}/{cfg['baseline_text_checkpoint']}_fold_{fold}.pth" if variant == 'baseline'
                         else f"{CHECKPOINTS_DIR}/{cfg['cascade_text_checkpoint']}_fold_{fold}.pth")
            iaf_ckpt = f"{CHECKPOINTS_DIR}/fusion_inverted_{fusion_tag}_{config_name}_fold_{fold}.pth"
            sym_ckpt = f"{CHECKPOINTS_DIR}/fusion_symmetric_{fusion_tag}_{config_name}_fold_{fold}.pth"

            # Skip combos whose checkpoints haven't been trained yet, rather
            # than crashing the whole run (and burning a Cell-2 training run
            # for a combo we can't complete anyway) — report and move on.
            missing = [p for p in (text_ckpt, iaf_ckpt, sym_ckpt) if not os.path.exists(p)]
            if missing:
                print(f"\nSKIPPED  config={config_name} fold={fold} variant={variant} — "
                      f"missing checkpoint(s): {missing}")
                continue

            if variant == 'baseline':
                text_model = TextBaseline(cfg['text_dim']).to(device)
            else:
                text_model = SimplifiedTextStudent(cfg['text_dim']).to(device)
            text_model.load_state_dict(torch.load(text_ckpt, map_location=device))

            t_train_f, a_train_f, v_train_f = extract_backbone_features(
                text_model, audio_model, video_model, X_text_train, X_audio_train, X_video_train)
            t_test_f, a_test_f, v_test_f = extract_backbone_features(
                text_model, audio_model, video_model, X_text_test, X_audio_test, X_video_test)

            train_loader = DataLoader(FusionDataset(t_train_f, a_train_f, v_train_f, y_train),
                                       batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(FusionDataset(t_test_f, a_test_f, v_test_f, y_test),
                                      batch_size=BATCH_SIZE, shuffle=False)
            a_test_dev, v_test_dev = a_test_f.to(device), v_test_f.to(device)

            iaf_model = InvertedAsymmetricFusion(text_model, audio_model, video_model).to(device)
            iaf_model.load_state_dict(torch.load(
                f"{CHECKPOINTS_DIR}/fusion_inverted_{fusion_tag}_{config_name}_fold_{fold}.pth", map_location=device))
            iaf_model.eval()

            sym_model = SymmetricFusion(text_model, audio_model, video_model).to(device)
            sym_model.load_state_dict(torch.load(
                f"{CHECKPOINTS_DIR}/fusion_symmetric_{fusion_tag}_{config_name}_fold_{fold}.pth", map_location=device))
            sym_model.eval()

            for dominant in DOMINANT_PATHWAYS:
                print(f"\n{'='*80}")
                print(f"CONFIG={config_name}  FOLD={fold}  VARIANT={variant}  DOMINANT={dominant}")
                print(f"{'='*80}")

                # ── Cell 1: No attn, frozen head (IAF PURE) — reuse checkpoint ──
                cell1 = evaluate_pure_pathway(iaf_model, a_test_dev, v_test_dev, y_test, dominant)
                print(f"  Cell 1 (IAF PURE, frozen head): {cell1:.2f}%")

                # ── Cell 3: Cross-attn, frozen head — reuse symmetric checkpoint ──
                frozen_head = copy.deepcopy(
                    audio_model.net[12:] if dominant == 'audio' else video_model.net[7:]).to(device)
                for p in frozen_head.parameters(): p.requires_grad = False
                with torch.no_grad():
                    feat = sym_model.attended_feature(
                        t_test_f.to(device), a_test_dev, v_test_dev, dominant)
                    logits = frozen_head(feat)
                    preds = logits.argmax(1).cpu().numpy()
                cell3 = accuracy_score(y_test, preds) * 100
                print(f"  Cell 3 (Symmetric, frozen head): {cell3:.2f}%")

                # ── Cell 4: Cross-attn, trainable head (standard symmetric) ──
                with torch.no_grad():
                    logits = sym_model.individual_pathway_logits(
                        t_test_f.to(device), a_test_dev, v_test_dev, dominant)
                    preds = logits.argmax(1).cpu().numpy()
                cell4 = accuracy_score(y_test, preds) * 100
                print(f"  Cell 4 (Symmetric, trainable head): {cell4:.2f}%")

                # ── Cell 2: No attn, trainable head — NEW TRAINING ──
                tph_model = TrainablePureHeadFusionMustard(
                    text_model, audio_model, video_model, ablate_dominant=dominant).to(device)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, tph_model.parameters()),
                                        lr=LR, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
                criterion = nn.CrossEntropyLoss()
                best_acc, trigger_times = 0.0, 0
                tph_ckpt = f"{CHECKPOINTS_DIR}/trainable_pure_head_ablation_mustard_{config_name}_{variant}_{dominant}_fold_{fold}.pth"

                for epoch in range(EPOCHS):
                    train_loss, train_acc = train_one_epoch(tph_model, train_loader, optimizer, criterion, epoch)
                    test_acc = evaluate(tph_model, test_loader)
                    scheduler.step()
                    if test_acc > best_acc:
                        best_acc, trigger_times = test_acc, 0
                        torch.save(tph_model.state_dict(), tph_ckpt)
                    else:
                        trigger_times += 1
                    if (epoch + 1) % 20 == 0:
                        print(f"    Epoch {epoch+1}: Train={train_acc:.2f}%  Test={test_acc:.2f}% (Best={best_acc:.2f}%)")
                    if trigger_times >= PATIENCE:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

                tph_model.load_state_dict(torch.load(tph_ckpt, map_location=device))
                cell2 = evaluate_pure_pathway(tph_model, a_test_dev, v_test_dev, y_test, dominant)
                print(f"  Cell 2 (No attn, trainable head): {cell2:.2f}%")

                print(f"\n  2x2 GRID — {config_name} [{variant}] dominant={dominant}, fold={fold}")
                print(f"                    Head frozen    Head trainable")
                print(f"  No cross-attn       {cell1:6.2f}%       {cell2:6.2f}%")
                print(f"  Cross-attn          {cell3:6.2f}%       {cell4:6.2f}%")

                attn_effect  = cell1 - cell3
                drift_effect = cell1 - cell2
                mechanism = 'attention' if attn_effect > drift_effect else 'classifier_drift'
                print(f"  Attention-alone effect  (Cell1 - Cell3): {attn_effect:+.2f}pp")
                print(f"  Classifier-drift effect (Cell1 - Cell2): {drift_effect:+.2f}pp")
                print(f"  -> Dominant collapse mechanism: {mechanism}")

                all_grid_rows.append({
                    'config': config_name, 'fold': fold, 'variant': variant, 'dominant': dominant,
                    'cell1_no_attn_frozen_head': cell1,
                    'cell2_no_attn_trainable_head': cell2,
                    'cell3_attn_frozen_head': cell3,
                    'cell4_attn_trainable_head': cell4,
                    'attention_effect_pp': attn_effect,
                    'classifier_drift_effect_pp': drift_effect,
                    'dominant_mechanism': mechanism,
                })

grid_df = pd.DataFrame(all_grid_rows)
grid_df.to_csv(f'{OUTPUT_DIR}/causal_decomposition_mustard_per_fold.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATE (mean ± std across folds)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"AGGREGATE (mean ± std across 5 folds)")
print(f"{'='*80}")

metric_cols = ['cell1_no_attn_frozen_head', 'cell2_no_attn_trainable_head',
               'cell3_attn_frozen_head', 'cell4_attn_trainable_head',
               'attention_effect_pp', 'classifier_drift_effect_pp']
summary_rows = []

for config_name in CONFIGS:
    for variant in VARIANTS:
        for dominant in DOMINANT_PATHWAYS:
            sub = grid_df[(grid_df.config == config_name) & (grid_df.variant == variant) & (grid_df.dominant == dominant)]
            if sub.empty:
                continue
            means = {c: sub[c].mean() for c in metric_cols}
            stds  = {c: sub[c].std()  for c in metric_cols}
            print(f"\n{config_name} [{variant}] dominant={dominant}:")
            print(f"  Cell1={means['cell1_no_attn_frozen_head']:.2f}±{stds['cell1_no_attn_frozen_head']:.2f}  "
                  f"Cell2={means['cell2_no_attn_trainable_head']:.2f}±{stds['cell2_no_attn_trainable_head']:.2f}  "
                  f"Cell3={means['cell3_attn_frozen_head']:.2f}±{stds['cell3_attn_frozen_head']:.2f}  "
                  f"Cell4={means['cell4_attn_trainable_head']:.2f}±{stds['cell4_attn_trainable_head']:.2f}")
            print(f"  Attention effect: {means['attention_effect_pp']:+.2f}pp   "
                  f"Classifier-drift effect: {means['classifier_drift_effect_pp']:+.2f}pp")
            row = {'config': config_name, 'variant': variant, 'dominant': dominant}
            for c in metric_cols:
                row[f'{c}_mean'] = means[c]
                row[f'{c}_std']  = stds[c]
            summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(f'{OUTPUT_DIR}/causal_decomposition_mustard_summary.csv', index=False)
with open(f'{OUTPUT_DIR}/causal_decomposition_mustard.json', 'w') as f:
    json.dump({'per_fold': all_grid_rows, 'summary': summary_rows}, f, indent=4, default=str)

print(f"\nSaved per-fold rows to {OUTPUT_DIR}/causal_decomposition_mustard_per_fold.csv")
print(f"Saved fold-aggregated summary to {OUTPUT_DIR}/causal_decomposition_mustard_summary.csv")
print(f"{'='*80}\n")
