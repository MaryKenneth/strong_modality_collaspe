"""
PATHWAY TESTING — SYMMETRIC & INVERTED ASYMMETRIC FUSION (MultiHUSE)
=====================================================================
Tests individual pathway performance for both architectures and both
backbone model sets (baseline and distilled(MAKD)), across all 4 combinations
and 5 folds.

2 MODEL_TYPE variants:
  baseline   — TextModel/BaselineAudioModel/BaselineVideoModel
  distilled  — TextModel(E5) or BERTStudent / AudioStudent / VideoStudent

Usage:
  Set MODEL_TYPE = 'baseline' or 'distilled'
  Run once per MODEL_TYPE — both architectures are tested in the same run.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import json
import copy

# CONFIGURATION — SET THIS FOR EACH RUN
MODEL_TYPE = 'distilled'   # ← CHANGE: 'baseline' or 'distilled'

# PATHS
SPLITS_DIR      = r'multihuse_dataset\splits_strict_v2'
FEATURES_DIR    = r'multihuse_dataset'
CHECKPOINTS_DIR = 'checkpoints'
OUTPUT_DIR      = f'pathway_results_{MODEL_TYPE}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Backbone checkpoint names per MODEL_TYPE ───────────────────────────────
# Baseline backbone checkpoints
BASELINE_BACKBONE = {
    'bert_audio_videomae': {
        'text_checkpoint':  'text_bert_text_embeddings',
        'text_type':        'text_model',   # TextModel (768)
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_checkpoint': 'video_videomae_visual_embeddings',
    },
    'bert_audio_pecore': {
        'text_checkpoint':  'text_bert_text_embeddings',
        'text_type':        'text_model',
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_checkpoint': 'video_pe_core_visual_embeddings',
    },
    'e5_audio_videomae': {
        'text_checkpoint':  'text_e5_text_embeddings',
        'text_type':        'text_model',
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_checkpoint': 'video_videomae_visual_embeddings',
    },
    'e5_audio_pecore': {
        'text_checkpoint':  'text_e5_text_embeddings',
        'text_type':        'text_model',
        'audio_checkpoint': 'audio_dasheng_audio_embeddings',
        'video_checkpoint': 'video_pe_core_visual_embeddings',
    },
}

# Distilled (MAKD) backbone checkpoints (E5 teacher → students)
DISTILLED_BACKBONE = {
    'bert_audio_videomae': {
        'text_checkpoint':  'multihuse_distill_bert_from_e5',
        'text_type':        'bert_student',
        'audio_checkpoint': 'multihuse_distill_audio_from_e5',
        'video_checkpoint': 'multihuse_distill_videomae_from_e5',
    },
    'bert_audio_pecore': {
        'text_checkpoint':  'multihuse_distill_bert_from_e5',
        'text_type':        'bert_student',
        'audio_checkpoint': 'multihuse_distill_audio_from_e5',
        'video_checkpoint': 'multihuse_distill_video_pe_from_e5',
    },
    'e5_audio_videomae': {
        'text_checkpoint':  'text_e5_text_embeddings',
        'text_type':        'e5_baseline',  # E5 is the teacher — no distillation
        'audio_checkpoint': 'multihuse_distill_audio_from_e5',
        'video_checkpoint': 'multihuse_distill_videomae_from_e5',
    },
    'e5_audio_pecore': {
        'text_checkpoint':  'text_e5_text_embeddings',
        'text_type':        'e5_baseline',
        'audio_checkpoint': 'multihuse_distill_audio_from_e5',
        'video_checkpoint': 'multihuse_distill_video_pe_from_e5',
    },
}

COMBINATIONS = [
    {
        'name':           'bert_audio_videomae',
        'text_features':  'bert_text_embeddings.csv',
        'text_dim':       768,
        'video_features': 'videomae_visual_embeddings.csv',
        'video_dim':      768,
    },
    {
        'name':           'bert_audio_pecore',
        'text_features':  'bert_text_embeddings.csv',
        'text_dim':       768,
        'video_features': 'pe_core_visual_embeddings.csv',
        'video_dim':      1024,
    },
    {
        'name':           'e5_audio_videomae',
        'text_features':  'e5_text_embeddings.csv',
        'text_dim':       1024,
        'video_features': 'videomae_visual_embeddings.csv',
        'video_dim':      768,
    },
    {
        'name':           'e5_audio_pecore',
        'text_features':  'e5_text_embeddings.csv',
        'text_dim':       1024,
        'video_features': 'pe_core_visual_embeddings.csv',
        'video_dim':      1024,
    },
]

ARCHITECTURES = ['symmetric', 'inverted_asymmetric']

AUDIO_DIM   = 1280
NUM_CLASSES = 5
LABEL_COLUMN = 'humor_style'
FUSION_DIM  = 512
SEED        = 999
BATCH_SIZE  = 32

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BACKBONE_CFG = DISTILLED_BACKBONE if MODEL_TYPE == 'distilled' else BASELINE_BACKBONE

print(f"\n{'='*80}")
print(f"PATHWAY TESTING — {MODEL_TYPE.upper()} MODELS")
print(f"{'='*80}")
print(f"Architectures : {ARCHITECTURES}")
print(f"Combinations  : {[c['name'] for c in COMBINATIONS]}")
print(f"Folds         : 1–5")
print(f"Tests/fold    : individual pathways (text/audio/video) + missing modality")
print(f"{'='*80}\n")


# ── BASELINE BACKBONE MODELS ─────────────────────────────────────────────
class TextModel(nn.Module):
    """Baseline text model (BERT 768 or E5 1024 → 512 → NUM_CLASSES)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

    def get_features(self, x):
        return self.net[:4](x)   # 512-dim

    def get_classifier(self):
        return self.net[4]      


class BaselineAudioModel(nn.Module):
    """Baseline Dasheng audio model (1280 → 1024 → 768 → 512 encoder)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUDIO_DIM, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),  # [0-3]
            nn.Linear(1024, 768),       nn.BatchNorm1d(768),  nn.ReLU(), nn.Dropout(0.3),  # [4-7]
            nn.Linear(768, 512),        nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.2),  # [8-11] ← feature
            nn.Linear(512, 256),        nn.ReLU(),            nn.Dropout(0.2),             # [12-14]
            nn.Linear(256, NUM_CLASSES)                                                    # [15]
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
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),  # [0-3]
            nn.Linear(1024, 512),       nn.ReLU(),            nn.Dropout(0.3),             # [4-6] ← feature
            nn.Linear(512, NUM_CLASSES)                                                    # [7]
        )

    def forward(self, x):
        return self.net(x)

    def get_features(self, x):
        return self.net[:7](x)   # 512-dim


# ── DISTILLED STUDENT MODELS ─────────────────────────────────────────────
class BERTStudent(nn.Module):
    """BERT student distilled from E5 (768 → 512 → 512 encoder)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def get_features(self, x):
        return self.encoder(x)   # 512-dim

    def get_classifier(self):
        return self.classifier   # Sequential(512→256→NUM_CLASSES)


class AudioStudent(nn.Module):
    """Audio student distilled from E5 (1280 → 1024 → 768 → 512 encoder)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(AUDIO_DIM, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 768),       nn.BatchNorm1d(768),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, 512),        nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def get_features(self, x):
        return self.encoder(x)   # 512-dim


class VideoStudent(nn.Module):
    """Video student distilled from E5 (input_dim → 1024 → 768 → 512 encoder)."""
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 768),       nn.BatchNorm1d(768),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, 512),        nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def get_features(self, x):
        return self.encoder(x)   # 512-dim


# BACKBONE FACTORY
def build_backbones(combo, fold):
    """Instantiate and load the correct backbone models for MODEL_TYPE."""
    bkp = BACKBONE_CFG[combo['name']]

    # Text
    text_type = bkp['text_type']
    if text_type == 'bert_student':
        txt = BERTStudent().to(device)
    else:
        txt = TextModel(combo['text_dim']).to(device)
    txt.load_state_dict(torch.load(
        f"{CHECKPOINTS_DIR}/{bkp['text_checkpoint']}_fold_{fold}.pth",
        map_location=device))
    txt.eval()

    # Audio
    if MODEL_TYPE == 'baseline':
        aud = BaselineAudioModel().to(device)
    else:
        aud = AudioStudent().to(device)
    aud.load_state_dict(torch.load(
        f"{CHECKPOINTS_DIR}/{bkp['audio_checkpoint']}_fold_{fold}.pth",
        map_location=device))
    aud.eval()

    # Video
    if MODEL_TYPE == 'baseline':
        vid = BaselineVideoModel(combo['video_dim']).to(device)
    else:
        vid = VideoStudent(combo['video_dim']).to(device)
    vid.load_state_dict(torch.load(
        f"{CHECKPOINTS_DIR}/{bkp['video_checkpoint']}_fold_{fold}.pth",
        map_location=device))
    vid.eval()

    return txt, aud, vid


# SHARED COMPONENT
class CrossModalAnchor(nn.Module):
    """Cross-attention: anchor attends to a 2-element context stack."""
    def __init__(self, d_model=FUSION_DIM):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, anchor, context):
        a_seq = anchor.unsqueeze(1)
        out, _ = self.cross_attn(query=a_seq, key=context, value=context)
        return self.norm(anchor + self.dropout(out.squeeze(1)))


# ==========================================================================
# FUSION ARCHITECTURES  (must match training scripts exactly)
class SymmetricTriModalFusion(nn.Module):
    def __init__(self, text_model, audio_model, video_model):
        super().__init__()
        self.text_backbone  = text_model
        self.audio_backbone = audio_model
        self.video_backbone = video_model
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters():
                p.requires_grad = False

        self.text_anchor  = CrossModalAnchor(FUSION_DIM)
        self.audio_anchor = CrossModalAnchor(FUSION_DIM)
        self.video_anchor = CrossModalAnchor(FUSION_DIM)

        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),
        )
        self.classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)  # SHARED

    def forward(self, t_feat, a_feat, v_feat):
        ctx_t = torch.stack([a_feat, v_feat], dim=1)
        ctx_a = torch.stack([t_feat, v_feat], dim=1)
        ctx_v = torch.stack([t_feat, a_feat], dim=1)
        out_t = self.text_anchor(t_feat,  ctx_t)
        out_a = self.audio_anchor(a_feat, ctx_a)
        out_v = self.video_anchor(v_feat, ctx_v)
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights  = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = (weights[:, i].unsqueeze(1) for i in range(3))
        fused = w_t * out_t + w_a * out_a + w_v * out_v
        return self.classifier(fused), weights

    @torch.no_grad()
    def forward_individual_pathway(self, t_feat, a_feat, v_feat, pathway):
        """
        Symmetric: route one attended feature through the SHARED classifier.
        The other two modalities' features are still used as context for attention.
        """
        ctx_t = torch.stack([a_feat, v_feat], dim=1)
        ctx_a = torch.stack([t_feat, v_feat], dim=1)
        ctx_v = torch.stack([t_feat, a_feat], dim=1)

        if pathway == 'text':
            out = self.text_anchor(t_feat, ctx_t)
        elif pathway == 'audio':
            out = self.audio_anchor(a_feat, ctx_a)
        elif pathway == 'video':
            out = self.video_anchor(v_feat, ctx_v)
        else:
            raise ValueError(f"Unknown pathway: {pathway}")
        return self.classifier(out)


class InvertedAsymmetricFusion(nn.Module):
    """
    MultiHUSE inverted: Text PURE | Audio+Video ATTEND.
    Per-pathway classifiers: frozen text_classifier + trainable audio/video classifiers.
    """
    def __init__(self, text_model, audio_model, video_model):
        super().__init__()
        self.text_backbone  = text_model
        self.audio_backbone = audio_model
        self.video_backbone = video_model
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters():
                p.requires_grad = False

        # Only weak modalities get anchors
        self.audio_anchor = CrossModalAnchor(FUSION_DIM)
        self.video_anchor = CrossModalAnchor(FUSION_DIM)

        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),
        )

        if hasattr(text_model, 'get_classifier'):
            self.text_classifier = copy.deepcopy(text_model.get_classifier())
        else:
            self.text_classifier = copy.deepcopy(text_model.net[4])
        for p in self.text_classifier.parameters():
            p.requires_grad = False

        # Audio & Video: new trainable heads
        self.audio_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)
        self.video_classifier  = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, t_feat, a_feat, v_feat):
        out_t = t_feat   # Text: PURE
        ctx_a = torch.stack([t_feat, v_feat], dim=1)
        ctx_v = torch.stack([t_feat, a_feat], dim=1)
        out_a = self.audio_anchor(a_feat, ctx_a)
        out_v = self.video_anchor(v_feat, ctx_v)
        gate_in = torch.cat([t_feat, a_feat, v_feat], dim=1)
        weights  = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = (weights[:, i].unsqueeze(1) for i in range(3))
        final = w_t * self.text_classifier(out_t) + \
                w_a * self.audio_classifier(out_a) + \
                w_v * self.video_classifier(out_v)
        return final, weights

    @torch.no_grad()
    def forward_individual_pathway(self, t_feat, a_feat, v_feat, pathway):
        """
        Inverted: Text uses its frozen backbone classifier (pure, no attention).
                  Audio/Video use their trained linear heads (with cross-attention context).
        """
        if pathway == 'text':
            return self.text_classifier(t_feat)
        elif pathway == 'audio':
            ctx_a = torch.stack([t_feat, v_feat], dim=1)
            out_a = self.audio_anchor(a_feat, ctx_a)
            return self.audio_classifier(out_a)
        elif pathway == 'video':
            ctx_v = torch.stack([t_feat, a_feat], dim=1)
            out_v = self.video_anchor(v_feat, ctx_v)
            return self.video_classifier(out_v)
        else:
            raise ValueError(f"Unknown pathway: {pathway}")


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


class TriModalDataset(Dataset):
    def __init__(self, t, a, v, y, fnames):
        self.t      = torch.tensor(t, dtype=torch.float32)
        self.a      = torch.tensor(a, dtype=torch.float32)
        self.v      = torch.tensor(v, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)
        self.fnames = fnames

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (self.t[idx], self.a[idx], self.v[idx],
                self.labels[idx], self.fnames[idx])


@torch.no_grad()
def extract_backbone_features(txt, aud, vid, X_t, X_a, X_v, batch_size=64):
    """Pre-extract 512-dim features once from all frozen backbones."""
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
    return (torch.cat(t_out), torch.cat(a_out), torch.cat(v_out))


# ==========================================================================
# TESTING FUNCTIONS
def build_and_load_fusion(architecture, txt, aud, vid, combo_name, fold):
    """Build fusion model and load its trained checkpoint."""
    if architecture == 'symmetric':
        model = SymmetricTriModalFusion(txt, aud, vid).to(device)
        ckpt  = f"{CHECKPOINTS_DIR}/symmetric_{MODEL_TYPE}_{combo_name}_fold_{fold}.pth"
    else:
        model = InvertedAsymmetricFusion(txt, aud, vid).to(device)
        ckpt  = f"{CHECKPOINTS_DIR}/inverted_asymmetric_{MODEL_TYPE}_{combo_name}_fold_{fold}.pth"

    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        print(f"  ✓ Loaded: {ckpt}")
        return model
    except FileNotFoundError:
        print(f"  ✗ Checkpoint not found: {ckpt}")
        return None


def run_pathway_test(model, loader, pathway):
    """Run forward_individual_pathway for one pathway, return accuracy."""
    all_preds, all_trues = [], []
    with torch.no_grad():
        for t_b, a_b, v_b, y_b, _ in loader:
            t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
            logits = model.forward_individual_pathway(t_b, a_b, v_b, pathway=pathway)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_trues.extend(y_b.numpy())
    return accuracy_score(all_trues, all_preds) * 100, all_trues, all_preds


def run_missing_modality_test(model, loader, pathway):
    """
    Zero-out all modalities except the target one, route through forward_individual_pathway.
    This tests robustness: can the pathway still classify when the other two are absent?
    """
    all_preds, all_trues = [], []
    with torch.no_grad():
        for t_b, a_b, v_b, y_b, _ in loader:
            t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
            t_in = t_b if pathway == 'text'  else torch.zeros_like(t_b)
            a_in = a_b if pathway == 'audio' else torch.zeros_like(a_b)
            v_in = v_b if pathway == 'video' else torch.zeros_like(v_b)
            logits = model.forward_individual_pathway(t_in, a_in, v_in, pathway=pathway)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_trues.extend(y_b.numpy())
    return accuracy_score(all_trues, all_preds) * 100


# ==========================================================================
# MAIN
all_results = {arch: {} for arch in ARCHITECTURES}   # arch → combo → list of fold dicts

for combo in COMBINATIONS:
    combo_name = combo['name']
    print(f"\n{'='*80}")
    print(f"COMBINATION: {combo_name.upper()}")
    print(f"{'='*80}")

    for arch in ARCHITECTURES:
        all_results[arch][combo_name] = []
        print(f"\n  ── {arch.upper()} ──")

        for fold in range(1, 6):
            print(f"\n  FOLD {fold}/5")

            # ── Load test data ──────────────────────────────────────────
            _, test_files = load_fold_split(fold)

            X_t, y_test, fnames = load_features(
                combo['text_features'], test_files, 'text_feat_')
            X_a, _, _ = load_features(
                'dasheng_audio_embeddings.csv', test_files, 'audio_feat_')
            X_v, _, _ = load_features(
                combo['video_features'], test_files, 'visual_feat_')

            le = LabelEncoder()
            y_enc = le.fit_transform(y_test)

            # ── Load backbones & extract features ───────────────────────
            try:
                txt, aud, vid = build_backbones(combo, fold)
            except FileNotFoundError as e:
                print(f"  ✗ Backbone not found: {e}")
                continue

            t_feat, a_feat, v_feat = extract_backbone_features(
                txt, aud, vid, X_t, X_a, X_v)

            ds     = TriModalDataset(t_feat.numpy(), a_feat.numpy(), v_feat.numpy(), y_enc, fnames)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

            # ── Build & load fusion model ────────────────────────────────
            model = build_and_load_fusion(arch, txt, aud, vid, combo_name, fold)
            if model is None:
                continue

            fold_result = {'fold': fold, 'pathway': {}, 'missing': {}}

            # ── Individual pathway tests ─────────────────────────────────
            print(f"  Individual pathways (all modalities present as context):")
            for pathway in ['text', 'audio', 'video']:
                acc, trues, preds = run_pathway_test(model, loader, pathway)
                fold_result['pathway'][pathway] = acc
                print(f"    {pathway.capitalize():6s} pathway: {acc:.2f}%")

            # ── Missing modality tests ───────────────────────────────────
            print(f"  Missing modality (zero out all but target):")
            for pathway in ['text', 'audio', 'video']:
                acc = run_missing_modality_test(model, loader, pathway)
                fold_result['missing'][pathway] = acc
                label = f"{pathway}_only"
                print(f"    {label:12s}: {acc:.2f}%")

            all_results[arch][combo_name].append(fold_result)


# ==========================================================================
# AGGREGATE RESULTS
print(f"\n\n{'='*80}")
print(f"COMPREHENSIVE RESULTS — {MODEL_TYPE.upper()} MODELS")
print(f"{'='*80}")

summary = {}

for combo in COMBINATIONS:
    combo_name = combo['name']
    print(f"\n{'─'*80}")
    print(f"COMBINATION: {combo_name.upper()}")
    print(f"{'─'*80}")
    summary[combo_name] = {}

    for arch in ARCHITECTURES:
        folds = all_results[arch][combo_name]
        if not folds:
            print(f"\n  {arch.upper()}: no results")
            continue

        print(f"\n  {arch.upper()}:")
        summary[combo_name][arch] = {}

        for test_type in ['pathway', 'missing']:
            label = 'Individual pathway' if test_type == 'pathway' else 'Missing modality'
            print(f"    {label}:")
            for pathway in ['text', 'audio', 'video']:
                accs = [f[test_type][pathway] for f in folds if pathway in f[test_type]]
                if accs:
                    mean, std = np.mean(accs), np.std(accs)
                    print(f"      {pathway.capitalize():6s}: {mean:.2f}% (±{std:.2f}%)  "
                          f"folds: {[f'{a:.1f}' for a in accs]}")
                    summary[combo_name][arch][f'{test_type}_{pathway}'] = {
                        'mean': mean, 'std': std}

# ── Cross-architecture comparison ──────────────────────────────────────────
print(f"\n\n{'='*80}")
print(f"CROSS-ARCHITECTURE COMPARISON — {MODEL_TYPE.upper()}")
print(f"{'='*80}")
print(f"\n{'Combination':30s}  {'Pathway':8s}  {'Symmetric':>12s}  {'Inverted':>12s}  {'Δ (Inv-Sym)':>12s}")
print(f"{'─'*30}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")

for combo in COMBINATIONS:
    cname = combo['name']
    if 'symmetric' not in summary.get(cname, {}) or 'inverted_asymmetric' not in summary.get(cname, {}):
        continue
    sym = summary[cname]['symmetric']
    inv = summary[cname]['inverted_asymmetric']
    for pathway in ['text', 'audio', 'video']:
        key = f'pathway_{pathway}'
        if key in sym and key in inv:
            s_m, i_m = sym[key]['mean'], inv[key]['mean']
            delta = i_m - s_m
            sign  = '+' if delta >= 0 else ''
            print(f"{cname:30s}  {pathway.capitalize():8s}  {s_m:>12.2f}%  {i_m:>12.2f}%  {sign}{delta:>+11.2f}%")
    print()

# ── Save results ────────────────────────────────────────────────────────────
out_path = f'{OUTPUT_DIR}/pathway_results_{MODEL_TYPE}.json'
with open(out_path, 'w') as f:
    json.dump({'all_results': all_results, 'summary': summary}, f, indent=4, default=str)

summary_path = f'{OUTPUT_DIR}/pathway_summary_{MODEL_TYPE}.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4, default=str)

print(f"\n{'='*80}")
print(f"TESTING COMPLETE — {MODEL_TYPE.upper()} MODELS")
print(f"{'='*80}")
print(f"Detailed results : {out_path}")
print(f"Summary          : {summary_path}")
print(f"{'='*80}\n")
