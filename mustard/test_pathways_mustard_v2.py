"""
PATHWAY TESTING – SYMMETRIC vs INVERTED ASYMMETRIC (MUStARD)
=============================================================
Tests individual pathway performance for both fusion architectures
across all 5 folds, covering:

  BASELINE  (4 configs): BERT/E5 × VideoMAE/PE-Core — TextBaseline backbones
  MAKD   (2 configs): BERT-cascade / E5-cascade  — SimplifiedTextStudent backbones

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import pickle
import json
import os
import copy
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MUSTARD_DIR       = r'mustard_dataset'
FEATURES_DIR      = f'{MUSTARD_DIR}/mustard_features_pkl_FIXED'
SPLITS_FILE       = f'{MUSTARD_DIR}/mustard_dataset_master/data/split_indices.p'
SARCASM_DATA_PATH = f'{MUSTARD_DIR}/mustard_dataset_master/data/sarcasm_data.json'
CHECKPOINTS_DIR   = 'checkpoints'
OUTPUT_DIR        = 'pathway_results_mustard'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/context/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/context/not_sarcasm',
}
PUNCHLINE_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/utterances/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/utterances/not_sarcasm',
}

# ── 4 baseline combinations (TextBaseline backbones) ─────────────────────────
BASELINE_COMBINATIONS = [
    {
        'name': 'bert_videomae',
        'text_type':         'bert_context_punchline',
        'text_checkpoint':   'mustard_text_bert_context_punchline',
        'text_dim':          1536,
        'text_model_type':   'baseline',
        'audio_checkpoint':  'mustard_audio_dasheng',
        'audio_dim':         1280,
        'video_type':        'videomae',
        'video_checkpoint':  'mustard_video_videomae',
        'video_dim':         768,
    },
    {
        'name': 'bert_pe_core',
        'text_type':         'bert_context_punchline',
        'text_checkpoint':   'mustard_text_bert_context_punchline',
        'text_dim':          1536,
        'text_model_type':   'baseline',
        'audio_checkpoint':  'mustard_audio_dasheng',
        'audio_dim':         1280,
        'video_type':        'pe_core',
        'video_checkpoint':  'mustard_video_pe_core',
        'video_dim':         1024,
    },
    {
        'name': 'e5_videomae',
        'text_type':         'e5_context_punchline',
        'text_checkpoint':   'mustard_text_e5_context_punchline',
        'text_dim':          2048,
        'text_model_type':   'baseline',
        'audio_checkpoint':  'mustard_audio_dasheng',
        'audio_dim':         1280,
        'video_type':        'videomae',
        'video_checkpoint':  'mustard_video_videomae',
        'video_dim':         768,
    },
    {
        'name': 'e5_pe_core',
        'text_type':         'e5_context_punchline',
        'text_checkpoint':   'mustard_text_e5_context_punchline',
        'text_dim':          2048,
        'text_model_type':   'baseline',
        'audio_checkpoint':  'mustard_audio_dasheng',
        'audio_dim':         1280,
        'video_type':        'pe_core',
        'video_checkpoint':  'mustard_video_pe_core',
        'video_dim':         1024,
    },
]

# ── 2 MAKD (Distillation) combinations (SimplifiedTextStudent backbones) ─────────────────
CASCADE_COMBINATIONS = [
    {
        'name': 'bert_cascade',
        'text_type':         'bert_context_punchline',
        'text_checkpoint':   'mustard_cascade_bert_audio_e5',
        'text_dim':          1536,
        'text_model_type':   'cascade',
        'audio_checkpoint':  'mustard_audio_dasheng',
        'audio_dim':         1280,
        'video_type':        'pe_core',
        'video_checkpoint':  'mustard_video_pe_core',
        'video_dim':         1024,
    },
    {
        'name': 'e5_cascade',
        'text_type':         'e5_context_punchline',
        'text_checkpoint':   'mustard_cascade_e5_audio_video',
        'text_dim':          2048,
        'text_model_type':   'cascade',
        'audio_checkpoint':  'mustard_audio_dasheng',
        'audio_dim':         1280,
        'video_type':        'pe_core',
        'video_checkpoint':  'mustard_video_pe_core',
        'video_dim':         1024,
    },
]

ALL_COMBINATIONS   = BASELINE_COMBINATIONS + CASCADE_COMBINATIONS
ARCHITECTURES      = ['symmetric', 'inverted_asymmetric']

NUM_CLASSES   = 2
FUSION_DIM    = 512
BATCH_SIZE    = 64
SEED          = 999

# MUStARD unimodal baselines (for delta reporting)
UNIMODAL_BASELINES = {
    'audio':          78.0,
    'video':          77.0,
    'text_baseline':  70.0,
    'text_cascade':   72.0,
}

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"\n{'='*70}")
print(f"PATHWAY TESTING — MUStARD (SYMMETRIC vs INVERTED ASYMMETRIC)")
print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE MODEL DEFINITIONS 
# ─────────────────────────────────────────────────────────────────────────────

class TextBaseline(nn.Module):
    """Baseline text model (E5 or BERT, context+punchline)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 3 and return_features:
                features = x.clone()
        return (x, features) if return_features else x


class SimplifiedTextStudent(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim > 1500:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(1024, 768),       nn.BatchNorm1d(768),  nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(768, 512),        nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.2),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits   = self.classifier(features)
        return (logits, features) if return_features else logits


class AudioTeacher(nn.Module):
    """Dasheng audio teacher (78%)."""
    def __init__(self, input_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 768),       nn.BatchNorm1d(768),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, 512),        nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
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
    """Video baseline (PE-Core or VideoMAE)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 6 and return_features:
                features = x.clone()
        return (x, features) if return_features else x


# ─────────────────────────────────────────────────────────────────────────────
# FUSION ARCHITECTURES WITH PATHWAY SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAnchor(nn.Module):
    def __init__(self, d_model=FUSION_DIM):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True)  # ← 8 heads
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, anchor, context):
        a_seq = anchor.unsqueeze(1)
        out, _ = self.cross_attn(query=a_seq, key=context, value=context)
        return self.norm(anchor + self.dropout(out.squeeze(1)))


class SymmetricFusionWithPathways(nn.Module):
    def __init__(self, text_backbone, audio_backbone, video_backbone):
        super().__init__()
        self.text_backbone  = text_backbone
        self.audio_backbone = audio_backbone
        self.video_backbone = video_backbone
        for b in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in b.parameters():
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
        self.classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, text_feat, audio_feat, video_feat):
        """All inputs are pre-extracted 512-dim features."""
        out_t = self.text_anchor( text_feat,  torch.stack([audio_feat, video_feat], dim=1))
        out_a = self.audio_anchor(audio_feat, torch.stack([text_feat,  video_feat], dim=1))
        out_v = self.video_anchor(video_feat, torch.stack([text_feat,  audio_feat], dim=1))

        gate_in = torch.cat([text_feat, audio_feat, video_feat], dim=1)
        weights = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = (weights[:, i].unsqueeze(1) for i in range(3))
        fused = w_t * out_t + w_a * out_a + w_v * out_v
        return self.classifier(fused), weights

    def forward_individual_pathway(self, text_feat, audio_feat, video_feat, pathway: str):
        """Isolate a single pathway through its anchor then shared classifier."""
        if pathway == 'text':
            out = self.text_anchor( text_feat,  torch.stack([audio_feat, video_feat], dim=1))
        elif pathway == 'audio':
            out = self.audio_anchor(audio_feat, torch.stack([text_feat,  video_feat], dim=1))
        elif pathway == 'video':
            out = self.video_anchor(video_feat, torch.stack([text_feat,  audio_feat], dim=1))
        else:
            raise ValueError(f"Unknown pathway: {pathway}")
        return self.classifier(out)


class InvertedAsymmetricWithPathways(nn.Module):
    def __init__(self, text_backbone, audio_backbone, video_backbone):
        super().__init__()
        self.text_backbone  = text_backbone
        self.audio_backbone = audio_backbone
        self.video_backbone = video_backbone
        for b in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in b.parameters():
                p.requires_grad = False

        # Only TEXT 
        self.text_anchor = CrossModalAnchor(FUSION_DIM)

        # gate_net WITH BatchNorm
        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),
        )

        # Frozen classifiers for strong modalities
        self.audio_classifier = copy.deepcopy(audio_backbone.net[12:])   # 512→256→2
        self.video_classifier  = copy.deepcopy(video_backbone.net[7:])   # 512→2
        for p in self.audio_classifier.parameters():
            p.requires_grad = False
        for p in self.video_classifier.parameters():
            p.requires_grad = False

        # Trainable text classifier head
        self.text_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def forward(self, text_feat, audio_feat, video_feat):
        """All inputs are pre-extracted 512-dim features."""
        out_audio = audio_feat    # PURE — no attention
        out_video = video_feat    # PURE — no attention
        out_text  = self.text_anchor(text_feat, torch.stack([audio_feat, video_feat], dim=1))

        gate_in = torch.cat([text_feat, audio_feat, video_feat], dim=1)
        weights = torch.softmax(self.gate_net(gate_in), dim=1)
        w_t, w_a, w_v = (weights[:, i].unsqueeze(1) for i in range(3))

        final = (w_t * self.text_classifier(out_text) +
                 w_a * self.audio_classifier(out_audio) +
                 w_v * self.video_classifier(out_video))
        return final, weights

    def forward_individual_pathway(self, text_feat, audio_feat, video_feat, pathway: str):
        """Isolate a single pathway through its own classifier."""
        if pathway == 'text':
            out = self.text_anchor(text_feat, torch.stack([audio_feat, video_feat], dim=1))
            return self.text_classifier(out)
        elif pathway == 'audio':
            return self.audio_classifier(audio_feat)   # RAW — frozen classifier
        elif pathway == 'video':
            return self.video_classifier(video_feat)   # RAW — frozen classifier
        else:
            raise ValueError(f"Unknown pathway: {pathway}")


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
    keys     = {'bert': 'text_bert', 'e5': 'text_e5', 'dasheng': 'audio_dasheng',
                'videomae': 'video_mae', 'pe_core': 'video_pe'}
    defaults = {'bert': 768, 'e5': 1024, 'dasheng': 1280, 'videomae': 768, 'pe_core': 1024}
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        feat = data.get(keys[feature_type], np.zeros((1, defaults[feature_type])))
        return feat[0] if len(feat.shape) > 1 else feat
    except:
        return np.zeros(defaults[feature_type])


def prepare_features(fold_indices, combination, sarcasm_data, identifiers):
    """Load raw features (before backbone encoding) for all samples in fold."""
    X_text, X_audio, X_video, y_list, fnames = [], [], [], [], []
    text_base_type = 'bert' if 'bert' in combination['text_type'] else 'e5'

    for idx in fold_indices:
        if idx >= len(identifiers):
            continue
        identifier = identifiers[idx]
        label      = 1 if sarcasm_data[identifier]['sarcasm'] else 0
        split_key  = 'sarcasm' if label == 1 else 'not_sarcasm'
        ctx_path   = Path(CONTEXT_PATHS[split_key])  / f"{identifier}.pkl"
        punch_path = Path(PUNCHLINE_PATHS[split_key]) / f"{identifier}.pkl"

        text_feat = np.concatenate([
            load_features_from_pkl(ctx_path,   text_base_type),
            load_features_from_pkl(punch_path, text_base_type),
        ])
        X_text.append(text_feat)
        X_audio.append(load_features_from_pkl(punch_path, 'dasheng'))
        X_video.append(load_features_from_pkl(punch_path, combination['video_type']))
        y_list.append(label)
        fnames.append(identifier)

    return (np.array(X_text,  dtype=np.float32),
            np.array(X_audio, dtype=np.float32),
            np.array(X_video, dtype=np.float32),
            np.array(y_list), fnames)


class FusionDataset(Dataset):
    """Stores pre-extracted 512-dim features."""
    def __init__(self, t, a, v, y, fnames):
        self.t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        self.a = a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)
        self.v = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
        self.fnames = fnames

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return self.t[i], self.a[i], self.v[i], self.y[i], self.fnames[i]


@torch.no_grad()
def extract_backbone_features(text_model, audio_model, video_model,
                               text_data, audio_data, video_data, batch_size=64):
    """Pre-extract 512-dim features from all frozen backbones once."""
    text_model.eval(); audio_model.eval(); video_model.eval()
    n = len(text_data)
    text_feats, audio_feats, video_feats = [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end - start == 1 and end < n:
            end = min(start + 2, n)

        t_b = torch.tensor(text_data[start:end],  dtype=torch.float32).to(device)
        a_b = torch.tensor(audio_data[start:end], dtype=torch.float32).to(device)
        v_b = torch.tensor(video_data[start:end], dtype=torch.float32).to(device)

        _, t_feat = text_model( t_b, return_features=True)
        _, a_feat = audio_model(a_b, return_features=True)
        _, v_feat = video_model(v_b, return_features=True)

        text_feats.append(t_feat.cpu())
        audio_feats.append(a_feat.cpu())
        video_feats.append(v_feat.cpu())

    return (torch.cat(text_feats,  dim=0),
            torch.cat(audio_feats, dim=0),
            torch.cat(video_feats, dim=0))


# ─────────────────────────────────────────────────────────────────────────────
# TESTING FUNCTIONS  (operate on pre-extracted 512-dim feature loaders)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_full_fusion(model, loader):
    model.eval()
    all_preds, all_labels, all_weights = [], [], []
    for t_b, a_b, v_b, y_b, _ in loader:
        t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
        logits, weights = model(t_b, a_b, v_b)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y_b.numpy())
        all_weights.append(weights.cpu().numpy())
    acc   = accuracy_score(all_labels, all_preds) * 100
    avg_w = np.concatenate(all_weights).mean(axis=0)
    return acc, avg_w


@torch.no_grad()
def test_individual_pathways(model, loader):
    model.eval()
    results = {}
    for pathway in ['text', 'audio', 'video']:
        all_preds, all_labels = [], []
        for t_b, a_b, v_b, y_b, _ in loader:
            t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
            logits = model.forward_individual_pathway(t_b, a_b, v_b, pathway)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_b.numpy())
        results[pathway] = accuracy_score(all_labels, all_preds) * 100
    return results


@torch.no_grad()
def test_missing_modalities(model, loader):
    model.eval()
    scenarios = {
        'text_only':   (True,  False, False),
        'audio_only':  (False, True,  False),
        'video_only':  (False, False, True),
        'text+audio':  (True,  True,  False),
        'text+video':  (True,  False, True),
        'audio+video': (False, True,  True),
    }
    results = {}
    for name, (kt, ka, kv) in scenarios.items():
        all_preds, all_labels = [], []
        for t_b, a_b, v_b, y_b, _ in loader:
            t_b, a_b, v_b = t_b.to(device), a_b.to(device), v_b.to(device)
            t_in = t_b if kt else torch.zeros_like(t_b)
            a_in = a_b if ka else torch.zeros_like(a_b)
            v_in = v_b if kv else torch.zeros_like(v_b)
            logits, _ = model(t_in, a_in, v_in)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_b.numpy())
        results[name] = accuracy_score(all_labels, all_preds) * 100
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_text_backbone(combination, fold):
    """Load the appropriate text backbone (baseline or cascade)."""
    dim    = combination['text_dim']
    ckpt   = f"{CHECKPOINTS_DIR}/{combination['text_checkpoint']}_fold_{fold}.pth"
    mtype  = combination['text_model_type']

    model = (TextBaseline(dim) if mtype == 'baseline'
             else SimplifiedTextStudent(dim))
    model = model.to(device)
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model
    except FileNotFoundError:
        print(f"  ✗ Text backbone checkpoint not found: {ckpt}")
        return None


def load_fusion_model(combination, fold, architecture, text_backbone,
                      audio_backbone, video_backbone):
    """
    Instantiate the fusion architecture and load saved checkpoint.
    Checkpoint names match what each training script saves.
    """
    config_name = combination['name']
    mtype       = combination['text_model_type']   # 'baseline' or 'cascade'

    if architecture == 'symmetric':
        model = SymmetricFusionWithPathways(
            text_backbone, audio_backbone, video_backbone).to(device)
        ckpt = (f"{CHECKPOINTS_DIR}/fusion_symmetric_{mtype}_{config_name}_fold_{fold}.pth")
    else:
        model = InvertedAsymmetricWithPathways(
            text_backbone, audio_backbone, video_backbone).to(device)
        ckpt = (f"{CHECKPOINTS_DIR}/fusion_inverted_{mtype}_{config_name}_fold_{fold}.pth")

    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  ✓ Loaded: {ckpt}")
        return model
    except FileNotFoundError:
        print(f"  ✗ Fusion checkpoint not found: {ckpt}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

splits_data  = load_pickle(SPLITS_FILE)
with open(SARCASM_DATA_PATH, 'r') as f:
    sarcasm_data = json.load(f)
identifiers = list(sarcasm_data.keys())
print(f"Total samples: {len(identifiers)}\n")

# {combo_name: {architecture: {fold_data}}}
all_results = {c['name']: {a: [] for a in ARCHITECTURES} for c in ALL_COMBINATIONS}

for combination in ALL_COMBINATIONS:
    cname  = combination['name']
    mtype  = combination['text_model_type']
    text_baseline_key = 'text_cascade' if mtype == 'cascade' else 'text_baseline'

    print(f"\n{'='*80}")
    print(f"COMBINATION: {cname.upper()}  [{mtype} text backbone]")
    print(f"{'='*80}")

    for architecture in ARCHITECTURES:
        print(f"\n  ARCHITECTURE: {architecture.upper()}")

        fold_full_accs, fold_pathway_results, fold_missing_results = [], [], []

        for fold in range(5):
            print(f"\n    ── Fold {fold} ──")

            # Load raw features (test fold only for evaluation)
            test_idx = splits_data[fold][1]
            X_text, X_audio, X_video, y_test, te_fnames = prepare_features(
                test_idx, combination, sarcasm_data, identifiers)

            # Load frozen backbones
            text_model = load_text_backbone(combination, fold)
            if text_model is None:
                continue

            audio_model = AudioTeacher(combination['audio_dim']).to(device)
            audio_model.load_state_dict(torch.load(
                f"{CHECKPOINTS_DIR}/{combination['audio_checkpoint']}_fold_{fold}.pth",
                map_location=device))

            video_model = VideoBaseline(combination['video_dim']).to(device)
            video_model.load_state_dict(torch.load(
                f"{CHECKPOINTS_DIR}/{combination['video_checkpoint']}_fold_{fold}.pth",
                map_location=device))

            # Pre-extract 512-dim features
            text_feats, audio_feats, video_feats = extract_backbone_features(
                text_model, audio_model, video_model,
                X_text, X_audio, X_video)

            test_ds     = FusionDataset(text_feats, audio_feats, video_feats,
                                         y_test, te_fnames)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            # Load fusion model
            fusion = load_fusion_model(
                combination, fold, architecture,
                text_model, audio_model, video_model)
            if fusion is None:
                continue

            # ── Tests ────────────────────────────────────────────────────────
            full_acc, avg_w = test_full_fusion(fusion, test_loader)
            pathway_res     = test_individual_pathways(fusion, test_loader)
            missing_res     = test_missing_modalities(fusion, test_loader)

            fold_full_accs.append(full_acc)
            fold_pathway_results.append(pathway_res)
            fold_missing_results.append(missing_res)

            text_base = UNIMODAL_BASELINES[text_baseline_key]
            print(f"      Full fusion: {full_acc:.2f}%  "
                  f"W[T={avg_w[0]:.3f} A={avg_w[1]:.3f} V={avg_w[2]:.3f}]")
            for p, base in [('text', text_base),
                             ('audio', UNIMODAL_BASELINES['audio']),
                             ('video', UNIMODAL_BASELINES['video'])]:
                delta = pathway_res[p] - base
                sym   = '↑' if delta >= 0 else '↓'
                print(f"      {p.capitalize():<6} pathway: {pathway_res[p]:.2f}%  "
                      f"(baseline={base:.1f}%  {sym}{abs(delta):.2f}%)")

        # Aggregate across folds
        if fold_pathway_results:
            all_results[cname][architecture] = {
                'full_fusion': {
                    'mean': np.mean(fold_full_accs),
                    'std':  np.std(fold_full_accs),
                },
                'pathways': {
                    p: {'mean': np.mean([f[p] for f in fold_pathway_results]),
                        'std':  np.std([f[p]  for f in fold_pathway_results])}
                    for p in ['text', 'audio', 'video']
                },
                'missing': {
                    s: np.mean([f[s] for f in fold_missing_results])
                    for s in fold_missing_results[0].keys()
                }
            }

            r = all_results[cname][architecture]
            print(f"\n    {architecture.upper()} — {cname} (mean over folds):")
            print(f"      Full fusion: {r['full_fusion']['mean']:.2f}% "
                  f"± {r['full_fusion']['std']:.2f}%")
            for p in ['audio', 'video', 'text']:
                pr = r['pathways'][p]
                print(f"      {p.capitalize():<6}: {pr['mean']:.2f}% ± {pr['std']:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-ARCHITECTURE COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"CROSS-ARCHITECTURE COMPARISON — ALL COMBINATIONS")
print(f"{'='*80}")

for combination in ALL_COMBINATIONS:
    cname = combination['name']
    mtype = combination['text_model_type']
    sym   = all_results[cname].get('symmetric', {})
    inv   = all_results[cname].get('inverted_asymmetric', {})
    if not sym or not inv or 'pathways' not in sym or 'pathways' not in inv:
        continue

    text_base = UNIMODAL_BASELINES['text_cascade' if mtype == 'cascade' else 'text_baseline']

    print(f"\n{cname.upper()}  [{mtype}]")
    print(f"{'Metric':<24} {'Symmetric':>12} {'Inverted':>12} {'Δ':>10}")
    print(f"{'─'*62}")

    s_full = sym['full_fusion']['mean']
    i_full = inv['full_fusion']['mean']
    print(f"{'Full Fusion':<24} {s_full:>10.2f}%  {i_full:>10.2f}%  {i_full-s_full:>+8.2f}%")
    print(f"{'─'*62}")

    for p, base in [('audio', UNIMODAL_BASELINES['audio']),
                    ('video', UNIMODAL_BASELINES['video']),
                    ('text',  text_base)]:
        s_acc = sym['pathways'][p]['mean']
        i_acc = inv['pathways'][p]['mean']
        print(f"  {(p.capitalize()+' pathway'):<22} {s_acc:>10.2f}%  {i_acc:>10.2f}%  "
              f"{i_acc-s_acc:>+8.2f}%  (baseline={base:.1f}%)")

    print(f"\n  Missing Modality (mean over folds):")
    for scenario in ['audio_only', 'video_only', 'text_only',
                     'text+audio', 'text+video', 'audio+video']:
        s_a = sym['missing'].get(scenario, 0)
        i_a = inv['missing'].get(scenario, 0)
        print(f"    {scenario:<18}: Sym={s_a:.2f}%  Inv={i_a:.2f}%  Δ={i_a-s_a:+.2f}%")


# Save results
out_path = f"{OUTPUT_DIR}/pathway_results_mustard.json"
safe = {}
for cname, arch_dict in all_results.items():
    safe[cname] = {}
    for arch, data in arch_dict.items():
        safe[cname][arch] = data if data else {}
with open(out_path, 'w') as f:
    json.dump(safe, f, indent=4, default=float)
print(f"\n✓ Results saved to: {out_path}")
