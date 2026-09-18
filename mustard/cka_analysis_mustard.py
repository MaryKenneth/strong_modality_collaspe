"""
CKA REPRESENTATIONAL SIMILARITY — MUStARD
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pickle
import json
import os
import copy
from pathlib import Path

MUSTARD_DIR       = r'mustard_dataset'
FEATURES_DIR      = f'{MUSTARD_DIR}/mustard_features_pkl_FIXED'
SPLITS_FILE       = f'{MUSTARD_DIR}/mustard_dataset_master/data/split_indices.p'
SARCASM_DATA_PATH = f'{MUSTARD_DIR}/mustard_dataset_master/data/sarcasm_data.json'
CHECKPOINTS_DIR   = 'checkpoints'
OUTPUT_DIR        = 'cka_results_mustard'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/context/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/context/not_sarcasm',
}
PUNCHLINE_PATHS = {
    'sarcasm':     f'{FEATURES_DIR}/utterances/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/utterances/not_sarcasm',
}

# Each entry carries BOTH the baseline text checkpoint stem and the MAKD
# (distilled) text checkpoint stem — audio/video checkpoints are shared.
CONFIGS = {
    'bert_videomae': {
        'name':                   'BERT + Audio + VideoMAE',
        'text_type':              'bert_context_punchline',
        'text_dim':               1536,
        'baseline_text_checkpoint': 'mustard_text_bert_context_punchline',
        'cascade_text_checkpoint':  'mustard_cascade_bert_audio_e5',
        'video_type':             'videomae',
        'video_checkpoint':       'mustard_video_videomae',
        'video_dim':              768,
    },
    'bert_pe_core': {
        'name':                   'BERT + Audio + Video_PE',
        'text_type':              'bert_context_punchline',
        'text_dim':               1536,
        'baseline_text_checkpoint': 'mustard_text_bert_context_punchline',
        'cascade_text_checkpoint':  'mustard_cascade_bert_audio_e5',
        'video_type':             'pe_core',
        'video_checkpoint':       'mustard_video_pe_core',
        'video_dim':              1024,
    },
    'e5_videomae': {
        'name':                   'E5 + Audio + VideoMAE',
        'text_type':              'e5_context_punchline',
        'text_dim':               2048,
        'baseline_text_checkpoint': 'mustard_text_e5_context_punchline',
        'cascade_text_checkpoint':  'mustard_cascade_e5_audio_video',
        'video_type':             'videomae',
        'video_checkpoint':       'mustard_video_videomae',
        'video_dim':              768,
    },
    'e5_pe_core': {
        'name':                   'E5 + Audio + Video_PE',
        'text_type':              'e5_context_punchline',
        'text_dim':               2048,
        'baseline_text_checkpoint': 'mustard_text_e5_context_punchline',
        'cascade_text_checkpoint':  'mustard_cascade_e5_audio_video',
        'video_type':             'pe_core',
        'video_checkpoint':       'mustard_video_pe_core',
        'video_dim':              1024,
    },
}

NUM_CLASSES = 2
FUSION_DIM  = 512
AUDIO_DIM   = 1280
SEED        = 999

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
class AudioTeacher(nn.Module):
    """Dasheng audio teacher (78%) — never distilled, shared across variants."""
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
    """Video baseline (PE-Core or VideoMAE) — never distilled, shared across variants."""
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
    """Baseline (non-distilled) text model — BASELINE variant only."""
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
    """distilled text student — distilled variant only."""
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

    def attended_feature(self, t_feat, a_feat, v_feat, pathway):
        if pathway == 'text':
            return self.text_anchor(t_feat, torch.stack([a_feat, v_feat], dim=1))
        elif pathway == 'audio':
            return self.audio_anchor(a_feat, torch.stack([t_feat, v_feat], dim=1))
        elif pathway == 'video':
            return self.video_anchor(v_feat, torch.stack([t_feat, a_feat], dim=1))
        raise ValueError(f"Unknown pathway: {pathway}")


class InvertedAsymmetricFusion(nn.Module):
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

    def attend_pathway_feature(self, t_feat, a_feat, v_feat):
        """Text is the only ATTEND pathway (Audio/Video are PURE)."""
        return self.text_anchor(t_feat, torch.stack([a_feat, v_feat], dim=1))


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019). X: (N, D1), Y: (N, D2) numpy arrays."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xty = X.T @ Y
    numerator = np.linalg.norm(xty, ord='fro') ** 2
    denom = np.linalg.norm(X.T @ X, ord='fro') * np.linalg.norm(Y.T @ Y, ord='fro')
    return float(numerator / denom) if denom > 0 else float('nan')


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════
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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — loop over config x fold x variant (baseline / distilled)
# ═══════════════════════════════════════════════════════════════════════════

splits_data = load_pickle(SPLITS_FILE)
with open(SARCASM_DATA_PATH, 'r') as f:
    sarcasm_data = json.load(f)
identifiers = list(sarcasm_data.keys())

VARIANTS = ['baseline', 'cascade']

print(f"\n{'='*80}")
print(f"CKA REPRESENTATIONAL SIMILARITY — MUStARD")
print(f"{'='*80}")

all_rows = []

for config_name, cfg in CONFIGS.items():
    print(f"\n{'-'*80}")
    print(f"CONFIG: {config_name} ({cfg['name']})  [Audio+Video=DOMINANT/PURE, Text=WEAK/ATTEND]")
    print(f"{'-'*80}")

    for fold in range(5):
        test_indices = splits_data[fold][1]
        X_text_test, X_audio_test, X_video_test, y_test, test_fnames = prepare_features(
            test_indices, cfg['text_type'], cfg['video_type'], sarcasm_data, identifiers)

        # Shared (never-distilled) backbones
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
            sym_ckpt = f"{CHECKPOINTS_DIR}/fusion_symmetric_{fusion_tag}_{config_name}_fold_{fold}.pth"
            iaf_ckpt = f"{CHECKPOINTS_DIR}/fusion_inverted_{fusion_tag}_{config_name}_fold_{fold}.pth"

            # Skip combos whose checkpoints haven't been trained yet
            missing = [p for p in (text_ckpt, sym_ckpt, iaf_ckpt) if not os.path.exists(p)]
            if missing:
                print(f"  [{variant}] Fold {fold}: SKIPPED — missing checkpoint(s): {missing}")
                continue

            if variant == 'baseline':
                text_model = TextBaseline(cfg['text_dim']).to(device)
            else:
                text_model = SimplifiedTextStudent(cfg['text_dim']).to(device)
            text_model.load_state_dict(torch.load(text_ckpt, map_location=device))

            t_feat, a_feat, v_feat = extract_backbone_features(
                text_model, audio_model, video_model, X_text_test, X_audio_test, X_video_test)
            t_feat, a_feat, v_feat = t_feat.to(device), a_feat.to(device), v_feat.to(device)
            raw_feat = {'text': t_feat, 'audio': a_feat, 'video': v_feat}

            sym_model = SymmetricFusion(text_model, audio_model, video_model).to(device)
            sym_model.load_state_dict(torch.load(sym_ckpt, map_location=device))
            sym_model.eval()

            iaf_model = InvertedAsymmetricFusion(text_model, audio_model, video_model).to(device)
            iaf_model.load_state_dict(torch.load(iaf_ckpt, map_location=device))
            iaf_model.eval()

            with torch.no_grad():
                # Cell 1: IAF PURE (audio, video) vs their own raw features — sanity, expect 1.0
                cka_iaf_pure_audio = linear_cka(raw_feat['audio'].cpu().numpy(), raw_feat['audio'].cpu().numpy())
                cka_iaf_pure_video = linear_cka(raw_feat['video'].cpu().numpy(), raw_feat['video'].cpu().numpy())

                # Cell 2: Symmetric audio, video (dominant) post-attention vs raw
                sym_audio_attended = sym_model.attended_feature(t_feat, a_feat, v_feat, 'audio')
                sym_video_attended = sym_model.attended_feature(t_feat, a_feat, v_feat, 'video')
                cka_sym_audio = linear_cka(sym_audio_attended.cpu().numpy(), raw_feat['audio'].cpu().numpy())
                cka_sym_video = linear_cka(sym_video_attended.cpu().numpy(), raw_feat['video'].cpu().numpy())

                # Cell 3: IAF ATTEND (text, weak) vs its own raw feature
                iaf_text_attended = iaf_model.attend_pathway_feature(t_feat, a_feat, v_feat)
                cka_iaf_text = linear_cka(iaf_text_attended.cpu().numpy(), raw_feat['text'].cpu().numpy())

                # Cell 4: Symmetric text (weak) post-attention vs its own raw feature
                sym_text_attended = sym_model.attended_feature(t_feat, a_feat, v_feat, 'text')
                cka_sym_text = linear_cka(sym_text_attended.cpu().numpy(), raw_feat['text'].cpu().numpy())

            print(f"  [{variant}] Fold {fold}: "
                  f"IAF_audio={cka_iaf_pure_audio:.4f} Sym_audio={cka_sym_audio:.4f}  "
                  f"IAF_video={cka_iaf_pure_video:.4f} Sym_video={cka_sym_video:.4f}  "
                  f"IAF_text={cka_iaf_text:.4f} Sym_text={cka_sym_text:.4f}")

            all_rows.append({
                'config': config_name, 'fold': fold, 'variant': variant,
                'cka_iaf_pure_audio': cka_iaf_pure_audio,
                'cka_symmetric_audio': cka_sym_audio,
                'cka_iaf_pure_video': cka_iaf_pure_video,
                'cka_symmetric_video': cka_sym_video,
                'cka_iaf_attend_text': cka_iaf_text,
                'cka_symmetric_text': cka_sym_text,
            })

rows_df = pd.DataFrame(all_rows)
rows_df.to_csv(f'{OUTPUT_DIR}/cka_mustard_per_fold.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATE (mean ± std across folds)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"AGGREGATE (mean ± std across 5 folds)")
print(f"{'='*80}")

metric_cols = ['cka_iaf_pure_audio', 'cka_symmetric_audio',
               'cka_iaf_pure_video', 'cka_symmetric_video',
               'cka_iaf_attend_text', 'cka_symmetric_text']
summary_rows = []

for config_name in CONFIGS:
    for variant in VARIANTS:
        sub = rows_df[(rows_df.config == config_name) & (rows_df.variant == variant)]
        if sub.empty:
            continue
        means = {c: sub[c].mean() for c in metric_cols}
        stds  = {c: sub[c].std()  for c in metric_cols}

        audio_drift = means['cka_iaf_pure_audio'] - means['cka_symmetric_audio']
        video_drift = means['cka_iaf_pure_video'] - means['cka_symmetric_video']

        print(f"\n{config_name} [{variant}]:")
        print(f"  Audio: IAF={means['cka_iaf_pure_audio']:.4f}±{stds['cka_iaf_pure_audio']:.4f}  "
              f"Sym={means['cka_symmetric_audio']:.4f}±{stds['cka_symmetric_audio']:.4f}  (drift={audio_drift:+.4f})")
        print(f"  Video: IAF={means['cka_iaf_pure_video']:.4f}±{stds['cka_iaf_pure_video']:.4f}  "
              f"Sym={means['cka_symmetric_video']:.4f}±{stds['cka_symmetric_video']:.4f}  (drift={video_drift:+.4f})")
        print(f"  Text : IAF={means['cka_iaf_attend_text']:.4f}±{stds['cka_iaf_attend_text']:.4f}  "
              f"Sym={means['cka_symmetric_text']:.4f}±{stds['cka_symmetric_text']:.4f}")

        row = {'config': config_name, 'variant': variant}
        for c in metric_cols:
            row[f'{c}_mean'] = means[c]
            row[f'{c}_std']  = stds[c]
        summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(f'{OUTPUT_DIR}/cka_mustard_summary.csv', index=False)
with open(f'{OUTPUT_DIR}/cka_mustard.json', 'w') as f:
    json.dump({'per_fold': all_rows, 'summary': summary_rows}, f, indent=4, default=str)

print(f"\nSaved per-fold rows to {OUTPUT_DIR}/cka_mustard_per_fold.csv")
print(f"Saved fold-aggregated summary to {OUTPUT_DIR}/cka_mustard_summary.csv")
print(f"{'='*80}\n")
