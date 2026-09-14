"""
CKA REPRESENTATIONAL SIMILARITY — MultiHUSE
===============================================================================
linear CKA (Kornblith et al. 2019) computed feature-wise (D=512 << N)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import copy
import json

SPLITS_DIR      = r'multihuse_dataset\splits_strict_v2'
FEATURES_DIR    = r'multihuse_dataset'
CHECKPOINTS_DIR = 'checkpoints'
OUTPUT_DIR      = 'cka_results_multihuse'
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
SEED         = 999

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================================================
# BASELINE MODEL DEFINITIONS
class TextModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES))
    def forward(self, x): return self.net(x)
    def get_features(self, x): return self.net[:4](x)


class BaselineAudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUDIO_DIM, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES))
    def forward(self, x): return self.net(x)
    def get_features(self, x): return self.net[:12](x)


class BaselineVideoModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES))
    def forward(self, x): return self.net(x)
    def get_features(self, x): return self.net[:7](x)


class CrossModalAnchor(nn.Module):
    def __init__(self, d_model=FUSION_DIM):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
    def forward(self, anchor, context):
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
    def __init__(self, text_model, audio_model, video_model):
        super().__init__()
        self.text_backbone  = text_model
        self.audio_backbone = audio_model
        self.video_backbone = video_model
        for bb in [self.text_backbone, self.audio_backbone, self.video_backbone]:
            for p in bb.parameters(): p.requires_grad = False

        self.audio_anchor = CrossModalAnchor(FUSION_DIM)
        self.video_anchor = CrossModalAnchor(FUSION_DIM)

        self.gate_net = nn.Sequential(
            nn.Linear(FUSION_DIM * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 3))

        self.text_classifier = copy.deepcopy(text_model.net[4])
        for p in self.text_classifier.parameters(): p.requires_grad = False
        self.audio_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)
        self.video_classifier = nn.Linear(FUSION_DIM, NUM_CLASSES)

    def attend_pathway_feature(self, t_feat, a_feat, v_feat, pathway):
        if pathway == 'audio':
            return self.audio_anchor(a_feat, torch.stack([t_feat, v_feat], dim=1))
        elif pathway == 'video':
            return self.video_anchor(v_feat, torch.stack([t_feat, a_feat], dim=1))
        raise ValueError(f"Unknown pathway: {pathway} (text is PURE, not attended)")


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019). X: (N, D1), Y: (N, D2) numpy arrays."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xty = X.T @ Y
    numerator = np.linalg.norm(xty, ord='fro') ** 2
    denom = np.linalg.norm(X.T @ X, ord='fro') * np.linalg.norm(Y.T @ Y, ord='fro')
    return float(numerator / denom) if denom > 0 else float('nan')


def load_fold_split(fold_num):
    train_df = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_train.csv')
    test_df  = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_test.csv')
    return train_df['file_name'].values, test_df['file_name'].values


def load_features(feature_file, file_names_subset, feat_prefix):
    df = pd.read_csv(f'{FEATURES_DIR}/{feature_file}')
    df = df[df['file_name'].isin(file_names_subset)].copy()
    df = df.sort_values('file_name').reset_index(drop=True)
    feat_cols = sorted([c for c in df.columns if c.startswith(feat_prefix)])
    return df[feat_cols].values.astype(np.float32), df[LABEL_COLUMN].values, df['file_name'].values


@torch.no_grad()
def extract_backbone_features(text_model, audio_model, video_model,
                               X_text, X_audio, X_video, batch_size=64):
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
    return torch.cat(t_feats, dim=0), torch.cat(a_feats, dim=0), torch.cat(v_feats, dim=0)


# ==========================================================================
# MAIN
print(f"\n{'='*80}")
print(f"CKA REPRESENTATIONAL SIMILARITY — MultiHUSE")
print(f"{'='*80}")

all_rows = []

for combo in COMBINATIONS:
    print(f"\n{'-'*80}")
    print(f"COMBINATION: {combo['name']}  (Text=STRONG/PURE, Audio+Video=WEAK/ATTEND)")
    print(f"{'-'*80}")

    for fold in range(1, 6):
        _, test_files = load_fold_split(fold)

        X_text_test,  _, _ = load_features(combo['text_features'], test_files, 'text_feat_')
        X_audio_test, _, _ = load_features('dasheng_audio_embeddings.csv', test_files, 'audio_feat_')
        X_video_test, _, _ = load_features(combo['video_features'], test_files, 'visual_feat_')

        txt_model = TextModel(combo['text_dim']).to(device)
        txt_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{combo['text_checkpoint']}_fold_{fold}.pth", map_location=device))
        aud_model = BaselineAudioModel().to(device)
        aud_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{combo['audio_checkpoint']}_fold_{fold}.pth", map_location=device))
        vid_model = BaselineVideoModel(combo['video_dim']).to(device)
        vid_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/{combo['video_checkpoint']}_fold_{fold}.pth", map_location=device))

        t_feat, a_feat, v_feat = extract_backbone_features(
            txt_model, aud_model, vid_model, X_text_test, X_audio_test, X_video_test)
        t_feat, a_feat, v_feat = t_feat.to(device), a_feat.to(device), v_feat.to(device)
        raw_feat = {'text': t_feat, 'audio': a_feat, 'video': v_feat}

        sym_model = SymmetricTriModalFusion(txt_model, aud_model, vid_model).to(device)
        sym_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/symmetric_baseline_{combo['name']}_fold_{fold}.pth", map_location=device))
        sym_model.eval()

        iaf_model = InvertedAsymmetricFusion(txt_model, aud_model, vid_model).to(device)
        iaf_model.load_state_dict(torch.load(
            f"{CHECKPOINTS_DIR}/inverted_asymmetric_baseline_{combo['name']}_fold_{fold}.pth", map_location=device))
        iaf_model.eval()

        with torch.no_grad():
            # Cell 1: IAF PURE (text) vs its own raw feature, expect 1.0
            cka_iaf_pure_text = linear_cka(raw_feat['text'].cpu().numpy(), raw_feat['text'].cpu().numpy())

            # Cell 2: Symmetric text (dominant) post-attention vs raw text
            sym_text_attended = sym_model.attended_feature(t_feat, a_feat, v_feat, 'text')
            cka_sym_text = linear_cka(sym_text_attended.cpu().numpy(), raw_feat['text'].cpu().numpy())

            # Cell 3: IAF ATTEND (audio, video) vs their own raw features
            iaf_audio = iaf_model.attend_pathway_feature(t_feat, a_feat, v_feat, 'audio')
            iaf_video = iaf_model.attend_pathway_feature(t_feat, a_feat, v_feat, 'video')
            cka_iaf_audio = linear_cka(iaf_audio.cpu().numpy(), raw_feat['audio'].cpu().numpy())
            cka_iaf_video = linear_cka(iaf_video.cpu().numpy(), raw_feat['video'].cpu().numpy())

            # Cell 4: Symmetric audio, video post-attention vs their own raw features
            sym_audio_attended = sym_model.attended_feature(t_feat, a_feat, v_feat, 'audio')
            sym_video_attended = sym_model.attended_feature(t_feat, a_feat, v_feat, 'video')
            cka_sym_audio = linear_cka(sym_audio_attended.cpu().numpy(), raw_feat['audio'].cpu().numpy())
            cka_sym_video = linear_cka(sym_video_attended.cpu().numpy(), raw_feat['video'].cpu().numpy())

        print(f"  Fold {fold}: IAF_pure_text={cka_iaf_pure_text:.4f}  Sym_text={cka_sym_text:.4f}  "
              f"IAF_audio={cka_iaf_audio:.4f}  Sym_audio={cka_sym_audio:.4f}  "
              f"IAF_video={cka_iaf_video:.4f}  Sym_video={cka_sym_video:.4f}")

        all_rows.append({
            'combination': combo['name'], 'fold': fold,
            'cka_iaf_pure_text': cka_iaf_pure_text,
            'cka_symmetric_text': cka_sym_text,
            'cka_iaf_audio': cka_iaf_audio,
            'cka_symmetric_audio': cka_sym_audio,
            'cka_iaf_video': cka_iaf_video,
            'cka_symmetric_video': cka_sym_video,
        })

rows_df = pd.DataFrame(all_rows)
rows_df.to_csv(f'{OUTPUT_DIR}/cka_multihuse_per_fold.csv', index=False)

# ==========================================================================
# AGGREGATE (mean ± std across folds)

print(f"\n{'='*80}")
print(f"AGGREGATE (mean ± std across 5 folds)")
print(f"{'='*80}")

summary_rows = []
metric_cols = ['cka_iaf_pure_text', 'cka_symmetric_text',
               'cka_iaf_audio', 'cka_symmetric_audio',
               'cka_iaf_video', 'cka_symmetric_video']

for combo in COMBINATIONS:
    name = combo['name']
    sub = rows_df[rows_df.combination == name]
    means = {c: sub[c].mean() for c in metric_cols}
    stds  = {c: sub[c].std()  for c in metric_cols}

    print(f"\n{name}:")
    text_drift = means['cka_iaf_pure_text'] - means['cka_symmetric_text']
    audio_drift = means['cka_iaf_audio'] - means['cka_symmetric_audio']
    video_drift = means['cka_iaf_video'] - means['cka_symmetric_video']
    print(f"  Text : IAF={means['cka_iaf_pure_text']:.4f}±{stds['cka_iaf_pure_text']:.4f}  "
          f"Sym={means['cka_symmetric_text']:.4f}±{stds['cka_symmetric_text']:.4f}  "
          f"(drift={text_drift:+.4f})")
    print(f"  Audio: IAF={means['cka_iaf_audio']:.4f}±{stds['cka_iaf_audio']:.4f}  "
          f"Sym={means['cka_symmetric_audio']:.4f}±{stds['cka_symmetric_audio']:.4f}  "
          f"(diff={audio_drift:+.4f})")
    print(f"  Video: IAF={means['cka_iaf_video']:.4f}±{stds['cka_iaf_video']:.4f}  "
          f"Sym={means['cka_symmetric_video']:.4f}±{stds['cka_symmetric_video']:.4f}  "
          f"(diff={video_drift:+.4f})")

    row = {'combination': name}
    for c in metric_cols:
        row[f'{c}_mean'] = means[c]
        row[f'{c}_std']  = stds[c]
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(f'{OUTPUT_DIR}/cka_multihuse_summary.csv', index=False)

with open(f'{OUTPUT_DIR}/cka_multihuse.json', 'w') as f:
    json.dump({'per_fold': all_rows, 'summary': summary_rows}, f, indent=4, default=str)

print(f"\nSaved per-fold rows to {OUTPUT_DIR}/cka_multihuse_per_fold.csv")
print(f"Saved fold-aggregated summary to {OUTPUT_DIR}/cka_multihuse_summary.csv")
print(f"{'='*80}\n")
