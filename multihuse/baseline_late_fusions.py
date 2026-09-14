import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# CONFIGURATION
SPLITS_DIR = r'multihuse_dataset\splits_strict_v2'
FEATURES_DIR = r'multihuse_dataset'
CHECKPOINTS_DIR = 'checkpoints'

# Define fusion combinations
FUSION_CONFIGS = [    
    {
        'name': 'text_bert_audio_video_pe',
        'text': 'bert_text_embeddings.csv',
        'audio': 'dasheng_audio_embeddings.csv',
        'video': 'pe_core_visual_embeddings.csv'
    },
    {
        'name': 'text_bert_audio_video_videomae',
        'text': 'bert_text_embeddings.csv',
        'audio': 'dasheng_audio_embeddings.csv',
        'video': 'videomae_visual_embeddings.csv'
    },
    {
        'name': 'text_e5_audio_video_pe',
        'text': 'e5_text_embeddings.csv',
        'audio': 'dasheng_audio_embeddings.csv',
        'video': 'pe_core_visual_embeddings.csv'
    },
    {
        'name': 'text_e5_audio_video_videomae',
        'text': 'e5_text_embeddings.csv',
        'audio': 'dasheng_audio_embeddings.csv',
        'video': 'videomae_visual_embeddings.csv'
    }
]

LABEL_COLUMN = 'humor_style'
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 999

PREDICTIONS_DIR = 'predictions_output'
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unimodal baseline model architectures
class TextModel(nn.Module):
    def __init__(self, input_dim):
        super(TextModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
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
    def __init__(self, input_dim):
        super(AudioModel, self).__init__()
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
        super(VideoModel, self).__init__()
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
    
    def forward(self, x):
        return self.net(x)
    
    def extract_features(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 6:
                return x
        return x

class FusionModel(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim):
        super(FusionModel, self).__init__()
        total_dim = text_dim + audio_dim + video_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )
    
    def forward(self, text_feat=None, audio_feat=None, video_feat=None):
        features = []
        if text_feat is not None:
            features.append(text_feat)
        if audio_feat is not None:
            features.append(audio_feat)
        if video_feat is not None:
            features.append(video_feat)
        
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)

class MultimodalDataset(Dataset):
    def __init__(self, text_data, audio_data, video_data, labels, file_names):
        self.text = text_data
        self.audio = audio_data
        self.video = video_data
        self.labels = labels
        self.file_names = file_names
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.text[idx], dtype=torch.float32) if self.text is not None else None
        audio_tensor = torch.tensor(self.audio[idx], dtype=torch.float32) if self.audio is not None else None
        video_tensor = torch.tensor(self.video[idx], dtype=torch.float32) if self.video is not None else None
        
        return text_tensor, audio_tensor, video_tensor, torch.tensor(self.labels[idx], dtype=torch.long), self.file_names[idx]

def custom_collate(batch):
    """Custom collate function that handles None values"""
    text_batch = []
    audio_batch = []
    video_batch = []
    labels = []
    fnames = []
    
    for text, audio, video, label, fname in batch:
        if text is not None:
            text_batch.append(text)
        if audio is not None:
            audio_batch.append(audio)
        if video is not None:
            video_batch.append(video)
        labels.append(label)
        fnames.append(fname)
    
    text_tensor = torch.stack(text_batch) if len(text_batch) > 0 else None
    audio_tensor = torch.stack(audio_batch) if len(audio_batch) > 0 else None
    video_tensor = torch.stack(video_batch) if len(video_batch) > 0 else None
    labels_tensor = torch.stack(labels)
    
    return text_tensor, audio_tensor, video_tensor, labels_tensor, fnames

def load_fold_split(fold_num):
    train_split = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_train.csv')
    test_split = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_test.csv')
    return train_split['file_name'].values, test_split['file_name'].values

def load_features(feature_file, file_names_subset, feat_prefix):
    if feature_file is None:
        return None
    
    df = pd.read_csv(f'{FEATURES_DIR}/{feature_file}')
    df_subset = df[df['file_name'].isin(file_names_subset)].copy()
    df_subset = df_subset.sort_values('file_name').reset_index(drop=True)
    
    feat_cols = sorted([c for c in df_subset.columns if c.startswith(feat_prefix)])
    X = df_subset[feat_cols].values.astype(np.float32)
    
    return X

def get_feature_dim(feature_file, feat_prefix):
    if feature_file is None:
        return 0
    df = pd.read_csv(f'{FEATURES_DIR}/{feature_file}')
    feat_cols = [c for c in df.columns if c.startswith(feat_prefix)]
    return len(feat_cols)

def load_unimodal_model(model_class, input_dim, checkpoint_path):
    if input_dim == 0:
        return None
    model = model_class(input_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def train_fusion(fusion_model, text_model, audio_model, video_model, 
                 train_loader, val_loader, fold_num, config_name):
    optimizer = optim.Adam(fusion_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch_data = None
    
    for epoch in range(EPOCHS):
        fusion_model.train()
        for text_b, audio_b, video_b, labels, _ in train_loader:
            labels = labels.to(device)
            
            with torch.no_grad():
                text_feat = text_model.extract_features(text_b.to(device)) if text_b is not None else None
                audio_feat = audio_model.extract_features(audio_b.to(device)) if audio_b is not None else None
                video_feat = video_model.extract_features(video_b.to(device)) if video_b is not None else None
            
            optimizer.zero_grad()
            outputs = fusion_model(text_feat, audio_feat, video_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        fusion_model.eval()
        epoch_predictions = []
        
        with torch.no_grad():
            for text_b, audio_b, video_b, labels, fnames in val_loader:
                text_feat = text_model.extract_features(text_b.to(device)) if text_b is not None else None
                audio_feat = audio_model.extract_features(audio_b.to(device)) if audio_b is not None else None
                video_feat = video_model.extract_features(video_b.to(device)) if video_b is not None else None
                
                outputs = fusion_model(text_feat, audio_feat, video_feat)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                for i, fname in enumerate(fnames):
                    epoch_predictions.append({
                        'file_name': fname,
                        'true_label': labels[i].item(),
                        'pred_label': preds[i],
                        'conf_0': probs[i, 0],
                        'conf_1': probs[i, 1],
                        'conf_2': probs[i, 2],
                        'conf_3': probs[i, 3],
                        'conf_4': probs[i, 4],
                        'fold': fold_num,
                        'epoch': epoch + 1
                    })
        
        trues = [p['true_label'] for p in epoch_predictions]
        preds = [p['pred_label'] for p in epoch_predictions]
        acc = accuracy_score(trues, preds)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch_data = epoch_predictions
            torch.save(fusion_model.state_dict(), f"{CHECKPOINTS_DIR}/fusion_{config_name}_fold_{fold_num}.pth")
    
    return best_acc, best_epoch_data

# MAIN
for config in FUSION_CONFIGS:
    config_name = config['name']
    print(f"\n{'='*60}")
    print(f"Fusion Configuration: {config_name}")
    print(f"{'='*60}")
    
    text_dim = get_feature_dim(config['text'], 'text_feat_') if config['text'] else 0
    audio_dim = get_feature_dim(config['audio'], 'audio_feat_') if config['audio'] else 0
    video_dim = get_feature_dim(config['video'], 'visual_feat_') if config['video'] else 0
    
    text_feat_dim = 512 if text_dim > 0 else 0
    audio_feat_dim = 512 if audio_dim > 0 else 0
    video_feat_dim = 512 if video_dim > 0 else 0
    
    print(f"Text: {config['text']} (dim: {text_dim})")
    print(f"Audio: {config['audio']} (dim: {audio_dim})")
    print(f"Video: {config['video']} (dim: {video_dim})")
    
    text_df = pd.read_csv(f'{FEATURES_DIR}/{config["text"]}') if config['text'] else None
    audio_df = pd.read_csv(f'{FEATURES_DIR}/{config["audio"]}') if config['audio'] else None
    video_df = pd.read_csv(f'{FEATURES_DIR}/{config["video"]}') if config['video'] else None
    
    fold_accs = []
    all_fold_predictions = []
    all_preds = []
    all_trues = []
    
    for fold in range(1, 6):
        print(f"\n--- Fold {fold} ---")
        train_files, test_files = load_fold_split(fold)
        
        text_model = None
        audio_model = None
        video_model = None
        
        if config['text']:
            text_name = config['text'].replace('.csv', '')
            checkpoint = f"{CHECKPOINTS_DIR}/text_{text_name}_fold_{fold}.pth"
            text_model = load_unimodal_model(TextModel, text_dim, checkpoint)
        
        if config['audio']:
            audio_name = config['audio'].replace('.csv', '')
            checkpoint = f"{CHECKPOINTS_DIR}/audio_{audio_name}_fold_{fold}.pth"
            audio_model = load_unimodal_model(AudioModel, audio_dim, checkpoint)
        
        if config['video']:
            video_name = config['video'].replace('.csv', '')
            checkpoint = f"{CHECKPOINTS_DIR}/video_{video_name}_fold_{fold}.pth"
            video_model = load_unimodal_model(VideoModel, video_dim, checkpoint)
        
        X_text_train = load_features(config['text'], train_files, 'text_feat_')
        X_audio_train = load_features(config['audio'], train_files, 'audio_feat_')
        X_video_train = load_features(config['video'], train_files, 'visual_feat_')
        
        X_text_test = load_features(config['text'], test_files, 'text_feat_')
        X_audio_test = load_features(config['audio'], test_files, 'audio_feat_')
        X_video_test = load_features(config['video'], test_files, 'visual_feat_')
        
        ref_df = text_df if text_df is not None else (audio_df if audio_df is not None else video_df)
        
        train_subset = ref_df[ref_df['file_name'].isin(train_files)].sort_values('file_name').reset_index(drop=True)
        test_subset = ref_df[ref_df['file_name'].isin(test_files)].sort_values('file_name').reset_index(drop=True)
        
        le = LabelEncoder()
        y_train = le.fit_transform(train_subset[LABEL_COLUMN].values)
        y_test = le.transform(test_subset[LABEL_COLUMN].values)
        
        train_fnames = train_subset['file_name'].values
        test_fnames = test_subset['file_name'].values
        
        train_ds = MultimodalDataset(X_text_train, X_audio_train, X_video_train, y_train, train_fnames)
        val_ds = MultimodalDataset(X_text_test, X_audio_test, X_video_test, y_test, test_fnames)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
        
        fusion_model = FusionModel(text_feat_dim, audio_feat_dim, video_feat_dim).to(device)
        
        acc, best_epoch_data = train_fusion(fusion_model, text_model, audio_model, video_model,
                                           train_loader, val_loader, fold, config_name)
        
        print(f"Fold {fold} Best Acc: {acc*100:.2f}%")
        fold_accs.append(acc)
        all_fold_predictions.extend(best_epoch_data)
        
        best_trues = [p['true_label'] for p in best_epoch_data]
        best_preds = [p['pred_label'] for p in best_epoch_data]
        all_preds.extend(best_preds)
        all_trues.extend(best_trues)
    
    predictions_df = pd.DataFrame(all_fold_predictions)
    predictions_df.to_csv(f'{PREDICTIONS_DIR}/fusion_{config_name}_predictions.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Results for {config_name}")
    print(f"Average Accuracy: {np.mean(fold_accs)*100:.2f}% (+/- {np.std(fold_accs)*100:.2f}%)")
    print(f"{'='*60}")
    print("Confusion Matrix:\n", confusion_matrix(all_trues, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds))