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

# Define early fusion combinations
FUSION_CONFIGS = [
    {
        'name': 'bert_audio_videomae',
        'text': 'bert_text_embeddings.csv',
        'audio': 'dasheng_audio_embeddings.csv',
        'video': 'videomae_visual_embeddings.csv'
    },
    {
        'name': 'bert_audio_pe',
        'text': 'bert_text_embeddings.csv',
        'audio': 'dasheng_audio_embeddings.csv',
        'video': 'pe_core_visual_embeddings.csv'
    },
    {
        'name': 'e5_audio_videomae',
        'text': 'e5_text_embeddings.csv',
        'audio': 'dasheng_audio_embeddings.csv',
        'video': 'videomae_visual_embeddings.csv'
    },
    {
        'name': 'e5_audio_pe',
        'text': 'e5_text_embeddings.csv',
        'audio': 'dasheng_audio_embeddings.csv',
        'video': 'pe_core_visual_embeddings.csv'
    }
]

LABEL_COLUMN = 'humor_style'
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
SEED = 999

PREDICTIONS_DIR = 'predictions_output'
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyFusionDataset(Dataset):
    def __init__(self, features, labels, file_names):
        self.features = features
        self.labels = labels
        self.file_names = file_names
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long), \
               self.file_names[idx]

class EarlyFusionModel(nn.Module):
    def __init__(self, input_dim):
        super(EarlyFusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, NUM_CLASSES)
        )
    
    def forward(self, x):
        return self.net(x)

def load_fold_split(fold_num):
    train_split = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_train.csv')
    test_split = pd.read_csv(f'{SPLITS_DIR}/fold_{fold_num}_test.csv')
    return train_split['file_name'].values, test_split['file_name'].values

def load_and_concatenate_features(text_file, audio_file, video_file, file_names_subset):
    # Load text features
    text_df = pd.read_csv(f'{FEATURES_DIR}/{text_file}')
    text_df = text_df[text_df['file_name'].isin(file_names_subset)].sort_values('file_name').reset_index(drop=True)
    text_cols = sorted([c for c in text_df.columns if c.startswith('text_feat_')])
    X_text = text_df[text_cols].values.astype(np.float32)
    
    # Load audio features
    audio_df = pd.read_csv(f'{FEATURES_DIR}/{audio_file}')
    audio_df = audio_df[audio_df['file_name'].isin(file_names_subset)].sort_values('file_name').reset_index(drop=True)
    audio_cols = sorted([c for c in audio_df.columns if c.startswith('audio_feat_')])
    X_audio = audio_df[audio_cols].values.astype(np.float32)
    
    # Load video features
    video_df = pd.read_csv(f'{FEATURES_DIR}/{video_file}')
    video_df = video_df[video_df['file_name'].isin(file_names_subset)].sort_values('file_name').reset_index(drop=True)
    video_cols = sorted([c for c in video_df.columns if c.startswith('visual_feat_')])
    X_video = video_df[video_cols].values.astype(np.float32)
    
    # Concatenate all features
    X_concat = np.concatenate([X_text, X_audio, X_video], axis=1)
    
    # Get labels and file names
    file_names = text_df['file_name'].values
    labels = text_df[LABEL_COLUMN].values
    
    return X_concat, labels, file_names

def get_concat_dim(text_file, audio_file, video_file):
    text_df = pd.read_csv(f'{FEATURES_DIR}/{text_file}')
    audio_df = pd.read_csv(f'{FEATURES_DIR}/{audio_file}')
    video_df = pd.read_csv(f'{FEATURES_DIR}/{video_file}')
    
    text_dim = len([c for c in text_df.columns if c.startswith('text_feat_')])
    audio_dim = len([c for c in audio_df.columns if c.startswith('audio_feat_')])
    video_dim = len([c for c in video_df.columns if c.startswith('visual_feat_')])
    
    return text_dim + audio_dim + video_dim, text_dim, audio_dim, video_dim

def train_fold(model, train_loader, val_loader, fold_num, config_name):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch_data = None
    
    for epoch in range(EPOCHS):
        model.train()
        for x_batch, y_batch, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch.to(device))
            loss = criterion(outputs, y_batch.to(device))
            loss.backward()
            optimizer.step()
        
        model.eval()
        epoch_predictions = []
        
        with torch.no_grad():
            for x_batch, y_batch, fnames in val_loader:
                outputs = model(x_batch.to(device))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                for i, fname in enumerate(fnames):
                    epoch_predictions.append({
                        'file_name': fname,
                        'true_label': y_batch[i].item(),
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
            torch.save(model.state_dict(), f"checkpoints/early_fusion_{config_name}_fold_{fold_num}.pth")
    
    return best_acc, best_epoch_data

# MAIN
for config in FUSION_CONFIGS:
    config_name = config['name']
    print(f"\n{'='*60}")
    print(f"Early Fusion: {config_name}")
    print(f"{'='*60}")
    
    total_dim, text_dim, audio_dim, video_dim = get_concat_dim(config['text'], config['audio'], config['video'])
    print(f"Text: {config['text']} ({text_dim})")
    print(f"Audio: {config['audio']} ({audio_dim})")
    print(f"Video: {config['video']} ({video_dim})")
    print(f"Total concatenated dimension: {total_dim}")
    
    fold_accs = []
    all_fold_predictions = []
    all_preds = []
    all_trues = []
    
    for fold in range(1, 6):
        print(f"\n--- Fold {fold} ---")
        train_files, test_files = load_fold_split(fold)
        
        # Load and concatenate features
        X_train, y_train_raw, train_fnames = load_and_concatenate_features(
            config['text'], config['audio'], config['video'], train_files
        )
        X_test, y_test_raw, test_fnames = load_and_concatenate_features(
            config['text'], config['audio'], config['video'], test_files
        )
        
        # Encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_test = le.transform(y_test_raw)
        
        # Create datasets
        train_ds = EarlyFusionDataset(X_train, y_train, train_fnames)
        val_ds = EarlyFusionDataset(X_test, y_test, test_fnames)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Create model
        model = EarlyFusionModel(total_dim).to(device)
        
        # Train
        acc, best_epoch_data = train_fold(model, train_loader, val_loader, fold, config_name)
        
        print(f"Fold {fold} Best Acc: {acc*100:.2f}%")
        fold_accs.append(acc)
        all_fold_predictions.extend(best_epoch_data)
        
        best_trues = [p['true_label'] for p in best_epoch_data]
        best_preds = [p['pred_label'] for p in best_epoch_data]
        all_preds.extend(best_preds)
        all_trues.extend(best_trues)
    
    # Save predictions
    predictions_df = pd.DataFrame(all_fold_predictions)
    predictions_df.to_csv(f'{PREDICTIONS_DIR}/early_fusion_{config_name}_predictions.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Results for {config_name}")
    print(f"Average Accuracy: {np.mean(fold_accs)*100:.2f}% (+/- {np.std(fold_accs)*100:.2f}%)")
    print(f"{'='*60}")
    print("Confusion Matrix:\n", confusion_matrix(all_trues, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds))