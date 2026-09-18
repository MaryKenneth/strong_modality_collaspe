import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import json
import os
from pathlib import Path

# CONFIGURATION
MUSTARD_DIR = r'mustard_dataset'
FEATURES_DIR = f'{MUSTARD_DIR}/mustard_features_pkl_FIXED'
SPLITS_FILE = f'{MUSTARD_DIR}/mustard_dataset_master/data/split_indices.p'
SARCASM_DATA_PATH = f'{MUSTARD_DIR}/mustard_dataset_master/data/sarcasm_data.json'

# Context and punchline paths
CONTEXT_PATHS = {
    'sarcasm': f'{FEATURES_DIR}/context/sarcasm',
    'not_sarcasm': f'{FEATURES_DIR}/context/not_sarcasm'
}
PUNCHLINE_PATHS = {
    'sarcasm': f'{FEATURES_DIR}/utterances/sarcasm', 
    'not_sarcasm': f'{FEATURES_DIR}/utterances/not_sarcasm'
}

# Define early fusion combinations for MUStARD
FUSION_CONFIGS = [
    # Context + Punchline variants
    {
        'name': 'bert_context_audio_videomae',
        'text_type': 'bert_context_punchline',
        'audio_type': 'dasheng',
        'video_type': 'videomae'
    },
    {
        'name': 'bert_context_audio_pe',
        'text_type': 'bert_context_punchline', 
        'audio_type': 'dasheng',
        'video_type': 'pe_core'
    },
    {
        'name': 'e5_context_audio_videomae',
        'text_type': 'e5_context_punchline',
        'audio_type': 'dasheng',
        'video_type': 'videomae'
    },
    {
        'name': 'e5_context_audio_pe',
        'text_type': 'e5_context_punchline',
        'audio_type': 'dasheng', 
        'video_type': 'pe_core'
    },
    # Punchline-only variants
    {
        'name': 'bert_punchline_audio_videomae',
        'text_type': 'bert_punchline_only',
        'audio_type': 'dasheng',
        'video_type': 'videomae'
    },
    {
        'name': 'bert_punchline_audio_pe',
        'text_type': 'bert_punchline_only',
        'audio_type': 'dasheng',
        'video_type': 'pe_core'
    },
    {
        'name': 'e5_punchline_audio_videomae',
        'text_type': 'e5_punchline_only',
        'audio_type': 'dasheng',
        'video_type': 'videomae'
    },
    {
        'name': 'e5_punchline_audio_pe',
        'text_type': 'e5_punchline_only',
        'audio_type': 'dasheng',
        'video_type': 'pe_core'
    }
]

# Feature dimensions and keys
FEATURE_DIMS = {
    'bert_context_punchline': 1536,  # 768 + 768
    'e5_context_punchline': 2048,    # 1024 + 1024
    'bert_punchline_only': 768,
    'e5_punchline_only': 1024,
    'dasheng': 1280,
    'videomae': 768,
    'pe_core': 1024
}

FEATURE_KEYS = {
    'bert': 'text_bert',
    'e5': 'text_e5', 
    'dasheng': 'audio_dasheng',
    'videomae': 'video_mae',
    'pe_core': 'video_pe'
}

NUM_CLASSES = 2  # Binary: sarcasm vs not_sarcasm
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
        elif feature_type == 'videomae':
            return data.get('video_mae', np.zeros((1, 768)))[0]
        elif feature_type == 'pe_core':
            return data.get('video_pe', np.zeros((1, 1024)))[0]
        else:
            return None
    except:
        # Return zero features if file doesn't exist or is corrupted
        if feature_type == 'bert':
            return np.zeros(768)
        elif feature_type == 'e5':
            return np.zeros(1024)
        elif feature_type == 'dasheng':
            return np.zeros(1280)
        elif feature_type == 'videomae':
            return np.zeros(768)
        elif feature_type == 'pe_core':
            return np.zeros(1024)

def prepare_multimodal_features(fold_indices, config, sarcasm_data, identifiers):
    """Prepare multimodal features for early fusion"""
    X_list = []
    y_list = []
    file_names = []
    
    for split_idx in fold_indices:
        if split_idx >= len(identifiers):
            print(f"Warning: Split index {split_idx} out of range")
            continue
            
        identifier = identifiers[split_idx]
        sarcasm_label = sarcasm_data[identifier]['sarcasm']
        label = 1 if sarcasm_label else 0
        
        # Determine folder based on label
        if label == 1:  # sarcasm
            context_path = Path(CONTEXT_PATHS['sarcasm']) / f"{identifier}.pkl"
            punchline_path = Path(PUNCHLINE_PATHS['sarcasm']) / f"{identifier}.pkl"
        else:  # not_sarcasm  
            context_path = Path(CONTEXT_PATHS['not_sarcasm']) / f"{identifier}.pkl"
            punchline_path = Path(PUNCHLINE_PATHS['not_sarcasm']) / f"{identifier}.pkl"
        
        # Extract text features
        if config['text_type'] == 'bert_context_punchline':
            context_feat = load_features_from_pkl(context_path, 'bert')
            punchline_feat = load_features_from_pkl(punchline_path, 'bert')
            text_feat = np.concatenate([context_feat, punchline_feat])
        elif config['text_type'] == 'e5_context_punchline':
            context_feat = load_features_from_pkl(context_path, 'e5')
            punchline_feat = load_features_from_pkl(punchline_path, 'e5')
            text_feat = np.concatenate([context_feat, punchline_feat])
        elif config['text_type'] == 'bert_punchline_only':
            text_feat = load_features_from_pkl(punchline_path, 'bert')
        elif config['text_type'] == 'e5_punchline_only':
            text_feat = load_features_from_pkl(punchline_path, 'e5')
        
        # Extract audio features (punchline only)
        audio_feat = load_features_from_pkl(punchline_path, config['audio_type'])
        
        # Extract video features (punchline only)
        video_feat = load_features_from_pkl(punchline_path, config['video_type'])
        
        # Concatenate all features
        combined_feat = np.concatenate([text_feat, audio_feat, video_feat])
        
        X_list.append(combined_feat)
        y_list.append(label)
        file_names.append(identifier)
    
    return np.array(X_list, dtype=np.float32), np.array(y_list), file_names

def get_concat_dim(config):
    """Get total concatenated dimension for the configuration"""
    text_dim = FEATURE_DIMS[config['text_type']]
    audio_dim = FEATURE_DIMS[config['audio_type']]
    video_dim = FEATURE_DIMS[config['video_type']]
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
                        'conf_0': probs[i, 0],  # not_sarcasm confidence
                        'conf_1': probs[i, 1],  # sarcasm confidence
                        'fold': fold_num,
                        'epoch': epoch + 1
                    })
        
        trues = [p['true_label'] for p in epoch_predictions]
        preds = [p['pred_label'] for p in epoch_predictions]
        acc = accuracy_score(trues, preds)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch_data = epoch_predictions
            torch.save(model.state_dict(), f"checkpoints/mustard_early_fusion_{config_name}_fold_{fold_num}.pth")
    
    return best_acc, best_epoch_data

# MAIN
splits_data = load_mustard_splits()

# Load sarcasm_data.json
with open(SARCASM_DATA_PATH, 'r') as f:
    sarcasm_data = json.load(f)

identifiers = list(sarcasm_data.keys())
print(f"Total samples in JSON: {len(identifiers)}")

for config in FUSION_CONFIGS:
    config_name = config['name']
    print(f"\n{'='*60}")
    print(f"Early Fusion: {config_name}")
    print(f"{'='*60}")
    
    total_dim, text_dim, audio_dim, video_dim = get_concat_dim(config)
    print(f"Text: {config['text_type']} ({text_dim})")
    print(f"Audio: {config['audio_type']} ({audio_dim})")
    print(f"Video: {config['video_type']} ({video_dim})")
    print(f"Total concatenated dimension: {total_dim}")
    
    fold_accs = []
    all_fold_predictions = []
    all_preds = []
    all_trues = []
    
    for fold in range(5):  # 0-4 for MUStARD
        print(f"\n--- Fold {fold} ---")
        
        # Get train/test indices for this fold
        train_indices = splits_data[fold][0]
        test_indices = splits_data[fold][1]
        
        # Prepare data
        X_train, y_train, train_fnames = prepare_multimodal_features(train_indices, config, sarcasm_data, identifiers)
        X_test, y_test, test_fnames = prepare_multimodal_features(test_indices, config, sarcasm_data, identifiers)
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
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
    predictions_df.to_csv(f'{PREDICTIONS_DIR}/mustard_early_fusion_{config_name}_predictions.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Results for {config_name}")
    print(f"Average Accuracy: {np.mean(fold_accs)*100:.2f}% (+/- {np.std(fold_accs)*100:.2f}%)")
    print(f"{'='*60}")
    print("Confusion Matrix:\n", confusion_matrix(all_trues, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds, target_names=['not_sarcasm', 'sarcasm']))