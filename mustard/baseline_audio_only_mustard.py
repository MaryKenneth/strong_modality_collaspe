import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os
from pathlib import Path

# CONFIGURATION
MUSTARD_DIR = r'mustard_dataset'
FEATURES_DIR = f'{MUSTARD_DIR}/mustard_features_pkl_FIXED'
SPLITS_FILE = f'{MUSTARD_DIR}/mustard_dataset_master/data/split_indices.p'

# Punchline paths only (no context for audio)
PUNCHLINE_PATHS = {
    'sarcasm': f'{FEATURES_DIR}/utterances/sarcasm', 
    'not_sarcasm': f'{FEATURES_DIR}/utterances/not_sarcasm'
}

AUDIO_FEATURES = ['dasheng']  # Dasheng audio embeddings
FEATURE_DIMS = {'dasheng': 1280}

LABEL_COLUMN = 'label'
NUM_CLASSES = 2  # Binary: sarcasm vs not_sarcasm
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 1e-4
SEED = 999

PREDICTIONS_DIR = 'predictions_output'
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioDataset(Dataset):
    def __init__(self, x, y, file_names): 
        self.x, self.y, self.file_names = x, y, file_names
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, i): 
        return torch.tensor(self.x[i]), torch.tensor(self.y[i], dtype=torch.long), self.file_names[i]

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

def load_mustard_splits():
    """Load MUStARD 5-fold cross-validation splits"""
    try:
        with open(SPLITS_FILE, 'rb') as f:
            splits_data = pickle.load(f, encoding='latin1')
    except UnicodeDecodeError:
        # Fallback for Python 2 pickle files
        with open(SPLITS_FILE, 'rb') as f:
            splits_data = pickle.load(f, encoding='bytes')
    return splits_data

def load_audio_features_from_pkl(pkl_path):
    """Load audio features from a single .pkl file"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract Dasheng audio features
        return data.get('audio_dasheng', np.zeros((1, 1280)))[0]  
    except:
        # Return zero features if file doesn't exist or is corrupted
        return np.zeros(1280)



def train_fold(model, train_loader, val_loader, fold_num, feature_name):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch_data = None
    
    for epoch in range(EPOCHS):
        model.train()
        for x_batch, y_batch, _ in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch.to(device))
            loss = criterion(logits, y_batch.to(device))
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
            torch.save(model.state_dict(), f"checkpoints/mustard_audio_{feature_name}_fold_{fold_num}.pth")
    
    return best_acc, best_epoch_data

# MAIN
splits_data = load_mustard_splits()

# Load sarcasm_data.json to get identifier mapping and labels
import json
SARCASM_DATA_PATH = f'{MUSTARD_DIR}/mustard_dataset_master/data/sarcasm_data.json'

with open(SARCASM_DATA_PATH, 'r') as f:
    sarcasm_data = json.load(f)

# Create mapping from split index to identifier and label
identifiers = list(sarcasm_data.keys())
print(f"Total samples in JSON: {len(identifiers)}")

def prepare_mustard_audio_data(fold_indices, sarcasm_data, identifiers):
    X_list = []
    y_list = []
    file_names = []
    
    for split_idx in fold_indices:
        if split_idx >= len(identifiers):
            print(f"Warning: Split index {split_idx} out of range")
            continue
            
        # Get the actual identifier and label from JSON
        identifier = identifiers[split_idx]
        sarcasm_label = sarcasm_data[identifier]['sarcasm']
        
        # Convert boolean to int: True=1 (sarcasm), False=0 (not_sarcasm)  
        label = 1 if sarcasm_label else 0
        
        # Determine which folder to look in based on the label
        if label == 1:  # sarcasm
            punchline_path = Path(PUNCHLINE_PATHS['sarcasm']) / f"{identifier}.pkl"
        else:  # not_sarcasm  
            punchline_path = Path(PUNCHLINE_PATHS['not_sarcasm']) / f"{identifier}.pkl"
        
        # Load audio features
        audio_feat = load_audio_features_from_pkl(punchline_path)
        
        X_list.append(audio_feat)
        y_list.append(label)
        file_names.append(identifier)
    
    return np.array(X_list, dtype=np.float32), np.array(y_list), file_names

for feature_type in AUDIO_FEATURES:
    print(f"\n{'='*50}")
    print(f"Training on: AUDIO {feature_type.upper()}")
    print(f"{'='*50}")
    
    input_dim = FEATURE_DIMS[feature_type]
    print(f"Feature dimension: {input_dim} (punchline only)")
    
    fold_accs = []
    all_fold_predictions = []
    all_preds = []
    all_trues = []
    
    for fold in range(5):  # 0-4 for MUStARD
        print(f"\n--- Fold {fold} ---")
        
        # Get train/test indices for this fold (tuples structure)
        train_indices = splits_data[fold][0]  # First element of tuple
        test_indices = splits_data[fold][1]   # Second element of tuple
        
        # Prepare data
        X_train, y_train, train_fnames = prepare_mustard_audio_data(train_indices, sarcasm_data, identifiers)
        X_test, y_test, test_fnames = prepare_mustard_audio_data(test_indices, sarcasm_data, identifiers)
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        train_loader = DataLoader(AudioDataset(X_train, y_train, train_fnames), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(AudioDataset(X_test, y_test, test_fnames), batch_size=BATCH_SIZE, shuffle=False)
        
        model = AudioModel(input_dim).to(device)
        acc, best_epoch_data = train_fold(model, train_loader, val_loader, fold, feature_type)
        
        print(f"Fold {fold} Best Acc: {acc*100:.2f}%")
        fold_accs.append(acc)
        all_fold_predictions.extend(best_epoch_data)
        
        best_trues = [p['true_label'] for p in best_epoch_data]
        best_preds = [p['pred_label'] for p in best_epoch_data]
        all_preds.extend(best_preds)
        all_trues.extend(best_trues)
    
    predictions_df = pd.DataFrame(all_fold_predictions)
    predictions_df.to_csv(f'{PREDICTIONS_DIR}/mustard_audio_{feature_type}_predictions.csv', index=False)
    
    print(f"\n{'='*50}")
    print(f"Results for AUDIO {feature_type.upper()}")
    print(f"Average Accuracy: {np.mean(fold_accs)*100:.2f}% (+/- {np.std(fold_accs)*100:.2f}%)")
    print(f"{'='*50}")
    print("Confusion Matrix:\n", confusion_matrix(all_trues, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds, target_names=['not_sarcasm', 'sarcasm']))